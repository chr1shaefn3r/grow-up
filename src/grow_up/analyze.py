"""Landmarking and quality scoring -- the core of the filter.

Runs one MediaPipe FaceLandmarker per worker process. The heavy work (JPEG
decode, inference, resampling) is all native and releases the GIL, so a process
pool over physical cores saturates the machine; the Python here is glue.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import images, metrics
from .immich import AspectMismatch, Face, scale_bbox
from .metrics import FaceMetrics

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path("models/face_landmarker.task")

# Sharpness is measured on the eye region resampled to this width so the value
# is comparable across source resolutions.
SHARPNESS_PATCH_WIDTH = 160

_LANDMARKER = None
_OPTS: "AnalyzeOptions | None" = None


@dataclass(frozen=True)
class AnalyzeOptions:
    model_path: str = str(DEFAULT_MODEL_PATH)
    bbox_margin: float = 0.8
    min_face_detection_confidence: float = 0.5
    oob_inset: float = 0.0


def physical_cores() -> int:
    """Physical cores, not hyperthreads.

    Inference is compute-bound, so scheduling two workers per physical core just
    adds contention.
    """
    try:
        count = os.cpu_count() or 1
        if hasattr(os, "sched_getaffinity"):
            count = len(os.sched_getaffinity(0))
    except OSError:
        count = 1
    return max(1, count // 2) if count > 2 else max(1, count)


def init_worker(opts: AnalyzeOptions) -> None:
    """Process-pool initializer: build one landmarker per worker."""
    global _LANDMARKER, _OPTS
    _OPTS = opts

    # Each worker gets one core; letting every library spin up its own thread
    # pool on top of the process pool is a net loss.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:
        pass

    _LANDMARKER = build_landmarker(opts)


def build_landmarker(opts: AnalyzeOptions):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = Path(opts.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"face landmarker model not found at {model_path}. "
            f"Run `grow-up fetch-model` or download {MODEL_URL}"
        )

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=opts.min_face_detection_confidence,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


def analyze_one(job: tuple[str, str, dict]) -> tuple[str, FaceMetrics]:
    """Worker entry point. Returns (asset_id, metrics); never raises for one bad file."""
    asset_id, path, face_row = job
    assert _LANDMARKER is not None and _OPTS is not None, "worker not initialized"
    try:
        result = analyze_image(_LANDMARKER, _OPTS, Path(path), face_row)
    except (AspectMismatch, FileNotFoundError, ValueError, OSError) as exc:
        result = FaceMetrics(detected=0, reject_reason=f"error:{type(exc).__name__}")
    return asset_id, result


def analyze_image(landmarker, opts: AnalyzeOptions, path: Path,
                  face_row: dict) -> FaceMetrics:
    import mediapipe as mp

    bgr = images.load_bgr(path)
    height, width = bgr.shape[:2]

    face = Face(
        x1=face_row["x1"], y1=face_row["y1"], x2=face_row["x2"], y2=face_row["y2"],
        image_width=face_row["image_width"], image_height=face_row["image_height"],
        source_type=face_row.get("source_type"),
    )
    box = scale_bbox(face, width, height)

    crop_box = images.expand_box(box, opts.bbox_margin, width, height)
    cx1, cy1, cx2, cy2 = crop_box
    if cx2 - cx1 < 16 or cy2 - cy1 < 16:
        return FaceMetrics(detected=0, reject_reason="face_too_small")

    crop = bgr[cy1:cy2, cx1:cx2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=images.to_rgb(crop))
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return FaceMetrics(
            detected=0,
            bbox_clipped=int(metrics.bbox_is_clipped(box, width, height)),
            reject_reason="no_face_detected",
        )

    crop_w, crop_h = cx2 - cx1, cy2 - cy1
    # Normalized landmarks are relative to the crop and may fall outside [0, 1]
    # where the mesh extrapolates past it. Mapping to full-image pixels keeps
    # that information, which is exactly what the out-of-frame check needs.
    pts = np.array(
        [[cx1 + lm.x * crop_w, cy1 + lm.y * crop_h] for lm in result.face_landmarks[0]],
        dtype=np.float64,
    )

    left_eye, right_eye = metrics.iris_centers(pts)
    interocular = float(np.hypot(*(right_eye - left_eye)))

    m = FaceMetrics(
        detected=1,
        oob_frac=metrics.out_of_bounds_fraction(pts, width, height, opts.oob_inset),
        bbox_clipped=int(metrics.bbox_is_clipped(box, width, height)),
        interocular_px=interocular,
        left_eye_x=float(left_eye[0]), left_eye_y=float(left_eye[1]),
        right_eye_x=float(right_eye[0]), right_eye_y=float(right_eye[1]),
    )

    if result.facial_transformation_matrixes:
        m.yaw, m.pitch, m.roll = metrics.euler_from_matrix(
            np.asarray(result.facial_transformation_matrixes[0])
        )

    if result.face_blendshapes:
        bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
        m.gaze_x, m.gaze_y = metrics.gaze_from_blendshapes(bs)
        m.blink_l, m.blink_r = metrics.blink_from_blendshapes(bs)

    gray_face = images.to_gray(bgr[cy1:cy2, cx1:cx2])
    m.exposure_lo, m.exposure_hi = metrics.exposure_percentiles(gray_face)
    m.sharpness = _eye_sharpness(bgr, left_eye, right_eye, interocular)
    return m


def _eye_sharpness(bgr: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray,
                   interocular: float) -> float:
    """Focus measured on the eye band specifically.

    A whole-face measure is dominated by background and hair texture; what
    matters for this timelapse is whether the eyes are sharp, since they are the
    alignment anchor and where the viewer looks.
    """
    height, width = bgr.shape[:2]
    pad_x = max(8.0, interocular * 0.6)
    pad_y = max(8.0, interocular * 0.35)
    x1 = int(max(0, min(left_eye[0], right_eye[0]) - pad_x))
    x2 = int(min(width, max(left_eye[0], right_eye[0]) + pad_x))
    y1 = int(max(0, min(left_eye[1], right_eye[1]) - pad_y))
    y2 = int(min(height, max(left_eye[1], right_eye[1]) + pad_y))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0.0

    gray = images.to_gray(bgr[y1:y2, x1:x2])
    try:
        gray = images.resize_gray(gray, SHARPNESS_PATCH_WIDTH)
    except ImportError:
        pass
    return metrics.laplacian_variance(gray)
