"""Landmarking and quality scoring -- the core of the filter.

Runs one MediaPipe FaceLandmarker per worker process. The heavy work (JPEG
decode, inference, resampling) is all native and releases the GIL, so a process
pool over physical cores saturates the machine; the Python here is glue.
"""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
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


# MediaPipe's graphs, TFLite and the GL context all log through C++ (glog/absl),
# writing to file descriptor 2 directly. Python's logging module cannot filter
# any of it, and these have to be set before the native libraries initialize --
# which is why workers set them ahead of importing mediapipe.
QUIET_ENV = {
    "GLOG_minloglevel": "2",        # ERROR and above only
    "GLOG_stderrthreshold": "3",
    "GLOG_logtostderr": "0",
    "ABSL_MIN_LOG_LEVEL": "2",
    "TF_CPP_MIN_LOG_LEVEL": "3",
}


@contextmanager
def suppress_native_output(enabled: bool = True):
    """Silence writes to stdout/stderr at the file-descriptor level.

    The env vars above quieten most of it, but a few messages -- notably
    TFLite's XNNPACK delegate banner -- are written to the descriptor
    regardless. Only descriptor redirection reliably catches those.

    Safe inside worker processes specifically: they never print anything of
    their own, returning results to the parent instead, so nothing worth seeing
    is lost. Failures still surface, because `analyze_one` reports them through
    its return value rather than by writing to stderr.
    """
    if not enabled:
        yield
        return

    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        for fd in (*saved, devnull):
            os.close(fd)


@dataclass(frozen=True)
class AnalyzeOptions:
    model_path: str = str(DEFAULT_MODEL_PATH)
    bbox_margin: float = 0.8
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    oob_inset: float = 0.0
    verbose: bool = False

    # -- time/accuracy dial (see EFFORT_PRESETS) --------------------------
    # Name of the preset the fields below came from, for reporting only.
    effort: str = "fast"
    # Extra crops to try when the first look finds nothing. Costs time only on
    # photos that already failed.
    retry_margins: tuple[float, ...] = ()
    # Degrees to rotate a failing crop by before retrying. BlazeFace is trained
    # on upright faces, so a strongly rolled head can defeat it outright.
    retry_rotations: tuple[float, ...] = ()
    # Retry a failure on a contrast-equalised copy, for dark and backlit shots.
    retry_equalize: bool = False
    # Number of jittered crops to combine per face. MediaPipe is deterministic
    # on a fixed crop, so varying the framing is the only way to sample its
    # error; the median of several looks is steadier than any one.
    ensemble: int = 1
    # Downscale crops longer than this before inference. 0 disables.
    max_crop_px: int = 0
    gaze_method: str = "blendshapes"


# The dial itself. `fast` is deliberately identical to the original behaviour:
# switching level should be the only thing that moves the numbers.
EFFORT_PRESETS: dict[str, dict] = {
    "fast": {
        "retry_margins": (),
        "retry_rotations": (),
        "retry_equalize": False,
        "ensemble": 1,
        "max_crop_px": 0,
    },
    "balanced": {
        "retry_margins": (1.6, 0.4),
        "retry_rotations": (),
        "retry_equalize": True,
        "ensemble": 1,
        "max_crop_px": 1024,
    },
    "thorough": {
        "retry_margins": (1.6, 0.4, 2.5),
        "retry_rotations": (-25.0, 25.0),
        "retry_equalize": True,
        "ensemble": 3,
        "max_crop_px": 1024,
    },
}

EFFORT_LEVELS = tuple(EFFORT_PRESETS)


def preset_for(effort: str) -> dict:
    """Settings for an effort level.

    Fails loudly on an unknown name rather than silently falling back: a typo in
    config.toml that quietly ran at a different level than intended would be a
    poor way to learn what the dial does.
    """
    try:
        return dict(EFFORT_PRESETS[effort])
    except KeyError:
        raise ValueError(
            f"unknown analyze.effort {effort!r}; expected one of {', '.join(EFFORT_LEVELS)}"
        ) from None


def available_cpus() -> int:
    """CPUs this process is actually allowed to run on.

    Respects affinity masks and container limits, which `os.cpu_count()` ignores.
    """
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _macos_physical_cores() -> int | None:
    try:
        result = subprocess.run(["sysctl", "-n", "hw.physicalcpu"],
                                capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        return int(result.stdout.strip()) or None
    except ValueError:
        return None


def _linux_physical_cores(sysfs: Path = Path("/sys/devices/system/cpu")) -> int | None:
    """Count distinct (package, core) pairs, so hyperthread siblings collapse."""
    pairs: set[tuple[str, str]] = set()
    try:
        cpu_dirs = sorted(sysfs.glob("cpu[0-9]*"))
    except OSError:
        return None
    for cpu_dir in cpu_dirs:
        topology = cpu_dir / "topology"
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError:
            continue
        pairs.add((package, core))
    return len(pairs) or None


def physical_cores() -> int:
    """Physical cores available to this process.

    Inference is compute-bound, so two workers per physical core mostly adds
    contention and memory pressure -- each worker holds its own landmarker.

    This has to be *detected*, not inferred by halving the logical count. That
    shortcut assumes simultaneous multithreading, which Apple Silicon does not
    have: an M1 Pro reports 8 logical and 8 physical cores (6 performance, 2
    efficiency), so halving would waste half the machine. The same is true of
    any x86 part with hyperthreading disabled.
    """
    detected = (_macos_physical_cores() if sys.platform == "darwin"
                else _linux_physical_cores())
    allowed = available_cpus()
    if detected:
        # Affinity still wins: a container pinned to fewer CPUs than the host
        # has physical cores must not oversubscribe them.
        return max(1, min(detected, allowed))
    return allowed


def init_worker(opts: AnalyzeOptions) -> None:
    """Process-pool initializer: build one landmarker per worker."""
    global _LANDMARKER, _OPTS
    _OPTS = opts

    # Each worker gets one core; letting every library spin up its own thread
    # pool on top of the process pool is a net loss.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    # Must happen before mediapipe is imported, hence here rather than at module
    # scope: the native log level is read once, when the library initializes.
    if not opts.verbose:
        os.environ.update(QUIET_ENV)

    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:
        pass

    with suppress_native_output(not opts.verbose):
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
        min_face_presence_confidence=opts.min_face_presence_confidence,
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


@dataclass(frozen=True)
class Attempt:
    """One framing of the face to hand the model."""

    margin: float
    equalize: bool = False
    rotation: float = 0.0

    def describe(self) -> str:
        parts = [f"margin={self.margin:g}"]
        if self.equalize:
            parts.append("equalized")
        if self.rotation:
            parts.append(f"rot={self.rotation:g}")
        return " ".join(parts)


@dataclass(frozen=True)
class Look:
    """A successful detection, already mapped into full-image coordinates."""

    points: np.ndarray
    blendshapes: dict[str, float]
    matrix: np.ndarray | None
    attempt: Attempt


def ensemble_margins(opts: AnalyzeOptions) -> tuple[float, ...]:
    """Crop margins for the ensemble, base first then alternating either side.

    Deterministic rather than random: re-running analysis must reproduce the
    same metrics, or the manifest stops being a reliable cache.
    """
    count = max(1, int(opts.ensemble))
    base = float(opts.bbox_margin)
    if count <= 1:
        return (base,)

    factors = [1.0]
    while len(factors) < count:
        index = len(factors)
        delta = 0.15 * ((index + 1) // 2)
        factors.append(1.0 - delta if index % 2 else 1.0 + delta)
    return tuple(round(base * factor, 4) for factor in factors[:count])


def attempt_ladder(opts: AnalyzeOptions) -> list[Attempt]:
    """Framings to try, in order, until enough of them succeed.

    The ensemble margins come first so a healthy photo is done after one look at
    `fast`. The retries below them only ever run when those come back empty, so
    they cost nothing on the photos that already worked.
    """
    base = float(opts.bbox_margin)
    ladder = [Attempt(margin=margin) for margin in ensemble_margins(opts)]
    ladder += [Attempt(margin=float(margin)) for margin in opts.retry_margins]
    if opts.retry_equalize:
        ladder.append(Attempt(margin=base, equalize=True))
    ladder += [Attempt(margin=base, rotation=float(deg)) for deg in opts.retry_rotations]
    return ladder


def detect_once(landmarker, opts: AnalyzeOptions, bgr: np.ndarray,
                box: tuple[int, int, int, int], attempt: Attempt) -> Look | None:
    """One inference on one framing, or None if nothing was found."""
    import mediapipe as mp

    height, width = bgr.shape[:2]
    cx1, cy1, cx2, cy2 = images.expand_box(box, attempt.margin, width, height)
    if cx2 - cx1 < 16 or cy2 - cy1 < 16:
        return None

    work = bgr[cy1:cy2, cx1:cx2]
    if attempt.equalize:
        work = images.equalize(work)

    work, scale = images.downscale_to(work, opts.max_crop_px)

    rotation_matrix = None
    if attempt.rotation:
        work, rotation_matrix = images.rotate(work, attempt.rotation)

    work_h, work_w = work.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=images.to_rgb(work))
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None

    # Normalized landmarks may fall outside [0, 1] where the mesh extrapolates
    # past the crop. Keeping those values is the whole basis of the
    # out-of-frame check, so nothing is clamped on the way back.
    pts = np.array([[lm.x * work_w, lm.y * work_h] for lm in result.face_landmarks[0]],
                   dtype=np.float64)
    if rotation_matrix is not None:
        pts = images.unrotate_points(pts, rotation_matrix)
    pts *= scale
    pts += np.array([cx1, cy1], dtype=np.float64)

    blendshapes = ({c.category_name: float(c.score) for c in result.face_blendshapes[0]}
                   if result.face_blendshapes else {})
    matrix = (np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64)
              if result.facial_transformation_matrixes else None)
    return Look(points=pts, blendshapes=blendshapes, matrix=matrix, attempt=attempt)


def gather_looks(landmarker, opts: AnalyzeOptions, bgr: np.ndarray,
                 box: tuple[int, int, int, int]) -> list[Look]:
    """Walk the ladder until `ensemble` looks succeed, or it runs out."""
    wanted = max(1, int(opts.ensemble))
    looks: list[Look] = []
    for attempt in attempt_ladder(opts):
        look = detect_once(landmarker, opts, bgr, box, attempt)
        if look is not None:
            looks.append(look)
            if len(looks) >= wanted:
                break
    return looks


def combine_looks(looks: list[Look]) -> tuple[np.ndarray, dict[str, float],
                                              tuple[float, float, float] | None]:
    """Median across looks: landmarks, blendshape scores and pose angles.

    The median rather than the mean, so a single bad framing that throws the
    mesh cannot drag the result with it.

    Pose is combined after decomposing to Euler angles, not by averaging the
    matrices -- the mean of rotation matrices is not itself a rotation.
    """
    points = np.median(np.stack([look.points for look in looks]), axis=0)

    names: set[str] = set()
    for look in looks:
        names.update(look.blendshapes)
    blendshapes = {
        name: float(np.median([look.blendshapes.get(name, 0.0) for look in looks]))
        for name in names
    }

    eulers = [metrics.euler_from_matrix(look.matrix)
              for look in looks if look.matrix is not None]
    pose = (tuple(float(np.median([e[axis] for e in eulers])) for axis in range(3))
            if eulers else None)
    return points, blendshapes, pose


def analyze_image(landmarker, opts: AnalyzeOptions, path: Path,
                  face_row: dict) -> FaceMetrics:
    bgr = images.load_bgr(path)
    height, width = bgr.shape[:2]

    face = Face(
        x1=face_row["x1"], y1=face_row["y1"], x2=face_row["x2"], y2=face_row["y2"],
        image_width=face_row["image_width"], image_height=face_row["image_height"],
        source_type=face_row.get("source_type"),
    )
    box = scale_bbox(face, width, height)
    clipped = int(metrics.bbox_is_clipped(box, width, height))

    looks = gather_looks(landmarker, opts, bgr, box)
    if not looks:
        return FaceMetrics(detected=0, bbox_clipped=clipped,
                           reject_reason="no_face_detected")

    pts, blendshapes, pose = combine_looks(looks)
    left_eye, right_eye = metrics.iris_centers(pts)
    interocular = float(np.hypot(*(right_eye - left_eye)))

    m = FaceMetrics(
        detected=1,
        oob_frac=metrics.out_of_bounds_fraction(pts, width, height, opts.oob_inset),
        bbox_clipped=clipped,
        interocular_px=interocular,
        left_eye_x=float(left_eye[0]), left_eye_y=float(left_eye[1]),
        right_eye_x=float(right_eye[0]), right_eye_y=float(right_eye[1]),
    )

    m.span_w, m.span_up, m.span_down = metrics.face_spans(
        pts, left_eye, right_eye, interocular)

    if pose is not None:
        m.yaw, m.pitch, m.roll = pose

    if blendshapes:
        m.blink_l, m.blink_r = metrics.blink_from_blendshapes(blendshapes)
    m.gaze_x, m.gaze_y = metrics.gaze(pts, blendshapes, opts.gaze_method)

    cx1, cy1, cx2, cy2 = images.expand_box(box, opts.bbox_margin, width, height)
    m.exposure_lo, m.exposure_hi = metrics.exposure_percentiles(
        images.to_gray(bgr[cy1:cy2, cx1:cx2]))
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
