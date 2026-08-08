"""Quality metrics derived from face landmarks.

Deliberately free of mediapipe, opencv and I/O: everything here is plain numpy
over arrays and dicts that `analyze.py` hands in. That keeps the filtering rules
-- the part of this project most likely to need tuning and the part most worth
testing -- runnable without a model download or a GPU.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

# MediaPipe FaceLandmarker emits 478 points: 468 mesh points plus two 5-point
# iris rings. The ring centres are the most stable anchors on the whole face,
# which is why alignment keys off them rather than eye corners.
IRIS_CENTER_A = 468
IRIS_CENTER_B = 473
N_LANDMARKS = 478


@dataclass
class FaceMetrics:
    detected: int = 0
    yaw: float | None = None
    pitch: float | None = None
    roll: float | None = None
    gaze_x: float | None = None
    gaze_y: float | None = None
    blink_l: float | None = None
    blink_r: float | None = None
    oob_frac: float | None = None
    bbox_clipped: int | None = None
    interocular_px: float | None = None
    left_eye_x: float | None = None
    left_eye_y: float | None = None
    right_eye_x: float | None = None
    right_eye_y: float | None = None
    sharpness: float | None = None
    exposure_lo: float | None = None
    exposure_hi: float | None = None
    reject_reason: str | None = None
    score: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def euler_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """Decompose a facial transformation matrix into (yaw, pitch, roll) degrees.

    Accepts the 4x4 MediaPipe emits or a bare 3x3. Columns are normalised first
    because the matrix carries scale as well as rotation.

    Uses the Tait-Bryan decomposition of R = Rz(roll) @ Ry(yaw) @ Rx(pitch).
    Sign conventions follow MediaPipe's axes and are not worth agonising over:
    filtering thresholds are applied to absolute values, and alignment takes its
    roll from the iris positions directly rather than from this matrix.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape == (4, 4):
        m = m[:3, :3]
    if m.shape != (3, 3):
        raise ValueError(f"expected a 3x3 or 4x4 matrix, got {m.shape}")

    norms = np.linalg.norm(m, axis=0)
    if np.any(norms < 1e-12):
        raise ValueError("degenerate rotation matrix")
    r = m / norms

    sy = math.hypot(r[0, 0], r[1, 0])
    if sy < 1e-6:
        # Gimbal lock: roll and yaw are no longer separable.
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    else:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])

    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def gaze_from_blendshapes(bs: dict[str, float]) -> tuple[float, float]:
    """Signed gaze direction in roughly [-1, 1], independent of head pose.

    This is the metric that answers "is she looking at the camera", which head
    pose alone cannot: the face can point straight at the lens while the eyes
    are turned away.

    "In" and "out" are relative to each eye's own nose side, so a subject
    looking to one side produces `out` on one eye and `in` on the other. The
    two are combined so they reinforce instead of cancelling.
    """
    g = lambda k: float(bs.get(k, 0.0))  # noqa: E731
    horizontal = ((g("eyeLookOutLeft") - g("eyeLookInLeft"))
                  + (g("eyeLookInRight") - g("eyeLookOutRight"))) / 2.0
    vertical = ((g("eyeLookUpLeft") + g("eyeLookUpRight"))
                - (g("eyeLookDownLeft") + g("eyeLookDownRight"))) / 2.0
    return horizontal, vertical


def blink_from_blendshapes(bs: dict[str, float]) -> tuple[float, float]:
    return float(bs.get("eyeBlinkLeft", 0.0)), float(bs.get("eyeBlinkRight", 0.0))


def iris_centers(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the two iris centres ordered left-to-right *in image space*.

    Ordering by x rather than trusting the subject-relative index naming keeps
    the transform's roll sign correct and survives mirrored source images. A
    head rolled past vertical would defeat this, but such frames are rejected on
    pose long before alignment sees them.
    """
    a = np.asarray(landmarks[IRIS_CENTER_A], dtype=np.float64)[:2]
    b = np.asarray(landmarks[IRIS_CENTER_B], dtype=np.float64)[:2]
    return (a, b) if a[0] <= b[0] else (b, a)


def out_of_bounds_fraction(landmarks: np.ndarray, width: int, height: int,
                           inset: float = 0.0) -> float:
    """Fraction of landmarks falling outside the frame.

    This is the "partially visible face" detector. MediaPipe happily
    extrapolates the mesh past the edge of the image, so a face half out of
    frame still yields a full 478 points -- their positions are what give it
    away, not their absence.
    """
    pts = np.asarray(landmarks, dtype=np.float64)[:, :2]
    outside = (
        (pts[:, 0] < inset)
        | (pts[:, 1] < inset)
        | (pts[:, 0] > width - 1 - inset)
        | (pts[:, 1] > height - 1 - inset)
    )
    return float(np.count_nonzero(outside) / len(pts))


def bbox_is_clipped(bbox: tuple[int, int, int, int], width: int, height: int,
                    margin: int = 2) -> bool:
    """True when the face box touches a frame edge, i.e. the face is cut off."""
    x1, y1, x2, y2 = bbox
    return bool(x1 <= margin or y1 <= margin
                or x2 >= width - 1 - margin or y2 >= height - 1 - margin)


def laplacian_variance(gray: np.ndarray) -> float:
    """Blur metric: variance of the 3x3 Laplacian.

    Callers must pass a patch already resampled to a canonical size, otherwise
    the value tracks image resolution instead of focus and is not comparable
    between a phone shot and a DSLR frame.
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2 or min(g.shape) < 3:
        return 0.0
    lap = (-4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def exposure_percentiles(gray: np.ndarray) -> tuple[float, float]:
    g = np.asarray(gray, dtype=np.float64)
    if g.size == 0:
        return 0.0, 0.0
    lo, hi = np.percentile(g, [5.0, 95.0])
    return float(lo), float(hi)


def hard_reject(m: FaceMetrics, limits: dict) -> str | None:
    """First failing hard filter, or None if the frame is a keeper.

    Returns the *reason* rather than a bool so the rejects gallery can show why
    a frame was dropped -- which is how over-aggressive thresholds get spotted.
    """
    if not m.detected:
        return "no_face_detected"

    checks: list[tuple[bool, str]] = [
        (m.bbox_clipped and not limits.get("allow_bbox_clipped", False), "bbox_clipped"),
        (m.oob_frac is not None and m.oob_frac > limits["max_oob_frac"], "partially_out_of_frame"),
        (m.yaw is not None and abs(m.yaw) > limits["max_yaw"], "head_turned"),
        (m.pitch is not None and abs(m.pitch) > limits["max_pitch"], "head_tilted"),
        (m.roll is not None and abs(m.roll) > limits["max_roll"], "head_rolled"),
        (m.gaze_x is not None and abs(m.gaze_x) > limits["max_gaze"], "looking_away"),
        (m.gaze_y is not None and abs(m.gaze_y) > limits["max_gaze"], "looking_away"),
        (max(m.blink_l or 0.0, m.blink_r or 0.0) > limits["max_blink"], "eyes_closed"),
        (m.interocular_px is not None
         and m.interocular_px < limits["min_interocular_px"], "face_too_small"),
        (m.sharpness is not None and m.sharpness < limits["min_sharpness"], "blurry"),
        (m.exposure_hi is not None and m.exposure_hi > limits["max_exposure_hi"], "overexposed"),
        (m.exposure_lo is not None and m.exposure_lo < limits["min_exposure_lo"], "underexposed"),
    ]
    for failed, reason in checks:
        if failed:
            return reason
    return None


def composite_score(m: FaceMetrics, limits: dict, weights: dict) -> float:
    """Rank surviving frames so the best one per time bucket wins.

    Each term is normalised to roughly [0, 1] against the same threshold that
    would have rejected the frame, so weights are comparable to one another.
    """
    def unit(value: float | None, limit: float) -> float:
        if value is None or limit <= 0:
            return 0.0
        return max(0.0, 1.0 - abs(value) / limit)

    pose = (unit(m.yaw, limits["max_yaw"])
            + unit(m.pitch, limits["max_pitch"])
            + unit(m.roll, limits["max_roll"])) / 3.0
    gaze = (unit(m.gaze_x, limits["max_gaze"]) + unit(m.gaze_y, limits["max_gaze"])) / 2.0
    eyes_open = 1.0 - min(1.0, max(m.blink_l or 0.0, m.blink_r or 0.0))

    sharp_ref = max(limits["min_sharpness"], 1e-6) * 8.0
    sharpness = min(1.0, (m.sharpness or 0.0) / sharp_ref)

    size_ref = max(limits["min_interocular_px"], 1e-6) * 4.0
    size = min(1.0, (m.interocular_px or 0.0) / size_ref)

    terms = {
        "w_pose": pose,
        "w_gaze": gaze,
        "w_eyes_open": eyes_open,
        "w_sharpness": sharpness,
        "w_size": size,
    }
    total_weight = sum(float(weights.get(k, 0.0)) for k in terms)
    if total_weight <= 0:
        return 0.0
    return sum(float(weights.get(k, 0.0)) * v for k, v in terms.items()) / total_weight
