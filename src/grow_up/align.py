"""Eye alignment.

The transform maths is pure numpy and unit-tested; opencv is imported lazily and
only for the actual pixel warp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TransformParams:
    tx: float
    ty: float
    angle: float   # radians
    scale: float


def similarity_transform(left_eye: np.ndarray, right_eye: np.ndarray,
                         dst_left: np.ndarray, dst_right: np.ndarray) -> np.ndarray:
    """2x3 affine mapping the two source eyes exactly onto the target positions.

    A similarity transform -- uniform scale, rotation, translation, no shear --
    is exactly the right family here: it levels the eyes and normalises face
    size without distorting facial proportions, which are the very thing the
    timelapse is meant to show changing.
    """
    src_v = np.asarray(right_eye, dtype=np.float64) - np.asarray(left_eye, dtype=np.float64)
    dst_v = np.asarray(dst_right, dtype=np.float64) - np.asarray(dst_left, dtype=np.float64)

    src_len = float(np.hypot(*src_v))
    if src_len < 1e-9:
        raise ValueError("source eyes are coincident")

    scale = float(np.hypot(*dst_v)) / src_len
    angle = math.atan2(dst_v[1], dst_v[0]) - math.atan2(src_v[1], src_v[0])
    return build_affine(TransformParams(*_translation_for(left_eye, dst_left, angle, scale),
                                        angle=angle, scale=scale))


def _translation_for(src_point: np.ndarray, dst_point: np.ndarray,
                     angle: float, scale: float) -> tuple[float, float]:
    cos_a, sin_a = math.cos(angle) * scale, math.sin(angle) * scale
    sx, sy = float(src_point[0]), float(src_point[1])
    tx = float(dst_point[0]) - (cos_a * sx - sin_a * sy)
    ty = float(dst_point[1]) - (sin_a * sx + cos_a * sy)
    return tx, ty


def build_affine(p: TransformParams) -> np.ndarray:
    cos_a, sin_a = math.cos(p.angle) * p.scale, math.sin(p.angle) * p.scale
    return np.array([[cos_a, -sin_a, p.tx],
                     [sin_a, cos_a, p.ty]], dtype=np.float64)


def decompose_affine(matrix: np.ndarray) -> TransformParams:
    m = np.asarray(matrix, dtype=np.float64)
    scale = float(np.hypot(m[0, 0], m[1, 0]))
    angle = float(math.atan2(m[1, 0], m[0, 0]))
    return TransformParams(tx=float(m[0, 2]), ty=float(m[1, 2]), angle=angle, scale=scale)


def target_eyes(width: int, height: int, left: tuple[float, float],
                right: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([left[0] * width, left[1] * height], dtype=np.float64),
            np.array([right[0] * width, right[1] * height], dtype=np.float64))


def transforms_for(eye_pairs: list[tuple[np.ndarray, np.ndarray]],
                   dst_left: np.ndarray, dst_right: np.ndarray) -> list[np.ndarray]:
    """Per-frame transforms for a whole sequence.

    Each frame is solved independently and exactly, which *is* the
    stabilisation: both eyes land on the canonical positions every time.

    There is deliberately no smoothing across frames here. An earlier version
    averaged the (tx, ty, angle, scale) series along the timeline, reasoning
    that it would calm residual wobble. That was unsound: those parameters live
    in each source photo's own pixel coordinate system, so across a library of
    differing resolutions and face sizes tx alone ranged over thousands of
    pixels. Averaging them produced a translation belonging to no photo, and
    pushed the face clean out of frame.

    Nor would a corrected version buy much: what remains after exact eye
    alignment is genuine head pose and expression change, which no similarity
    transform can smooth away -- only unpin the eyes while trying.
    """
    return [similarity_transform(left, right, dst_left, dst_right)
            for left, right in eye_pairs]


def warp(image: np.ndarray, matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    """Apply the transform. Lanczos, because these frames get upscaled."""
    import cv2

    return cv2.warpAffine(
        image, matrix, (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )


def match_luma(frame: np.ndarray, reference_median: float) -> np.ndarray:
    """Damp exposure flicker by nudging each frame toward a rolling reference.

    A gain rather than a full histogram match: it removes the flicker between
    consecutive frames without touching colour or crushing highlights.
    """
    import cv2

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    current = float(np.median(lab[:, :, 0]))
    if current <= 1e-6:
        return frame
    gain = np.clip(reference_median / current, 0.85, 1.18)
    lab[:, :, 0] = np.clip(lab[:, :, 0].astype(np.float64) * gain, 0, 255).astype(lab.dtype)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
