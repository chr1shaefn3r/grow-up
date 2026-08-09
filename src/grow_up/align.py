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


def target_eyes_from(width: int, height: int, eye_distance: float,
                     eye_level: float) -> tuple[np.ndarray, np.ndarray]:
    """Canonical eye positions from a separation and a height.

    `eye_distance` is what actually controls how much of the frame the face
    occupies, and so how much room is left around it: a head is roughly 2.4-3.0x
    the interocular distance across, so 0.20 puts it at about half the frame
    width. Expressing it this way means "zoom out" is one number, rather than
    two coordinates that have to stay symmetric about the centre.
    """
    if not 0.0 < eye_distance < 1.0:
        raise ValueError(
            f"align.eye_distance must be between 0 and 1, got {eye_distance!r}")
    if not 0.0 < eye_level < 1.0:
        raise ValueError(f"align.eye_level must be between 0 and 1, got {eye_level!r}")

    half = eye_distance / 2.0
    return target_eyes(width, height, (0.5 - half, eye_level), (0.5 + half, eye_level))


def head_fits(eye_distance: float, eye_level: float, span_w: float,
              span_up: float, span_down: float, aspect: float,
              margin: float = 1.0) -> bool:
    """Whether a face of these proportions stays inside the frame.

    Spans are measured in units of interocular distance, which is exactly what
    the transform normalises, so this holds regardless of the source photo's
    resolution or how close the camera was.

    `aspect` is width/height of the output frame: a horizontal span is a
    fraction of the width, a vertical one of the height, and the two only relate
    through the frame's shape.
    """
    half_w = eye_distance * span_w * margin / 2.0
    if half_w > 0.5:
        return False
    # Vertical extents scale by the same pixels-per-interocular factor, then
    # convert to a fraction of height via the aspect ratio.
    per_eye_height = eye_distance * aspect * margin
    return (eye_level - span_up * per_eye_height >= 0.0
            and eye_level + span_down * per_eye_height <= 1.0)


def fitting_eye_distance(span_w: float, span_up: float, span_down: float,
                         eye_level: float, aspect: float,
                         margin: float = 1.0) -> float:
    """Largest `eye_distance` at which a face of these proportions still fits."""
    limits = []
    if span_w > 0:
        limits.append(1.0 / (span_w * margin))
    if span_up > 0:
        limits.append(eye_level / (span_up * aspect * margin))
    if span_down > 0:
        limits.append((1.0 - eye_level) / (span_down * aspect * margin))
    return min(limits) if limits else 1.0


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


FILL_MODES = ("blur", "edge", "black")


def source_footprint(image: np.ndarray, matrix: np.ndarray, width: int,
                     height: int) -> np.ndarray:
    """Mask of the output frame that real source pixels actually reach.

    Warping a white image with a zero border says exactly where the photo ends,
    which is what makes it possible to treat the rest differently instead of
    smearing edge pixels across it.
    """
    import cv2

    solid = np.full(image.shape[:2], 255, dtype=np.uint8)
    return cv2.warpAffine(solid, matrix, (width, height), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def warp(image: np.ndarray, matrix: np.ndarray, width: int, height: int,
         fill: str = "edge") -> np.ndarray:
    """Apply the transform. Lanczos, because these frames get upscaled.

    Framing the face loosely means the output often reaches past the edge of the
    source photo. `edge` stretches the outermost pixels across that area, which
    reads as smeared bars; `blur` keeps the same content but defocused, which
    reads as a deliberate surround.
    """
    import cv2

    if fill not in FILL_MODES:
        raise ValueError(f"unknown align.fill {fill!r}; expected one of "
                         f"{', '.join(FILL_MODES)}")

    warped = cv2.warpAffine(
        image, matrix, (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if fill == "edge":
        return warped

    inside = source_footprint(image, matrix, width, height)
    if inside.all():
        # The frame is covered by real pixels; nothing to fill.
        return warped

    if fill == "black":
        background = np.zeros_like(warped)
    else:
        radius = max(3, (min(width, height) // 20) | 1)
        background = cv2.GaussianBlur(warped, (radius, radius), 0)

    mask = (inside > 127)[:, :, None]
    return np.where(mask, warped, background)


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
