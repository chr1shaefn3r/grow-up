"""Image decoding.

Isolated so the EXIF-orientation rule lives in exactly one place: Immich reports
face boxes against a rotation-applied rendition, so every decode in this project
must apply orientation too or every crop lands on the wrong part of the photo.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def _register_heif() -> None:
    """iPhone libraries are largely HEIC; Pillow needs the plugin registered."""
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass


def load_bgr(path: str | Path) -> np.ndarray:
    """Decode to an EXIF-oriented BGR array (opencv's channel order)."""
    from PIL import Image, ImageOps

    _register_heif()
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        rgb = np.asarray(im.convert("RGB"))
    return rgb[:, :, ::-1].copy()


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    return bgr[:, :, ::-1].copy()


def to_gray(bgr: np.ndarray) -> np.ndarray:
    b, g, r = bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2]
    return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.float64)


def expand_box(box: tuple[int, int, int, int], margin: float,
               width: int, height: int) -> tuple[int, int, int, int]:
    """Grow a face box by `margin` on each side, clamped to the image.

    MediaPipe's mesh fit degrades on a box cropped tight to the face; it wants
    surrounding context.
    """
    x1, y1, x2, y2 = box
    dx = (x2 - x1) * margin / 2.0
    dy = (y2 - y1) * margin / 2.0
    return (
        max(0, int(round(x1 - dx))),
        max(0, int(round(y1 - dy))),
        min(width, int(round(x2 + dx))),
        min(height, int(round(y2 + dy))),
    )


def downscale_to(bgr: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Shrink a crop so its longest side is at most `max_side`.

    Returns (image, scale) where `scale` maps the returned image's coordinates
    back to the input's -- landmarks come out in the smaller space and must be
    multiplied by it.

    MediaPipe resizes to its own 256px input with a plain bilinear filter, so
    handing it a 2000px crop wastes time and loses detail to aliasing.
    INTER_AREA averages the discarded pixels instead.
    """
    if max_side <= 0:
        return bgr, 1.0
    height, width = bgr.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return bgr, 1.0

    import cv2

    scale = longest / float(max_side)
    resized = cv2.resize(bgr, (max(1, int(round(width / scale))),
                               max(1, int(round(height / scale)))),
                         interpolation=cv2.INTER_AREA)
    return resized, scale


def equalize(bgr: np.ndarray) -> np.ndarray:
    """Local contrast equalisation, for faces lost in shadow or backlight.

    CLAHE on the L channel only, so colour is untouched -- this exists to give
    the detector something to find, not to alter how the photo looks.
    """
    import cv2

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def rotation_matrix(degrees: float, width: int, height: int) -> np.ndarray:
    """2x3 rotation about the centre of a `width` x `height` image.

    Written out rather than taken from `cv2.getRotationMatrix2D` -- same result,
    but it keeps the coordinate round trip testable without opencv installed,
    which is what lets the suite stay dependency-light.
    """
    angle = math.radians(degrees)
    alpha, beta = math.cos(angle), math.sin(angle)
    cx, cy = width / 2.0, height / 2.0
    return np.array([
        [alpha, beta, (1.0 - alpha) * cx - beta * cy],
        [-beta, alpha, beta * cx + (1.0 - alpha) * cy],
    ], dtype=np.float64)


def rotate(bgr: np.ndarray, degrees: float) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a crop about its centre, keeping the same canvas size.

    Returns (rotated, matrix). Landmarks detected on the rotated copy are in
    that copy's coordinates, so `unrotate_points` must undo the same matrix
    before they mean anything in the original.
    """
    import cv2

    height, width = bgr.shape[:2]
    matrix = rotation_matrix(degrees, width, height)
    rotated = cv2.warpAffine(bgr, matrix, (width, height),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, matrix


def unrotate_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Map points from a rotated image back to the original.

    Pure numpy: inverting a 2x3 affine is a 2x2 solve plus a translation, and
    keeping it dependency-free means the round trip is testable without opencv.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    linear, offset = matrix[:, :2], matrix[:, 2]
    return (np.linalg.solve(linear, (np.asarray(points, dtype=np.float64) - offset).T)).T


def resize_gray(gray: np.ndarray, target_width: int) -> np.ndarray:
    """Resample a grayscale patch to a fixed width.

    Sharpness must be measured at a canonical scale, otherwise the metric ranks
    high-megapixel photos above sharp ones.
    """
    import cv2

    h, w = gray.shape[:2]
    if w == 0 or h == 0:
        return gray
    scale = target_width / float(w)
    return cv2.resize(gray, (target_width, max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_AREA)
