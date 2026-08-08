"""Image decoding.

Isolated so the EXIF-orientation rule lives in exactly one place: Immich reports
face boxes against a rotation-applied rendition, so every decode in this project
must apply orientation too or every crop lands on the wrong part of the photo.
"""

from __future__ import annotations

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
