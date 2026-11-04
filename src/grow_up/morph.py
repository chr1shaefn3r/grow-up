"""Landmark-driven face morphing, for `transition = "face-morph"`.

`morph` hands the job to ffmpeg's `minterpolate`, which estimates where each
pixel went from the pixels themselves. Between two photographs months apart the
clothes, the light and the background have all changed, so what it finds is not
motion -- it is two unrelated images dragged into each other, and the result
smears. The filter is not badly built; it is asked a question the pixels cannot
answer.

Here that question is already answered. MediaPipe puts 478 labelled points on
every face, so the nose in one photograph *is* point 4 in the next and there is
nothing to estimate. Each in-between frame warps both photographs onto the mesh
part of the way between them and dissolves the two: an eye becomes an eye, and
the background -- which genuinely has no correspondence -- dissolves rather than
smears. The approach, and the two triangulation details called out below, come
from a fork of this project (firepinn/grow-up), where they were seen working on
real photographs.

The points come from landmarking the *aligned* frames again. The manifest keeps
the metrics the filters need, not the mesh, and the aligned frames are already
in output coordinates, so nothing has to be transformed -- and only photographs
that reach the video are looked at.

The warp is one sampling map per frame: every output pixel lies in one triangle
of the in-between mesh, and that triangle's affine says where in the photograph
to read it from. A single `cv2.remap` then fills every pixel, so there is no
per-triangle compositing to leave seams, wedges or holes. The maths -- the
interpolation, the affines and the map -- is plain numpy and tested with nothing
installed, like `align.py`; opencv is imported lazily, for the Delaunay, the
rasterising and the remap.
"""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

# The mesh covers the face and nothing else, so a warp driven by it alone would
# leave everything around it undefined. Anchoring the frame's corners and edge
# midpoints pins the border still and lets the triangles between the face and
# the edge stretch smoothly, so hair and shoulders follow the head rather than
# tearing away from it.
BORDER_FRACTIONS = (
    (0.0, 0.0), (0.5, 0.0), (1.0, 0.0),
    (0.0, 0.5), (1.0, 0.5),
    (0.0, 1.0), (0.5, 1.0), (1.0, 1.0),
)

# What an in-between frame is saved at. The same as `align` writes the frames
# themselves, so a still and the morph leaving it are not visibly different.
JPEG_QUALITY = 95


class MorphUnavailable(RuntimeError):
    """face-morph cannot run here at all.

    Raised rather than quietly rendering something else: the render is
    fingerprinted as face-morph, so a fallback video would be kept as the
    finished face-morph by every later run.
    """


def border_points(width: int, height: int) -> np.ndarray:
    """The frame anchors, in pixels, on the last row and column rather than past them."""
    w, h = float(width - 1), float(height - 1)
    return np.array([(fx * w, fy * h) for fx, fy in BORDER_FRACTIONS], dtype=np.float64)


def with_border(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Face points first, then the frame anchors, as one point set.

    Face first because the triangles index into this, and the same index has to
    mean the same point in every frame.
    """
    return np.vstack([np.asarray(points, dtype=np.float64)[:, :2],
                      border_points(width, height)])


def interpolate_points(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """The mesh `t` of the way from `a` to `b`.

    A straight line per point. `align` has already taken the rotation, scale and
    position out of both, so what is left is the change in shape the timelapse
    exists to show, and a straight line through it neither overshoots nor curves.
    `t` is clamped: past either mesh there is no face, only invention.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"mesh shapes differ: {a.shape} vs {b.shape}")
    t = float(np.clip(t, 0.0, 1.0))
    return (1.0 - t) * a + t * b


def inverse_affines(src_points: np.ndarray, dst_points: np.ndarray,
                    triangles: np.ndarray) -> np.ndarray:
    """Per triangle, the affine that finds a destination pixel in the source.

    Shaped (n, 3, 2): `[x, y, 1] @ m` is where the destination point (x, y) is
    read from. Three point pairs fix an affine exactly, so for a real triangle
    this is a solve. A triangle squashed flat has no exact answer, and `pinv`
    gives it the least-squares one rather than raising or putting NaN into the
    picture -- it covers almost no pixels either way.
    """
    triangles = np.asarray(triangles, dtype=np.intp)
    src = np.asarray(src_points, dtype=np.float64)[triangles]
    dst = np.asarray(dst_points, dtype=np.float64)[triangles]
    system = np.concatenate([dst, np.ones(dst.shape[:2] + (1,))], axis=2)
    return np.linalg.pinv(system) @ src


@lru_cache(maxsize=4)
def _pixel_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Every pixel's (x, y), built once per frame size. Read-only to callers."""
    ys, xs = np.indices((height, width), dtype=np.float32)
    xs.flags.writeable = False
    ys.flags.writeable = False
    return xs, ys


def sample_map(labels: np.ndarray, affines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where each output pixel is read from, as the two float32 maps `remap` takes.

    `labels` names the triangle each pixel lies in, -1 for none. With the
    frame's corners in the mesh every pixel is in one, but an unlabelled pixel
    reads from where it already is -- the photograph left in place rather than
    a hole.
    """
    labels = np.asarray(labels)
    xs, ys = _pixel_grid(*labels.shape)
    affines = np.asarray(affines, dtype=np.float32)
    if len(affines) == 0:
        return xs.copy(), ys.copy()
    # One coefficient at a time: six flat lookups are cheaper than gathering a
    # (height, width, 6) block, and this runs twice for every frame made.
    table = affines.reshape(len(affines), 6)
    which = np.maximum(labels, 0)
    c = [np.take(table[:, j], which) for j in range(6)]
    map_x = c[0] * xs
    map_x += c[2] * ys
    map_x += c[4]
    map_y = c[1] * xs
    map_y += c[3] * ys
    map_y += c[5]
    uncovered = labels < 0
    if uncovered.any():
        map_x[uncovered] = xs[uncovered]
        map_y[uncovered] = ys[uncovered]
    return map_x, map_y


def rasterize(points: np.ndarray, triangles: np.ndarray, width: int,
              height: int) -> np.ndarray:
    """Which triangle of `points` each pixel lies in; -1 where none does.

    Neighbouring triangles share their corners, so they share their rounded
    edges too and tile the frame with nothing left between them. Where two
    claim a pixel on a shared edge the later one wins, and both affines agree
    along that edge anyway.
    """
    import cv2

    labels = np.full((height, width), -1, dtype=np.int32)
    corners = np.rint(np.asarray(points, dtype=np.float64)[triangles]).astype(np.int32)
    for index, corner in enumerate(corners):
        cv2.fillConvexPoly(labels, corner, int(index))
    return labels


def warp(image: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray,
         triangles: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    """Move `image` from one mesh onto another, every pixel at once.

    A sample that lands outside the photograph -- a triangle reaching past the
    frame edge -- reflects back into it rather than reading black.
    """
    import cv2

    image = np.asarray(image)
    height, width = image.shape[:2]
    if labels is None:
        labels = rasterize(dst_points, triangles, width, height)
    map_x, map_y = sample_map(labels, inverse_affines(src_points, dst_points, triangles))
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


def blend(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Cross-dissolve, in integers.

    The weight is `t` in 256ths: a fourteenth of the cost of doing it in
    floats, and never more than one level of 255 away from it. Both ends are
    exact -- 0 is A and 1 is B, pixel for pixel.
    """
    w = int(round(float(np.clip(t, 0.0, 1.0)) * 256))
    mixed = (np.asarray(a, dtype=np.uint16) * (256 - w)
             + np.asarray(b, dtype=np.uint16) * w + 128) >> 8
    return mixed.astype(np.uint8)


def morph_pair(image_a: np.ndarray, image_b: np.ndarray, points_a: np.ndarray,
               points_b: np.ndarray, triangles: np.ndarray, t: float) -> np.ndarray:
    """One in-between frame, `t` of the way from A to B.

    Both photographs are warped onto the *same* in-between mesh before they are
    dissolved. Dissolving first would lay two differently shaped faces over each
    other -- the doubled eyes and mouths that make a plain crossfade look wrong
    on a face.
    """
    height, width = np.asarray(image_a).shape[:2]
    middle = interpolate_points(points_a, points_b, t)
    labels = rasterize(middle, triangles, width, height)
    return blend(warp(image_a, points_a, middle, triangles, labels),
                 warp(image_b, points_b, middle, triangles, labels), t)


def in_between_ts(count: int) -> list[float]:
    """How far along each of `count` in-between frames is.

    Evenly spaced and never 0 or 1: those are the two photographs themselves,
    which are already on screen either side.
    """
    return [(k + 1) / (count + 1) for k in range(max(0, int(count)))]


def triangulate(points: np.ndarray, width: int, height: int) -> np.ndarray:
    """Delaunay triangles over `points`, as indices into it.

    Computed once per render from the mean of every frame's mesh, and used for
    every pair. A triangulation per pair would flip a triangle here and there
    between neighbouring frames, and a flipped triangle is a visible twitch in
    an otherwise still face.

    Two details from the fork, both found as holes in a real morph. Subdiv2D
    silently merges points closer together than its own tolerance, and the
    478-point mesh has ten iris points sitting almost on eye-contour ones -- so
    corners are matched back to the input by *nearest* point, not by exact
    coordinate, or the triangles around the eyes go missing. And Subdiv2D adds
    four virtual points of its own far outside the frame; a corner nowhere near
    any real point is one of those, and its triangle is dropped.
    """
    import cv2

    # Subdiv2D refuses a point outside its rectangle, and a landmark can sit
    # just past the frame edge. The clamped copy is what is inserted and what
    # the corners are matched against, so indices still mean the input's points.
    inserted = np.asarray(points, dtype=np.float64).copy()
    inserted[:, 0] = np.clip(inserted[:, 0], 0.0, width - 1.0)
    inserted[:, 1] = np.clip(inserted[:, 1], 0.0, height - 1.0)
    subdiv = cv2.Subdiv2D((-1, -1, width + 2, height + 2))
    for x, y in inserted:
        subdiv.insert((float(x), float(y)))

    raw = np.asarray(subdiv.getTriangleList(), dtype=np.float64)
    if raw.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    corners = raw.reshape(-1, 2)
    gaps = ((corners[:, None, :] - inserted[None, :, :]) ** 2).sum(axis=2)
    nearest = gaps.argmin(axis=1)
    near = gaps[np.arange(len(corners)), nearest] <= 1.0

    index = nearest.reshape(-1, 3)
    real = near.reshape(-1, 3).all(axis=1)
    # A merged pair can leave a triangle naming one point twice.
    distinct = ((index[:, 0] != index[:, 1]) & (index[:, 1] != index[:, 2])
                & (index[:, 0] != index[:, 2]))
    return index[real & distinct].astype(np.int32)


def landmarks_for(paths: Sequence[Path], opts) -> list[np.ndarray | None]:
    """The face mesh of each aligned frame, or None where no face is found.

    The whole frame is the crop: `align` has already centred the face and put
    the eyes where they belong, so there is nothing left to search for.
    """
    try:
        import cv2

        from . import analyze

        with analyze.suppress_native_output(not opts.verbose):
            landmarker = analyze.build_landmarker(opts)
    except ImportError as exc:
        raise MorphUnavailable(
            f'transition = "face-morph" finds the face in every frame with the same '
            f"landmarker `analyze` uses, and it cannot load here ({exc}).") from exc

    found: list[np.ndarray | None] = []
    try:
        with analyze.suppress_native_output(not opts.verbose):
            for path in paths:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise MorphUnavailable(f"could not read the aligned frame {path}")
                height, width = image.shape[:2]
                look = analyze.detect_once(landmarker, opts, image,
                                           (0, 0, width, height),
                                           analyze.Attempt(margin=0.0))
                found.append(None if look is None
                             else np.asarray(look.points, dtype=np.float64)[:, :2].copy())
    finally:
        # Closed here rather than left to the finalizer, which runs during
        # interpreter teardown and prints a traceback over a run that worked.
        try:
            landmarker.close()
        except Exception:                                  # pragma: no cover
            pass
    return found


def _render_pair(job: tuple) -> list[str]:
    """Every in-between frame for one pair of photographs. The worker entry point.

    Pairs rather than frames are the unit of work: a pair reads its two
    photographs once and reuses them for every step between them.
    """
    import cv2

    path_a, path_b, mesh_a, mesh_b, triangles, ts, out_dir = job
    image_a = cv2.imread(path_a, cv2.IMREAD_COLOR)
    image_b = cv2.imread(path_b, cv2.IMREAD_COLOR)
    if image_a is None or image_b is None:
        raise MorphUnavailable(f"could not read {path_a} or {path_b}")
    if image_a.shape != image_b.shape:
        raise MorphUnavailable(
            f"{Path(path_a).name} and {Path(path_b).name} differ in size; "
            "re-run `grow-up align` so every frame is the configured size")

    # Named after the two photographs, never after a place in the timeline.
    stem = f"{Path(path_a).stem}--{Path(path_b).stem}"
    written = []
    for k, t in enumerate(ts, start=1):
        if mesh_a is None or mesh_b is None:
            # No face to follow in one of them: the plain dissolve, which is
            # what the background gets anyway.
            frame = blend(image_a, image_b, t)
        else:
            frame = morph_pair(image_a, image_b, mesh_a, mesh_b, triangles, t)
        target = Path(out_dir) / f"{stem}-{k:03d}.jpg"
        if not cv2.imwrite(str(target), frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]):
            raise MorphUnavailable(f"could not write {target}")
        written.append(str(target))
    return written


def render(frames: Sequence[Path], meshes: Sequence[np.ndarray | None], out_dir: Path,
           count: int, workers: int = 0,
           log: Callable[[str], None] = lambda _m: None) -> list[list[Path]]:
    """Write `count` in-between frames for every consecutive pair, and list them per pair.

    `out_dir` is emptied first and nothing in it is ever reused. A frame left by
    an earlier run -- one that failed part-way, or ran before the selection
    changed -- would be a morph between two other photographs, and it would
    look entirely plausible.

    A pair where either photograph has no mesh dissolves instead of morphing.
    Every pair still gets `count` frames, so the timing is the same throughout.

    Run across processes, like `analyze`: the pairs are independent, and the
    warp is numpy and opencv work that one process would do on one core.
    """
    import cv2

    from . import analyze
    from .progress import Progress

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    pairs = max(0, len(frames) - 1)
    if count <= 0 or pairs == 0:
        return [[] for _ in range(pairs)]

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise MorphUnavailable(f"could not read the aligned frame {frames[0]}")
    height, width = first.shape[:2]

    bordered = [None if mesh is None else with_border(mesh, width, height)
                for mesh in meshes]
    found = [mesh for mesh in bordered if mesh is not None]
    if not found:
        raise MorphUnavailable("no face was found in any aligned frame")
    triangles = triangulate(np.mean(found, axis=0), width, height)

    ts = in_between_ts(count)
    jobs = [(str(frames[i]), str(frames[i + 1]), bordered[i], bordered[i + 1],
             triangles, ts, str(out_dir)) for i in range(pairs)]
    processes = min(workers or analyze.physical_cores(), len(jobs))
    log(f"  morph: {len(triangles)} triangles, {pairs * count} in-between frames "
        f"across {processes} worker{'s' if processes != 1 else ''}")

    made: list[list[str]] = []
    bar = Progress("morph", len(jobs), emit=log)
    if processes <= 1:
        for job in jobs:
            made.append(_render_pair(job))
            bar.advance()
    else:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # Spawned, not forked. This process has already run opencv, which keeps
        # a thread pool, and forking a process with threads in it can deadlock
        # the child -- Python warns about exactly that. macOS spawns anyway.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=processes, mp_context=context) as pool:
            for result in pool.map(_render_pair, jobs):
                made.append(result)
                bar.advance()
    bar.close()
    return [[Path(p) for p in pair] for pair in made]
