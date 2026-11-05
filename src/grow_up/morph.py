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
that reach the video are looked at. That second look has to find the *same
person*: `analyze` landmarked inside Immich's box for the tagged face, while
this sees the whole frame and everyone in it. Asked for one face, the model
hands back whichever it ranks first, and a stranger in the background often
wins. So every face is asked for, and the one used is the one whose eyes sit
where `align` put the subject's -- known exactly, from the eyes `analyze` found
and the transform `align` stored.

The warp is one sampling map per frame: every output pixel lies in one triangle
of the in-between mesh, and that triangle's affine says where in the photograph
to read it from. A single `cv2.remap` then fills every pixel, so there is no
per-triangle compositing to leave seams, wedges or holes. The maths -- the
interpolation, the affines and the map -- is plain numpy and tested with nothing
installed, like `align.py`; opencv is imported lazily, for the Delaunay, the
rasterising and the remap.

`neutral_background` goes one step further, and also comes from the fork. The
background is the one part of two photographs months apart with nothing in
common -- another room, another light, another year -- so it is where a morph
ghosts most and the part least worth keeping. MediaPipe's selfie segmenter cuts
the person out and every frame is put on one plain field, the held ones as well
as the in-betweens: a film that switched between a real background and a plain
one on every photograph would flicker. The cut-out moves no pixel, so the meshes
measured on the aligned frames stay valid for it.
"""

from __future__ import annotations

import os
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

# How far, as a fraction of the distance between them, the eyes of the face
# found may sit from where the subject's eyes are known to be. The same face
# found again lands within a few pixels; anyone else's eyes are a whole face
# away, so the margin can be generous without letting a stranger through.
EYE_TOLERANCE = 0.25

# Which version of this code made a face-morph video, recorded in its
# fingerprint. A render is skipped when its inputs are unchanged, and for
# face-morph this code is one of them: the frames it invents are part of the
# picture. Without it, the run after a fix found the old video "already
# rendered" and kept it -- which is how the fix for morphing a person in the
# background never reached the video it was made for. Bump it whenever the
# frames made from the same inputs change, the cut-out onto the neutral
# background included.
REVISION = 2

# What an in-between frame is saved at. The same as `align` writes the frames
# themselves, so a still and the morph leaving it are not visibly different.
JPEG_QUALITY = 95

# MediaPipe's multiclass selfie segmenter: background, hair, body skin, face
# skin, clothes, and everything else on the person. mediapipe is already a
# dependency, so the neutral background costs one model file and no library.
SEGMENT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
SEGMENT_MODEL_NAME = "selfie_multiclass.tflite"

# The one class that is not the person. Hair, skin, clothes and accessories are
# all kept: a cut-out that dropped any of them would be the wrong shape.
BACKGROUND_CLASS = 0

# A soft, slightly cool grey, in opencv's BGR. Dark enough that fair hair and
# skin stand out against it, light enough not to read as a void, and neutral
# enough that years of different colour temperature sit on it without argument.
NEUTRAL_BACKGROUND = (232, 230, 226)

# How far the edge of the cut-out is softened, in pixels of the frame. The
# segmenter sees 256x256, so its edge scaled up to a 1080-wide frame is a
# staircase. A couple of pixels of blur reads as an ordinary depth-of-field
# edge; the staircase reads as scissors.
FEATHER = 2.5


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


def pick_subject(candidates: Sequence[np.ndarray], left_eye: np.ndarray,
                 right_eye: np.ndarray,
                 tolerance: float = EYE_TOLERANCE) -> np.ndarray | None:
    """The face among `candidates` whose eyes are the subject's, or None.

    `left_eye` and `right_eye` are where the subject's irises are in this
    frame. The candidate whose irises are nearest wins, and only if both are
    within `tolerance` of the distance between the eyes: a frame where the
    subject was not found at all must not fall back to whoever was, which is
    exactly the mistake this exists to prevent.
    """
    from .metrics import iris_centers

    left_eye = np.asarray(left_eye, dtype=np.float64)
    right_eye = np.asarray(right_eye, dtype=np.float64)
    limit = tolerance * float(np.hypot(*(right_eye - left_eye)))
    best, best_off = None, np.inf
    for points in candidates:
        found_left, found_right = iris_centers(np.asarray(points, dtype=np.float64))
        off = max(float(np.hypot(*(found_left - left_eye))),
                  float(np.hypot(*(found_right - right_eye))))
        if off < best_off:
            best, best_off = points, off
    if best is None or best_off > limit:
        return None
    return np.asarray(best, dtype=np.float64)[:, :2].copy()


def landmarks_for(paths: Sequence[Path], opts,
                  eyes: Sequence[tuple[np.ndarray, np.ndarray] | None]
                  ) -> list[np.ndarray | None]:
    """The subject's face mesh in each aligned frame, or None where it is not found.

    `eyes` gives, per frame, where the subject's irises are -- or None if that
    is not known, which is treated as not found rather than guessed at. The
    whole frame is looked at, since `align` has already centred the face, and
    the face kept is the one at those eyes, whoever else is in the picture.
    """
    try:
        import cv2

        from . import analyze

        with analyze.suppress_native_output(not opts.verbose):
            landmarker = analyze.build_landmarker(opts, num_faces=analyze.MAX_FACES)
    except ImportError as exc:
        raise MorphUnavailable(
            f'transition = "face-morph" finds the face in every frame with the same '
            f"landmarker `analyze` uses, and it cannot load here ({exc}).") from exc

    found: list[np.ndarray | None] = []
    try:
        with analyze.suppress_native_output(not opts.verbose):
            for path, subject in zip(paths, eyes, strict=True):
                if subject is None:
                    found.append(None)
                    continue
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise MorphUnavailable(f"could not read the aligned frame {path}")
                height, width = image.shape[:2]
                looks = analyze.detect_all(landmarker, opts, image,
                                           (0, 0, width, height),
                                           analyze.Attempt(margin=0.0))
                found.append(pick_subject([look.points for look in looks], *subject))
    finally:
        # Closed here rather than left to the finalizer, which runs during
        # interpreter teardown and prints a traceback over a run that worked.
        try:
            landmarker.close()
        except Exception:                                  # pragma: no cover
            pass
    return found


def segment_model_beside(landmarker: Path | str) -> Path:
    """Where the segmenter is looked for: beside the face landmarker, which is
    where `grow-up fetch-model` puts it."""
    return Path(landmarker).parent / SEGMENT_MODEL_NAME


def subject_alpha(mask: np.ndarray, feather: float = FEATHER) -> np.ndarray:
    """How much of each pixel is the person, 0 to 1, softened at the silhouette."""
    import cv2

    alpha = (np.asarray(mask) != BACKGROUND_CLASS).astype(np.float32)
    if feather <= 0:
        return alpha
    return cv2.GaussianBlur(alpha, (0, 0), sigmaX=float(feather))


def on_neutral(image: np.ndarray, mask: np.ndarray,
               colour: tuple[int, int, int] = NEUTRAL_BACKGROUND,
               feather: float = FEATHER) -> np.ndarray:
    """The person over the plain field, with their own background thrown away."""
    image = np.asarray(image)
    mask = np.asarray(mask)
    if mask.shape != image.shape[:2]:
        raise ValueError(f"a {mask.shape} mask does not cover a {image.shape[:2]} frame")
    alpha = subject_alpha(mask, feather)[..., None]
    field = np.asarray(colour, dtype=np.float32)
    blended = alpha * image.astype(np.float32) + (1.0 - alpha) * field
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def subject_at(mask: np.ndarray, eyes: tuple | None) -> bool:
    """Did the cut-out keep the subject?

    Asked at the bridge of the nose, half way between the eyes `align` put
    there. A cut-out that does not reach it has lost the face, and a frame of
    plain grey with nobody on it would be the worst frame in the film -- worse
    than one photograph keeping its own background. Without known eyes,
    anything on the person will do.
    """
    person = np.asarray(mask) != BACKGROUND_CLASS
    if eyes is None:
        return bool(person.any())
    left, right = (np.asarray(e, dtype=np.float64)[:2] for e in eyes)
    x, y = (int(v) for v in np.rint((left + right) / 2.0))
    height, width = person.shape[:2]
    return 0 <= x < width and 0 <= y < height and bool(person[y, x])


def build_segmenter(model_path: Path):
    """The selfie segmenter, asked for a class per pixel and nothing else."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    return vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        output_category_mask=True,
        output_confidence_masks=False))


def segment(segmenter, image: np.ndarray) -> np.ndarray:
    """A class label per pixel of `image`, at the image's own size."""
    import cv2
    import mediapipe as mp

    frame = mp.Image(image_format=mp.ImageFormat.SRGB,
                     data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    mask = segmenter.segment(frame).category_mask.numpy_view()
    return np.ascontiguousarray(mask).reshape(mask.shape[:2])


def _cut_out(job: tuple) -> list[bool]:
    """Put every frame on the neutral field, and say which kept their subject.

    The entry point of the segmenter's own process. One segmenter for the whole
    render, built and closed here: the fork first built one per pair, which was
    119 model loads across 12 processes and killed its pool.
    """
    from . import analyze

    sources, targets, eyes, model, verbose = job
    # Before mediapipe is imported, which is when the native log level is read.
    if not verbose:
        os.environ.update(analyze.QUIET_ENV)

    import cv2

    found = []
    with analyze.suppress_native_output(not verbose):
        segmenter = build_segmenter(Path(model))
        try:
            for source, target, subject in zip(sources, targets, eyes, strict=True):
                image = cv2.imread(source, cv2.IMREAD_COLOR)
                if image is None:
                    raise MorphUnavailable(f"could not read the aligned frame {source}")
                mask = segment(segmenter, image)
                if not subject_at(mask, subject):
                    found.append(False)
                    continue
                if not cv2.imwrite(target, on_neutral(image, mask),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]):
                    raise MorphUnavailable(f"could not write {target}")
                found.append(True)
        finally:
            try:
                segmenter.close()
            except Exception:                              # pragma: no cover
                pass
    return found


def _in_a_child(work: Callable, job: tuple):
    """`work(job)`, run in a fresh process of its own.

    For the segmenter only. In the fork this comes from, a process that had
    built one could no longer start a worker pool: the parent segfaulted as
    the pool was created, at any worker count, spawned or forked, and closing
    the segmenter first did not help. `render` starts its pool straight after.
    It did not reproduce here (Linux, mediapipe 0.10.35, a landmarker and a
    segmenter both live in the parent, then a 12-worker pool), but the crash
    leaves nothing in the log except "terminated abruptly", and a second for a
    fresh interpreter is a small price for never finding out.
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        return pool.submit(work, job).result()


def neutral_frames(frames: Sequence[Path], eyes: Sequence[tuple | None], model: Path,
                   out_dir: Path, verbose: bool = False,
                   log: Callable[[str], None] = lambda _m: None) -> list[Path | None]:
    """Every frame cut out onto the neutral field, or None where the cut-out missed.

    `eyes` is where the subject's irises are in each frame, as for
    `landmarks_for`. A None in the result is a frame whose cut-out lost the
    subject; the caller shows the photograph there instead, and names it.

    Each is named after the frame it was cut from, never after its place in
    the timeline, and `out_dir` is emptied first: a cut-out left behind by
    another run is of another photograph or another selection, and would look
    entirely plausible.
    """
    from concurrent.futures.process import BrokenProcessPool

    model = Path(model)
    if not model.exists():
        raise MorphUnavailable(
            f"neutral_background = true cuts the subject out with MediaPipe's selfie "
            f"segmenter, and there is none at {model}. Run `grow-up fetch-model`, "
            "which puts it beside the face landmarker.")

    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    if not frames:
        return []
    targets = [out_dir / f"{Path(frame).stem}.jpg" for frame in frames]
    job = ([str(f) for f in frames], [str(t) for t in targets],
           [None if e is None else tuple(tuple(float(v) for v in point[:2]) for point in e)
            for e in eyes],
           str(model), bool(verbose))

    log(f"  morph: cutting the subject out of {len(frames)} frames onto a neutral background")
    try:
        found = _in_a_child(_cut_out, job)
    except ImportError as exc:
        raise MorphUnavailable(
            f"neutral_background = true cuts the subject out with MediaPipe's segmenter, "
            f"and it cannot load here ({exc}).") from exc
    except BrokenProcessPool as exc:
        raise MorphUnavailable(
            "the segmenter's process died before it finished, so nothing was cut out. "
            "`grow-up -v encode` shows what MediaPipe printed.") from exc
    return [target if kept else None for target, kept in zip(targets, found, strict=True)]


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
