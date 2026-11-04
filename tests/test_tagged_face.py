"""Every look at a photograph finds the person tagged in Immich, not whoever else is in it.

MediaPipe returns the faces it finds in its own order, which is neither by size
nor by who was tagged. Asked for one, it can hand back a person behind the
subject -- and that person's eyes were then stored, aligned on and morphed as
the subject's. It happened on a correctly tagged photograph: the second look
face-morph takes at each aligned frame was fixed first, and the photograph still
morphed the person in the background, because `analyze`'s own look at the
original has the same flaw. Each look now asks for several faces and keeps the
one that is where the tagged person is.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

from grow_up import analyze, db, metrics, pipeline
from grow_up.analyze import AnalyzeOptions, Attempt, Look

BOX = (300, 200, 500, 400)      # Immich's box for the tagged face, in image pixels


def mesh(left, right) -> np.ndarray:
    points = np.zeros((metrics.N_LANDMARKS, 2))
    points[metrics.IRIS_CENTER_A] = left
    points[metrics.IRIS_CENTER_B] = right
    return points


def look(left, right) -> Look:
    return Look(points=mesh(left, right), blendshapes={}, matrix=None,
                attempt=Attempt(margin=0.8))


TAGGED = ((360.0, 280.0), (440.0, 280.0))       # eyes inside the box
BEHIND = ((150.0, 160.0), (210.0, 160.0))       # a person over the subject's shoulder


class TestTheTaggedFaceIsTheOneInItsBox:

    def test_it_wins_even_when_the_model_lists_someone_else_first(self):
        stranger, tagged = look(*BEHIND), look(*TAGGED)
        assert analyze.pick_tagged([stranger, tagged], BOX) is tagged

    def test_only_someone_else_found_is_nobody_found(self):
        """Taking whoever was there is the mistake itself."""
        assert analyze.pick_tagged([look(*BEHIND)], BOX) is None

    def test_a_face_whose_eyes_only_graze_the_box_loses_to_the_centred_one(self):
        """With the crop grown for context, a neighbour can reach into the box's
        slack; the face nearest the box centre is the one it was drawn round."""
        grazing = look((280.0, 230.0), (330.0, 230.0))
        tagged = look(*TAGGED)
        assert analyze.pick_tagged([grazing, tagged], BOX) is tagged

    def test_the_box_is_given_slack_for_two_detectors_drawing_edges_differently(self):
        box_w = BOX[2] - BOX[0]
        just_outside = ((BOX[0] - 0.1 * box_w, 250.0), (BOX[0] + 0.05 * box_w, 250.0))
        assert analyze.eyes_in_box(*just_outside, BOX)
        far_outside = ((BOX[0] - 0.6 * box_w, 250.0), (BOX[0] - 0.4 * box_w, 250.0))
        assert not analyze.eyes_in_box(*far_outside, BOX)


# --------------------------------------------------------------------------- #
# analyze's ladder, with a stand-in landmarker
# --------------------------------------------------------------------------- #

class _Point:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _Result:
    def __init__(self, faces):
        self.face_landmarks = faces
        self.face_blendshapes = []
        self.facial_transformation_matrixes = []


class TwoPeople:
    """Answers every crop with a stranger first, then -- optionally -- the
    tagged face. Positions are given in image pixels and converted into the
    crop's normalised coordinates, as the real model reports them."""

    def __init__(self, tagged=True):
        self.tagged = tagged
        self.crop = None

    def face(self, left, right):
        x1, y1, x2, y2 = self.crop
        w, h = x2 - x1, y2 - y1
        points = [_Point(0.5, 0.5) for _ in range(metrics.N_LANDMARKS)]
        for index, (x, y) in ((metrics.IRIS_CENTER_A, left), (metrics.IRIS_CENTER_B, right)):
            points[index] = _Point((x - x1) / w, (y - y1) / h)
        return points

    def detect(self, image):
        faces = [self.face(*BEHIND)] + ([self.face(*TAGGED)] if self.tagged else [])
        return _Result(faces)


@pytest.fixture()
def fake_mediapipe(monkeypatch):
    module = types.ModuleType("mediapipe")
    module.Image = lambda image_format=None, data=None: types.SimpleNamespace(shape=data.shape)
    module.ImageFormat = types.SimpleNamespace(SRGB="srgb")
    monkeypatch.setitem(sys.modules, "mediapipe", module)


class TestAnalyzeKeepsTheTaggedFace:

    def gather(self, landmarker, monkeypatch):
        opts = AnalyzeOptions(bbox_margin=0.8)
        crop = analyze.images.expand_box(BOX, opts.bbox_margin, 800, 600)
        landmarker.crop = crop
        photo = np.zeros((600, 800, 3), dtype=np.uint8)
        return analyze.gather_looks(landmarker, opts, photo, BOX)

    def test_the_stranger_listed_first_is_passed_over(self, fake_mediapipe, monkeypatch):
        [found] = self.gather(TwoPeople(tagged=True), monkeypatch)
        left, right = metrics.iris_centers(found.points)
        assert left == pytest.approx(TAGGED[0]) and right == pytest.approx(TAGGED[1])

    def test_with_only_the_stranger_in_view_nothing_is_kept(self, fake_mediapipe, monkeypatch):
        """The photo is then filed as no face detected, and filtered out --
        rather than analyzed, aligned and morphed as somebody else."""
        assert self.gather(TwoPeople(tagged=False), monkeypatch) == []

    def test_the_workers_ask_for_more_than_one_face(self, monkeypatch):
        asked = {}

        def build(opts, num_faces=1):
            asked["num_faces"] = num_faces
            return object()

        monkeypatch.setattr(analyze, "build_landmarker", build)
        analyze.init_worker(AnalyzeOptions(verbose=True))
        assert asked["num_faces"] > 1


# --------------------------------------------------------------------------- #
# align catches what an older analyze stored
# --------------------------------------------------------------------------- #

class TestAlignNamesFramesCentredOnSomeoneElse:
    """`analyze` keeps its results, so a photo analyzed before it chose the
    tagged face can still hold a stranger's eyes. `align` reads every selected
    photo anyway, so it checks them against Immich's box for free."""

    @pytest.fixture()
    def library(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        pytest.importorskip("PIL.Image", reason="needs Pillow to decode")
        conn = db.connect(tmp_path / "t.sqlite")
        stamp = "2026-01-01T00:00:00.000Z"
        photos = {"right": ("2024-03-10", TAGGED), "wrong": ("2024-05-02", BEHIND)}
        for asset, (date, (left, right)) in photos.items():
            path = tmp_path / f"{asset}.png"
            cv2.imwrite(str(path), np.full((600, 800, 3), 120, dtype=np.uint8))
            conn.execute("INSERT INTO assets (id, local_datetime, indexed_at) VALUES (?, ?, ?)",
                         (asset, f"{date}T10:00:00", stamp))
            conn.execute("INSERT INTO downloads (asset_id, path, source, fetched_at)"
                         " VALUES (?, ?, 'original', ?)", (asset, str(path), stamp))
            # Immich's detection space at half size: the box has to be scaled.
            conn.execute("INSERT INTO faces (asset_id, status, x1, y1, x2, y2, image_width,"
                         " image_height, fetched_at) VALUES (?, 'ok', ?, ?, ?, ?, 400, 300, ?)",
                         (asset, BOX[0] // 2, BOX[1] // 2, BOX[2] // 2, BOX[3] // 2, stamp))
            conn.execute("INSERT INTO metrics (asset_id, detected, interocular_px, left_eye_x,"
                         " left_eye_y, right_eye_x, right_eye_y, analyzed_at)"
                         " VALUES (?, 1, 80, ?, ?, ?, ?, ?)", (asset, *left, *right, stamp))
            conn.execute("INSERT INTO selection (asset_id, bucket, rank, alternate, selected_at)"
                         " VALUES (?, ?, 0, 0, ?)", (asset, date[:7], stamp))
        return conn, tmp_path

    def align(self, library):
        conn, root = library
        lines = []
        pipeline.stage_align(conn, root / "frames", {"width": 64, "height": 80},
                             lines.append, workers=1)
        return lines

    def test_the_stray_one_is_named_by_date_with_what_to_run(self, library):
        lines = self.align(library)
        warning = next(i for i, line in enumerate(lines) if "someone other than" in line)
        assert "2024-05-02" in lines[warning] and "2024-03-10" not in lines[warning]
        assert "`grow-up analyze --reanalyze`" in lines[warning + 1]
        assert "`grow-up select && grow-up align && grow-up encode`" in lines[warning + 1]

    def test_a_library_analyzed_on_the_tagged_faces_says_nothing(self, library):
        conn, _ = library
        conn.execute("DELETE FROM selection WHERE asset_id = 'wrong'")
        assert not any("someone other than" in line for line in self.align(library))

    def test_a_photo_with_no_box_to_check_against_is_left_alone(self, library):
        conn, _ = library
        conn.execute("DELETE FROM faces WHERE asset_id = 'wrong'")
        assert not any("someone other than" in line for line in self.align(library))
