"""The landmark morph behind `transition = "face-morph"`.

The maths -- the in-between mesh, the per-triangle affines, the sampling map,
the dissolve, the steps -- is plain numpy and runs with nothing installed, like
`align.py`. The pixel half is gated on opencv per class: a module-level
`importorskip` would take the numpy half down with it, and that half is why the
maths lives outside opencv at all.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from grow_up import morph

requires_cv2 = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None, reason="the warp needs opencv")


class TestTheBorderIsAnchored:
    """The mesh covers the face and nothing else, so a warp driven by it alone
    would leave everything around it undefined."""

    def test_the_anchors_sit_on_the_frame_edge(self):
        points = morph.border_points(100, 200)
        assert points[:, 0].min() == 0.0 and points[:, 1].min() == 0.0
        assert points[:, 0].max() == 99.0 and points[:, 1].max() == 199.0

    def test_every_corner_is_anchored(self):
        points = {tuple(p) for p in morph.border_points(11, 21)}
        for corner in ((0.0, 0.0), (10.0, 0.0), (0.0, 20.0), (10.0, 20.0)):
            assert corner in points

    def test_the_face_points_come_first(self):
        """The triangles index into this, so an index must mean the same point
        in every frame."""
        face = np.arange(20, dtype=np.float64).reshape(10, 2)
        combined = morph.with_border(face, 64, 64)
        assert combined.shape == (10 + len(morph.BORDER_FRACTIONS), 2)
        assert np.array_equal(combined[:10], face)


class TestTheInBetweenMesh:

    def test_the_ends_are_the_two_meshes(self):
        a = np.array([[0.0, 0.0], [10.0, 0.0]])
        b = np.array([[0.0, 4.0], [10.0, 8.0]])
        assert np.allclose(morph.interpolate_points(a, b, 0.0), a)
        assert np.allclose(morph.interpolate_points(a, b, 1.0), b)

    def test_the_middle_is_the_average(self):
        assert np.allclose(morph.interpolate_points(np.zeros((3, 2)),
                                                    np.full((3, 2), 8.0), 0.5), 4.0)

    def test_it_never_extrapolates_past_either_face(self):
        a, b = np.zeros((2, 2)), np.ones((2, 2))
        assert np.allclose(morph.interpolate_points(a, b, 2.0), b)
        assert np.allclose(morph.interpolate_points(a, b, -1.0), a)

    def test_meshes_of_different_sizes_are_refused(self):
        with pytest.raises(ValueError):
            morph.interpolate_points(np.zeros((3, 2)), np.zeros((4, 2)), 0.5)


class TestTheTriangleAffines:
    """Each affine answers: for this pixel of the in-between mesh, where in the
    photograph is it read from?"""

    SRC = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    DST = np.array([[3.0, 4.0], [17.0, 1.0], [1.0, 22.0], [15.0, 19.0]])
    TRIANGLES = np.array([[0, 1, 2], [1, 3, 2]])

    def test_every_corner_is_found_exactly_where_it_came_from(self):
        affines = morph.inverse_affines(self.SRC, self.DST, self.TRIANGLES)
        for tri, m in zip(self.TRIANGLES, affines):
            homogeneous = np.hstack([self.DST[tri], np.ones((3, 1))])
            assert np.allclose(homogeneous @ m, self.SRC[tri])

    def test_an_unmoved_mesh_reads_every_pixel_from_itself(self):
        affines = morph.inverse_affines(self.SRC, self.SRC, self.TRIANGLES)
        assert np.allclose(affines, np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))

    def test_a_triangle_squashed_flat_gives_numbers_not_nan(self):
        """A squashed mesh can produce one. NaN would reach the picture, and an
        exception would cost the whole render for a sliver of no pixels."""
        flat = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        affines = morph.inverse_affines(flat + 1.0, flat, np.array([[0, 1, 2]]))
        assert np.all(np.isfinite(affines))


class TestTheSamplingMap:
    """Every output pixel reads from the source through its own triangle's
    affine -- built by hand here, so no rasteriser is needed."""

    def test_each_pixel_follows_its_own_triangle(self):
        labels = np.zeros((4, 6), dtype=np.int32)
        labels[:, 3:] = 1
        shift_right = [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]
        shift_down = [[1.0, 0.0], [0.0, 1.0], [0.0, 3.0]]
        map_x, map_y = morph.sample_map(labels, np.array([shift_right, shift_down]))
        assert map_x[1, 1] == 3.0 and map_y[1, 1] == 1.0      # (1,1) reads (3,1)
        assert map_x[1, 4] == 4.0 and map_y[1, 4] == 4.0      # (4,1) reads (4,4)

    def test_a_pixel_in_no_triangle_stays_where_it_is(self):
        """Left in place rather than read from somewhere arbitrary, or left black."""
        labels = np.full((3, 3), -1, dtype=np.int32)
        labels[0, 0] = 0
        far = np.array([[[1.0, 0.0], [0.0, 1.0], [50.0, 50.0]]])
        map_x, map_y = morph.sample_map(labels, far)
        assert (map_x[2, 1], map_y[2, 1]) == (1.0, 2.0)
        assert map_x[0, 0] == 50.0

    def test_without_triangles_the_picture_is_left_alone(self):
        map_x, map_y = morph.sample_map(np.full((2, 3), -1, dtype=np.int32),
                                        np.empty((0, 3, 2)))
        assert map_x.tolist() == [[0, 1, 2], [0, 1, 2]]
        assert map_y.tolist() == [[0, 0, 0], [1, 1, 1]]

    def test_the_maps_are_what_remap_takes(self):
        labels = np.zeros((2, 2), dtype=np.int32)
        map_x, map_y = morph.sample_map(labels, np.array([np.eye(3, 2)]))
        assert map_x.dtype == np.float32 and map_y.dtype == np.float32


class TestTheSteps:

    def test_they_are_evenly_spaced_and_never_the_photographs(self):
        """0 and 1 are the two photographs themselves, already on screen."""
        assert morph.in_between_ts(3) == [0.25, 0.5, 0.75]

    def test_none_asked_for_none_made(self):
        assert morph.in_between_ts(0) == []


class TestTheDissolve:

    A = np.array([[[0, 100, 255]]], dtype=np.uint8)
    B = np.array([[[255, 0, 10]]], dtype=np.uint8)

    def test_both_ends_are_exact(self):
        assert np.array_equal(morph.blend(self.A, self.B, 0.0), self.A)
        assert np.array_equal(morph.blend(self.A, self.B, 1.0), self.B)

    def test_it_is_within_one_level_of_the_exact_dissolve(self):
        """Integer weights, for speed; the cost must be invisible."""
        rng = np.random.default_rng(3)
        a = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        b = rng.integers(0, 256, (20, 20, 3), dtype=np.uint8)
        for t in (0.1, 0.33, 0.5, 0.77):
            exact = (1 - t) * a.astype(float) + t * b.astype(float)
            assert np.abs(morph.blend(a, b, t).astype(float) - exact).max() <= 1.0

    def test_it_stays_in_bytes(self):
        full = np.full((2, 2, 3), 255, dtype=np.uint8)
        out = morph.blend(full, full, 0.5)
        assert out.dtype == np.uint8 and out.max() == 255


def face_at(left: tuple[float, float], right: tuple[float, float]) -> np.ndarray:
    """A 478-point mesh whose irises are at `left` and `right`."""
    from grow_up import metrics

    mesh = np.zeros((478, 2))
    mesh[metrics.IRIS_CENTER_A] = left
    mesh[metrics.IRIS_CENTER_B] = right
    return mesh


class TestOnlyTheSubjectIsMorphed:
    """The second look sees the whole aligned frame and everyone in it. Asked
    for one face, the model returned whichever it ranked first, and a second
    person in the background was morphed as the subject. This shipped, on a
    correctly tagged photograph. The subject's eyes are known exactly in every
    aligned frame, so the face used is the one with its eyes there."""

    LEFT, RIGHT = (432.0, 567.0), (648.0, 567.0)

    def test_the_face_at_the_subjects_eyes_wins_even_when_it_comes_second(self):
        stranger = face_at((130.0, 184.0), (205.0, 184.0))
        subject = face_at((434.0, 569.0), (646.0, 566.0))
        chosen = morph.pick_subject([stranger, subject], self.LEFT, self.RIGHT)
        assert np.array_equal(chosen, subject)

    def test_a_frame_with_only_someone_else_in_it_has_no_subject(self):
        """Falling back to whoever was found is exactly the mistake."""
        stranger = face_at((130.0, 184.0), (205.0, 184.0))
        assert morph.pick_subject([stranger], self.LEFT, self.RIGHT) is None

    def test_no_faces_is_no_subject(self):
        assert morph.pick_subject([], self.LEFT, self.RIGHT) is None

    def test_a_face_beside_the_subject_cheek_to_cheek_is_not_them(self):
        """One eye can be close to the subject's while the other is a face away."""
        neighbour = face_at((648.0, 570.0), (860.0, 570.0))
        assert morph.pick_subject([neighbour], self.LEFT, self.RIGHT) is None

    def test_the_tolerance_is_a_share_of_the_eye_distance(self):
        """So it means the same at every output size."""
        near = face_at((432.0 + 50.0, 567.0), (648.0 + 50.0, 567.0))     # 50 of 216px
        far = face_at((432.0 + 60.0, 567.0), (648.0 + 60.0, 567.0))      # 60 of 216px
        assert morph.pick_subject([near], self.LEFT, self.RIGHT) is not None
        assert morph.pick_subject([far], self.LEFT, self.RIGHT) is None

    def test_eyes_are_matched_left_to_right_in_the_picture(self):
        """The mesh's own left and right are the subject's, mirrored in the
        picture; `analyze` stores them ordered by x, and so must this."""
        swapped = face_at((648.0, 567.0), (432.0, 567.0))
        assert morph.pick_subject([swapped], self.LEFT, self.RIGHT) is not None


# --------------------------------------------------------------------------- #
# The pixel half
# --------------------------------------------------------------------------- #

@requires_cv2
class TestTheTriangulation:

    def test_it_uses_only_the_points_it_was_given(self):
        """Subdiv2D adds four virtual corners of its own; none may survive."""
        points = morph.with_border(np.array([[30.0, 30.0], [70.0, 30.0], [50.0, 60.0]]),
                                   100, 100)
        triangles = morph.triangulate(points, 100, 100)
        assert len(triangles) > 0
        assert triangles.min() >= 0 and triangles.max() < len(points)

    def test_near_duplicate_points_keep_their_triangles(self):
        """Subdiv2D merges points closer than its tolerance, and the real mesh
        has ten iris points almost on top of eye-contour ones. Matched back by
        exact coordinate, their triangles were dropped and the fork's first real
        morph had holes across the eyes, nose and mouth."""
        face = np.array([[30.0, 30.0], [70.0, 30.0], [50.0, 62.0],
                         [50.0, 45.0], [50.0005, 45.0002], [30.0002, 30.0001]])
        points = morph.with_border(face, 100, 100)
        triangles = morph.triangulate(points, 100, 100)
        labels = morph.rasterize(points, triangles, 100, 100)
        assert (labels < 0).sum() == 0, "every pixel lies in a triangle"
        assert all(len(set(t)) == 3 for t in triangles), "no triangle names a point twice"

    def test_a_landmark_just_past_the_edge_does_not_stop_it(self):
        """MediaPipe extrapolates past the crop, and Subdiv2D refuses points
        outside its rectangle."""
        face = np.array([[30.0, 30.0], [70.0, 30.0], [50.0, 103.0]])
        triangles = morph.triangulate(morph.with_border(face, 100, 100), 100, 100)
        assert 2 in set(triangles.ravel())

    def test_the_same_points_give_the_same_triangles(self):
        """One shared topology per render is what keeps a still face from
        twitching between frames."""
        points = morph.with_border(np.array([[30.0, 30.0], [70.0, 30.0], [50.0, 60.0]]),
                                   100, 100)
        assert np.array_equal(morph.triangulate(points, 100, 100),
                              morph.triangulate(points, 100, 100))


@requires_cv2
class TestTheWarp:

    @staticmethod
    def scene():
        import cv2

        image = np.zeros((80, 80, 3), dtype=np.uint8)
        image[:] = (30, 90, 200)
        cv2.rectangle(image, (25, 25), (55, 55), (40, 160, 230), -1)
        points = morph.with_border(np.array([[25.0, 25.0], [55.0, 25.0], [55.0, 55.0],
                                             [25.0, 55.0], [40.0, 40.0]]), 80, 80)
        return image, points, morph.triangulate(points, 80, 80)

    def test_an_unmoved_mesh_returns_the_picture_exactly(self):
        image, points, triangles = self.scene()
        assert np.array_equal(morph.warp(image, points, points, triangles), image)

    def test_a_moved_mesh_moves_the_content(self):
        image, points, triangles = self.scene()
        moved = points.copy()
        moved[:5, 0] += 12.0
        out = morph.warp(image, points, moved, triangles)
        before = np.argwhere(image[:, :, 1] > 150)[:, 1].mean()
        after = np.argwhere(out[:, :, 1] > 150)[:, 1].mean()
        assert after > before + 8

    def test_every_pixel_is_written_even_with_the_face_pushed_off_the_edge(self):
        """The fork's first real morph left black wedges where triangles
        reached past the frame. A remap writes every pixel by construction;
        this holds it to that."""
        image, points, triangles = self.scene()
        moved = points.copy()
        moved[:5, 0] -= 30.0
        out = morph.warp(image, points, moved, triangles)
        assert np.all(out == 0, axis=2).sum() == 0


@requires_cv2
class TestMorphingAPair:

    @staticmethod
    def pair():
        """A white disc at two well-separated places, with a mesh on it."""
        import cv2

        a = np.zeros((80, 80, 3), dtype=np.uint8)
        b = np.zeros((80, 80, 3), dtype=np.uint8)
        cv2.circle(a, (22, 40), 8, (255, 255, 255), -1)
        cv2.circle(b, (58, 40), 8, (255, 255, 255), -1)

        def ring(cx):
            return np.array([[cx, 40.0], [cx - 8.0, 40.0], [cx + 8.0, 40.0],
                             [cx, 32.0], [cx, 48.0]])

        pa = morph.with_border(ring(22.0), 80, 80)
        pb = morph.with_border(ring(58.0), 80, 80)
        return a, b, pa, pb, morph.triangulate((pa + pb) / 2.0, 80, 80)

    def test_the_feature_travels_instead_of_appearing_twice(self):
        """What a crossfade does to a face: both places at half strength. A
        morph leaves one solid disc between them."""
        a, b, pa, pb, triangles = self.pair()
        morphed = morph.morph_pair(a, b, pa, pb, triangles, 0.5)[40, :, 0].astype(int)
        solid = np.flatnonzero(morphed > 200)
        assert solid.size > 8
        assert 30 < solid.mean() < 50

    def test_a_dissolve_really_does_show_both_places(self):
        """The baseline the test above is set against, so a change to the
        fixture cannot quietly make that comparison mean nothing."""
        a, b, _, _, _ = self.pair()
        row = morph.blend(a, b, 0.5)[40, :, 0].astype(int)
        assert np.flatnonzero(row > 200).size == 0
        lit = np.flatnonzero(row > 100)
        assert np.max(np.diff(lit)) > 5

    def test_the_ends_are_the_photographs(self):
        a, b, pa, pb, triangles = self.pair()
        assert np.array_equal(morph.morph_pair(a, b, pa, pb, triangles, 0.0), a)
        assert np.array_equal(morph.morph_pair(a, b, pa, pb, triangles, 1.0), b)


@requires_cv2
class TestTheSecondLookAsksForEveryFace:
    """Driven with a stand-in landmarker: the real one needs mediapipe."""

    def run(self, tmp_path, monkeypatch, faces, eyes):
        import cv2

        from grow_up import analyze

        asked = {}

        class Landmarker:
            def close(self):
                pass

        def build(opts, num_faces=1):
            asked["num_faces"] = num_faces
            return Landmarker()

        def detect_all(landmarker, opts, image, box, attempt):
            return [analyze.Look(points=f, blendshapes={}, matrix=None, attempt=attempt)
                    for f in faces]

        monkeypatch.setattr(analyze, "build_landmarker", build)
        monkeypatch.setattr(analyze, "detect_all", detect_all)
        frame = tmp_path / "a.jpg"
        cv2.imwrite(str(frame), np.zeros((1350, 1080, 3), dtype=np.uint8))
        found = morph.landmarks_for([frame], analyze.AnalyzeOptions(), [eyes])
        return found[0], asked

    def test_it_asks_for_more_than_one_face(self, tmp_path, monkeypatch):
        _, asked = self.run(tmp_path, monkeypatch, [], ((432.0, 567.0), (648.0, 567.0)))
        assert asked["num_faces"] > 1

    def test_it_keeps_the_subject_when_a_stranger_comes_first(self, tmp_path, monkeypatch):
        stranger = face_at((130.0, 184.0), (205.0, 184.0))
        subject = face_at((433.0, 568.0), (647.0, 567.0))
        found, _ = self.run(tmp_path, monkeypatch, [stranger, subject],
                            ((432.0, 567.0), (648.0, 567.0)))
        assert np.array_equal(found, subject)

    def test_unknown_eyes_are_not_guessed_at(self, tmp_path, monkeypatch):
        found, _ = self.run(tmp_path, monkeypatch, [face_at((432.0, 567.0), (648.0, 567.0))],
                            None)
        assert found is None


@requires_cv2
class TestRenderingThePairs:

    @staticmethod
    def frames(tmp_path, count=3):
        import cv2

        paths, meshes = [], []
        for i in range(count):
            image = np.zeros((60, 60, 3), dtype=np.uint8)
            cv2.circle(image, (20 + 8 * i, 30), 9, (255, 255, 255), -1)
            path = tmp_path / f"asset-{i}.jpg"
            cv2.imwrite(str(path), image)
            paths.append(path)
            meshes.append(np.array([[20.0 + 8 * i, 30.0], [11.0 + 8 * i, 30.0],
                                    [29.0 + 8 * i, 30.0]]))
        return paths, meshes

    def test_each_pair_gets_its_frames_named_after_its_two_photographs(self, tmp_path):
        """Never after a place in the timeline -- see the trap of that name."""
        paths, meshes = self.frames(tmp_path)
        made = morph.render(paths, meshes, tmp_path / "morph", 4, workers=1)
        assert [len(pair) for pair in made] == [4, 4]
        assert made[1][0].name == "asset-1--asset-2-001.jpg"
        assert all(p.exists() for pair in made for p in pair)

    def test_nothing_left_by_an_earlier_run_survives(self, tmp_path):
        """A frame from a run that failed part-way, or from before the
        selection changed, is a morph between two other photographs."""
        paths, meshes = self.frames(tmp_path)
        stale = tmp_path / "morph" / "asset-0--asset-1-001.jpg"
        stale.parent.mkdir()
        stale.write_bytes(b"from another run")
        leftover = tmp_path / "morph" / "asset-7--asset-9-003.jpg"
        leftover.write_bytes(b"from another selection")

        morph.render(paths, meshes, tmp_path / "morph", 2, workers=1)
        assert stale.read_bytes() != b"from another run"
        assert not leftover.exists()

    def test_a_pair_with_no_face_dissolves(self, tmp_path):
        import cv2

        paths, meshes = self.frames(tmp_path)
        meshes[1] = None
        made = morph.render(paths, meshes, tmp_path / "morph", 1, workers=1)
        a, b = (cv2.imread(str(p)) for p in paths[:2])
        middle = cv2.imread(str(made[0][0]))
        # Mean, not max: the frame went through JPEG, which rings at hard edges.
        assert np.abs(middle.astype(int) - morph.blend(a, b, 0.5).astype(int)).mean() < 1.5

    def test_no_face_anywhere_is_refused(self, tmp_path):
        paths, _ = self.frames(tmp_path)
        with pytest.raises(morph.MorphUnavailable):
            morph.render(paths, [None] * len(paths), tmp_path / "morph", 2, workers=1)

    def test_no_room_for_in_betweens_makes_none(self, tmp_path):
        paths, meshes = self.frames(tmp_path)
        assert morph.render(paths, meshes, tmp_path / "morph", 0, workers=1) == [[], []]

    def test_several_workers_make_the_same_frames_as_one(self, tmp_path):
        import cv2

        paths, meshes = self.frames(tmp_path)
        one = morph.render(paths, meshes, tmp_path / "one", 2, workers=1)
        one_pixels = [cv2.imread(str(p)) for pair in one for p in pair]
        many = morph.render(paths, meshes, tmp_path / "many", 2, workers=2)
        many_pixels = [cv2.imread(str(p)) for pair in many for p in pair]
        assert all(np.array_equal(x, y) for x, y in zip(one_pixels, many_pixels))


def _pid(_job) -> int:
    """Which process ran this. Top-level so a spawned child can import it."""
    import os

    return os.getpid()


class TestTheCutOutKeepsTheSubject:
    """A cut-out that lost the face would be a frame of plain grey with nobody
    on it -- worse than one photograph keeping its own background."""

    EYES = ((30.0, 40.0), (50.0, 40.0))

    @staticmethod
    def mask(person=True):
        mask = np.zeros((80, 80), dtype=np.uint8)
        if person:
            mask[20:70, 20:60] = 3
        return mask

    def test_the_bridge_of_the_nose_on_the_person_is_kept(self):
        assert morph.subject_at(self.mask(), self.EYES)

    def test_a_person_somewhere_else_in_the_frame_is_not_the_subject(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[60:80, 0:15] = 3
        assert not morph.subject_at(mask, self.EYES)

    def test_eyes_off_the_frame_are_not_kept(self):
        assert not morph.subject_at(self.mask(), ((300.0, 40.0), (320.0, 40.0)))

    def test_without_known_eyes_anyone_on_the_person_will_do(self):
        assert morph.subject_at(self.mask(), None)
        assert not morph.subject_at(self.mask(person=False), None)


def test_the_segmenter_is_looked_for_beside_the_landmarker():
    """Where `grow-up fetch-model` puts it, wherever the landmarker is configured."""
    from pathlib import Path

    assert (morph.segment_model_beside("elsewhere/face_landmarker.task")
            == Path("elsewhere") / morph.SEGMENT_MODEL_NAME)


@requires_cv2
class TestTheNeutralField:

    @staticmethod
    def scene():
        """A textured background with a person-shaped block on it."""
        rng = np.random.default_rng(7)
        image = rng.integers(0, 255, (90, 90, 3), dtype=np.uint8)
        mask = np.zeros((90, 90), dtype=np.uint8)
        mask[20:80, 25:65] = 3
        return image, mask

    def test_the_background_becomes_the_field(self):
        image, mask = self.scene()
        out = morph.on_neutral(image, mask)
        assert np.array_equal(out[:8, :8], np.broadcast_to(morph.NEUTRAL_BACKGROUND, (8, 8, 3)))

    def test_the_person_is_left_as_they_were(self):
        image, mask = self.scene()
        out = morph.on_neutral(image, mask)
        assert np.array_equal(out[35:65, 35:55], image[35:65, 35:55])

    def test_every_class_but_the_background_is_the_person(self):
        """Hair, skin, clothes and accessories: a cut-out without any one of
        them would be the wrong shape."""
        mask = np.repeat(np.arange(6, dtype=np.uint8), 20)[None, :].repeat(20, axis=0)
        alpha = morph.subject_alpha(mask, feather=0)
        assert alpha[:, :20].max() == 0.0
        assert alpha[:, 20:].min() == 1.0

    def test_the_silhouette_is_softened_not_cut_with_scissors(self):
        image, mask = self.scene()
        row = morph.subject_alpha(mask)[50, 15:35]
        assert ((row > 0.05) & (row < 0.95)).sum() >= 3

    def test_a_mask_that_does_not_fit_the_frame_is_refused(self):
        image, _ = self.scene()
        with pytest.raises(ValueError):
            morph.on_neutral(image, np.zeros((45, 45), dtype=np.uint8))


class TestTheSegmenterIsRequired:

    def test_without_it_the_cut_out_is_refused_and_says_how_to_get_it(self, tmp_path):
        """Never the photographs instead: the render is fingerprinted as the
        neutral one, and every later run would keep it."""
        with pytest.raises(morph.MorphUnavailable, match="grow-up fetch-model"):
            morph.neutral_frames([tmp_path / "a.jpg"], [None], tmp_path / "missing.tflite",
                                 tmp_path / "neutral")

    @pytest.mark.skipif(importlib.util.find_spec("mediapipe") is not None,
                        reason="only says something where mediapipe is missing")
    def test_without_mediapipe_it_stops_and_says_so(self, tmp_path):
        """Through the real child process: what fails there arrives here."""
        model = tmp_path / morph.SEGMENT_MODEL_NAME
        model.write_bytes(b"not a model")
        with pytest.raises(morph.MorphUnavailable, match="cannot load here"):
            morph.neutral_frames([tmp_path / "a.jpg"], [None], model, tmp_path / "neutral")


def test_the_segmenter_runs_in_a_process_of_its_own():
    """In the fork, a process that had built a segmenter could no longer start
    the worker pool `render` starts next."""
    import os

    assert morph._in_a_child(_pid, None) != os.getpid()


@requires_cv2
class TestCuttingOutTheFrames:
    """Driven with a stand-in segmenter, in this process: the real one needs mediapipe."""

    @pytest.fixture()
    def segmenter(self, monkeypatch, tmp_path):
        state = {"built": 0, "closed": 0, "runner": None, "lose": set()}

        class Segmenter:
            def close(self):
                state["closed"] += 1

        def build(model_path):
            state["built"] += 1
            return Segmenter()

        def segment(segmenter, image):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            if int(image[0, 0, 0]) not in state["lose"]:
                mask[10:50, 15:45] = 3
            return mask

        def in_a_child(work, job):
            state["runner"] = work
            return work(job)

        monkeypatch.setattr(morph, "build_segmenter", build)
        monkeypatch.setattr(morph, "segment", segment)
        monkeypatch.setattr(morph, "_in_a_child", in_a_child)
        model = tmp_path / morph.SEGMENT_MODEL_NAME
        model.write_bytes(b"model")
        state["model"] = model
        return state

    @staticmethod
    def frames(tmp_path, count=3):
        import cv2

        paths = []
        for i in range(count):
            image = np.full((60, 60, 3), 40 * (i + 1), dtype=np.uint8)
            path = tmp_path / f"asset-{i}.jpg"
            cv2.imwrite(str(path), image)
            paths.append(path)
        return paths

    EYES = ((24.0, 30.0), (36.0, 30.0))

    def cut(self, segmenter, paths, out):
        return morph.neutral_frames(paths, [self.EYES] * len(paths), segmenter["model"], out)

    def test_each_is_named_after_its_frame(self, segmenter, tmp_path):
        """Never after a place in the timeline -- see the trap of that name."""
        paths = self.frames(tmp_path)
        made = self.cut(segmenter, paths, tmp_path / "neutral")
        assert [p.name for p in made] == ["asset-0.jpg", "asset-1.jpg", "asset-2.jpg"]
        assert all(p.parent == tmp_path / "neutral" and p.exists() for p in made)

    def test_the_background_is_the_field_and_the_person_is_not(self, segmenter, tmp_path):
        import cv2

        [made] = self.cut(segmenter, self.frames(tmp_path, 1), tmp_path / "neutral")
        image = cv2.imread(str(made)).astype(int)
        # Within JPEG's error of the field in the corner, of the photograph inside.
        assert np.abs(image[0, 0] - np.array(morph.NEUTRAL_BACKGROUND)).max() <= 3
        assert np.abs(image[30, 30] - 40).max() <= 3

    def test_nothing_left_by_an_earlier_run_survives(self, segmenter, tmp_path):
        stale = tmp_path / "neutral" / "asset-9.jpg"
        stale.parent.mkdir()
        stale.write_bytes(b"from another selection")
        self.cut(segmenter, self.frames(tmp_path), tmp_path / "neutral")
        assert not stale.exists()

    def test_one_segmenter_serves_the_whole_render_and_is_closed(self, segmenter, tmp_path):
        """The fork built one per pair: 119 model loads across 12 processes."""
        self.cut(segmenter, self.frames(tmp_path), tmp_path / "neutral")
        assert segmenter["built"] == 1 and segmenter["closed"] == 1

    def test_the_work_is_handed_to_a_process_of_its_own(self, segmenter, tmp_path):
        self.cut(segmenter, self.frames(tmp_path), tmp_path / "neutral")
        assert segmenter["runner"] is morph._cut_out

    def test_a_frame_whose_cut_out_lost_the_subject_comes_back_empty(self, segmenter, tmp_path):
        segmenter["lose"] = {80}                  # the second frame's grey level
        made = self.cut(segmenter, self.frames(tmp_path), tmp_path / "neutral")
        assert made[1] is None
        assert made[0] is not None and made[2] is not None
        assert not (tmp_path / "neutral" / "asset-1.jpg").exists()


class TestFetchModelBringsTheSegmenter:
    """`fetch-model` is the only setup step, and everyone who set up before the
    segmenter existed already has the landmarker."""

    @pytest.fixture()
    def fetched(self, monkeypatch):
        from grow_up import cli

        urls = []

        def retrieve(url, path):
            urls.append(url)
            path.write_bytes(b"model")

        monkeypatch.setattr(cli.urllib.request, "urlretrieve", retrieve)
        monkeypatch.setattr(cli, "log", lambda _m: None)
        return urls

    @staticmethod
    def run(target, force=False):
        import argparse

        from grow_up import cli

        cli.cmd_fetch_model(argparse.Namespace(output=str(target), force=force))

    def test_an_existing_landmarker_does_not_stop_the_segmenter(self, tmp_path, fetched):
        target = tmp_path / "models" / "face_landmarker.task"
        target.parent.mkdir()
        target.write_bytes(b"already here")
        self.run(target)
        assert fetched == [morph.SEGMENT_MODEL_URL]
        assert target.read_bytes() == b"already here"

    def test_it_lands_beside_the_landmarker(self, tmp_path, fetched):
        target = tmp_path / "elsewhere" / "face_landmarker.task"
        self.run(target)
        assert target.exists()
        assert morph.segment_model_beside(target).exists()

    def test_with_both_present_nothing_is_downloaded(self, tmp_path, fetched):
        target = tmp_path / "face_landmarker.task"
        self.run(target)
        fetched.clear()
        self.run(target)
        assert fetched == []

    def test_force_fetches_both_again(self, tmp_path, fetched):
        target = tmp_path / "face_landmarker.task"
        self.run(target)
        fetched.clear()
        self.run(target, force=True)
        assert len(fetched) == 2
