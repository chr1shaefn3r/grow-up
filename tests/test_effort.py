"""The analyze stage's time/accuracy dial.

The retry ladder and ensemble are driven here by a fake landmarker, so the
ordering and short-circuiting are pinned down without mediapipe, a model
download or a single real photo.
"""

from __future__ import annotations

import numpy as np
import pytest

from grow_up import analyze, images, metrics
from grow_up.analyze import Attempt, AnalyzeOptions, Look


class FakeLandmark:
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y


class FakeCategory:
    def __init__(self, name: str, score: float):
        self.category_name, self.score = name, score


class FakeResult:
    def __init__(self, landmarks=None, blendshapes=None, matrix=None):
        self.face_landmarks = [landmarks] if landmarks else []
        self.face_blendshapes = [blendshapes] if blendshapes else []
        self.facial_transformation_matrixes = [matrix] if matrix is not None else []


class FakeLandmarker:
    """Records every crop it is handed and answers by a scripted policy."""

    def __init__(self, succeed_on=None, always=True):
        self.calls: list[tuple[int, int]] = []
        self.succeed_on = succeed_on          # indices that return a face
        self.always = always

    def detect(self, image):
        index = len(self.calls)
        self.calls.append(image.shape[:2])
        wanted = (self.succeed_on is None and self.always) or (
            self.succeed_on is not None and index in self.succeed_on)
        if not wanted:
            return FakeResult()

        # A plausible mesh: 478 points spread over the middle of the crop, with
        # the two iris centres placed apart on the horizontal.
        pts = [FakeLandmark(0.5, 0.5) for _ in range(metrics.N_LANDMARKS)]
        pts[metrics.IRIS_CENTER_A] = FakeLandmark(0.40, 0.45)
        pts[metrics.IRIS_CENTER_B] = FakeLandmark(0.60, 0.45)
        blend = [FakeCategory("eyeBlinkLeft", 0.1), FakeCategory("eyeBlinkRight", 0.1)]
        return FakeResult(pts, blend, np.eye(4))


class FakeImage:
    def __init__(self, data, **_):
        self.shape = data.shape


@pytest.fixture(autouse=True)
def fake_mediapipe(monkeypatch):
    """Stand in for `import mediapipe as mp` inside detect_once."""
    import sys
    import types

    module = types.ModuleType("mediapipe")
    module.Image = lambda image_format=None, data=None: FakeImage(data)
    module.ImageFormat = types.SimpleNamespace(SRGB="srgb")
    monkeypatch.setitem(sys.modules, "mediapipe", module)
    return module


def photo(width=800, height=600) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


BOX = (300, 200, 500, 400)


class TestPresets:
    def test_fast_matches_the_original_behaviour(self):
        preset = analyze.preset_for("fast")
        assert preset["retry_margins"] == ()
        assert preset["retry_rotations"] == ()
        assert preset["retry_equalize"] is False
        assert preset["ensemble"] == 1
        assert preset["max_crop_px"] == 0, "fast must stay bit-identical to before"

    def test_levels_escalate(self):
        fast, balanced, thorough = (analyze.preset_for(n) for n in analyze.EFFORT_LEVELS)
        assert len(fast["retry_margins"]) < len(balanced["retry_margins"])
        assert len(balanced["retry_margins"]) < len(thorough["retry_margins"])
        assert thorough["ensemble"] > balanced["ensemble"] == fast["ensemble"]
        assert thorough["retry_rotations"] and not balanced["retry_rotations"]

    def test_unknown_level_fails_loudly(self):
        """A typo silently running at a different level would be a poor way to
        learn what the dial does."""
        with pytest.raises(ValueError, match="unknown analyze.effort"):
            analyze.preset_for("thourough")

    def test_preset_is_a_copy(self):
        analyze.preset_for("fast")["ensemble"] = 99
        assert analyze.preset_for("fast")["ensemble"] == 1


class TestEnsembleMargins:
    def test_single_look_uses_the_configured_margin(self):
        assert analyze.ensemble_margins(AnalyzeOptions(bbox_margin=0.8)) == (0.8,)

    def test_base_margin_comes_first(self):
        margins = analyze.ensemble_margins(AnalyzeOptions(bbox_margin=0.8, ensemble=3))
        assert margins[0] == 0.8

    def test_spreads_either_side(self):
        margins = analyze.ensemble_margins(AnalyzeOptions(bbox_margin=0.8, ensemble=3))
        assert len(margins) == 3
        assert min(margins) < 0.8 < max(margins)

    def test_is_deterministic(self):
        """Re-analysis must reproduce the same metrics or the manifest stops
        being a trustworthy cache."""
        opts = AnalyzeOptions(bbox_margin=0.8, ensemble=5)
        assert analyze.ensemble_margins(opts) == analyze.ensemble_margins(opts)


class TestAttemptLadder:
    def test_fast_tries_exactly_one_framing(self):
        ladder = analyze.attempt_ladder(AnalyzeOptions(**analyze.preset_for("fast")))
        assert ladder == [Attempt(margin=0.8)]

    def test_order_is_ensemble_then_margins_then_equalize_then_rotation(self):
        opts = AnalyzeOptions(**analyze.preset_for("thorough"))
        ladder = analyze.attempt_ladder(opts)

        kinds = []
        for attempt in ladder:
            if attempt.rotation:
                kinds.append("rotate")
            elif attempt.equalize:
                kinds.append("equalize")
            else:
                kinds.append("margin")
        assert kinds == sorted(kinds, key=["margin", "equalize", "rotate"].index)
        assert kinds.count("rotate") == 2

    def test_balanced_has_no_rotations(self):
        opts = AnalyzeOptions(**analyze.preset_for("balanced"))
        assert not any(a.rotation for a in analyze.attempt_ladder(opts))


class TestRetryLadder:
    """Margin-only options throughout: retrying a different crop touches no
    opencv, so these run in CI alongside everything else. The equalise and
    rotate rungs are covered separately, where opencv is available."""

    def margins_only(self, retries=(1.6, 0.4), ensemble=1) -> AnalyzeOptions:
        return AnalyzeOptions(bbox_margin=0.8, retry_margins=retries,
                              ensemble=ensemble)

    def test_fast_never_retries(self):
        """The whole point of the default: one look, no matter what."""
        opts = AnalyzeOptions(**analyze.preset_for("fast"))
        landmarker = FakeLandmarker(succeed_on=set())

        assert analyze.gather_looks(landmarker, opts, photo(), BOX) == []
        assert len(landmarker.calls) == 1

    def test_success_short_circuits_the_ladder(self):
        landmarker = FakeLandmarker(succeed_on={0})
        looks = analyze.gather_looks(landmarker, self.margins_only(), photo(), BOX)

        assert len(looks) == 1
        assert len(landmarker.calls) == 1, "retries cost nothing when the first look works"

    def test_retries_rescue_a_first_pass_failure(self):
        """The 124 no_face_detected photos are exactly this case."""
        landmarker = FakeLandmarker(succeed_on={2})
        looks = analyze.gather_looks(landmarker, self.margins_only(), photo(), BOX)

        assert len(looks) == 1
        assert len(landmarker.calls) == 3

    def test_walks_the_whole_ladder_before_giving_up(self):
        opts = self.margins_only()
        landmarker = FakeLandmarker(succeed_on=set())

        assert analyze.gather_looks(landmarker, opts, photo(), BOX) == []
        assert len(landmarker.calls) == len(analyze.attempt_ladder(opts)) == 3

    def test_retries_are_tried_in_ladder_order(self):
        opts = self.margins_only(retries=(1.6, 0.4))
        landmarker = FakeLandmarker(succeed_on=set())
        analyze.gather_looks(landmarker, opts, photo(), BOX)

        # Wider margin means a bigger crop; the ladder goes base, wider, tighter.
        areas = [h * w for h, w in landmarker.calls]
        assert areas[1] > areas[0] > areas[2]

    def test_stops_once_the_ensemble_is_full(self):
        opts = self.margins_only(ensemble=3)
        landmarker = FakeLandmarker()  # everything succeeds
        looks = analyze.gather_looks(landmarker, opts, photo(), BOX)

        assert len(looks) == 3
        assert len(landmarker.calls) == 3, "no retries once the ensemble is satisfied"

    def test_ensemble_keeps_looking_when_members_fail(self):
        """A partly-failing ensemble should fall through to the retries rather
        than settle for one look."""
        opts = self.margins_only(ensemble=3)
        landmarker = FakeLandmarker(succeed_on={0, 3, 4})
        looks = analyze.gather_looks(landmarker, opts, photo(), BOX)

        assert len(looks) == 3
        assert len(landmarker.calls) == 5


class TestPixelRungs:
    """The equalise and rotate rungs, which do touch opencv."""

    @pytest.fixture(autouse=True)
    def needs_cv2(self):
        pytest.importorskip("cv2")

    def test_equalize_rung_runs_after_the_margins(self):
        opts = AnalyzeOptions(bbox_margin=0.8, retry_margins=(1.6,),
                              retry_equalize=True)
        landmarker = FakeLandmarker(succeed_on={2})

        looks = analyze.gather_looks(landmarker, opts, photo(), BOX)
        assert len(looks) == 1
        assert looks[0].attempt.equalize is True

    def test_rotation_rung_runs_last(self):
        opts = AnalyzeOptions(bbox_margin=0.8, retry_equalize=True,
                              retry_rotations=(-25.0, 25.0))
        landmarker = FakeLandmarker(succeed_on={2})

        looks = analyze.gather_looks(landmarker, opts, photo(), BOX)
        assert looks[0].attempt.rotation == -25.0

    def test_thorough_walks_every_rung(self):
        opts = AnalyzeOptions(**analyze.preset_for("thorough"))
        landmarker = FakeLandmarker(succeed_on=set())

        assert analyze.gather_looks(landmarker, opts, photo(), BOX) == []
        assert len(landmarker.calls) == len(analyze.attempt_ladder(opts)) == 9


class TestDetectOnce:
    def test_maps_landmarks_into_full_image_coordinates(self):
        landmarker = FakeLandmarker()
        opts = AnalyzeOptions(bbox_margin=0.0)

        look = analyze.detect_once(landmarker, opts, photo(), BOX, Attempt(margin=0.0))

        # The crop is the box itself, so 0.5/0.5 is its centre in full coords.
        assert look is not None
        assert look.points[0] == pytest.approx([400.0, 300.0], abs=1.0)

    def test_returns_none_when_nothing_is_found(self):
        landmarker = FakeLandmarker(succeed_on=set())
        look = analyze.detect_once(landmarker, AnalyzeOptions(), photo(), BOX,
                                   Attempt(margin=0.8))
        assert look is None

    def test_rejects_a_degenerate_crop(self):
        landmarker = FakeLandmarker()
        look = analyze.detect_once(landmarker, AnalyzeOptions(), photo(),
                                   (10, 10, 14, 14), Attempt(margin=0.0))
        assert look is None
        assert landmarker.calls == [], "not worth an inference"

    def test_downscale_still_yields_full_resolution_coordinates(self):
        """max_crop_px changes what the model sees, never what we record."""
        pytest.importorskip("cv2")
        landmarker = FakeLandmarker()
        big_box = (0, 0, 4000, 3000)
        image = photo(4000, 3000)

        plain = analyze.detect_once(landmarker, AnalyzeOptions(max_crop_px=0),
                                    image, big_box, Attempt(margin=0.0))
        scaled = analyze.detect_once(landmarker, AnalyzeOptions(max_crop_px=512),
                                     image, big_box, Attempt(margin=0.0))

        assert landmarker.calls[1][0] <= 512 and landmarker.calls[1][1] <= 512
        assert scaled.points[0] == pytest.approx(plain.points[0], abs=2.0)


class TestRotationRoundTrip:
    @pytest.mark.parametrize("degrees", [-25.0, -10.0, 10.0, 25.0])
    def test_points_return_to_their_original_position(self, degrees):
        matrix = images.rotation_matrix(degrees, 400, 300)
        point = np.array([[137.0, 208.0]])

        rotated = (matrix[:, :2] @ point.T).T + matrix[:, 2]
        assert images.unrotate_points(rotated, matrix) == pytest.approx(point, abs=1e-6)

    def test_rotated_detection_lands_in_original_coordinates(self):
        pytest.importorskip("cv2")
        landmarker = FakeLandmarker()
        opts = AnalyzeOptions(bbox_margin=0.0)
        image = photo()

        upright = analyze.detect_once(landmarker, opts, image, BOX, Attempt(margin=0.0))
        turned = analyze.detect_once(landmarker, opts, image, BOX,
                                     Attempt(margin=0.0, rotation=25.0))

        # The fake mesh sits at the crop centre, which rotation leaves fixed.
        assert turned.points[0] == pytest.approx(upright.points[0], abs=1.0)


class TestCombineLooks:
    def look(self, x: float, blink: float = 0.1, yaw_matrix=None) -> Look:
        pts = np.full((metrics.N_LANDMARKS, 2), x, dtype=np.float64)
        return Look(points=pts, blendshapes={"eyeBlinkLeft": blink},
                    matrix=yaw_matrix if yaw_matrix is not None else np.eye(4),
                    attempt=Attempt(margin=0.8))

    def test_takes_the_median_landmark(self):
        pts, _, _ = analyze.combine_looks(
            [self.look(10.0), self.look(12.0), self.look(11.0)])
        assert pts[0][0] == pytest.approx(11.0)

    def test_one_bad_look_cannot_drag_the_result(self):
        """The reason for a median rather than a mean."""
        pts, _, _ = analyze.combine_looks(
            [self.look(10.0), self.look(11.0), self.look(900.0)])
        assert pts[0][0] == pytest.approx(11.0)

    def test_medians_blendshapes_per_name(self):
        _, blend, _ = analyze.combine_looks(
            [self.look(1, blink=0.1), self.look(1, blink=0.9), self.look(1, blink=0.2)])
        assert blend["eyeBlinkLeft"] == pytest.approx(0.2)

    def test_missing_blendshape_counts_as_zero(self):
        looks = [self.look(1, blink=0.4), self.look(1, blink=0.4)]
        looks[1] = Look(points=looks[1].points, blendshapes={}, matrix=np.eye(4),
                        attempt=Attempt(margin=0.8))
        _, blend, _ = analyze.combine_looks(looks)
        assert blend["eyeBlinkLeft"] == pytest.approx(0.2)

    def test_pose_is_medianed_as_angles_not_matrices(self):
        """The mean of rotation matrices is not itself a rotation."""
        import math

        def rot_y(deg):
            r = math.radians(deg)
            m = np.eye(4)
            m[:3, :3] = [[math.cos(r), 0, math.sin(r)], [0, 1, 0],
                         [-math.sin(r), 0, math.cos(r)]]
            return m

        _, _, pose = analyze.combine_looks(
            [self.look(1, yaw_matrix=rot_y(d)) for d in (10.0, 20.0, 15.0)])
        assert pose[0] == pytest.approx(15.0, abs=1e-6)

    def test_single_look_passes_through(self):
        pts, blend, pose = analyze.combine_looks([self.look(7.0)])
        assert pts[0][0] == pytest.approx(7.0)
        assert pose == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)

    def test_survives_looks_without_a_pose_matrix(self):
        look = Look(points=np.zeros((metrics.N_LANDMARKS, 2)), blendshapes={},
                    matrix=None, attempt=Attempt(margin=0.8))
        assert analyze.combine_looks([look])[2] is None


class TestOptionResolution:
    """`analyze.effort` picks a bundle; a lone setting beside it still wins."""

    def build(self, raw: dict, **kwargs) -> AnalyzeOptions:
        from pathlib import Path

        from grow_up import cli, config

        return cli._analyze_options(config.Config(raw={"analyze": raw},
                                                  root=Path(".")), **kwargs)

    def test_defaults_to_fast(self):
        opts = self.build({})
        assert opts.effort == "fast"
        assert opts.ensemble == 1 and opts.retry_margins == ()

    def test_effort_selects_the_whole_bundle(self):
        opts = self.build({"effort": "thorough"})
        assert opts.ensemble == 3
        assert opts.retry_rotations == (-25.0, 25.0)
        assert opts.retry_equalize is True

    def test_explicit_setting_overrides_the_preset(self):
        opts = self.build({"effort": "thorough", "ensemble": 5})
        assert opts.ensemble == 5, "the lone setting wins"
        assert opts.retry_rotations == (-25.0, 25.0), "the rest of the preset stands"

    def test_flag_overrides_the_configured_level(self):
        opts = self.build({"effort": "fast"}, effort="balanced")
        assert opts.effort == "balanced" and opts.retry_equalize is True

    def test_unknown_level_in_config_fails_loudly(self):
        with pytest.raises(ValueError, match="unknown analyze.effort"):
            self.build({"effort": "maximum"})

    def test_presence_confidence_is_exposed(self):
        assert self.build({"min_face_presence_confidence": 0.2}
                          ).min_face_presence_confidence == 0.2


class FakePool:
    """Stands in for ProcessPoolExecutor with precomputed results."""

    def __init__(self, results):
        self.results = results

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, jobs, chunksize=1):
        return iter(self.results)


class TestPersist:
    """`trial --compare` must not leave stored metrics at whichever level
    happened to run last."""

    def prepare(self, tmp_path, monkeypatch, results):
        from grow_up import db, pipeline

        conn = db.connect(tmp_path / "t.sqlite")
        stamp = "2026-01-01T00:00:00.000Z"
        conn.execute("BEGIN")
        for asset_id, _ in results:
            conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                         " VALUES (?, '2026-01-01', ?)", (asset_id, stamp))
            conn.execute("INSERT INTO faces (asset_id, status, x1, y1, x2, y2,"
                         " image_width, image_height, n_candidates, fetched_at)"
                         " VALUES (?, 'ok', 0, 0, 10, 10, 100, 100, 1, ?)",
                         (asset_id, stamp))
            conn.execute("INSERT INTO downloads (asset_id, path, source, fetched_at)"
                         " VALUES (?, '/x.jpg', 'original', ?)", (asset_id, stamp))
        conn.execute("COMMIT")

        monkeypatch.setattr(pipeline, "ProcessPoolExecutor",
                            lambda **kwargs: FakePool(results))
        return conn, pipeline

    def results(self):
        return [("a0", metrics.FaceMetrics(detected=1, yaw=1.0)),
                ("a1", metrics.FaceMetrics(detected=0))]

    def test_persist_false_writes_nothing(self, tmp_path, monkeypatch):
        conn, pipeline = self.prepare(tmp_path, monkeypatch, self.results())

        done = pipeline.stage_analyze(conn, AnalyzeOptions(), 1, lambda _: None,
                                      persist=False)

        assert done == 2
        assert conn.execute("SELECT count(*) FROM metrics").fetchone()[0] == 0

    def test_persist_true_still_writes(self, tmp_path, monkeypatch):
        conn, pipeline = self.prepare(tmp_path, monkeypatch, self.results())

        pipeline.stage_analyze(conn, AnalyzeOptions(), 1, lambda _: None, persist=True)

        assert conn.execute("SELECT count(*) FROM metrics").fetchone()[0] == 2

    def test_collect_hands_back_the_metrics(self, tmp_path, monkeypatch):
        """--compare scores these in memory instead of reading them back."""
        conn, pipeline = self.prepare(tmp_path, monkeypatch, self.results())
        collected: list = []

        pipeline.stage_analyze(conn, AnalyzeOptions(), 1, lambda _: None,
                               persist=False, collect=collected)

        assert [asset_id for asset_id, _ in collected] == ["a0", "a1"]
        assert sum(1 for _, m in collected if m.detected) == 1


class TestGeometricGaze:
    def mesh(self, iris_offset: float = 0.0) -> np.ndarray:
        pts = np.zeros((metrics.N_LANDMARKS, 2), dtype=np.float64)
        pts[metrics.LEFT_EYE_CORNERS[0]] = [100.0, 200.0]
        pts[metrics.LEFT_EYE_CORNERS[1]] = [140.0, 200.0]
        pts[metrics.RIGHT_EYE_CORNERS[0]] = [200.0, 200.0]
        pts[metrics.RIGHT_EYE_CORNERS[1]] = [240.0, 200.0]
        pts[metrics.IRIS_CENTER_A] = [120.0 + iris_offset, 200.0]
        pts[metrics.IRIS_CENTER_B] = [220.0 + iris_offset, 200.0]
        return pts

    def test_centred_irises_read_as_looking_ahead(self):
        assert metrics.gaze_from_geometry(self.mesh()) == pytest.approx((0.0, 0.0))

    def test_offset_irises_read_as_looking_aside(self):
        right = metrics.gaze_from_geometry(self.mesh(iris_offset=10.0))[0]
        left = metrics.gaze_from_geometry(self.mesh(iris_offset=-10.0))[0]
        assert right > 0 > left
        assert right == pytest.approx(-left)

    def test_is_independent_of_face_size(self):
        """A distant face and a close-up looking the same way must agree."""
        near = self.mesh(iris_offset=10.0)
        far = near * 0.25
        assert metrics.gaze_from_geometry(far)[0] == pytest.approx(
            metrics.gaze_from_geometry(near)[0])

    def test_degenerate_mesh_reads_as_neutral(self):
        assert metrics.gaze_from_geometry(
            np.zeros((metrics.N_LANDMARKS, 2))) == (0.0, 0.0)

    def test_reads_vertical_gaze_too(self):
        """Looking up moves both irises above the corner midpoint."""
        pts = self.mesh()
        pts[metrics.IRIS_CENTER_A] = [120.0, 190.0]
        pts[metrics.IRIS_CENTER_B] = [220.0, 190.0]
        assert metrics.gaze_from_geometry(pts)[1] > 0
