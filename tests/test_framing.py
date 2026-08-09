"""Output framing: how much room is left around the face.

Faces were being cut off at the frame edges. Not a bug -- the framing constant.
The eyes spanned 0.28 of the width, which puts a head (2.4-3.0x the interocular
distance across) at 67-84% of the frame, touching the edges by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from grow_up import align, config, metrics, pipeline

WIDTH, HEIGHT = 1080, 1350
ASPECT = WIDTH / HEIGHT
LEVEL = 0.42

# Landmark-mesh extents in interocular units. The mesh covers the face but not
# hair, ears or the cranium above it, which is what fit_margin allows for.
CHILD = dict(span_w=2.2, span_up=1.1, span_down=1.75)
INFANT = dict(span_w=2.6, span_up=1.3, span_down=1.9)
MARGIN = 1.5


class TestTargetEyesFrom:
    def test_places_eyes_symmetrically_about_the_centre(self):
        left, right = align.target_eyes_from(WIDTH, HEIGHT, 0.20, LEVEL)
        assert (left[0] + right[0]) / 2 == pytest.approx(WIDTH / 2)

    def test_separation_is_the_requested_fraction_of_width(self):
        left, right = align.target_eyes_from(WIDTH, HEIGHT, 0.20, LEVEL)
        assert right[0] - left[0] == pytest.approx(0.20 * WIDTH)

    def test_both_eyes_sit_on_the_eye_line(self):
        left, right = align.target_eyes_from(WIDTH, HEIGHT, 0.20, 0.35)
        assert left[1] == right[1] == pytest.approx(0.35 * HEIGHT)

    def test_reproduces_the_previous_hand_written_coordinates(self):
        """0.28 is what [0.36, 0.42] / [0.64, 0.42] expressed."""
        old = align.target_eyes(WIDTH, HEIGHT, (0.36, LEVEL), (0.64, LEVEL))
        new = align.target_eyes_from(WIDTH, HEIGHT, 0.28, LEVEL)
        assert new[0] == pytest.approx(old[0])
        assert new[1] == pytest.approx(old[1])

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5])
    def test_rejects_impossible_separations(self, bad):
        with pytest.raises(ValueError, match="eye_distance"):
            align.target_eyes_from(WIDTH, HEIGHT, bad, LEVEL)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.1])
    def test_rejects_impossible_eye_levels(self, bad):
        with pytest.raises(ValueError, match="eye_level"):
            align.target_eyes_from(WIDTH, HEIGHT, 0.2, bad)


class TestHeadFits:
    """The arithmetic this whole change rests on."""

    def test_the_old_framing_clips_and_the_new_one_does_not(self):
        for face in (CHILD, INFANT):
            assert not align.head_fits(0.28, LEVEL, aspect=ASPECT, margin=MARGIN, **face)
            assert align.head_fits(0.20, LEVEL, aspect=ASPECT, margin=MARGIN, **face)

    def test_a_wider_face_is_the_first_to_clip(self):
        """An infant's cranium is large relative to eye spacing, which is why
        the earliest photos suffer worst."""
        # The child tolerates up to ~0.276, the infant only ~0.254.
        distance = 0.26
        assert align.head_fits(distance, LEVEL, aspect=ASPECT, margin=MARGIN, **CHILD)
        assert not align.head_fits(distance, LEVEL, aspect=ASPECT, margin=MARGIN, **INFANT)

    def test_zooming_out_always_helps(self):
        fits = [align.head_fits(d, LEVEL, aspect=ASPECT, margin=MARGIN, **INFANT)
                for d in (0.40, 0.30, 0.25, 0.20, 0.15)]
        assert fits == sorted(fits), "monotonic: never fits, then clips, then fits"

    def test_catches_a_chin_below_the_frame(self):
        assert not align.head_fits(0.20, LEVEL, span_w=1.0, span_up=0.1, span_down=9.0,
                                   aspect=ASPECT, margin=1.0)

    def test_catches_a_forehead_above_the_frame(self):
        assert not align.head_fits(0.20, LEVEL, span_w=1.0, span_up=9.0, span_down=0.1,
                                   aspect=ASPECT, margin=1.0)

    def test_catches_a_face_wider_than_the_frame(self):
        assert not align.head_fits(0.20, LEVEL, span_w=20.0, span_up=0.1, span_down=0.1,
                                   aspect=ASPECT, margin=1.0)

    def test_margin_makes_the_check_stricter(self):
        loose = align.head_fits(0.26, LEVEL, aspect=ASPECT, margin=1.0, **INFANT)
        tight = align.head_fits(0.26, LEVEL, aspect=ASPECT, margin=MARGIN, **INFANT)
        assert loose and not tight


class TestFittingEyeDistance:
    def test_the_suggestion_actually_fits(self):
        for face in (CHILD, INFANT):
            suggested = align.fitting_eye_distance(eye_level=LEVEL, aspect=ASPECT,
                                                   margin=MARGIN, **face)
            assert align.head_fits(suggested, LEVEL, aspect=ASPECT, margin=MARGIN, **face)

    @pytest.mark.parametrize("seed", range(20))
    def test_holds_for_arbitrary_face_proportions(self, seed):
        """Randomised spans: the suggestion must fit, and anything larger must not."""
        rng = np.random.default_rng(seed)
        face = dict(span_w=rng.uniform(1.5, 4.0), span_up=rng.uniform(0.5, 2.5),
                    span_down=rng.uniform(0.8, 3.0))
        level = rng.uniform(0.3, 0.6)

        suggested = align.fitting_eye_distance(eye_level=level, aspect=ASPECT,
                                               margin=MARGIN, **face)

        assert align.head_fits(suggested * 0.999, level, aspect=ASPECT,
                               margin=MARGIN, **face)
        assert not align.head_fits(suggested * 1.02, level, aspect=ASPECT,
                                   margin=MARGIN, **face)

    def test_degenerate_spans_do_not_divide_by_zero(self):
        assert align.fitting_eye_distance(0.0, 0.0, 0.0, LEVEL, ASPECT) == 1.0


class TestFaceSpans:
    def mesh(self, half_width=100.0, up=60.0, down=90.0) -> np.ndarray:
        pts = np.zeros((metrics.N_LANDMARKS, 2), dtype=np.float64)
        pts[:, 0] = 500.0
        pts[:, 1] = 400.0
        pts[0] = [500.0 - half_width, 400.0 - up]
        pts[1] = [500.0 + half_width, 400.0 + down]
        return pts

    def test_measures_in_interocular_units(self):
        pts = self.mesh(half_width=100.0, up=60.0, down=90.0)
        left, right = np.array([450.0, 400.0]), np.array([550.0, 400.0])

        span_w, span_up, span_down = metrics.face_spans(pts, left, right, 100.0)

        assert span_w == pytest.approx(2.0)
        assert span_up == pytest.approx(0.6)
        assert span_down == pytest.approx(0.9)

    def test_is_independent_of_photo_resolution(self):
        """The point of the normalisation: the same face at any scale reads the
        same, because alignment holds interocular distance constant."""
        pts = self.mesh()
        left, right = np.array([450.0, 400.0]), np.array([550.0, 400.0])
        near = metrics.face_spans(pts, left, right, 100.0)
        far = metrics.face_spans(pts * 0.25, left * 0.25, right * 0.25, 25.0)
        assert far == pytest.approx(near)

    def test_uses_the_eye_line_not_the_mesh_centre(self):
        pts = self.mesh(up=60.0, down=90.0)
        left, right = np.array([450.0, 380.0]), np.array([550.0, 420.0])  # tilted
        _, span_up, span_down = metrics.face_spans(pts, left, right, 100.0)
        assert span_up == pytest.approx(0.6) and span_down == pytest.approx(0.9)

    def test_degenerate_interocular_is_survivable(self):
        assert metrics.face_spans(self.mesh(), np.zeros(2), np.zeros(2), 0.0) == (
            0.0, 0.0, 0.0)


class TestFitReport:
    def rows(self, spans: list[tuple | None]) -> list[dict]:
        return [{"span_w": s[0], "span_up": s[1], "span_down": s[2]} if s
                else {"span_w": None, "span_up": None, "span_down": None}
                for s in spans]

    def report(self, rows, eye_distance=0.28) -> list[str]:
        lines: list[str] = []
        pipeline._report_fit(rows, eye_distance, LEVEL, ASPECT, MARGIN, lines.append)
        return lines

    def test_names_the_count_and_a_working_suggestion(self):
        lines = self.report(self.rows([tuple(INFANT.values())] * 3))

        assert len(lines) == 1
        assert "3 frames clip" in lines[0]
        suggested = float(lines[0].split("; ")[1].split()[0])
        assert align.head_fits(suggested, LEVEL, aspect=ASPECT, margin=MARGIN, **INFANT)

    def test_silent_when_everything_fits(self):
        assert self.report(self.rows([tuple(CHILD.values())]), eye_distance=0.20) == []

    def test_counts_unjudged_frames_separately(self):
        """Rows analyzed before spans were recorded must not be assumed to fit."""
        lines = self.report(self.rows([None, None, tuple(CHILD.values())]))
        joined = "\n".join(lines)
        assert "2 frames analyzed before face spans" in joined
        assert "--reanalyze" in joined

    def test_all_unjudged_reports_only_that(self):
        lines = self.report(self.rows([None, None]))
        assert len(lines) == 1 and "not checked" in lines[0]

    def test_suggestion_is_the_tightest_across_frames(self):
        lines = self.report(self.rows([tuple(CHILD.values()), tuple(INFANT.values())]))
        suggested = float(lines[0].split("; ")[1].split()[0])
        for face in (CHILD, INFANT):
            assert align.head_fits(suggested, LEVEL, aspect=ASPECT, margin=MARGIN, **face)


class TestRemovedConfigKeys:
    """The old keys are gone; a config still carrying them must say so rather
    than be quietly ignored, which would read as the new setting not working."""

    @pytest.mark.parametrize("key", ["left_eye", "right_eye"])
    def test_old_eye_coordinates_are_rejected(self, key):
        with pytest.raises(RuntimeError, match="no longer exist"):
            config.check_removed({"output": {key: [0.36, 0.42]}})

    def test_the_message_names_the_replacement(self):
        with pytest.raises(RuntimeError, match="align.eye_distance"):
            config.check_removed({"output": {"left_eye": [0.36, 0.42]}})

    def test_obsolete_smoothing_keys_are_rejected(self):
        with pytest.raises(RuntimeError, match="smoothing_window"):
            config.check_removed({"output": {"smoothing_window": 9}})

    def test_a_clean_config_passes(self):
        config.check_removed({"output": {"width": 1080}, "align": {"eye_distance": 0.2}})

    def test_reports_every_stale_key_at_once(self):
        with pytest.raises(RuntimeError) as excinfo:
            config.check_removed({"output": {"left_eye": [0, 0], "right_eye": [1, 1]}})
        assert "left_eye" in str(excinfo.value) and "right_eye" in str(excinfo.value)


class TestMigration:
    """`CREATE TABLE IF NOT EXISTS` leaves an existing table alone, so columns
    added later have to be applied explicitly or an older database keeps the old
    shape and every query for them fails."""

    def old_database(self, tmp_path):
        import sqlite3

        path = tmp_path / "old.sqlite"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE metrics (asset_id TEXT PRIMARY KEY,"
                     " detected INTEGER NOT NULL, analyzed_at TEXT NOT NULL)")
        conn.execute("INSERT INTO metrics VALUES ('a0', 1, 'now')")
        conn.commit()
        conn.close()
        return path

    def test_adds_the_span_columns(self, tmp_path):
        from grow_up import db

        conn = db.connect(self.old_database(tmp_path))
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(metrics)")}
        assert {"span_w", "span_up", "span_down"} <= columns

    def test_existing_rows_survive_with_null_spans(self, tmp_path):
        """Which is what makes them 'unjudged' rather than assumed to fit."""
        from grow_up import db

        conn = db.connect(self.old_database(tmp_path))
        row = conn.execute("SELECT asset_id, span_w FROM metrics").fetchone()
        assert row["asset_id"] == "a0" and row["span_w"] is None

    def test_is_idempotent(self, tmp_path):
        from grow_up import db

        conn = db.connect(self.old_database(tmp_path))
        assert db.migrate(conn) == [], "nothing left to add on a second pass"

    def test_a_fresh_database_needs_no_migration(self, tmp_path):
        from grow_up import db

        conn = db.connect(tmp_path / "fresh.sqlite")
        assert db.migrate(conn) == []


class TestFill:
    @pytest.fixture(autouse=True)
    def needs_cv2(self):
        pytest.importorskip("cv2")

    def setup_warp(self):
        """A transform that leaves part of the output past the source edge."""
        rng = np.random.default_rng(1)
        image = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
        matrix = np.array([[1.0, 0.0, 100.0], [0.0, 1.0, 100.0]])
        return image, matrix

    def test_footprint_marks_where_real_pixels_land(self):
        image, matrix = self.setup_warp()
        mask = align.source_footprint(image, matrix, 400, 400)

        assert mask[150, 150] > 0, "inside the shifted source"
        assert mask[10, 10] == 0, "outside it"

    def test_fill_leaves_real_pixels_untouched(self):
        image, matrix = self.setup_warp()
        edge = align.warp(image, matrix, 400, 400, fill="edge")
        blurred = align.warp(image, matrix, 400, 400, fill="blur")

        mask = align.source_footprint(image, matrix, 400, 400) > 127
        assert np.array_equal(edge[mask], blurred[mask]), "only outside may differ"

    def test_blur_and_black_differ_outside_the_footprint(self):
        image, matrix = self.setup_warp()
        blurred = align.warp(image, matrix, 400, 400, fill="blur")
        black = align.warp(image, matrix, 400, 400, fill="black")

        outside = ~(align.source_footprint(image, matrix, 400, 400) > 127)
        assert black[outside].max() == 0
        assert blurred[outside].max() > 0

    def test_edge_mode_is_the_original_behaviour(self):
        import cv2

        image, matrix = self.setup_warp()
        expected = cv2.warpAffine(image, matrix, (400, 400), flags=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_REPLICATE)
        assert np.array_equal(align.warp(image, matrix, 400, 400, fill="edge"), expected)

    def test_a_fully_covered_frame_is_unchanged_by_fill(self):
        rng = np.random.default_rng(2)
        image = rng.integers(0, 255, size=(400, 400, 3), dtype=np.uint8)
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        assert np.array_equal(align.warp(image, identity, 200, 200, fill="blur"),
                              align.warp(image, identity, 200, 200, fill="edge"))

    def test_unknown_fill_fails_loudly(self):
        image, matrix = self.setup_warp()
        with pytest.raises(ValueError, match="unknown align.fill"):
            align.warp(image, matrix, 400, 400, fill="vignette")
