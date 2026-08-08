from __future__ import annotations

from datetime import datetime

import pytest

from grow_up import db, select

LIMITS = {
    "max_yaw": 20.0, "max_pitch": 18.0, "max_roll": 25.0, "max_gaze": 0.35,
    "max_blink": 0.45, "max_oob_frac": 0.005, "allow_bbox_clipped": False,
    "min_interocular_px": 60.0, "min_sharpness": 12.0,
    "min_exposure_lo": 8.0, "max_exposure_hi": 250.0,
}
WEIGHTS = {"w_pose": 1.0, "w_gaze": 1.0, "w_eyes_open": 1.0,
           "w_sharpness": 1.0, "w_size": 0.5}


@pytest.fixture()
def conn(tmp_path):
    return db.connect(tmp_path / "test.sqlite")


def add(conn, asset_id: str, when: str, *, sharpness: float = 100.0, yaw: float = 2.0,
        gaze: float = 0.0, interocular: float = 150.0) -> None:
    conn.execute(
        "INSERT INTO assets (id, local_datetime, indexed_at) VALUES (?, ?, '2026-01-01T00:00:00.000Z')",
        (asset_id, when),
    )
    conn.execute(
        "INSERT INTO metrics (asset_id, detected, yaw, pitch, roll, gaze_x, gaze_y,"
        " blink_l, blink_r, oob_frac, bbox_clipped, interocular_px, sharpness,"
        " exposure_lo, exposure_hi, analyzed_at)"
        " VALUES (?, 1, ?, 1.0, 1.0, ?, 0.0, 0.05, 0.05, 0.0, 0, ?, ?, 40.0, 200.0,"
        " '2026-01-01T00:00:00.000Z')",
        (asset_id, yaw, gaze, interocular, sharpness),
    )


class TestBucketKey:
    @pytest.mark.parametrize("cadence,expected", [
        ("day", "2026-03-14"),
        ("week", "2026-W11"),
        ("month", "2026-03"),
    ])
    def test_labels(self, cadence, expected):
        assert select.bucket_key(datetime(2026, 3, 14, 9, 30), cadence) == expected

    def test_iso_week_spans_the_new_year(self):
        # 2025-12-29 is ISO week 1 of 2026; a naive %Y-%W would call it week 52 of 2025
        # and split one week across two buckets.
        assert select.bucket_key(datetime(2025, 12, 29), "week") == "2026-W01"

    def test_all_gives_every_photo_its_own_bucket(self):
        a = select.bucket_key(datetime(2026, 3, 14, 9, 30, 0), "all")
        b = select.bucket_key(datetime(2026, 3, 14, 9, 30, 1), "all")
        assert a != b

    def test_unknown_cadence_is_rejected(self):
        with pytest.raises(ValueError):
            select.bucket_key(datetime(2026, 3, 14), "fortnight")


class TestParseWhen:
    @pytest.mark.parametrize("value", [
        "2026-03-14T09:30:00.000Z",
        "2026-03-14T09:30:00+02:00",
        "2026-03-14T09:30:00",
        "2026-03-14",
    ])
    def test_accepts_the_shapes_immich_emits(self, value):
        parsed = select.parse_when(value)
        assert parsed is not None and parsed.year == 2026 and parsed.month == 3

    def test_none_and_garbage_are_survivable(self):
        assert select.parse_when(None) is None
        assert select.parse_when("not a date") is None


class TestApplyFilters:
    def test_recomputes_from_stored_metrics_without_reanalysis(self, conn):
        """Threshold tuning must be a SQL pass, not an ML re-run."""
        add(conn, "a", "2026-01-05T10:00:00.000Z", yaw=30.0)

        kept, total = select.apply_filters(conn, LIMITS, WEIGHTS)
        assert (kept, total) == (0, 1)
        assert conn.execute("SELECT reject_reason FROM metrics").fetchone()[0] == "head_turned"

        loosened = {**LIMITS, "max_yaw": 40.0}
        kept, _ = select.apply_filters(conn, loosened, WEIGHTS)
        assert kept == 1
        row = conn.execute("SELECT reject_reason, score FROM metrics").fetchone()
        assert row["reject_reason"] is None and row["score"] > 0


class TestSelectFrames:
    def test_keeps_the_best_frame_per_bucket(self, conn):
        add(conn, "sharp", "2026-03-10T10:00:00.000Z", sharpness=400.0)
        add(conn, "soft", "2026-03-11T10:00:00.000Z", sharpness=20.0)
        select.apply_filters(conn, LIMITS, WEIGHTS)

        assert select.select_frames(conn, "week", 1) == 1
        assert conn.execute("SELECT asset_id FROM selection").fetchone()[0] == "sharp"

    def test_per_bucket_caps_the_count(self, conn):
        for i in range(5):
            add(conn, f"a{i}", f"2026-03-1{i}T10:00:00.000Z", sharpness=100.0 + i)
        select.apply_filters(conn, LIMITS, WEIGHTS)

        assert select.select_frames(conn, "week", 2) == 2

    def test_evens_out_photo_heavy_periods(self, conn):
        """The reason bucketing exists: a busy holiday must not dominate the video."""
        for i in range(30):  # a burst in one week
            add(conn, f"holiday{i}", f"2026-03-09T{i % 24:02d}:00:00.000Z")
        for week, day in enumerate(("2026-04-06", "2026-04-13", "2026-04-20"), start=1):
            add(conn, f"quiet{week}", f"{day}T10:00:00.000Z")
        select.apply_filters(conn, LIMITS, WEIGHTS)

        select.select_frames(conn, "week", 1)
        rows = conn.execute("SELECT asset_id FROM selection").fetchall()
        chosen = {r["asset_id"] for r in rows}
        assert len(chosen) == 4, "one per week, not 30 from the busy one"
        assert {"quiet1", "quiet2", "quiet3"} <= chosen

    def test_rejected_frames_are_never_selected(self, conn):
        add(conn, "bad", "2026-03-10T10:00:00.000Z", yaw=45.0)
        select.apply_filters(conn, LIMITS, WEIGHTS)
        assert select.select_frames(conn, "week", 1) == 0

    def test_selection_spans_the_whole_corpus_not_just_recent_assets(self, conn):
        """Regression guard for the watermark leaking past the index stage.

        If --since ever reached selection, the timelapse would contain only the
        newest frames -- a failure that produces a plausible-looking video and so
        would not be obvious from the output alone.
        """
        for year in range(2019, 2027):
            add(conn, f"y{year}", f"{year}-06-10T10:00:00.000Z")
        select.apply_filters(conn, LIMITS, WEIGHTS)
        select.select_frames(conn, "month", 1)

        years = {r["asset_id"] for r in conn.execute("SELECT asset_id FROM selection")}
        assert years == {f"y{y}" for y in range(2019, 2027)}

    def test_reselect_replaces_rather_than_accumulates(self, conn):
        for i in range(4):
            add(conn, f"a{i}", f"2026-0{i + 1}-10T10:00:00.000Z")
        select.apply_filters(conn, LIMITS, WEIGHTS)

        select.select_frames(conn, "month", 1)
        first = conn.execute("SELECT count(*) FROM selection").fetchone()[0]
        select.select_frames(conn, "month", 1)
        assert conn.execute("SELECT count(*) FROM selection").fetchone()[0] == first

    def test_ordering_is_chronological_not_by_score(self, conn):
        add(conn, "later_but_better", "2026-05-10T10:00:00.000Z", sharpness=500.0)
        add(conn, "earlier", "2026-01-10T10:00:00.000Z", sharpness=50.0)
        conn.execute("INSERT INTO downloads (asset_id, path, source, fetched_at)"
                     " VALUES ('later_but_better', '/x.jpg', 'original', 'now')")
        conn.execute("INSERT INTO downloads (asset_id, path, source, fetched_at)"
                     " VALUES ('earlier', '/y.jpg', 'original', 'now')")
        select.apply_filters(conn, LIMITS, WEIGHTS)
        select.select_frames(conn, "month", 1)

        order = [r["asset_id"] for r in select.selected_in_order(conn)]
        assert order == ["earlier", "later_but_better"]


def test_reject_summary_counts_by_reason(conn):
    add(conn, "ok", "2026-03-10T10:00:00.000Z")
    add(conn, "turned", "2026-03-11T10:00:00.000Z", yaw=45.0)
    add(conn, "blurry", "2026-03-12T10:00:00.000Z", sharpness=1.0)
    select.apply_filters(conn, LIMITS, WEIGHTS)

    summary = dict(select.reject_summary(conn))
    assert summary == {"accepted": 1, "head_turned": 1, "blurry": 1}
