"""Ordering of the CLI's own output.

The filter-outcome table used to appear directly after selection in `run` but
adrift at the very end in `trial`, after the timing report, because the two
paths spelled the same steps out separately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grow_up import cli, config, db

CONFIG = config.Config(raw={
    "filter": {
        "max_yaw": 20.0, "max_pitch": 18.0, "max_roll": 25.0, "max_gaze": 0.35,
        "max_blink": 0.45, "max_oob_frac": 0.005, "allow_bbox_clipped": False,
        "min_interocular_px": 60.0, "min_sharpness": 12.0,
        "min_exposure_lo": 8.0, "max_exposure_hi": 250.0,
    },
    "score": {"w_pose": 1.0, "w_gaze": 1.0, "w_eyes_open": 1.0,
              "w_sharpness": 1.0, "w_size": 0.5},
    "select": {"cadence": "week", "per_bucket": 1},
}, root=Path("."))


@pytest.fixture()
def conn(tmp_path):
    conn = db.connect(tmp_path / "t.sqlite")
    stamp = "2026-01-01T00:00:00.000Z"
    conn.execute("BEGIN")
    for i, yaw in enumerate([2.0, 2.0, 45.0]):
        conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                     " VALUES (?, ?, ?)", (f"a{i}", f"2026-0{i + 1}-10", stamp))
        conn.execute(
            "INSERT INTO metrics (asset_id, detected, yaw, pitch, roll, gaze_x, gaze_y,"
            " blink_l, blink_r, oob_frac, bbox_clipped, interocular_px, sharpness,"
            " exposure_lo, exposure_hi, analyzed_at)"
            " VALUES (?, 1, ?, 1, 1, 0.02, 0, 0.05, 0.05, 0, 0, 150, 100, 40, 200, ?)",
            (f"a{i}", yaw, stamp))
    conn.execute("COMMIT")
    return conn


@pytest.fixture()
def lines(monkeypatch):
    captured: list[str] = []
    monkeypatch.setattr(cli, "log", captured.append)
    monkeypatch.setattr(cli.pipeline, "log", captured.append, raising=False)
    return captured


class TestSelectFrames:
    def test_reports_the_filter_outcome_with_the_selection(self, conn, lines):
        """The outcome belongs to selection, so it must be emitted by the same
        helper rather than left to each caller to remember."""
        cli._select_frames(CONFIG, conn)

        text = "\n".join(lines)
        assert "pass the hard filters" in text
        assert "filter outcome:" in text
        assert text.index("pass the hard filters") < text.index("filter outcome:")

    def test_outcome_lists_every_reason_and_the_acceptances(self, conn, lines):
        cli._select_frames(CONFIG, conn)
        text = "\n".join(lines)

        assert "accepted" in text
        assert "head_turned" in text

    def test_returns_the_counts_its_callers_need(self, conn):
        kept, scored, frames = cli._select_frames(CONFIG, conn)
        assert (kept, scored) == (2, 3)
        assert frames >= 1

    def test_cadence_argument_overrides_config(self, conn, lines):
        cli._select_frames(CONFIG, conn, "month")
        assert any("cadence=month" in line for line in lines)

    def test_falls_back_to_configured_cadence(self, conn, lines):
        cli._select_frames(CONFIG, conn, None)
        assert any("cadence=week" in line for line in lines)

    def test_nothing_is_printed_after_the_outcome_table(self, conn, lines):
        """Whatever a caller prints next starts a new section; the helper must
        not trail extra output that would separate the table from selection."""
        cli._select_frames(CONFIG, conn)

        tail = [line for line in lines if line.strip()][-1]
        assert tail.strip().split()[0] in {"accepted", "head_turned", "no_face_detected",
                                           "blurry", "looking_away", "head_tilted",
                                           "head_rolled", "eyes_closed", "bbox_clipped",
                                           "face_too_small", "overexposed", "underexposed",
                                           "partially_out_of_frame"}
