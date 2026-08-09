"""The filter outcome as reported by `status`.

Shared with the stages through one renderer, because the last time this table
had two call sites spelling out their own version, it ended up printed in the
wrong place in `trial`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from grow_up import cli, config, db, pipeline, select

CONFIG = config.Config(raw={
    "paths": {"db": "t.sqlite", "cache": "c", "frames": "f", "out": "o"},
    "immich": {"person_id": "person-1"},
}, root=Path("."))


@pytest.fixture()
def conn(tmp_path):
    return db.connect(tmp_path / "t.sqlite")


def add(conn, asset_id: str, reason: str | None, score: float | None):
    stamp = "2026-01-01T00:00:00.000Z"
    conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                 " VALUES (?, '2026-01-01', ?)", (asset_id, stamp))
    conn.execute("INSERT INTO metrics (asset_id, detected, reject_reason, score,"
                 " analyzed_at) VALUES (?, 1, ?, ?, ?)",
                 (asset_id, reason, score, stamp))


def filtered(conn, accepted=2, turned=1, blurry=1):
    """A database that has been through `select`, marker and all."""
    conn.execute("BEGIN")
    n = 0
    for _ in range(accepted):
        add(conn, f"a{n}", None, 0.8); n += 1
    for _ in range(turned):
        add(conn, f"a{n}", "head_turned", None); n += 1
    for _ in range(blurry):
        add(conn, f"a{n}", "blurry", None); n += 1
    conn.execute("COMMIT")
    conn.execute("UPDATE metrics SET filtered_at = '2026-01-01T00:00:00.000Z'")
    return conn


class TestFormatting:
    def test_lists_every_reason_with_counts(self, conn):
        lines = select.format_reject_summary(filtered(conn))
        joined = "\n".join(lines)

        assert "filter outcome:" in joined
        for reason, count in select.reject_summary(conn):
            assert f"{reason:<26} {count:>6}" in joined

    def test_shares_are_against_the_total(self, conn):
        """Not against the largest bucket, which would make them meaningless."""
        lines = select.format_reject_summary(filtered(conn, accepted=1, turned=3,
                                                      blurry=0))
        assert "25.0%" in "\n".join(lines)   # 1 of 4
        assert "75.0%" in "\n".join(lines)   # 3 of 4

    def test_shares_sum_to_one_hundred(self, conn):
        import re

        lines = select.format_reject_summary(filtered(conn, accepted=3, turned=2,
                                                      blurry=1))
        shares = [float(x) for x in re.findall(r"(\d+\.\d)%", "\n".join(lines))]
        assert sum(shares) == pytest.approx(100.0, abs=0.2)

    def test_indent_is_the_only_structural_difference(self, conn):
        filtered(conn)
        stage = select.format_reject_summary(conn, indent="  ")
        status = select.format_reject_summary(conn, indent="")

        assert [line.strip() for line in stage] == [line.strip() for line in status]

    def test_label_is_configurable(self, conn):
        lines = select.format_reject_summary(filtered(conn), indent="",
                                             label="filter outcome (x)")
        assert lines[0].startswith("filter outcome (x)")


class TestNotYetEvaluated:
    """`reject_reason` is written by select, not analyze. In between, every
    surviving face reads as `accepted` -- photos never evaluated, reported as
    having passed."""

    def test_analyzed_but_unfiltered_says_so(self, conn):
        conn.execute("BEGIN")
        add(conn, "a0", None, None)
        add(conn, "a1", "no_face_detected", None)
        conn.execute("COMMIT")

        lines = select.format_reject_summary(conn)

        assert len(lines) == 1
        assert "not yet evaluated" in lines[0]
        assert "accepted" not in lines[0], "the misreading this guard exists for"

    def test_the_marker_is_what_decides(self, conn):
        conn.execute("BEGIN")
        add(conn, "a0", None, 0.8)   # a score, but select never ran
        conn.execute("COMMIT")
        assert not select.filters_applied(conn)

        conn.execute("UPDATE metrics SET filtered_at = '2026-01-01T00:00:00.000Z'")
        assert select.filters_applied(conn)

    def test_empty_database_says_nothing_analyzed(self, conn):
        lines = select.format_reject_summary(conn)
        assert len(lines) == 1 and "nothing analyzed" in lines[0]

    def test_all_rejected_still_counts_as_evaluated(self, conn):
        """Rejected rows carry no score, so inferring "filtered" from a score
        would call a library with thresholds set too tight -- every photo
        failing -- unevaluated. Entirely reachable while tuning."""
        conn.execute("BEGIN")
        add(conn, "a0", "head_turned", None)
        add(conn, "a1", "blurry", None)
        conn.execute("COMMIT")
        conn.execute("UPDATE metrics SET filtered_at = '2026-01-01T00:00:00.000Z'")

        lines = select.format_reject_summary(conn)

        assert select.filters_applied(conn)
        assert "not yet evaluated" not in "\n".join(lines)
        assert "head_turned" in "\n".join(lines)
        assert "accepted" not in "\n".join(lines), "nothing passed, so nothing accepted"

    def test_apply_filters_records_that_it_ran(self, conn):
        """End to end: the real select path must set the marker."""
        limits = {"max_yaw": 20.0, "max_pitch": 18.0, "max_roll": 25.0,
                  "max_gaze": 0.35, "max_blink": 0.45, "max_oob_frac": 0.005,
                  "allow_bbox_clipped": False, "min_interocular_px": 60.0,
                  "min_sharpness": 12.0, "min_exposure_lo": 8.0,
                  "max_exposure_hi": 250.0}
        conn.execute("BEGIN")
        add(conn, "a0", None, None)
        conn.execute("COMMIT")
        assert not select.filters_applied(conn)

        select.apply_filters(conn, limits, {"w_pose": 1.0})

        assert select.filters_applied(conn)


class TestStatusCommand:
    def run_status(self, tmp_path, monkeypatch, seed=True):
        conn = db.connect(tmp_path / "t.sqlite")
        if seed:
            filtered(conn)

        lines: list[str] = []
        monkeypatch.setattr(cli, "log", lines.append)
        monkeypatch.setattr(cli, "_open", lambda args: (CONFIG, conn))
        cli.cmd_status(type("Args", (), {"config": "config.toml"})())
        return conn, lines

    def test_includes_the_filter_outcome(self, tmp_path, monkeypatch):
        _, lines = self.run_status(tmp_path, monkeypatch)
        joined = "\n".join(lines)

        assert "filter outcome (last select)" in joined
        assert "head_turned" in joined and "accepted" in joined

    def test_outcome_sits_with_the_counts_it_describes(self, tmp_path, monkeypatch):
        _, lines = self.run_status(tmp_path, monkeypatch)
        joined = "\n".join(lines)

        assert joined.index("metrics") < joined.index("filter outcome")
        assert joined.index("filter outcome") < joined.index("watermark")

    def test_heading_names_the_vintage(self, tmp_path, monkeypatch):
        """The stored verdicts are from the last select, not from whatever
        config.toml says now."""
        _, lines = self.run_status(tmp_path, monkeypatch)
        assert any("last select" in line for line in lines)

    def test_status_writes_nothing(self, tmp_path, monkeypatch):
        conn, _ = self.run_status(tmp_path, monkeypatch)
        before = conn.execute(
            "SELECT asset_id, reject_reason, score FROM metrics ORDER BY asset_id"
        ).fetchall()

        cli.cmd_status(type("Args", (), {"config": "config.toml"})())

        after = conn.execute(
            "SELECT asset_id, reject_reason, score FROM metrics ORDER BY asset_id"
        ).fetchall()
        assert [tuple(r) for r in before] == [tuple(r) for r in after]

    def test_empty_database_is_survivable(self, tmp_path, monkeypatch):
        _, lines = self.run_status(tmp_path, monkeypatch, seed=False)
        assert any("nothing analyzed" in line for line in lines)


def test_stage_and_status_report_the_same_numbers(tmp_path, monkeypatch):
    """The point of the shared renderer."""
    conn = db.connect(tmp_path / "t.sqlite")
    filtered(conn)

    stage: list[str] = []
    pipeline.report_rejects(conn, stage.append)

    status = select.format_reject_summary(conn, indent="",
                                          label="filter outcome (last select)")

    assert [line.strip() for line in stage[1:]] == [line.strip() for line in status[1:]]
