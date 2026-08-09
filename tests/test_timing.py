from __future__ import annotations

import pytest

from grow_up import db, pipeline
from grow_up.timing import StageTiming, Trial, format_bytes, format_duration


class TestFormatDuration:
    @pytest.mark.parametrize("seconds,expected", [
        (0.021, "21ms"),
        (0.4837, "484ms"),
        (12.34, "12.3s"),
        (59.9, "59.9s"),
        (95, "1m 35s"),
        (402, "6m 42s"),
        (3600, "1h 00m"),
        (8130, "2h 15m"),
    ])
    def test_scales_units(self, seconds, expected):
        assert format_duration(seconds) == expected


class TestFormatBytes:
    @pytest.mark.parametrize("count,expected", [
        (512, "512 B"),
        (2048, "2.0 kB"),
        (5 * 1024 ** 2, "5.0 MB"),
        (3 * 1024 ** 3, "3.0 GB"),
    ])
    def test_scales_units(self, count, expected):
        assert format_bytes(count) == expected


class TestStageTiming:
    def test_per_item_and_projection(self):
        stage = StageTiming("fetch", processed=100, elapsed=50.0, remaining=832)
        assert stage.per_item == pytest.approx(0.5)
        assert stage.projected == pytest.approx(416.0)

    def test_nothing_processed_does_not_divide_by_zero(self):
        stage = StageTiming("faces", processed=0, elapsed=0.0, remaining=832)
        assert stage.per_item == 0.0
        assert stage.projected == 0.0


class TestTrial:
    def build(self) -> Trial:
        return Trial(sample_size=100, total_assets=832, stages=[
            StageTiming("faces", 100, 2.0, 832),
            StageTiming("fetch", 100, 50.0, 832),
            StageTiming("analyze", 100, 12.0, 832),
            StageTiming("align", 18, 4.0, 132, unit="frame"),
        ])

    def test_totals(self):
        trial = self.build()
        assert trial.elapsed == pytest.approx(68.0)
        assert trial.per_picture == pytest.approx(0.68)

    def test_projection_sums_the_stages(self):
        trial = self.build()
        # faces 2/100*832 + fetch 50/100*832 + analyze 12/100*832 + align 4/18*132
        expected = 16.64 + 416.0 + 99.84 + (4 / 18 * 132)
        assert trial.projected == pytest.approx(expected)

    def test_align_projects_on_frames_not_assets(self):
        """Bucketing means a bigger library yields proportionally more frames,
        not one frame per photo, so align must not scale on the asset count."""
        trial = self.build()
        align = trial.stages[-1]
        assert align.remaining == 132
        assert align.projected < 40, "832 assets would wildly overstate it"

    def test_render_reports_time_per_picture(self):
        text = "\n".join(self.build().render())
        assert "Time per picture" in text
        assert "680ms" in text

    def test_outstanding_discounts_the_trial_s_own_work(self):
        """The trial is a partial real run, so what it processed is already done."""
        stage = StageTiming("fetch", processed=100, elapsed=50.0, remaining=832)
        assert stage.outstanding == 732
        assert stage.projected_outstanding == pytest.approx(366.0)
        assert stage.projected == pytest.approx(416.0), "full set still reported"

    def test_outstanding_never_goes_negative(self):
        stage = StageTiming("faces", processed=50, elapsed=1.0, remaining=10)
        assert stage.outstanding == 0

    def test_render_separates_full_set_from_what_is_left(self):
        text = "\n".join(self.build().render())
        assert "Full set:" in text and "Still to go:" in text

    def test_render_includes_every_stage(self):
        text = "\n".join(self.build().render())
        for name in ("faces", "fetch", "analyze", "align", "total"):
            assert name in text

    def test_render_marks_stages_with_nothing_pending(self):
        trial = Trial(100, 832, [StageTiming("faces", 0, 0.0, 0)])
        assert "nothing pending" in "\n".join(trial.render())

    def test_render_states_the_linearity_assumption(self):
        assert "linear scaling" in "\n".join(self.build().render())


class TestPendingCounts:
    @pytest.fixture()
    def conn(self, tmp_path):
        conn = db.connect(tmp_path / "t.sqlite")
        stamp = "2026-01-01T00:00:00.000Z"
        for i in range(10):
            conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                         " VALUES (?, ?, ?)", (f"a{i}", f"2026-01-{i + 1:02d}", stamp))
        # 4 have faces, 2 of those are downloaded, 1 of those is analyzed
        for i in range(4):
            conn.execute("INSERT INTO faces (asset_id, status, x1, y1, x2, y2,"
                         " image_width, image_height, n_candidates, fetched_at)"
                         " VALUES (?, 'ok', 0, 0, 10, 10, 100, 100, 1, ?)", (f"a{i}", stamp))
        for i in range(2):
            conn.execute("INSERT INTO downloads (asset_id, path, source, fetched_at)"
                         " VALUES (?, '/x.jpg', 'original', ?)", (f"a{i}", stamp))
        conn.execute("INSERT INTO metrics (asset_id, detected, analyzed_at)"
                     " VALUES ('a0', 1, ?)", (stamp,))
        return conn

    def test_counts_outstanding_work_per_stage(self, conn):
        counts = pipeline.pending_counts(conn)
        assert counts == {"faces": 6, "fetch": 2, "analyze": 1}


class TestEventualWorkload:
    """Projections need the eventual population, not what is actionable now."""

    def build(self, tmp_path, *, assets=10, faces_ok=0, faces_bad=0,
              downloads=0, metrics=0):
        conn = db.connect(tmp_path / f"w{assets}{faces_ok}{downloads}{metrics}.sqlite")
        stamp = "2026-01-01T00:00:00.000Z"
        # One transaction for the lot; the connection autocommits otherwise and
        # a few hundred fsyncs per fixture dominates the suite's runtime.
        conn.execute("BEGIN")
        for i in range(assets):
            conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                         " VALUES (?, '2026-01-01', ?)", (f"a{i}", stamp))
        for i in range(faces_ok + faces_bad):
            status = "ok" if i < faces_ok else "no_face"
            conn.execute("INSERT INTO faces (asset_id, status, n_candidates, fetched_at)"
                         " VALUES (?, ?, 0, ?)", (f"a{i}", status, stamp))
        for i in range(downloads):
            conn.execute("INSERT INTO downloads (asset_id, path, source, fetched_at)"
                         " VALUES (?, '/x.jpg', 'original', ?)", (f"a{i}", stamp))
        for i in range(metrics):
            conn.execute("INSERT INTO metrics (asset_id, detected, analyzed_at)"
                         " VALUES (?, 1, ?)", (f"a{i}", stamp))
        conn.execute("COMMIT")
        return conn

    def test_analyze_is_sized_before_anything_is_downloaded(self, tmp_path):
        """The reported bug: 10 images at 203ms projected a full run as 0ms,
        because the actionable analyze count joins on downloads."""
        conn = self.build(tmp_path, assets=832, faces_ok=832)

        assert pipeline.pending_counts(conn)["analyze"] == 0
        assert pipeline.eventual_workload(conn)["analyze"] == 832

    def test_fetch_is_sized_before_anything_is_downloaded(self, tmp_path):
        conn = self.build(tmp_path, assets=832, faces_ok=832)
        assert pipeline.eventual_workload(conn)["fetch"] == 832

    def test_discounts_work_already_done(self, tmp_path):
        conn = self.build(tmp_path, assets=832, faces_ok=832, downloads=10, metrics=10)
        counts = pipeline.eventual_workload(conn)
        assert counts == {"faces": 0, "fetch": 822, "analyze": 822}

    def test_excludes_assets_with_no_usable_face(self, tmp_path):
        """Those are never downloaded or analyzed, so they must not inflate it."""
        conn = self.build(tmp_path, assets=100, faces_ok=90, faces_bad=10)
        counts = pipeline.eventual_workload(conn)
        assert counts["fetch"] == 90 and counts["analyze"] == 90

    def test_estimates_unchecked_assets_from_the_observed_rate(self, tmp_path):
        """Half the library checked at a 90% hit rate implies ~90% of the rest."""
        conn = self.build(tmp_path, assets=100, faces_ok=45, faces_bad=5)
        counts = pipeline.eventual_workload(conn)
        assert counts["faces"] == 50
        assert counts["fetch"] == 90  # 45 known + 90% of the 50 unchecked

    def test_assumes_the_best_when_nothing_is_checked_yet(self, tmp_path):
        conn = self.build(tmp_path, assets=832)
        counts = pipeline.eventual_workload(conn)
        assert counts == {"faces": 832, "fetch": 832, "analyze": 832}

    def test_empty_library(self, tmp_path):
        conn = self.build(tmp_path, assets=0)
        assert pipeline.eventual_workload(conn) == {"faces": 0, "fetch": 0, "analyze": 0}

    def test_never_goes_negative(self, tmp_path):
        """Downloads can outlive the faces they came from if a person is retagged."""
        conn = self.build(tmp_path, assets=10, faces_ok=2, downloads=8, metrics=8)
        counts = pipeline.eventual_workload(conn)
        assert all(value >= 0 for value in counts.values())

    def test_projection_is_non_zero_for_a_measured_stage(self, tmp_path):
        """End to end: the exact shape from the bug report."""
        conn = self.build(tmp_path, assets=832, faces_ok=832)
        remaining = pipeline.eventual_workload(conn)["analyze"]

        stage = StageTiming("analyze", processed=10, elapsed=2.03, remaining=remaining)
        assert stage.per_item == pytest.approx(0.203)
        assert stage.projected == pytest.approx(0.203 * 832)
        assert format_duration(stage.projected) == "2m 48s"


class TestLimitClause:
    def test_no_limit_still_orders_deterministically(self):
        assert pipeline._limit_clause(None) == " ORDER BY a.id"

    def test_limit_is_applied(self):
        assert pipeline._limit_clause(100) == " ORDER BY a.id LIMIT 100"

    def test_limit_is_coerced_to_int(self):
        """The value is interpolated into SQL, so it must never be free text."""
        assert pipeline._limit_clause("100") == " ORDER BY a.id LIMIT 100"
        with pytest.raises(ValueError):
            pipeline._limit_clause("1; DROP TABLE assets")

    def test_sampling_is_stable_across_runs(self, tmp_path):
        conn = db.connect(tmp_path / "t.sqlite")
        stamp = "2026-01-01T00:00:00.000Z"
        for i in range(50):
            conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                         " VALUES (?, ?, ?)", (f"asset-{i:03d}", "2026-01-01", stamp))

        query = "SELECT a.id FROM assets a" + pipeline._limit_clause(10)
        first = [r[0] for r in conn.execute(query)]
        second = [r[0] for r in conn.execute(query)]
        assert first == second and len(first) == 10
