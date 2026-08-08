from __future__ import annotations

import io

import pytest

from grow_up.progress import BAR_WIDTH, Progress


class FakeStream(io.StringIO):
    def __init__(self, tty: bool = True):
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class FakeClock:
    """Deterministic monotonic clock; tests advance it explicitly."""

    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += seconds


def make(total=100, tty=True, **kwargs):
    clock = FakeClock()
    stream = FakeStream(tty=tty)
    lines: list[str] = []
    bar = Progress("fetch", total, stream=stream, emit=lines.append,
                   clock=clock, **kwargs)
    return bar, stream, lines, clock


class TestRendering:
    def test_bar_fills_proportionally(self):
        bar, _, _, _ = make(total=100)
        bar.completed = 50
        rendered = bar.render()
        assert rendered.count("=") == BAR_WIDTH // 2
        assert " 50%" in rendered

    def test_shows_counts(self):
        bar, _, _, _ = make(total=832)
        bar.completed = 312
        assert "312/832" in bar.render()

    def test_empty_and_full(self):
        bar, _, _, _ = make(total=10)
        assert bar.render().count("=") == 0
        bar.completed = 10
        assert bar.render().count("=") == BAR_WIDTH
        assert "100%" in bar.render()

    def test_zero_total_does_not_divide_by_zero(self):
        bar, _, _, _ = make(total=0)
        assert "100%" in bar.render()
        assert bar.eta is None

    def test_failures_are_surfaced(self):
        bar, _, _, _ = make(total=10)
        bar.advance(failed=1)
        assert "1 failed" in bar.render()

    def test_throughput_shown_when_tracking_bytes(self):
        bar, _, _, clock = make(total=10, show_bytes=True)
        clock.tick(10.0)
        bar.advance(nbytes=50 * 1024 * 1024)
        rendered = bar.render()
        assert "50.0 MB" in rendered
        assert "MB/s" in rendered

    def test_bytes_hidden_when_not_requested(self):
        bar, _, _, clock = make(total=10, show_bytes=False)
        clock.tick(1.0)
        bar.advance(nbytes=1024 * 1024)
        assert "MB" not in bar.render()


class TestEta:
    def test_extrapolates_from_observed_rate(self):
        bar, _, _, clock = make(total=100)
        clock.tick(10.0)
        bar.completed = 10
        assert bar.eta == pytest.approx(90.0)

    def test_none_before_any_progress(self):
        bar, _, _, _ = make(total=100)
        assert bar.eta is None

    def test_none_once_complete(self):
        bar, _, _, clock = make(total=10)
        clock.tick(5.0)
        bar.completed = 10
        assert bar.eta is None

    def test_appears_in_the_rendered_bar(self):
        bar, _, _, clock = make(total=100)
        clock.tick(10.0)
        bar.completed = 10
        assert "eta" in bar.render()


class TestTerminalOutput:
    def test_repaints_in_place(self):
        bar, stream, _, clock = make(total=10, min_interval=0.0)
        bar.advance()
        bar.advance()
        assert stream.getvalue().count("\r") >= 2

    def test_throttles_repaints(self):
        bar, stream, _, clock = make(total=100, min_interval=1.0)
        for _ in range(10):
            clock.tick(0.01)
            bar.advance()
        # Ten quick updates well inside the interval paint at most once.
        assert stream.getvalue().count("\r") <= 1

    def test_always_paints_the_final_item(self):
        """Otherwise a bar can sit at 99% after the work is done."""
        bar, stream, _, clock = make(total=3, min_interval=999.0)
        bar.advance()
        bar.advance()
        bar.advance()
        assert "100%" in stream.getvalue()

    def test_close_leaves_a_summary_line(self):
        bar, _, lines, clock = make(total=10)
        for _ in range(10):
            bar.advance()
        clock.tick(42.0)
        bar.close()
        assert lines[-1].strip().startswith("fetch: 10/10")
        assert "42.0s" in lines[-1]

    def test_summary_counts_against_the_whole_job(self):
        """After `trial -n 100` against 832 outstanding assets, "100/100" would
        hide that 732 remain -- which is the number worth knowing."""
        bar, _, lines, clock = make(total=100, overall=832)
        clock.tick(48.3)
        bar.advance(count=100)
        bar.close()

        assert "fetch: 100/832" in lines[-1]
        assert "732 still pending" in lines[-1]

    def test_summary_includes_work_from_earlier_runs(self):
        bar, _, lines, _ = make(total=100, overall=832, already_done=100)
        bar.advance(count=100)
        bar.close()

        assert "fetch: 200/832" in lines[-1]
        assert "632 still pending" in lines[-1]

    def test_no_pending_note_when_the_job_is_finished(self):
        bar, _, lines, _ = make(total=832, overall=832)
        bar.advance(count=832)
        bar.close()

        assert "fetch: 832/832" in lines[-1]
        assert "pending" not in lines[-1]

    def test_summary_is_a_single_line(self):
        bar, _, lines, clock = make(total=100, overall=832, show_bytes=True)
        clock.tick(10.0)
        bar.advance(count=100, nbytes=1024 * 1024)
        bar.close()

        assert len(lines) == 1
        assert "\n" not in lines[0]

    def test_bar_itself_still_tracks_the_batch(self):
        """The bar must reach 100% and give a batch-accurate ETA, even though
        the summary reports the wider job."""
        bar, _, _, clock = make(total=100, overall=832)
        clock.tick(10.0)
        bar.advance(count=100)

        assert "100/100" in bar.render()
        assert "100%" in bar.render()
        assert bar.eta is None

    def test_close_reports_failures_and_bytes(self):
        bar, _, lines, clock = make(total=10, show_bytes=True)
        clock.tick(2.0)
        bar.advance(count=9, nbytes=20 * 1024 * 1024)
        bar.advance(failed=1)
        bar.close()
        assert "1 failed" in lines[-1]
        assert "MB" in lines[-1]


class TestNonTerminalOutput:
    """Redirected output and CI transcripts must not fill with \\r repaints."""

    def test_never_writes_carriage_returns(self):
        bar, stream, lines, clock = make(total=100, tty=False, quiet_interval=0.0)
        for _ in range(20):
            clock.tick(1.0)
            bar.advance()
        assert "\r" not in stream.getvalue()
        assert stream.getvalue() == "", "nothing is painted to a non-tty stream"

    def test_emits_periodic_plain_lines(self):
        bar, _, lines, clock = make(total=100, tty=False, quiet_interval=10.0)
        for _ in range(20):
            clock.tick(1.0)
            bar.advance()
        # 20 seconds at a 10s cadence: a couple of updates, not twenty.
        assert 1 <= len(lines) <= 3

    def test_stays_quiet_between_intervals(self):
        bar, _, lines, clock = make(total=1000, tty=False, quiet_interval=60.0)
        for _ in range(30):
            clock.tick(1.0)
            bar.advance()
        assert lines == []


class TestLogInterleaving:
    def test_log_clears_the_bar_first(self):
        """A line printed mid-bar would otherwise be overwritten by the repaint."""
        bar, stream, lines, clock = make(total=10, min_interval=0.0)
        bar.advance()
        stream.seek(0, io.SEEK_END)
        before = stream.tell()

        bar.log("  ! download failed")

        written = stream.getvalue()[before:]
        assert lines == ["  ! download failed"]
        assert written.startswith("\r" + " " * 100 + "\r"), "bar cleared before the log"
        assert written.rstrip().endswith("%") or "=" in written, "bar redrawn afterwards"

    def test_log_on_a_non_tty_does_not_duplicate_the_bar(self):
        bar, _, lines, clock = make(total=10, tty=False, quiet_interval=999.0)
        bar.advance()
        bar.log("  ! download failed")
        assert lines == ["  ! download failed"]
