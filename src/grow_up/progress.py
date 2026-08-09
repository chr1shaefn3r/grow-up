"""A small progress bar for the long stages.

Hand-rolled rather than pulled in, to keep the runtime dependency list short and
CI installing only what the tests import.

Two behaviours matter more than looks:

* **Not a terminal?** Repainting with a carriage return turns a redirected log
  or a CI transcript into a single unreadable line, so a non-TTY gets periodic
  plain lines instead.
* **Something else needs to print?** A log line emitted mid-bar would be
  overwritten by the next repaint, so `log()` clears the bar first and redraws
  it afterwards.
"""

from __future__ import annotations

import shutil
import sys
import threading
import time
from typing import Callable, TextIO

from .timing import format_bytes, format_duration

BAR_WIDTH = 24

# ANSI "erase from cursor to end of line".
ERASE_LINE = "\x1b[K"


class Progress:
    """Tracks completion of a stage and renders it at a bounded frame rate."""

    def __init__(self, label: str, total: int, *, stream: TextIO | None = None,
                 emit: Callable[[str], None] | None = None, show_bytes: bool = False,
                 min_interval: float = 0.1, quiet_interval: float = 15.0,
                 clock: Callable[[], float] = time.monotonic,
                 overall: int | None = None, already_done: int = 0):
        self.label = label
        self.total = max(0, int(total))
        # The bar tracks this batch, so it fills to 100% and its ETA is right.
        # The closing summary instead reports position in the *whole* job:
        # after `trial -n 100` against 832 outstanding assets, "100/100" hides
        # that 732 remain, which is the number worth knowing.
        self.overall = self.total if overall is None else max(0, int(overall))
        self.already_done = max(0, int(already_done))
        self.show_bytes = show_bytes
        self.min_interval = min_interval
        self.quiet_interval = quiet_interval

        self._stream = stream if stream is not None else sys.stdout
        self._emit = emit or (lambda message: print(message, flush=True))
        self._clock = clock

        self.completed = 0
        self.failed = 0
        self.bytes = 0

        self._start = clock()
        self._last_paint = 0.0
        self._painted = False
        self._lock = threading.Lock()

    @property
    def interactive(self) -> bool:
        try:
            return bool(self._stream.isatty())
        except (AttributeError, ValueError):
            return False

    @property
    def elapsed(self) -> float:
        return max(self._clock() - self._start, 1e-9)

    @property
    def eta(self) -> float | None:
        if not self.completed or self.completed >= self.total:
            return None
        rate = self.elapsed / self.completed
        return rate * (self.total - self.completed)

    def advance(self, count: int = 1, nbytes: int = 0, failed: int = 0) -> None:
        with self._lock:
            self.completed += count
            self.failed += failed
            self.bytes += nbytes
            self._maybe_paint()

    def log(self, message: str) -> None:
        """Emit a line without the bar eating it."""
        with self._lock:
            self._clear()
            self._emit(message)
            self._paint(force=True)

    @property
    def overall_done(self) -> int:
        return self.already_done + self.completed

    def close(self) -> None:
        """Finish the bar, leaving exactly one durable summary line behind."""
        with self._lock:
            self._clear()
            denominator = max(self.overall, self.overall_done)
            summary = (f"  {self.label}: {self.overall_done}/{denominator}"
                       f" in {format_duration(self.elapsed)}")
            if self.failed:
                summary += f", {self.failed} failed"
            if self.show_bytes and self.bytes:
                summary += (f", {format_bytes(self.bytes)} at "
                            f"{format_bytes(self.bytes / self.elapsed)}/s")
            remaining = denominator - self.overall_done
            if remaining:
                summary += f" ({remaining} still pending)"
            self._emit(summary)
            self._painted = False

    # -- rendering ---------------------------------------------------------

    def render(self) -> str:
        fraction = self.completed / self.total if self.total else 1.0
        filled = int(round(BAR_WIDTH * min(fraction, 1.0)))
        bar = "=" * filled + " " * (BAR_WIDTH - filled)

        parts = [f"  {self.label:<8} [{bar}] {self.completed:>5}/{self.total} "
                 f"{fraction * 100:3.0f}%"]
        if self.show_bytes and self.bytes:
            parts.append(f"{format_bytes(self.bytes)} at "
                         f"{format_bytes(self.bytes / self.elapsed)}/s")
        eta = self.eta
        if eta is not None:
            parts.append(f"eta {format_duration(eta)}")
        if self.failed:
            parts.append(f"{self.failed} failed")
        return "   ".join(parts)

    def _maybe_paint(self) -> None:
        now = self._clock()
        interval = self.min_interval if self.interactive else self.quiet_interval
        if now - self._last_paint < interval and self.completed < self.total:
            return
        self._paint()

    def _width(self) -> int:
        """Usable columns, so a long bar cannot wrap and defeat the repaint."""
        try:
            return max(20, shutil.get_terminal_size((100, 24)).columns - 1)
        except (OSError, ValueError):
            return 100

    def _paint(self, force: bool = False) -> None:
        if not self.interactive:
            # Without a terminal there is nothing to repaint over, so emit a
            # plain line -- but only on the throttled schedule, never on force,
            # or a burst of log lines would each drag a duplicate along.
            if not force:
                self._last_paint = self._clock()
                self._emit(self.render())
            return
        # Erase to end of line rather than padding with spaces. Padding leaves
        # them sitting in the terminal's line buffer past the text, so the
        # summary that replaces the bar copies out with a trail of blanks.
        self._stream.write("\r" + self.render()[: self._width()] + ERASE_LINE)
        self._stream.flush()
        self._painted = True
        self._last_paint = self._clock()

    def _clear(self) -> None:
        if self.interactive and self._painted:
            self._stream.write("\r" + ERASE_LINE)
            self._stream.flush()
            self._painted = False
