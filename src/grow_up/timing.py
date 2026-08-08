"""Stage timing and full-run projection.

Used by `grow-up trial` to answer "how long will the whole library take?" from a
small sample, without pretending the extrapolation is more precise than it is.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60):02d}m"


def format_bytes(count: float) -> str:
    for unit, size in (("GB", 1 << 30), ("MB", 1 << 20), ("kB", 1 << 10)):
        if count >= size:
            return f"{count / size:.1f} {unit}"
    return f"{int(count)} B"


@dataclass
class StageTiming:
    """One stage's measured throughput and what it implies for the full set."""

    name: str
    processed: int
    elapsed: float
    remaining: int
    unit: str = "asset"
    note: str = ""

    @property
    def per_item(self) -> float:
        return self.elapsed / self.processed if self.processed else 0.0

    @property
    def projected(self) -> float:
        """Wall time for the whole set, measured as outstanding at trial start.

        `remaining` is captured before the trial runs, so this is the cost of
        the full job -- including the part the trial itself just completed.
        """
        return self.per_item * self.remaining

    @property
    def outstanding(self) -> int:
        """Items still to do once this trial's own work is discounted."""
        return max(0, self.remaining - self.processed)

    @property
    def projected_outstanding(self) -> float:
        return self.per_item * self.outstanding


@dataclass
class Trial:
    """A sampled run, plus what it projects for the rest of the library."""

    sample_size: int
    total_assets: int
    stages: list[StageTiming] = field(default_factory=list)

    @property
    def elapsed(self) -> float:
        return sum(s.elapsed for s in self.stages)

    @property
    def per_picture(self) -> float:
        return self.elapsed / self.sample_size if self.sample_size else 0.0

    @property
    def projected(self) -> float:
        """Full-set wall time, the figure a trial exists to produce."""
        return sum(s.projected for s in self.stages)

    @property
    def projected_outstanding(self) -> float:
        """What is left after this trial, which already banked real work."""
        return sum(s.projected_outstanding for s in self.stages)

    def render(self) -> list[str]:
        header = f"{'stage':<10}{'items':>7}{'elapsed':>10}{'per item':>11}{'projected':>12}"
        lines = ["", header, "-" * len(header)]

        for s in self.stages:
            if not s.processed:
                lines.append(f"{s.name:<10}{'-':>7}{'-':>10}{'-':>11}{'-':>12}"
                             f"   (nothing pending)")
                continue
            row = (f"{s.name:<10}{s.processed:>7}{format_duration(s.elapsed):>10}"
                   f"{format_duration(s.per_item):>11}{format_duration(s.projected):>12}")
            if s.note:
                row += f"   {s.note}"
            lines.append(row)

        lines.append("-" * len(header))
        lines.append(
            f"{'total':<10}{self.sample_size:>7}{format_duration(self.elapsed):>10}"
            f"{format_duration(self.per_picture):>11}{format_duration(self.projected):>12}"
        )
        lines += [
            "",
            f"Time per picture:  {format_duration(self.per_picture)} "
            f"across all stages ({self.sample_size} sampled)",
            f"Full set:          {format_duration(self.projected)} "
            f"for all {self.total_assets} assets",
            f"Still to go:       {format_duration(self.projected_outstanding)} "
            "(this trial already banked its own work — nothing is repeated)",
            "",
            "Assumes linear scaling. Real runs tend to come in slightly under this: "
            "the sample",
            "paid the model-load cost once per worker across few images, and a "
            "re-run skips",
            "everything already cached.",
        ]
        return lines


@contextmanager
def stopwatch():
    """Yield a callable returning elapsed seconds, frozen once the block exits."""
    start = time.perf_counter()
    result = {"elapsed": 0.0}
    yield lambda: result["elapsed"] or (time.perf_counter() - start)
    result["elapsed"] = time.perf_counter() - start
