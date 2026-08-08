"""Worker-count detection.

Regression cover for a real misconfiguration: worker count was derived by
halving the logical CPU count, which assumes simultaneous multithreading. An
M1 Pro reports 8 logical and 8 physical cores (6 performance + 2 efficiency),
so the machine ran at half capacity.
"""

from __future__ import annotations

import subprocess

import pytest

from grow_up import analyze


def write_topology(root, layout: list[tuple[int, int]]) -> None:
    """Build a fake sysfs tree: one entry per logical CPU as (package, core)."""
    for index, (package, core) in enumerate(layout):
        topology = root / f"cpu{index}" / "topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text(f"{package}\n")
        (topology / "core_id").write_text(f"{core}\n")


class TestLinuxDetection:
    def test_collapses_hyperthread_siblings(self, tmp_path):
        """8 logical on 4 physical: each core_id appears twice."""
        write_topology(tmp_path, [(0, 0), (0, 1), (0, 2), (0, 3),
                                  (0, 0), (0, 1), (0, 2), (0, 3)])
        assert analyze._linux_physical_cores(tmp_path) == 4

    def test_counts_every_core_without_smt(self, tmp_path):
        write_topology(tmp_path, [(0, i) for i in range(8)])
        assert analyze._linux_physical_cores(tmp_path) == 8

    def test_handles_multiple_sockets(self, tmp_path):
        write_topology(tmp_path, [(0, 0), (0, 1), (1, 0), (1, 1)])
        assert analyze._linux_physical_cores(tmp_path) == 4, "core ids repeat per package"

    def test_missing_sysfs_returns_none(self, tmp_path):
        assert analyze._linux_physical_cores(tmp_path / "absent") is None

    def test_unreadable_topology_is_skipped(self, tmp_path):
        (tmp_path / "cpu0").mkdir()
        assert analyze._linux_physical_cores(tmp_path) is None


class TestMacosDetection:
    def test_reads_sysctl(self, monkeypatch):
        def fake_run(cmd, **kwargs):
            assert cmd == ["sysctl", "-n", "hw.physicalcpu"]
            return subprocess.CompletedProcess(cmd, 0, stdout="8\n", stderr="")

        monkeypatch.setattr(analyze.subprocess, "run", fake_run)
        assert analyze._macos_physical_cores() == 8

    def test_unparseable_output_returns_none(self, monkeypatch):
        monkeypatch.setattr(analyze.subprocess, "run",
                            lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""))
        assert analyze._macos_physical_cores() is None

    def test_missing_sysctl_returns_none(self, monkeypatch):
        def boom(cmd, **kwargs):
            raise FileNotFoundError("sysctl")

        monkeypatch.setattr(analyze.subprocess, "run", boom)
        assert analyze._macos_physical_cores() is None


class TestPhysicalCores:
    def test_apple_silicon_uses_every_core(self, monkeypatch):
        """The reported bug: an M1 Pro must yield 8 workers, not 4."""
        monkeypatch.setattr(analyze.sys, "platform", "darwin")
        monkeypatch.setattr(analyze, "_macos_physical_cores", lambda: 8)
        monkeypatch.setattr(analyze, "available_cpus", lambda: 8)
        assert analyze.physical_cores() == 8

    def test_hyperthreaded_x86_uses_physical_cores(self, monkeypatch):
        monkeypatch.setattr(analyze.sys, "platform", "linux")
        monkeypatch.setattr(analyze, "_linux_physical_cores", lambda *a: 8)
        monkeypatch.setattr(analyze, "available_cpus", lambda: 16)
        assert analyze.physical_cores() == 8

    def test_affinity_limit_wins_over_hardware(self, monkeypatch):
        """A container pinned to 4 CPUs must not spawn 8 workers."""
        monkeypatch.setattr(analyze.sys, "platform", "linux")
        monkeypatch.setattr(analyze, "_linux_physical_cores", lambda *a: 8)
        monkeypatch.setattr(analyze, "available_cpus", lambda: 4)
        assert analyze.physical_cores() == 4

    def test_falls_back_to_visible_cpus(self, monkeypatch):
        monkeypatch.setattr(analyze.sys, "platform", "linux")
        monkeypatch.setattr(analyze, "_linux_physical_cores", lambda *a: None)
        monkeypatch.setattr(analyze, "available_cpus", lambda: 6)
        assert analyze.physical_cores() == 6

    def test_never_returns_zero(self, monkeypatch):
        monkeypatch.setattr(analyze.sys, "platform", "linux")
        monkeypatch.setattr(analyze, "_linux_physical_cores", lambda *a: 0)
        monkeypatch.setattr(analyze, "available_cpus", lambda: 1)
        assert analyze.physical_cores() == 1

    def test_real_machine_reports_something_sane(self):
        cores = analyze.physical_cores()
        assert 1 <= cores <= analyze.available_cpus()


class TestAvailableCpus:
    def test_at_least_one(self):
        assert analyze.available_cpus() >= 1

    def test_uses_affinity_when_present(self, monkeypatch):
        monkeypatch.setattr(analyze.os, "sched_getaffinity", lambda _: {0, 1, 2},
                            raising=False)
        assert analyze.available_cpus() == 3


class TestChunkSize:
    """Coarse fixed chunks strand work on efficiency cores at the tail."""

    def chunksize(self, jobs: int, workers: int) -> int:
        return max(1, min(16, jobs // (workers * 4) or 1))

    def test_small_batches_chunk_finely(self):
        # A 100-image trial on 8 workers: chunks of 8 would give ~12 chunks for
        # 8 workers, so a single slow chunk doubles the tail.
        assert self.chunksize(100, 8) == 3

    def test_large_batches_stay_bounded(self):
        assert self.chunksize(832, 8) == 16

    def test_never_zero(self):
        assert self.chunksize(3, 8) == 1
        assert self.chunksize(0, 8) == 1
