"""Suppression of MediaPipe/TFLite native logging.

The noise comes from C++ (glog/absl and TFLite) writing to file descriptors
directly, so Python's logging module cannot filter it. These tests cover the two
mechanisms that can: environment variables read at library init, and
descriptor-level redirection for the messages that ignore them.
"""

from __future__ import annotations

import os

import pytest

from grow_up import analyze


class TestSuppressNativeOutput:
    def test_swallows_direct_descriptor_writes(self, capfd):
        """Python-level redirection would miss these; they bypass sys.stderr."""
        with analyze.suppress_native_output(True):
            os.write(2, b"I0000 init-domain.cc:132] Fiber init\n")
            os.write(1, b"INFO: Created TensorFlow Lite XNNPACK delegate\n")

        out, err = capfd.readouterr()
        assert "Fiber init" not in err
        assert "XNNPACK" not in out

    def test_passes_output_through_when_disabled(self, capfd):
        with analyze.suppress_native_output(False):
            os.write(2, b"visible warning\n")

        assert "visible warning" in capfd.readouterr()[1]

    def test_restores_descriptors_afterwards(self, capfd):
        with analyze.suppress_native_output(True):
            os.write(2, b"hidden\n")
        os.write(2, b"visible again\n")

        _, err = capfd.readouterr()
        assert "hidden" not in err
        assert "visible again" in err

    def test_restores_descriptors_after_an_exception(self, capfd):
        with pytest.raises(ValueError):
            with analyze.suppress_native_output(True):
                raise ValueError("boom")
        os.write(2, b"still working\n")

        assert "still working" in capfd.readouterr()[1]

    def test_leaks_no_descriptors(self):
        """Called once per worker; a leak here would exhaust the process."""
        def open_count() -> int:
            return len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else -1

        before = open_count()
        if before < 0:
            pytest.skip("/proc/self/fd unavailable")
        for _ in range(50):
            with analyze.suppress_native_output(True):
                pass
        assert open_count() == before

    def test_nesting_is_safe(self, capfd):
        with analyze.suppress_native_output(True):
            with analyze.suppress_native_output(True):
                os.write(2, b"inner\n")
            os.write(2, b"outer\n")
        os.write(2, b"after\n")

        _, err = capfd.readouterr()
        assert "inner" not in err and "outer" not in err
        assert "after" in err


class TestQuietEnv:
    def test_covers_the_logging_frameworks_in_play(self):
        assert analyze.QUIET_ENV["GLOG_minloglevel"] == "2", "ERROR and above only"
        assert "TF_CPP_MIN_LOG_LEVEL" in analyze.QUIET_ENV
        assert "ABSL_MIN_LOG_LEVEL" in analyze.QUIET_ENV

    def test_quiet_worker_sets_them_before_importing_mediapipe(self, monkeypatch):
        """The native log level is read once, at library init, so the ordering
        inside init_worker is what makes this work at all."""
        monkeypatch.delenv("GLOG_minloglevel", raising=False)
        seen: dict[str, str] = {}

        def fake_build(opts):
            seen.update(os.environ)
            return object()

        monkeypatch.setattr(analyze, "build_landmarker", fake_build)
        analyze.init_worker(analyze.AnalyzeOptions(verbose=False))

        assert seen["GLOG_minloglevel"] == "2"

    def test_verbose_worker_leaves_the_environment_alone(self, monkeypatch):
        monkeypatch.delenv("GLOG_minloglevel", raising=False)
        monkeypatch.setattr(analyze, "build_landmarker", lambda opts: object())

        analyze.init_worker(analyze.AnalyzeOptions(verbose=True))
        assert "GLOG_minloglevel" not in os.environ

    def test_worker_pins_thread_counts(self, monkeypatch):
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.setattr(analyze, "build_landmarker", lambda opts: object())

        analyze.init_worker(analyze.AnalyzeOptions())
        assert os.environ["OMP_NUM_THREADS"] == "1"


class TestAnalyzeOptions:
    def test_quiet_by_default(self):
        assert analyze.AnalyzeOptions().verbose is False
