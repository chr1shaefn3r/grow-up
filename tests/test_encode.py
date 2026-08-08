from __future__ import annotations

from pathlib import Path

import pytest

from grow_up import encode


@pytest.fixture()
def frames(tmp_path) -> list[Path]:
    paths = []
    for i in range(1, 4):
        p = tmp_path / f"frame_{i:06d}.jpg"
        p.write_bytes(b"\xff\xd8\xff")
        paths.append(p)
    return paths


class TestConcatList:
    def test_lists_every_frame_with_a_duration(self, frames, tmp_path):
        listing = encode.write_concat_list(frames, 10.0, tmp_path / "frames.ffconcat")
        text = listing.read_text()

        assert text.startswith("ffconcat version 1.0")
        assert text.count("duration 0.100000") == 3
        for frame in frames:
            assert frame.resolve().as_posix() in text

    def test_repeats_the_last_frame(self, frames, tmp_path):
        """ffmpeg ignores the final entry's duration, so the last frame would
        otherwise flash past in a single tick."""
        text = encode.write_concat_list(frames, 10.0, tmp_path / "l.ffconcat").read_text()
        assert text.count(frames[-1].resolve().as_posix()) == 2

    def test_omitting_a_frame_needs_no_renumbering_on_disk(self, frames, tmp_path):
        """Manual rejects drop out of the list; the files and manifest stay put."""
        kept = [frames[0], frames[2]]
        text = encode.write_concat_list(kept, 10.0, tmp_path / "l.ffconcat").read_text()

        assert frames[1].resolve().as_posix() not in text
        assert frames[0].resolve().as_posix() in text

    def test_empty_input_is_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            encode.write_concat_list([], 10.0, tmp_path / "l.ffconcat")

    def test_fps_drives_the_duration(self, frames, tmp_path):
        text = encode.write_concat_list(frames, 4.0, tmp_path / "l.ffconcat").read_text()
        assert "duration 0.250000" in text


class TestBuildCommand:
    @pytest.fixture(autouse=True)
    def fake_ffmpeg(self, monkeypatch):
        monkeypatch.setattr(encode.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    def test_uses_the_concat_demuxer(self, tmp_path):
        cmd = encode.build_command(tmp_path / "l.ffconcat", tmp_path / "o.mp4",
                                   10.0, "libx264", 18)
        assert "-f" in cmd and cmd[cmd.index("-f") + 1] == "concat"
        assert "-safe" in cmd

    def test_forces_even_dimensions(self, tmp_path):
        """yuv420p rejects odd geometry, and it fails at the very last step."""
        cmd = encode.build_command(tmp_path / "l.ffconcat", tmp_path / "o.mp4",
                                   10.0, "libx264", 18)
        assert "scale=trunc(iw/2)*2:trunc(ih/2)*2" in cmd[cmd.index("-vf") + 1]

    def test_crf_only_for_codecs_that_support_it(self, tmp_path):
        x264 = encode.build_command(tmp_path / "l", tmp_path / "o.mp4", 10.0, "libx264", 20)
        assert "-crf" in x264 and x264[x264.index("-crf") + 1] == "20"

        videotoolbox = encode.build_command(
            tmp_path / "l", tmp_path / "o.mp4", 10.0, "hevc_videotoolbox", 20
        )
        assert "-crf" not in videotoolbox, "videotoolbox has no CRF and would error"
        assert "-q:v" in videotoolbox

    def test_extra_filters_come_before_the_scale_guard(self, tmp_path):
        cmd = encode.build_command(tmp_path / "l", tmp_path / "o.mp4", 10.0, "libx264", 18,
                                   ["minterpolate=fps=30"])
        chain = cmd[cmd.index("-vf") + 1]
        assert chain.index("minterpolate") < chain.index("scale=trunc")


class TestStageEncode:
    """The trial reuses this stage with an overridden filename, so that a
    two-second sample cannot silently replace a finished full render."""

    @pytest.fixture()
    def conn(self, tmp_path):
        from grow_up import db

        conn = db.connect(tmp_path / "t.sqlite")
        stamp = "2026-01-01T00:00:00.000Z"
        for i in range(1, 4):
            frame = tmp_path / f"frame_{i:06d}.jpg"
            frame.write_bytes(b"\xff\xd8\xff")
            conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                         " VALUES (?, ?, ?)", (f"a{i}", f"2026-0{i}-01", stamp))
            conn.execute("INSERT INTO frames (asset_id, path, seq, warped_at)"
                         " VALUES (?, ?, ?, ?)", (f"a{i}", str(frame), i, stamp))
        return conn

    def captured(self, monkeypatch):
        from grow_up import pipeline

        seen = {}

        def fake_encode(frames, out_path, **kwargs):
            seen["frames"] = list(frames)
            seen["out"] = out_path
            seen.update(kwargs)
            return out_path

        monkeypatch.setattr(pipeline, "encode", fake_encode)
        return seen

    def test_uses_the_configured_filename(self, conn, tmp_path, monkeypatch):
        from grow_up import pipeline

        seen = self.captured(monkeypatch)
        out = pipeline.stage_encode(conn, tmp_path / "out",
                                    {"filename": "trial-timelapse.mp4", "fps": 10},
                                    lambda _: None)

        assert out.name == "trial-timelapse.mp4"
        assert seen["out"].name == "trial-timelapse.mp4"
        assert len(seen["frames"]) == 3

    def test_defaults_the_filename(self, conn, tmp_path, monkeypatch):
        from grow_up import pipeline

        self.captured(monkeypatch)
        out = pipeline.stage_encode(conn, tmp_path / "out", {}, lambda _: None)
        assert out.name == "timelapse.mp4"

    def test_honours_manual_rejects(self, conn, tmp_path, monkeypatch):
        from grow_up import pipeline

        seen = self.captured(monkeypatch)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "rejects.json").write_text('{"rejected": ["a2"]}')

        pipeline.stage_encode(conn, out_dir, {}, lambda _: None)
        assert len(seen["frames"]) == 2
        assert all("000002" not in str(f) for f in seen["frames"])

    def test_raises_when_everything_is_rejected(self, conn, tmp_path, monkeypatch):
        from grow_up import pipeline

        self.captured(monkeypatch)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "rejects.json").write_text('{"rejected": ["a1", "a2", "a3"]}')

        with pytest.raises(RuntimeError, match="no frames"):
            pipeline.stage_encode(conn, out_dir, {}, lambda _: None)

    def test_ffmpeg_missing_is_a_runtime_error_subclass(self):
        """cmd_trial catches FFmpegMissing before RuntimeError, so the order of
        those except clauses depends on this relationship."""
        assert issubclass(encode.FFmpegMissing, RuntimeError)


def test_missing_ffmpeg_says_how_to_install_it(monkeypatch):
    monkeypatch.setattr(encode.shutil, "which", lambda _: None)
    with pytest.raises(encode.FFmpegMissing, match="(?i)install it"):
        encode.ffmpeg_binary()
