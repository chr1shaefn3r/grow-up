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


def test_missing_ffmpeg_says_how_to_install_it(monkeypatch):
    monkeypatch.setattr(encode.shutil, "which", lambda _: None)
    with pytest.raises(encode.FFmpegMissing, match="(?i)install it"):
        encode.ffmpeg_binary()
