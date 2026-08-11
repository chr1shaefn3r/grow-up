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

        assert [v.name for v in out] == ["trial-timelapse.mp4"]
        assert seen["out"].name == "trial-timelapse.mp4"
        assert len(seen["frames"]) == 3

    def test_defaults_the_filename(self, conn, tmp_path, monkeypatch):
        from grow_up import pipeline

        self.captured(monkeypatch)
        out = pipeline.stage_encode(conn, tmp_path / "out", {}, lambda _: None)
        assert [v.name for v in out] == ["timelapse.mp4"]

    def test_annotation_off_writes_exactly_one_video(self, conn, tmp_path, monkeypatch):
        """The default has to stay a single render with no annotated leftovers."""
        from grow_up import pipeline

        self.captured(monkeypatch)
        out = pipeline.stage_encode(conn, tmp_path / "out",
                                    {"annotate": {"enabled": False}}, lambda _: None)

        assert len(out) == 1
        assert not (tmp_path / "frames" / "annotated").exists()

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


class TestBothVideos:
    """With a footer configured, the plain render is still produced.

    A date format or a language is a preference, and getting one wrong should
    never be able to cost you the clean video.
    """

    @pytest.fixture()
    def conn(self, tmp_path):
        pytest.importorskip("PIL.Image", reason="needs Pillow")
        from PIL import Image

        from grow_up import db

        conn = db.connect(tmp_path / "t.sqlite")
        stamp = "2026-01-01T00:00:00.000Z"
        frames = tmp_path / "frames"
        frames.mkdir()
        for i in range(1, 4):
            frame = frames / f"frame_{i:06d}.jpg"
            Image.new("RGB", (200, 250), "white").save(frame)
            conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                         " VALUES (?, ?, ?)", (f"a{i}", f"2026-0{i}-01", stamp))
            conn.execute("INSERT INTO frames (asset_id, path, seq, warped_at)"
                         " VALUES (?, ?, ?, ?)", (f"a{i}", str(frame), i, stamp))
        db.upsert_person(conn, "p1", "me", "Kid", "2020-03-14")
        return conn

    def captured(self, monkeypatch):
        from grow_up import pipeline

        seen = []
        monkeypatch.setattr(pipeline, "encode",
                            lambda frames, out, **kw: seen.append((list(frames), out)) or out)
        return seen

    def test_two_videos_are_written(self, conn, tmp_path, monkeypatch):
        from grow_up import pipeline

        seen = self.captured(monkeypatch)
        out = pipeline.stage_encode(conn, tmp_path / "out",
                                    {"annotate": {"enabled": True}}, lambda _: None)

        assert [v.name for v in out] == ["timelapse.mp4", "timelapse-annotated.mp4"]
        assert len(seen) == 2

    def test_the_plain_video_uses_the_untouched_frames(self, conn, tmp_path, monkeypatch):
        from grow_up import pipeline

        seen = self.captured(monkeypatch)
        pipeline.stage_encode(conn, tmp_path / "out",
                              {"annotate": {"enabled": True}}, lambda _: None)

        plain, annotated = (paths for paths, _ in seen)
        assert all("annotated" not in str(p) for p in plain)
        assert all(p.parent.name == "annotated" for p in annotated)
        assert len(plain) == len(annotated) == 3

    def test_the_annotated_frames_differ_from_the_originals(self, conn, tmp_path,
                                                            monkeypatch):
        from grow_up import pipeline

        seen = self.captured(monkeypatch)
        pipeline.stage_encode(conn, tmp_path / "out",
                              {"annotate": {"enabled": True}}, lambda _: None)

        (_, _), (annotated, _) = ((p, o) for p, o in seen)
        assert annotated[0].read_bytes() != (tmp_path / "frames" / "frame_000001.jpg").read_bytes()

    def test_a_missing_birth_date_warns_and_still_annotates(self, conn, tmp_path,
                                                            monkeypatch):
        from grow_up import pipeline

        conn.execute("UPDATE people SET birth_date = NULL")
        self.captured(monkeypatch)
        said: list[str] = []
        out = pipeline.stage_encode(conn, tmp_path / "out",
                                    {"annotate": {"enabled": True}}, said.append)

        assert len(out) == 2, "the annotated video is still produced"
        assert any("birth date" in line for line in said)

    def test_no_warning_when_the_age_is_switched_off(self, conn, tmp_path, monkeypatch):
        from grow_up import pipeline

        conn.execute("UPDATE people SET birth_date = NULL")
        self.captured(monkeypatch)
        said: list[str] = []
        pipeline.stage_encode(conn, tmp_path / "out",
                              {"annotate": {"enabled": True, "age": "off"}}, said.append)

        assert not any("birth date" in line for line in said)

    def test_manual_rejects_apply_to_both(self, conn, tmp_path, monkeypatch):
        seen = self.captured(monkeypatch)
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "rejects.json").write_text('{"rejected": ["a2"]}')

        from grow_up import pipeline
        pipeline.stage_encode(conn, out_dir, {"annotate": {"enabled": True}},
                              lambda _: None)

        assert [len(paths) for paths, _ in seen] == [2, 2]


def test_missing_ffmpeg_says_how_to_install_it(monkeypatch):
    monkeypatch.setattr(encode.shutil, "which", lambda _: None)
    with pytest.raises(encode.FFmpegMissing, match="(?i)install it"):
        encode.ffmpeg_binary()
