from __future__ import annotations

import json
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
            # Every warped frame has a selection row in a real run; the join to
            # it is what keeps runner-ups out of the video.
            conn.execute("INSERT INTO selection (asset_id, bucket, rank, alternate,"
                         " selected_at) VALUES (?, ?, 0, 0, ?)",
                         (f"a{i}", f"2026-0{i}", stamp))
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
            conn.execute("INSERT INTO selection (asset_id, bucket, rank, alternate,"
                         " selected_at) VALUES (?, ?, 0, 0, ?)",
                         (f"a{i}", f"2026-0{i}", stamp))
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


class TestAlternatesNeverReachTheVideo:
    """align warps runner-ups so the contact sheet can show them, which means
    they have frames rows like anything else. The join to selection is the only
    thing keeping them out of the render -- a silent duplicate week otherwise."""

    @pytest.fixture()
    def conn(self, tmp_path):
        from grow_up import db

        conn = db.connect(tmp_path / "t.sqlite")
        stamp = "2026-01-01T00:00:00.000Z"
        for i, (asset, alternate) in enumerate(
                [("pick", 0), ("spare", 1), ("spare2", 1)], start=1):
            frame = tmp_path / f"frame_{i:06d}.jpg"
            frame.write_bytes(b"\xff\xd8\xff")
            conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                         " VALUES (?, '2026-03-02', ?)", (asset, stamp))
            conn.execute("INSERT INTO frames (asset_id, path, seq, warped_at)"
                         " VALUES (?, ?, ?, ?)", (asset, str(frame), i, stamp))
            conn.execute("INSERT INTO selection (asset_id, bucket, rank, alternate,"
                         " selected_at) VALUES (?, '2026-W10', ?, ?, ?)",
                         (asset, i - 1, alternate, stamp))
        return conn

    def test_only_the_pick_is_encoded(self, conn, tmp_path, monkeypatch):
        from grow_up import pipeline

        seen = {}
        monkeypatch.setattr(pipeline, "encode",
                            lambda frames, out, **kw: seen.update(frames=list(frames)) or out)
        pipeline.stage_encode(conn, tmp_path / "out", {}, lambda _: None)

        assert len(seen["frames"]) == 1
        assert "frame_000001" in str(seen["frames"][0])


class TestTransitionFilters:
    """The filter strings are pure arithmetic so they stay testable here, where
    ffmpeg is never installed."""

    def test_none_adds_no_filter_and_keeps_the_hold_rate(self):
        assert encode.transition_filters("none", 0.5, 30.0, 1.0) == ([], 0.5)

    def test_crossfade_lifts_the_rate_to_playback(self):
        _, rate = encode.transition_filters("crossfade", 0.5, 30.0, 1.0)
        assert rate == 30.0

    def test_a_one_second_dissolve_on_a_two_second_hold(self):
        """Half the interval dissolving, centred on the boundary."""
        filters, _ = encode.transition_filters("crossfade", 0.5, 30.0, 1.0)
        assert "interp_start=64" in filters[0]
        assert "interp_end=191" in filters[0]

    def test_scene_detection_is_switched_off(self):
        """Its default treats consecutive photographs as a cut and declines to
        blend them, which is every pair in a timelapse -- the filter would run
        and hand back the hard cuts it was added to remove."""
        filters, _ = encode.transition_filters("crossfade", 0.5, 30.0, 1.0)
        assert "scene=100" in filters[0]

    def test_a_dissolve_longer_than_the_hold_is_clamped(self):
        filters, _ = encode.transition_filters("crossfade", 0.5, 30.0, 99.0)
        assert "interp_start=0" in filters[0] and "interp_end=255" in filters[0]

    def test_a_zero_dissolve_never_blends(self):
        filters, _ = encode.transition_filters("crossfade", 0.5, 30.0, 0.0)
        assert "interp_start=128" in filters[0] and "interp_end=128" in filters[0]

    def test_morph_asks_for_motion_compensation(self):
        filters, rate = encode.transition_filters("morph", 0.5, 30.0, 1.0)
        assert "minterpolate" in filters[0] and "mi_mode=mci" in filters[0]
        assert rate == 30.0

    def test_an_unknown_transition_fails_loudly(self):
        with pytest.raises(ValueError, match="unknown transition"):
            encode.transition_filters("dissolve", 0.5, 30.0, 1.0)


class TestTheOutputRateMatchesTheFilter:
    """The defect this feature was built on top of.

    `-r` runs after the filter chain, so an output rate left at the hold rate
    drops every frame the interpolation just synthesised. The expensive work
    happens and none of it reaches the file -- which is what `interpolate = true`
    did for three releases.
    """

    @pytest.fixture(autouse=True)
    def fake_ffmpeg(self, monkeypatch):
        monkeypatch.setattr(encode.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    def rate_of(self, cmd) -> str:
        return cmd[cmd.index("-r") + 1]

    def test_a_transition_free_command_still_uses_the_hold_rate(self, tmp_path):
        cmd = encode.build_command(tmp_path / "l", tmp_path / "o.mp4", 10.0,
                                   "libx264", 18)
        assert self.rate_of(cmd) == "10.0"

    def test_the_output_rate_follows_the_filter(self, tmp_path):
        filters, rate = encode.transition_filters("crossfade", 0.5, 30.0, 1.0)
        cmd = encode.build_command(tmp_path / "l", tmp_path / "o.mp4", 0.5,
                                   "libx264", 18, filters, output_fps=rate)
        assert self.rate_of(cmd) == "30.0", "the synthesised frames would be discarded"

    def test_morph_is_not_decimated_either(self, tmp_path):
        filters, rate = encode.transition_filters("morph", 0.5, 30.0, 1.0)
        cmd = encode.build_command(tmp_path / "l", tmp_path / "o.mp4", 0.5,
                                   "libx264", 18, filters, output_fps=rate)
        assert self.rate_of(cmd) == "30.0"


class TestTheFooterDoesNotDissolve:
    """The date and age must switch cleanly while the photographs melt.

    Baked into the frames they would go through the same filter as the picture:
    ghosting between two dates under `crossfade`, and warped glyphs under
    `morph`. So they ride a second input and are composited afterwards.
    """

    @pytest.fixture(autouse=True)
    def fake_ffmpeg(self, monkeypatch):
        monkeypatch.setattr(encode.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    def overlaid(self, tmp_path):
        filters, rate = encode.transition_filters("crossfade", 0.5, 30.0, 1.0)
        return encode.build_command(
            tmp_path / "l", tmp_path / "o.mp4", 0.5, "libx264", 18, filters,
            output_fps=rate, overlay_list=tmp_path / "footer.ffconcat")

    def graph(self, cmd) -> str:
        return cmd[cmd.index("-filter_complex") + 1]

    def test_the_footer_arrives_as_a_second_input(self, tmp_path):
        cmd = self.overlaid(tmp_path)
        assert cmd.count("-i") == 2
        assert "-map" in cmd and cmd[cmd.index("-map") + 1] == "[v]"

    def test_the_footer_is_repeated_never_blended(self, tmp_path):
        """`fps` duplicates frames; `framerate` would interpolate them, which is
        the whole thing being avoided."""
        footer = self.graph(self.overlaid(tmp_path)).split("[1:v]")[1].split(";")[0]
        assert "fps=30" in footer
        assert "framerate" not in footer
        assert "minterpolate" not in footer

    def test_the_picture_still_dissolves(self, tmp_path):
        picture = self.graph(self.overlaid(tmp_path)).split("[0:v]")[1].split(";")[0]
        assert "framerate=fps=30" in picture and "scene=100" in picture

    def test_the_alpha_channel_is_demanded_explicitly(self, tmp_path):
        """Losing it shows up as a black band behind the text, not as an error."""
        assert "format=rgba" in self.graph(self.overlaid(tmp_path))

    def test_the_overlay_runs_before_the_even_dimension_guard(self, tmp_path):
        graph = self.graph(self.overlaid(tmp_path))
        assert graph.index("overlay=0:0") < graph.index("scale=trunc")

    def test_without_an_overlay_the_command_keeps_its_old_shape(self, tmp_path):
        filters, rate = encode.transition_filters("crossfade", 0.5, 30.0, 1.0)
        cmd = encode.build_command(tmp_path / "l", tmp_path / "o.mp4", 0.5,
                                   "libx264", 18, filters, output_fps=rate)
        assert "-filter_complex" not in cmd and "-vf" in cmd


class TestTheFooterSwitchesMidDissolve:
    """Photographs carry timestamps at the *end* of the dissolve leading into
    them. A footer list following them directly would leave the next photo on
    screen already, still labelled with the previous date."""

    def test_the_first_entry_is_half_a_hold(self):
        assert encode.footer_concat_offset(0.5) == 1.0      # half of a 2s hold
        assert encode.footer_concat_offset(10.0) == 0.05

    def test_only_the_opening_entry_is_shortened(self, frames, tmp_path):
        listing = encode.write_concat_list(frames, 0.5, tmp_path / "f.ffconcat",
                                           first_duration=1.0)
        durations = [line for line in listing.read_text().splitlines()
                     if line.startswith("duration")]
        assert durations[0] == "duration 1.000000"
        assert all(d == "duration 2.000000" for d in durations[1:])

    def test_the_frame_list_is_untouched_without_it(self, frames, tmp_path):
        listing = encode.write_concat_list(frames, 0.5, tmp_path / "f.ffconcat")
        durations = [line for line in listing.read_text().splitlines()
                     if line.startswith("duration")]
        assert all(d == "duration 2.000000" for d in durations)


class TestTheTransitionReachesTheStage:
    """What stage_encode resolves from config, including the legacy spelling."""

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
            conn.execute("INSERT INTO selection (asset_id, bucket, rank, alternate,"
                         " selected_at) VALUES (?, ?, 0, 0, ?)",
                         (f"a{i}", f"2026-0{i}", stamp))
        return conn

    def run_with(self, conn, tmp_path, monkeypatch, cfg):
        from grow_up import pipeline

        seen = {}
        monkeypatch.setattr(pipeline, "encode",
                            lambda frames, out, **kw: seen.update(kw) or out)
        lines = []
        pipeline.stage_encode(conn, tmp_path / "out", cfg, lines.append)
        return seen, lines

    def test_the_settings_are_passed_through(self, conn, tmp_path, monkeypatch):
        seen, _ = self.run_with(conn, tmp_path, monkeypatch, {
            "fps": 0.5, "transition": "crossfade", "playback_fps": 30,
            "transition_seconds": 1.0})
        assert seen["transition"] == "crossfade"
        assert seen["playback_fps"] == 30.0
        assert seen["transition_seconds"] == 1.0

    def test_the_default_is_no_transition(self, conn, tmp_path, monkeypatch):
        seen, _ = self.run_with(conn, tmp_path, monkeypatch, {})
        assert seen["transition"] == "none"

    def test_the_run_reports_what_it_resolved(self, conn, tmp_path, monkeypatch):
        """A config edit has no other feedback until the video appears."""
        _, lines = self.run_with(conn, tmp_path, monkeypatch, {
            "fps": 0.5, "transition": "crossfade", "transition_seconds": 1.0})
        assert any("crossfade at 30 fps" in line for line in lines)
        assert any("1s moving and 1s still per 2s photo" in line for line in lines)

    def test_a_clamped_dissolve_says_so(self, conn, tmp_path, monkeypatch):
        _, lines = self.run_with(conn, tmp_path, monkeypatch, {
            "fps": 0.5, "transition": "crossfade", "transition_seconds": 9.0})
        assert any("clamped" in line for line in lines)

    def test_a_transition_free_run_stays_quiet(self, conn, tmp_path, monkeypatch):
        _, lines = self.run_with(conn, tmp_path, monkeypatch, {"fps": 10})
        assert not any("dissolving" in line for line in lines)

    def test_the_old_interpolate_setting_still_means_morph(self, conn, tmp_path,
                                                           monkeypatch):
        """Published in 1.0.0, so a config carrying it must keep working."""
        seen, lines = self.run_with(conn, tmp_path, monkeypatch, {"interpolate": True})
        assert seen["interpolate"] is True and seen["transition"] == "none"
        assert any("morph at" in line for line in lines), "encode() resolves it"

    def test_an_explicit_transition_beats_the_old_setting(self, conn, tmp_path,
                                                          monkeypatch):
        _, lines = self.run_with(conn, tmp_path, monkeypatch, {
            "interpolate": True, "transition": "crossfade", "fps": 0.5})
        assert any("crossfade at" in line for line in lines)


class TestTheAnnotatedVideoPicksTheRightPath:
    """Two ways the footer reaches the video, and only one is right per case.

    Without a transition the footer is baked into the frames as it always was.
    With one, the *unbaked* frames are encoded and the footer is handed over as
    an overlay -- passing the baked frames there would dissolve the text and
    draw it twice.
    """

    @pytest.fixture()
    def conn(self, tmp_path):
        from grow_up import db

        pytest.importorskip("PIL.Image", reason="needs Pillow")
        from PIL import Image

        conn = db.connect(tmp_path / "t.sqlite")
        stamp = "2026-01-01T00:00:00.000Z"
        db.upsert_person(conn, "p1", "me", "Kid", "2020-03-14")
        for i in range(1, 4):
            frame = tmp_path / f"frame_{i:06d}.jpg"
            Image.new("RGB", (80, 100), (90, 90, 90)).save(frame)
            conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                         " VALUES (?, ?, ?)", (f"a{i}", f"2026-0{i}-01", stamp))
            conn.execute("INSERT INTO frames (asset_id, path, seq, warped_at)"
                         " VALUES (?, ?, ?, ?)", (f"a{i}", str(frame), i, stamp))
            conn.execute("INSERT INTO selection (asset_id, bucket, rank, alternate,"
                         " selected_at) VALUES (?, ?, 0, 0, ?)",
                         (f"a{i}", f"2026-0{i}", stamp))
        return conn

    def calls(self, conn, tmp_path, monkeypatch, cfg):
        from grow_up import pipeline

        seen = []
        monkeypatch.setattr(pipeline, "encode",
                            lambda frames, out, **kw: seen.append((list(frames), kw)) or out)
        cfg = {**cfg, "annotate": {"enabled": True, "age": "year_months"}}
        pipeline.stage_encode(conn, tmp_path / "out", cfg, lambda _: None)
        return seen

    def test_without_a_transition_the_footer_is_baked_in(self, conn, tmp_path,
                                                          monkeypatch):
        _, annotated = self.calls(conn, tmp_path, monkeypatch, {"fps": 10})
        assert annotated[1]["overlays"] is None
        assert all("annotated" in str(f) for f in annotated[0])

    def test_with_a_transition_the_footer_becomes_an_overlay(self, conn, tmp_path,
                                                              monkeypatch):
        _, annotated = self.calls(conn, tmp_path, monkeypatch, {
            "fps": 0.5, "transition": "crossfade"})
        overlays = annotated[1]["overlays"]

        assert overlays and all(str(f).endswith(".png") for f in overlays)
        assert all("footers" in str(f) for f in overlays)

    def test_the_overlaid_video_encodes_the_unbaked_frames(self, conn, tmp_path,
                                                           monkeypatch):
        """Otherwise the footer is drawn twice, and the baked one dissolves."""
        plain, annotated = self.calls(conn, tmp_path, monkeypatch, {
            "fps": 0.5, "transition": "crossfade"})
        assert annotated[0] == plain[0]
        assert not any("annotated" in str(f) for f in annotated[0])

    def test_the_layers_are_transparent_where_the_picture_shows(self, conn, tmp_path,
                                                                 monkeypatch):
        from PIL import Image

        _, annotated = self.calls(conn, tmp_path, monkeypatch, {
            "fps": 0.5, "transition": "crossfade"})
        with Image.open(annotated[1]["overlays"][0]) as layer:
            assert layer.mode == "RGBA"
            assert layer.getpixel((40, 5))[3] == 0


class TestTheLastFooterSurvivesTheTail:
    """Shortening the first footer entry makes the whole footer stream half a
    hold shorter than the picture. Dropping it at that point would leave the
    final photograph unlabelled for the last half-second."""

    @pytest.fixture(autouse=True)
    def fake_ffmpeg(self, monkeypatch):
        monkeypatch.setattr(encode.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    def test_the_overlay_repeats_past_the_end_of_the_footer(self, tmp_path):
        filters, rate = encode.transition_filters("crossfade", 0.5, 30.0, 1.0)
        cmd = encode.build_command(tmp_path / "l", tmp_path / "o.mp4", 0.5,
                                   "libx264", 18, filters, output_fps=rate,
                                   overlay_list=tmp_path / "f.ffconcat")
        assert "eof_action=repeat" in cmd[cmd.index("-filter_complex") + 1]

    def test_the_footer_list_is_shorter_by_half_a_hold(self, frames, tmp_path):
        """The arithmetic behind it, so the reason is not just a comment."""
        listing = encode.write_concat_list(frames, 0.5, tmp_path / "f.ffconcat",
                                           first_duration=encode.footer_concat_offset(0.5))
        held = [float(line.split()[1]) for line in listing.read_text().splitlines()
                if line.startswith("duration")]
        assert sum(held) == len(frames) * 2.0 - 1.0


class TestTheTransitionLengthReachesBothBranches:
    """The defect that shipped: `morph` took the timing argument and dropped it.

    Every morph test passed 1.0 and asserted only that `minterpolate` appeared,
    so deleting the parameter outright would have left the suite green. These
    assert the value changes the output, which is the only thing that could have
    caught it.
    """

    def entries(self, transition, seconds, frames):
        return encode.concat_entries(frames, 0.5, transition, seconds)

    def test_morph_timing_changes_the_entries(self, frames):
        half = self.entries("morph", 0.5, frames)
        full = self.entries("morph", 1.0, frames)
        assert half != full, "transition_seconds is being ignored on the morph path"

    def test_morph_holds_then_moves(self, frames):
        """1.5s still and 0.5s morphing, out of a 2s slot."""
        held = [duration for _, duration in self.entries("morph", 0.5, frames)]
        assert held[:2] == [1.5, 0.5]

    def test_each_photo_keeps_its_whole_slot(self, frames):
        entries = self.entries("morph", 0.5, frames)
        assert sum(d for _, d in entries) == len(frames) * 2.0

    def test_a_photo_appears_twice_under_morph(self, frames):
        """The still half and the moving half are the same picture; motion
        estimation finds nothing between them, which is what makes it hold."""
        paths = [path for path, _ in self.entries("morph", 0.5, frames)]
        assert paths[0] == paths[1] and paths[2] == paths[3]

    def test_morph_timing_is_clamped_to_the_hold(self, frames):
        """Nothing left to hold, so the morph fills the slot and the doubling
        goes away -- a `duration 0.000000` entry would be handed to ffmpeg for
        no reason."""
        entries = self.entries("morph", 99.0, frames)
        assert len(entries) == len(frames)
        assert all(d == 2.0 for _, d in entries)

    def test_the_default_timing_leaves_a_fast_run_continuous(self, frames):
        """At fps = 10 a photo is up for 0.1s, so the 1.0s default clamps and
        morph behaves exactly as it did before it had a hold."""
        entries = encode.concat_entries(frames, 10.0, "morph", 1.0)
        assert len(entries) == len(frames)

    def test_crossfade_is_not_doubled(self, frames):
        """Its hold comes from interp_start/interp_end, not from the input."""
        entries = self.entries("crossfade", 0.5, frames)
        assert len(entries) == len(frames)
        assert all(d == 2.0 for _, d in entries)

    def test_no_transition_is_not_doubled_either(self, frames):
        entries = self.entries("none", 0.5, frames)
        assert len(entries) == len(frames)


class TestMorphSceneDetection:
    """minterpolate has its own scene-change detection, and it defaults on.

    When it fires it stops interpolating and duplicates frames instead, so a
    timelapse -- where every consecutive pair is a huge difference -- gets hard
    cuts at full motion-compensation cost. Exactly the trap `scene=100` guards
    on the crossfade path, missed on this one.
    """

    def test_scene_detection_is_switched_off(self):
        filters, _ = encode.transition_filters("morph", 0.5, 30.0, 1.0)
        assert "scd=none" in filters[0]

    def test_both_transitions_disable_their_own_detector(self):
        crossfade, _ = encode.transition_filters("crossfade", 0.5, 30.0, 1.0)
        morph, _ = encode.transition_filters("morph", 0.5, 30.0, 1.0)
        assert "scene=100" in crossfade[0] and "scd=none" in morph[0]


class TestTheFooterFollowsTheTransition:
    """Half a hold is right only while the transition straddles the boundary.

    Morph's sits at the end of each photo's slot, because a duplicated frame can
    only hold from the start of one. A footer still switching at the midpoint of
    the slot would change the date before the morph had begun.
    """

    def test_crossfade_switches_at_the_middle_of_the_slot(self):
        assert encode.footer_concat_offset(0.5, "crossfade", 1.0) == 1.0

    def test_morph_switches_in_the_middle_of_the_morph(self):
        """2s slot, 0.5s morphing at the end -> midpoint at 1.75s."""
        assert encode.footer_concat_offset(0.5, "morph", 0.5) == 1.75

    def test_a_longer_morph_moves_the_switch_earlier(self):
        assert encode.footer_concat_offset(0.5, "morph", 1.0) == 1.5


class TestTheOlderSpellingKeepsWorking:
    """`crossfade_seconds` named the setting while it only governed a crossfade.

    It governs the morph's hold too now, so the general name is the honest one --
    but the old one is in live config files and must not start being ignored,
    which is the very failure this change exists to fix.
    """

    @pytest.fixture()
    def conn(self, tmp_path):
        from grow_up import db

        conn = db.connect(tmp_path / "t.sqlite")
        stamp = "2026-01-01T00:00:00.000Z"
        frame = tmp_path / "frame_000001.jpg"
        frame.write_bytes(b"\xff\xd8\xff")
        conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                     " VALUES ('a1', '2026-01-01', ?)", (stamp,))
        conn.execute("INSERT INTO frames (asset_id, path, seq, warped_at)"
                     " VALUES ('a1', ?, 1, ?)", (str(frame), stamp))
        conn.execute("INSERT INTO selection (asset_id, bucket, rank, alternate,"
                     " selected_at) VALUES ('a1', '2026-01', 0, 0, ?)", (stamp,))
        return conn

    def seconds_for(self, conn, tmp_path, monkeypatch, cfg):
        from grow_up import pipeline

        seen = {}
        monkeypatch.setattr(pipeline, "encode",
                            lambda frames, out, **kw: seen.update(kw) or out)
        pipeline.stage_encode(conn, tmp_path / "out",
                              {"fps": 0.5, "transition": "morph", **cfg}, lambda _: None)
        return seen["transition_seconds"]

    def test_the_old_name_is_still_read(self, conn, tmp_path, monkeypatch):
        assert self.seconds_for(conn, tmp_path, monkeypatch,
                                {"crossfade_seconds": 0.5}) == 0.5

    def test_the_new_name_wins_where_both_appear(self, conn, tmp_path, monkeypatch):
        assert self.seconds_for(conn, tmp_path, monkeypatch,
                                {"crossfade_seconds": 0.5,
                                 "transition_seconds": 1.5}) == 1.5

    def test_morph_reports_its_split(self, conn, tmp_path, monkeypatch):
        """The report that was missing: a morph printed no timing at all, so a
        value being silently dropped looked exactly like one being honoured."""
        from grow_up import pipeline

        monkeypatch.setattr(pipeline, "encode", lambda frames, out, **kw: out)
        lines = []
        pipeline.stage_encode(conn, tmp_path / "out",
                              {"fps": 0.5, "transition": "morph",
                               "transition_seconds": 0.5}, lines.append)
        assert any("0.5s moving and 1.5s still per 2s photo" in line for line in lines)


class TestTheStageReportsWhatItCost:
    """encode is the slowest stage once a transition synthesises frames, and it
    used to go silent between announcing the work and the files appearing."""

    @pytest.fixture()
    def conn(self, tmp_path):
        from grow_up import db

        conn = db.connect(tmp_path / "t.sqlite")
        stamp = "2026-01-01T00:00:00.000Z"
        db.upsert_person(conn, "p1", "me", "Kid", "2020-03-14")
        # Real images: the annotated path opens them with Pillow, and the
        # timing is measured around a monkeypatched encode either way.
        Image = pytest.importorskip("PIL.Image", reason="needs Pillow")
        for i in range(1, 3):
            frame = tmp_path / f"frame_{i:06d}.jpg"
            Image.new("RGB", (80, 100), (90, 90, 90)).save(frame)
            conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                         " VALUES (?, ?, ?)", (f"a{i}", f"2026-0{i}-01", stamp))
            conn.execute("INSERT INTO frames (asset_id, path, seq, warped_at)"
                         " VALUES (?, ?, ?, ?)", (f"a{i}", str(frame), i, stamp))
            conn.execute("INSERT INTO selection (asset_id, bucket, rank, alternate,"
                         " selected_at) VALUES (?, ?, 0, 0, ?)",
                         (f"a{i}", f"2026-0{i}", stamp))
        return conn

    def lines_for(self, conn, tmp_path, monkeypatch, cfg, clock=None):
        from grow_up import pipeline, timing

        monkeypatch.setattr(pipeline, "encode", lambda frames, out, **kw: out)
        if clock is not None:
            ticks = iter(clock)
            monkeypatch.setattr(timing.time, "perf_counter", lambda: next(ticks))
        lines = []
        pipeline.stage_encode(conn, tmp_path / "out", cfg, lines.append)
        return lines

    def timing_lines(self, lines):
        return [line for line in lines if " in " in line or "stage took" in line]

    def test_a_render_reports_its_own_time(self, conn, tmp_path, monkeypatch):
        lines = self.lines_for(conn, tmp_path, monkeypatch, {})
        assert any("timelapse.mp4 in " in line for line in lines)

    def test_the_path_is_left_to_the_wrote_line(self, conn, tmp_path, monkeypatch):
        """cli prints the full path right below; repeating it here is noise."""
        lines = self.timing_lines(self.lines_for(conn, tmp_path, monkeypatch, {}))
        assert not any(str(tmp_path) in line for line in lines)

    def test_one_render_gets_no_total(self, conn, tmp_path, monkeypatch):
        """It would restate the line above it, which teaches you to skip both."""
        lines = self.lines_for(conn, tmp_path, monkeypatch, {})
        assert not any("stage took" in line for line in lines)
        assert len(self.timing_lines(lines)) == 1

    def test_both_renders_report_and_a_total_follows(self, conn, tmp_path, monkeypatch):
        pytest.importorskip("PIL.Image", reason="needs Pillow")
        lines = self.lines_for(conn, tmp_path, monkeypatch, {
            "annotate": {"enabled": True, "age": "year_months"}})
        timed = self.timing_lines(lines)

        assert len(timed) == 3
        assert "timelapse.mp4 in " in timed[0]
        assert "timelapse-annotated.mp4 in " in timed[1]
        assert "stage took" in timed[2]

    def test_the_annotated_render_is_shown_costing_its_own_time(self, conn, tmp_path,
                                                                monkeypatch):
        """The point of reporting both: with a footer on, ffmpeg runs twice."""
        pytest.importorskip("PIL.Image", reason="needs Pillow")
        lines = self.lines_for(conn, tmp_path, monkeypatch, {
            "annotate": {"enabled": True, "age": "year_months"}})
        assert sum(1 for line in lines if ".mp4 in " in line) == 2

    def test_durations_are_formatted_not_raw_seconds(self, conn, tmp_path, monkeypatch):
        """A bare 128.42131 in the output is the regression this guards."""
        pytest.importorskip("PIL.Image", reason="needs Pillow")
        # stage start, render 1 in/out, render 2 in/out, stage end.
        clock = [0.0, 0.0, 126.0, 126.0, 257.0, 259.0]
        timed = self.timing_lines(self.lines_for(
            conn, tmp_path, monkeypatch,
            {"annotate": {"enabled": True, "age": "year_months"}}, clock=clock))

        assert timed[0].endswith("timelapse.mp4 in 2m 06s")
        assert timed[1].endswith("timelapse-annotated.mp4 in 2m 11s")
        assert timed[2].endswith("stage took 4m 19s")

    def test_a_quick_render_reads_in_milliseconds(self, conn, tmp_path, monkeypatch):
        timed = self.timing_lines(self.lines_for(
            conn, tmp_path, monkeypatch, {}, clock=[0.0, 0.0, 0.012, 0.02]))
        assert timed[0].endswith("in 12ms")


class TestAStaleRejectIsNotSilent:
    """`grow-up encode` alone can drop a frame but cannot reconsider a bucket.

    Following that advice used to leave a week missing and print
    `honouring 11 manual rejects`, which reads like success. The video looks
    entirely plausible; only the calendar is wrong.
    """

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
            conn.execute("INSERT INTO selection (asset_id, bucket, rank, alternate,"
                         " selected_at) VALUES (?, ?, 0, 0, ?)",
                         (f"a{i}", f"2026-0{i}", stamp))
        return conn

    def run_with(self, conn, tmp_path, monkeypatch, rejected):
        from grow_up import pipeline

        monkeypatch.setattr(pipeline, "encode", lambda frames, out, **kw: out)
        out_dir = tmp_path / "out"
        out_dir.mkdir(exist_ok=True)
        (out_dir / "rejects.json").write_text(json.dumps({"rejected": rejected}))
        lines = []
        pipeline.stage_encode(conn, out_dir, {}, lines.append)
        return lines

    def test_a_rejection_select_has_not_seen_is_reported(self, conn, tmp_path,
                                                          monkeypatch):
        lines = self.run_with(conn, tmp_path, monkeypatch, ["a2"])
        assert any("not been through select" in line for line in lines)

    def test_the_warning_names_all_three_stages(self, conn, tmp_path, monkeypatch):
        """One command is what caused this; the fix has to spell out three."""
        lines = self.run_with(conn, tmp_path, monkeypatch, ["a2"])
        advice = next(line for line in lines if "Run:" in line)
        assert "grow-up select && grow-up align && grow-up encode" == advice.split("Run: ")[1]

    def test_it_counts_only_what_this_stage_dropped(self, conn, tmp_path, monkeypatch):
        """Ids already gone from selection are select's work, not a warning."""
        conn.execute("DELETE FROM selection WHERE asset_id = 'a3'")
        lines = self.run_with(conn, tmp_path, monkeypatch, ["a2", "a3"])
        assert any("1 rejection has not been" in line for line in lines)

    def test_the_plural_agrees(self, conn, tmp_path, monkeypatch):
        lines = self.run_with(conn, tmp_path, monkeypatch, ["a1", "a2"])
        assert any("2 rejections have not been" in line for line in lines)

    def test_a_rejection_select_applied_says_nothing(self, conn, tmp_path, monkeypatch):
        """After select the id is gone from selection, so this stage drops
        nothing and has nothing to report -- select already did."""
        conn.execute("DELETE FROM selection WHERE asset_id = 'a2'")
        lines = self.run_with(conn, tmp_path, monkeypatch, ["a2"])

        assert not any("not been through select" in line for line in lines)
        assert not any("honouring" in line for line in lines)

    def test_a_clean_run_never_mentions_rejects(self, conn, tmp_path, monkeypatch):
        lines = self.run_with(conn, tmp_path, monkeypatch, [])
        assert not any("reject" in line for line in lines)

    def test_the_frame_is_still_dropped(self, conn, tmp_path, monkeypatch):
        """The warning explains the gap; it does not avert it. Encoding alone
        must keep honouring the file, which is what makes a quick re-render
        possible when nothing needs promoting."""
        from grow_up import pipeline

        seen = {}
        monkeypatch.setattr(pipeline, "encode",
                            lambda frames, out, **kw: seen.update(f=list(frames)) or out)
        out_dir = tmp_path / "out"
        out_dir.mkdir(exist_ok=True)
        (out_dir / "rejects.json").write_text('{"rejected": ["a2"]}')
        pipeline.stage_encode(conn, out_dir, {}, lambda _: None)

        assert len(seen["f"]) == 2
        assert all("000002" not in str(f) for f in seen["f"])
