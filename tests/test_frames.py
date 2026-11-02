"""A frame file holds the photograph it is named after, whatever else changes.

`align` used to name frames by their place in the timeline -- picks and
runner-ups together, in date order -- so any change to the selection renumbered
everything after it. One rejection in a month with few photos moved every later
frame along by one, and `frame_000123.jpg` quietly became a different
photograph.

Two things held on to the old names. The contact sheet, which the advice for
applying rejections -- `select && align && encode` -- never rewrote: re-opened
after following it, the sheet showed other photos in months nobody had touched,
and a reject clicked there would have been recorded against a photo the user
never saw. And the manifest, while an `align` was part-way through: the files
are written before the rows that point at them, so a run that failed left
`encode` reading the neighbours of the chosen photos.

The video was right whenever `align` finished, which is why every test that
looked only at the video passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from grow_up import cli, config, db, pipeline, review, select

LIMITS = {
    "max_yaw": 20.0, "max_pitch": 18.0, "max_roll": 25.0, "max_gaze": 0.35,
    "max_blink": 0.45, "max_oob_frac": 0.005, "allow_bbox_clipped": False,
    "min_interocular_px": 10.0, "min_sharpness": 12.0,
    "min_exposure_lo": 8.0, "max_exposure_hi": 250.0,
}
WEIGHTS = {"w_pose": 1.0, "w_gaze": 1.0, "w_eyes_open": 1.0,
           "w_sharpness": 1.0, "w_size": 0.5}
STAMP = "2026-01-01T00:00:00.000Z"
BIRTH = "2024-03-14"

# Photos per month of age. The second month has fewer than a pick and its four
# runner-ups, so rejecting its pick shrinks the selection by one -- which is
# what shifted every later frame.
PER_MONTH = [7, 3, 6, 2, 7, 4]


def colour(i: int) -> tuple[int, int, int]:
    return (37 * i % 200 + 20, 91 * i % 200 + 20, 53 * i % 200 + 20)


def configured(tmp_path: Path, **paths: str) -> config.Config:
    return config.Config(raw={
        "paths": {"frames": "frames", "out": "out", **paths},
        "filter": LIMITS,
        "score": WEIGHTS,
        "select": {"cadence": select.BIRTHDAY_MONTHS, "per_bucket": 1, "alternates": 4},
        # Exact colours: flicker matching would shift them.
        "output": {"width": 64, "height": 80, "flicker_match": False},
    }, root=tmp_path)


@pytest.fixture()
def opened(tmp_path, monkeypatch):
    """Point every `cli.cmd_*` at one database and config in `tmp_path`."""
    conn = db.connect(tmp_path / "t.sqlite")
    cfg = configured(tmp_path)
    monkeypatch.setattr(cli, "_open", lambda _args: (cfg, conn))
    monkeypatch.setattr(cli, "log", lambda _msg: None)
    return cfg, conn


class Library:
    """Solid-colour photographs, so a frame's pixels say which one it holds."""

    def __init__(self, conn, root: Path):
        cv2 = pytest.importorskip("cv2")
        pytest.importorskip("PIL.Image", reason="needs Pillow to decode")
        self.conn = conn
        self.palette: dict[str, np.ndarray] = {}
        db.upsert_person(conn, "p1", "me", "Kid", BIRTH)
        cache = root / "cache"
        cache.mkdir()
        n = 0
        for month, count in enumerate(PER_MONTH):
            for k in range(count):
                n += 1
                asset = f"m{month}-{k}"
                image = np.zeros((200, 200, 3), np.uint8)
                image[:] = colour(n)
                path = cache / f"{asset}.png"
                cv2.imwrite(str(path), image)
                self.palette[asset] = np.array(colour(n), float)
                # Two days apart from the 15th: every photo stays inside the
                # month of age that opens on the 14th.
                when = f"2024-{3 + month:02d}-{15 + 2 * k:02d}T10:00:00.000Z"
                conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                             " VALUES (?, ?, ?)", (asset, when, STAMP))
                conn.execute("INSERT INTO downloads (asset_id, path, source, fetched_at)"
                             " VALUES (?, ?, 'original', ?)", (asset, str(path), STAMP))
                # Sharpness falls with k, so the first photo of a month is its pick.
                conn.execute(
                    "INSERT INTO metrics (asset_id, detected, yaw, pitch, roll, gaze_x,"
                    " gaze_y, blink_l, blink_r, oob_frac, bbox_clipped, interocular_px,"
                    " left_eye_x, left_eye_y, right_eye_x, right_eye_y, sharpness,"
                    " exposure_lo, exposure_hi, analyzed_at)"
                    " VALUES (?, 1, 2, 1, 1, 0, 0, .05, .05, 0, 0, 40, 80, 100, 120, 100,"
                    " ?, 40, 200, ?)", (asset, 100.0 - 10 * k, STAMP))

    def holds(self, path: str | Path) -> str | None:
        """Which photograph a frame file really contains, or None if it is gone."""
        import cv2

        image = cv2.imread(str(path))
        if image is None:
            return None
        mean = image.reshape(-1, 3).mean(axis=0)
        return min(self.palette, key=lambda a: np.abs(self.palette[a] - mean).sum())

    def frames(self) -> dict[str, str]:
        return {r["asset_id"]: r["path"] for r in self.conn.execute(
            "SELECT asset_id, path FROM frames")}


def run(*commands) -> None:
    args = argparse.Namespace(cadence=None)
    for command in commands:
        command(args)


def sheet_images(page: Path) -> list[tuple[str, Path]]:
    """Every (asset id, image file) pair the contact sheet would act on."""
    data = json.loads(page.read_text().split('id="buckets">')[1].split("</script>")[0])
    return [(c["id"], page.parent / c["src"]) for b in data for c in b["candidates"]]


class TestTheAdvisedSequenceLeavesATruthfulSheet:
    """The reported sequence, end to end: a clean run, rejections on the
    contact sheet, then exactly what the page says to run."""

    @pytest.fixture()
    def library(self, opened, tmp_path):
        library = Library(opened[1], tmp_path)
        run(cli.cmd_select, cli.cmd_align, cli.cmd_review)
        return library

    def reject(self, tmp_path, *ids: str) -> None:
        (tmp_path / "out" / "rejects.json").write_text(json.dumps({"rejected": list(ids)}))
        run(cli.cmd_select, cli.cmd_align)

    def test_the_first_sheet_shows_what_it_says(self, library, tmp_path):
        for asset, image in sheet_images(tmp_path / "out" / "contact-sheet.html"):
            assert library.holds(image) == asset

    def test_after_rejecting_every_image_on_the_sheet_is_the_photo_it_names(
            self, library, tmp_path):
        """A reject or a choice made on the page is recorded by id. A picture
        that is not that id's photo gets the wrong one dropped."""
        self.reject(tmp_path, "m1-0", "m0-0", "m0-1")

        for asset, image in sheet_images(tmp_path / "out" / "contact-sheet.html"):
            assert library.holds(image) == asset, f"{asset}'s card shows another photo"

    def test_after_rejecting_the_sheet_shows_the_new_picks(self, library, tmp_path):
        self.reject(tmp_path, "m1-0", "m0-0", "m0-1")

        page = (tmp_path / "out" / "contact-sheet.html").read_text()
        data = json.loads(page.split('id="buckets">')[1].split("</script>")[0])
        assert [b["candidates"][0]["id"] for b in data] == [
            "m0-2", "m1-1", "m2-0", "m3-0", "m4-0", "m5-0"]

    def test_a_tab_left_open_on_the_old_sheet_never_shows_another_photo(
            self, library, tmp_path):
        """Nobody has to reload for a picture to be the photo its id says.
        One that has since left the selection may be gone, which reads as out
        of date; it may not be some other photo."""
        before = sheet_images(tmp_path / "out" / "contact-sheet.html")
        self.reject(tmp_path, "m1-0", "m0-0", "m0-1")

        for asset, image in before:
            assert library.holds(image) in (asset, None)


class TestAFramePathAlwaysMeansOnePhotograph:
    @pytest.fixture()
    def library(self, opened, tmp_path):
        library = Library(opened[1], tmp_path)
        run(cli.cmd_select, cli.cmd_align)
        return library

    def reselect(self, tmp_path, *ids: str) -> None:
        (tmp_path / "out").mkdir(exist_ok=True)
        (tmp_path / "out" / "rejects.json").write_text(json.dumps({"rejected": list(ids)}))
        run(cli.cmd_select)

    def test_a_rejection_moves_no_other_photo_to_a_new_path(self, library, tmp_path):
        before = library.frames()
        self.reselect(tmp_path, "m1-0")
        run(cli.cmd_align)

        after = library.frames()
        assert {a: after[a] for a in after if a in before} == {
            a: before[a] for a in after if a in before}

    def test_every_frame_holds_its_own_photo_after_a_rejection(self, library, tmp_path):
        self.reselect(tmp_path, "m1-0")
        run(cli.cmd_align)

        for asset, path in library.frames().items():
            assert library.holds(path) == asset

    def test_the_frame_of_a_photo_no_longer_selected_is_removed(self, library, tmp_path):
        rejected = library.frames()["m1-0"]
        self.reselect(tmp_path, "m1-0")
        run(cli.cmd_align)

        assert not Path(rejected).exists()
        assert sorted(p.name for p in (tmp_path / "frames").glob("*.jpg")) == sorted(
            Path(p).name for p in library.frames().values())

    def test_an_align_that_fails_part_way_leaves_every_frame_its_own_photo(
            self, library, tmp_path):
        """The files are written before the rows that point at them, so a
        failure leaves the old rows in place. They must still be right."""
        self.reselect(tmp_path, "m1-0")
        broken = library.conn.execute(
            "SELECT path FROM downloads WHERE asset_id = 'm5-3'").fetchone()[0]
        Path(broken).write_bytes(b"not an image")

        with pytest.raises(Exception):
            run(cli.cmd_align)

        encoded = library.conn.execute(
            "SELECT f.asset_id, f.path FROM frames f JOIN selection s USING (asset_id)"
            " WHERE s.alternate = 0").fetchall()
        assert encoded
        for asset, path in encoded:
            assert library.holds(path) == asset, f"encode would show another photo for {asset}"


class TestOnlySupersededFramesAreRemoved:
    """`align` deletes what it wrote before and did not write again -- and
    nothing it cannot be sure is its own."""

    def frames(self, tmp_path, *names: str) -> list[str]:
        home = tmp_path / "frames"
        home.mkdir(exist_ok=True)
        paths = []
        for name in names:
            (home / name).write_bytes(b"x")
            paths.append(str(home / name))
        return paths

    def test_a_frame_written_again_is_kept(self, tmp_path):
        kept = self.frames(tmp_path, "a.jpg")
        pipeline._remove_superseded(kept, kept, tmp_path / "frames")
        assert Path(kept[0]).exists()

    def test_a_frame_not_written_again_is_removed(self, tmp_path):
        old, new = self.frames(tmp_path, "old.jpg", "new.jpg")
        pipeline._remove_superseded([old, new], [new], tmp_path / "frames")
        assert not Path(old).exists() and Path(new).exists()

    def test_the_same_file_spelled_another_way_is_still_kept(self, tmp_path, monkeypatch):
        """A run started from another directory records the same frame under a
        different spelling; that must not read as a superseded file."""
        (path,) = self.frames(tmp_path, "a.jpg")
        monkeypatch.chdir(tmp_path)
        pipeline._remove_superseded([path], ["frames/a.jpg"], Path("frames"))
        assert Path(path).exists()

    def test_a_frame_outside_the_frames_directory_is_left_alone(self, tmp_path):
        """A row from before `paths.frames` moved points at someone else's
        directory now."""
        elsewhere = tmp_path / "moved"
        elsewhere.mkdir()
        (elsewhere / "a.jpg").write_bytes(b"x")
        pipeline._remove_superseded([str(elsewhere / "a.jpg")], [], tmp_path / "frames")
        assert (elsewhere / "a.jpg").exists()

    def test_a_frame_already_gone_is_not_an_error(self, tmp_path):
        (tmp_path / "frames").mkdir()
        pipeline._remove_superseded([str(tmp_path / "frames" / "gone.jpg")], [],
                                    tmp_path / "frames")


class TestAlignRewritesTheContactSheet:
    """The page is a view of the frames, so the stage that rewrites them
    rewrites it. Runs with no opencv: the warp itself is stubbed out."""

    @pytest.fixture()
    def conn(self, opened, tmp_path, monkeypatch):
        cfg, conn = opened
        for i in range(1, 3):
            conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                         " VALUES (?, ?, ?)", (f"a{i}", f"2026-0{i}-10T10:00:00Z", STAMP))
            conn.execute("INSERT INTO selection (asset_id, bucket, rank, selected_at)"
                         " VALUES (?, ?, 0, ?)", (f"a{i}", f"2026-0{i}", STAMP))

        def stage_align(conn, frames_dir, *_args, **_kwargs):
            conn.execute("DELETE FROM frames")
            for seq, (asset,) in enumerate(conn.execute(
                    "SELECT asset_id FROM selection ORDER BY asset_id").fetchall(), 1):
                conn.execute("INSERT INTO frames (asset_id, path, seq, warped_at)"
                             " VALUES (?, ?, ?, ?)",
                             (asset, str(frames_dir / f"{asset}.jpg"), seq, STAMP))
            return seq

        monkeypatch.setattr(cli.pipeline, "stage_align", stage_align)
        return conn

    def page(self, tmp_path) -> str:
        return (tmp_path / "out" / "contact-sheet.html").read_text()

    def test_align_alone_writes_the_sheet(self, conn, tmp_path):
        run(cli.cmd_align)
        assert 'data-id="a1"' in self.page(tmp_path)
        assert "../frames/a1.jpg" in self.page(tmp_path)

    def test_a_sheet_from_an_earlier_selection_is_replaced(self, conn, tmp_path):
        run(cli.cmd_align, cli.cmd_review)
        conn.execute("DELETE FROM selection WHERE asset_id = 'a1'")

        run(cli.cmd_align)

        assert 'data-id="a1"' not in self.page(tmp_path)
        assert 'data-id="a2"' in self.page(tmp_path)

    def test_it_is_seeded_with_the_rejects_file(self, conn, tmp_path):
        """A page seeded from before the rejections would download a file
        without them -- and the download replaces the whole file."""
        (tmp_path / "out").mkdir()
        (tmp_path / "out" / "rejects.json").write_text(json.dumps({"rejected": ["gone"]}))

        run(cli.cmd_align)

        assert '"rejected-seed">["gone"]</script>' in self.page(tmp_path)

    def test_a_config_without_an_out_directory_still_aligns(self, conn, tmp_path,
                                                            monkeypatch):
        """`align` never needed `paths.out` before it started writing the sheet."""
        cfg = config.Config(raw={"paths": {"frames": "frames"}}, root=tmp_path)
        monkeypatch.setattr(cli, "_open", lambda _args: (cfg, conn))

        run(cli.cmd_align)

        assert not (tmp_path / "out").exists()


class TestAnnotatedCopiesFollowTheVideo:
    """Footer copies are named after their frame, so one left by an earlier
    encode belongs to a photo the video no longer has. Nothing reads it, and
    with frames named by asset they would otherwise pile up."""

    @pytest.fixture()
    def conn(self, tmp_path, monkeypatch):
        pytest.importorskip("PIL.Image", reason="needs Pillow")
        from PIL import Image

        conn = db.connect(tmp_path / "t.sqlite")
        frames = tmp_path / "frames"
        frames.mkdir()
        for i in range(1, 3):
            frame = frames / f"a{i}.jpg"
            Image.new("RGB", (200, 250), "white").save(frame)
            conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                         " VALUES (?, ?, ?)", (f"a{i}", f"2026-0{i}-01", STAMP))
            conn.execute("INSERT INTO frames (asset_id, path, seq, warped_at)"
                         " VALUES (?, ?, ?, ?)", (f"a{i}", str(frame), i, STAMP))
            conn.execute("INSERT INTO selection (asset_id, bucket, rank, alternate,"
                         " selected_at) VALUES (?, ?, 0, 0, ?)",
                         (f"a{i}", f"2026-0{i}", STAMP))
        monkeypatch.setattr(pipeline, "encode", lambda frames, out, **kw: out)
        return conn

    @pytest.mark.parametrize("transition,folder,suffix", [
        ("none", "annotated", ".jpg"),
        ("crossfade", "footers", ".png"),
    ])
    def test_a_copy_of_a_photo_no_longer_in_the_video_is_removed(
            self, conn, tmp_path, transition, folder, suffix):
        stale = tmp_path / "frames" / folder / f"gone{suffix}"
        stale.parent.mkdir()
        stale.write_bytes(b"x")

        pipeline.stage_encode(conn, tmp_path / "out",
                              {"annotate": {"enabled": True}, "transition": transition},
                              lambda _: None)

        assert not stale.exists()
        assert sorted(p.name for p in stale.parent.iterdir()) == [
            f"a1{suffix}", f"a2{suffix}"]
