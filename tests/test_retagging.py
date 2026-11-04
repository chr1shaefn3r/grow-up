"""Tagging changes in Immich reach the video, for photos the manifest already holds.

A stored face verdict was treated as settled forever. Found on a real library by
a fork (starryalley/grow-up): an earlier index had pulled tens of thousands of
photos in while pointed at a different person, and `faces` had written a verdict
against each. Tagging the subject in 2,551 of them then changed nothing:
`index` reported no new assets because it knew the ids, `faces` skipped them
because a row existed, and `fetch` said "nothing new to download" run after run.
Re-assigning a face moves neither the id nor `updatedAt`, so the person search
returning an asset is the only evidence there is.

The same blind spot ran the other way. Untag the subject from a photo, delete
it, or correct a run made against the wrong person, and its box, its
measurements and its frame stayed in the video for good.
"""

from __future__ import annotations

import asyncio
import sqlite3

import pytest

from grow_up import db, pipeline

PERSON = "person-astrid"
OTHER = "person-hailey"
STAMP = "2026-01-01T00:00:00.000Z"


@pytest.fixture()
def conn(tmp_path):
    conn = db.connect(tmp_path / "t.sqlite")
    for i in range(1, 6):
        add_asset(conn, f"a{i}")
    return conn


def add_asset(conn, asset_id, source="default"):
    db.upsert_asset(conn, {
        "id": asset_id, "local_datetime": STAMP, "file_created_at": None,
        "updated_at": None, "width": None, "height": None, "checksum": None,
        "original_file_name": None, "source": source})


def store_face(conn, asset_id, status="ok", person_id=PERSON, box=(10, 10, 30, 30)):
    x1, y1, x2, y2 = box if box else (None, None, None, None)
    conn.execute(
        "INSERT OR REPLACE INTO faces (asset_id, status, x1, y1, x2, y2, image_width,"
        " image_height, source_type, person_id, stale, n_candidates, fetched_at)"
        " VALUES (?, ?, ?, ?, ?, ?, 100, 100, 'machine-learning', ?, 0, 1, ?)",
        (asset_id, status, x1, y1, x2, y2, person_id, STAMP))


def store_metrics(conn, asset_id):
    conn.execute("INSERT OR REPLACE INTO metrics (asset_id, detected, analyzed_at)"
                 " VALUES (?, 1, ?)", (asset_id, STAMP))


def has_metrics(conn, asset_id) -> bool:
    return conn.execute("SELECT 1 FROM metrics WHERE asset_id = ?",
                        (asset_id,)).fetchone() is not None


def face(conn, asset_id) -> sqlite3.Row:
    return conn.execute("SELECT * FROM faces WHERE asset_id = ?", (asset_id,)).fetchone()


class FakeSearch:
    """Just enough client for `stage_index`: the person search, as Immich pages it."""

    def __init__(self, ids):
        self.ids = list(ids)

    async def search_assets(self, person_id, updated_after, page_size):
        for asset_id in self.ids:
            yield {"id": asset_id, "localDateTime": STAMP}


class FakeFaces:
    """Just enough client for `stage_faces`: who is where, per asset."""

    def __init__(self, boxes: dict[str, tuple | None]):
        self.boxes = boxes
        self.asked: list[str] = []

    async def faces_for_asset(self, asset_id):
        self.asked.append(asset_id)
        box = self.boxes.get(asset_id)
        if box is None:
            return []
        x1, y1, x2, y2 = box
        return [{"person": {"id": PERSON}, "boundingBoxX1": x1, "boundingBoxY1": y1,
                 "boundingBoxX2": x2, "boundingBoxY2": y2, "imageWidth": 100,
                 "imageHeight": 100, "sourceType": "machine-learning"}]


def index(conn, ids, full=True, source="default"):
    watermark = pipeline.Watermark(None if full else "2026-01-01T00:00:00Z", "test")
    lines: list[str] = []
    seen_new = asyncio.run(pipeline.stage_index(FakeSearch(ids), conn, PERSON, watermark,
                                                1000, lines.append, source))
    return seen_new, lines


def faces(conn, boxes):
    client = FakeFaces(boxes)
    lines: list[str] = []
    asyncio.run(pipeline.stage_faces(client, conn, PERSON, lines.append))
    return client.asked, lines


class TestTheSearchOverrulesAStoredVerdict:

    def test_a_no_face_verdict_loses(self, conn):
        store_face(conn, "a1", status="no_face", box=None)
        assert db.mark_face_stale(conn, "a1", PERSON) == 1

    def test_a_failed_lookup_is_tried_again(self, conn):
        """A request that errored once was never retried at all."""
        store_face(conn, "a1", status="error", box=None)
        assert db.mark_face_stale(conn, "a1", PERSON) == 1

    def test_this_persons_own_box_is_left_alone(self, conn):
        store_face(conn, "a1")
        assert db.mark_face_stale(conn, "a1", PERSON) == 0

    def test_a_box_looked_up_for_somebody_else_loses(self, conn):
        """What a run against the wrong person leaves behind."""
        store_face(conn, "a1", person_id=OTHER)
        assert db.mark_face_stale(conn, "a1", PERSON) == 1

    def test_a_box_nobody_can_vouch_for_loses(self, conn):
        store_face(conn, "a1", person_id=None)
        assert db.mark_face_stale(conn, "a1", PERSON) == 1

    def test_requeueing_twice_counts_once(self, conn):
        store_face(conn, "a1", status="no_face", box=None)
        db.mark_face_stale(conn, "a1", PERSON)
        assert db.mark_face_stale(conn, "a1", PERSON) == 0


class TestIndexingRequeuesWhatItCannotReportAsNew:
    """The whole failure: nothing is new, and yet something has changed."""

    def test_a_photo_tagged_after_its_face_was_looked_up_is_requeued(self, conn):
        store_face(conn, "a1", status="no_face", box=None)
        (seen, new), lines = index(conn, ["a1"], full=False)
        assert (seen, new) == (1, 0), "the id was already known, so nothing is new"
        assert face(conn, "a1")["stale"] == 1
        assert any("queued to be looked up again" in line for line in lines)

    def test_it_becomes_work_for_faces_again(self, conn):
        store_face(conn, "a1", status="no_face", box=None)
        store_face(conn, "a2", status="no_face", box=None)
        for asset_id in ("a3", "a4", "a5"):
            store_face(conn, asset_id)
        assert pipeline.pending_counts(conn)["faces"] == 0

        index(conn, ["a1", "a3", "a4", "a5"])
        # a1 is tagged now; a2 is not in the search, so its no_face stands.
        assert pipeline.pending_counts(conn)["faces"] == 1
        assert face(conn, "a2")["stale"] == 0

    def test_a_settled_photo_is_not_looked_up_on_every_index(self, conn):
        for asset_id in ("a1", "a2", "a3", "a4", "a5"):
            store_face(conn, asset_id)
        index(conn, ["a1", "a2", "a3", "a4", "a5"])
        assert pipeline.pending_counts(conn)["faces"] == 0


class TestAFullListingSettlesWhatItLeavesOut:
    """It enumerates the person completely, so its omissions are evidence too."""

    def test_a_photo_no_longer_tagged_leaves_the_video(self, conn):
        store_face(conn, "a1")
        store_face(conn, "a2")
        store_metrics(conn, "a2")
        _, lines = index(conn, ["a1"])
        assert face(conn, "a2")["status"] == "no_face"
        assert not has_metrics(conn, "a2"), "select reads metrics, so they must go too"
        assert any("no longer tagged" in line for line in lines)

    def test_a_box_from_a_run_against_somebody_else_leaves_too(self, conn):
        """Three frames of the wrong child reached a finished timelapse this way
        in the fork's library. Within one account, only the subject's photos
        belong in the video, whoever a stale box claims to be."""
        store_face(conn, "a1")
        store_face(conn, "a2", person_id=OTHER)
        store_metrics(conn, "a2")
        index(conn, ["a1"])
        assert face(conn, "a2")["status"] == "no_face"
        assert not has_metrics(conn, "a2")

    def test_the_withdrawn_photo_is_not_looked_up_again(self, conn):
        """The listing already answered the question `/faces` would be asked."""
        store_face(conn, "a1")
        store_face(conn, "a2")
        index(conn, ["a1"])
        assert face(conn, "a2")["stale"] == 0
        assert pipeline.pending_counts(conn)["faces"] == 3, "only a3-a5, never indexed"

    def test_another_accounts_photos_are_not_its_business(self, conn):
        add_asset(conn, "b1", source="partner")
        store_face(conn, "a1")
        store_face(conn, "b1")
        store_metrics(conn, "b1")
        index(conn, ["a1"])
        assert face(conn, "b1")["status"] == "ok" and has_metrics(conn, "b1")

    def test_an_incremental_window_withdraws_nothing(self, conn):
        """It leaves out nearly the whole library by design."""
        store_face(conn, "a1")
        store_face(conn, "a2")
        index(conn, ["a1"], full=False)
        assert face(conn, "a2")["status"] == "ok"

    def test_an_empty_listing_withdraws_nothing(self, conn):
        """Far likelier a wrong person id than a library-wide untagging."""
        store_face(conn, "a1")
        index(conn, [])
        assert face(conn, "a1")["status"] == "ok"


class TestFacesActsOnTheRequeue:

    def test_it_looks_up_a_requeued_photo_and_settles_it(self, conn):
        store_face(conn, "a1", status="no_face", box=None)
        index(conn, ["a1"], full=False)
        asked, _ = faces(conn, {"a1": (12, 12, 40, 40)})
        assert "a1" in asked
        row = face(conn, "a1")
        assert (row["status"], row["stale"], row["person_id"]) == ("ok", 0, PERSON)

    def test_it_leaves_a_settled_verdict_alone(self, conn):
        store_face(conn, "a1", status="no_face", box=None)
        asked, _ = faces(conn, {})
        assert "a1" not in asked

    def test_a_box_that_moved_takes_its_measurements_with_it(self, conn):
        """`analyze` skips anything already measured, so numbers taken around
        the wrong face would outlive the correction."""
        store_face(conn, "a1", person_id=OTHER)
        store_metrics(conn, "a1")
        index(conn, ["a1"], full=False)
        _, lines = faces(conn, {"a1": (50, 50, 80, 80)})
        assert not has_metrics(conn, "a1")
        assert any("different face box" in line for line in lines)

    def test_a_box_that_came_back_the_same_keeps_them(self, conn):
        """Re-tagging must not cost a re-analysis of the library."""
        store_face(conn, "a1", person_id=None)
        store_metrics(conn, "a1")
        index(conn, ["a1"], full=False)
        faces(conn, {"a1": (10, 10, 30, 30)})
        assert has_metrics(conn, "a1")
        assert face(conn, "a1")["person_id"] == PERSON

    def test_a_face_that_is_gone_takes_them_too(self, conn):
        store_face(conn, "a1", person_id=OTHER)
        store_metrics(conn, "a1")
        index(conn, ["a1"], full=False)
        faces(conn, {})
        assert face(conn, "a1")["status"] == "no_face"
        assert not has_metrics(conn, "a1")

    def test_a_requeued_box_is_not_analyzed_until_it_is_looked_up(self, conn):
        store_face(conn, "a1", person_id=OTHER)
        conn.execute("INSERT INTO downloads (asset_id, path, source, fetched_at)"
                     " VALUES ('a1', 'x.jpg', 'original', ?)", (STAMP,))
        index(conn, ["a1"], full=False)
        assert pipeline.pending_counts(conn)["analyze"] == 0


class TestTheHiddenPhotoReachesTheVideo:
    """End to end over index and faces: the 2,551 photos of the report."""

    def test_a_photo_once_judged_faceless_becomes_one_to_download(self, conn):
        store_face(conn, "a1", status="no_face", box=None)
        assert pipeline.pending_counts(conn)["fetch"] == 0
        index(conn, ["a1"])
        faces(conn, {"a1": (12, 12, 40, 40)})
        assert pipeline.pending_counts(conn)["fetch"] == 1


class TestAttributingRowsFromAnOlderDatabase:
    """Face rows written before `faces.person_id` existed name nobody. Where an
    account has only ever known one person they are that person's; anywhere
    else, a guess could keep somebody else's face in the video."""

    def older(self, tmp_path, people: list[tuple[str, str]], assets: list[tuple[str, str]]):
        path = tmp_path / "old.sqlite"
        schema = db.SCHEMA.replace(
            "    person_id    TEXT,                   -- whose face this row is a verdict about;\n"
            "                                         -- a box means nothing without it\n"
            "    stale        INTEGER NOT NULL DEFAULT 0,  -- the person search has since\n"
            "                                              -- contradicted this verdict\n", "")
        assert "stale" not in schema.split("CREATE TABLE IF NOT EXISTS faces")[1].split(");")[0]
        old = sqlite3.connect(path)
        old.executescript(schema)
        for person_id, source in people:
            old.execute("INSERT INTO people (person_id, source, name, fetched_at)"
                        " VALUES (?, ?, 'x', ?)", (person_id, source, STAMP))
        for asset_id, source in assets:
            old.execute("INSERT INTO assets (id, local_datetime, indexed_at, source)"
                        " VALUES (?, ?, ?, ?)", (asset_id, STAMP, STAMP, source))
            old.execute("INSERT INTO faces (asset_id, status, x1, y1, x2, y2, image_width,"
                        " image_height, n_candidates, fetched_at)"
                        " VALUES (?, 'ok', 1, 2, 3, 4, 100, 100, 1, ?)", (asset_id, STAMP))
        old.commit()
        old.close()
        return db.connect(path)

    def test_the_boxes_survive_the_upgrade(self, tmp_path):
        conn = self.older(tmp_path, [(PERSON, "default")], [("a1", "default")])
        row = face(conn, "a1")
        assert (row["x1"], row["y2"], row["stale"]) == (1, 4, 0)

    def test_one_person_ever_claims_every_row(self, tmp_path):
        conn = self.older(tmp_path, [(PERSON, "default")], [("a1", "default"), ("a2", None)])
        assert {face(conn, a)["person_id"] for a in ("a1", "a2")} == {PERSON}

    def test_each_account_with_one_person_claims_its_own(self, tmp_path):
        conn = self.older(tmp_path, [(PERSON, "default"), (OTHER, "partner")],
                          [("a1", "default"), ("b1", "partner")])
        assert face(conn, "a1")["person_id"] == PERSON
        assert face(conn, "b1")["person_id"] == OTHER

    def test_an_account_that_knew_two_people_claims_nothing(self, tmp_path):
        """Either could have written the box; the next index re-checks it."""
        conn = self.older(tmp_path, [(PERSON, "default"), (OTHER, "default")],
                          [("a1", "default")])
        assert face(conn, "a1")["person_id"] is None


def test_status_counts_what_is_waiting_to_be_looked_up_again(tmp_path, capsys):
    from grow_up import cli

    conn = db.connect(tmp_path / "g.sqlite")
    add_asset(conn, "a1")
    store_face(conn, "a1", status="no_face", box=None)
    db.mark_face_stale(conn, "a1", PERSON)
    conn.close()
    (tmp_path / "config.toml").write_text('[paths]\ndb = "g.sqlite"\nout = "out"\n')
    assert cli.app(["--config", str(tmp_path / "config.toml"), "status"]) == 0
    assert "(re-check)" in capsys.readouterr().out
