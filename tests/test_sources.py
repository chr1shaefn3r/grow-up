"""Two Immich accounts, one subject.

Immich scopes face recognition per account, so a partner's photos are invisible
to this account's person id no matter how the search is phrased. The fix is to
ask again as them -- which means credentials, a person id and now every asset
carry an account with them.

The first test here is the load-bearing one: 1.0.0 shipped a config shape, and a
file written against it has to keep working untouched. Everything else in this
file is new behaviour; that one is a promise.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from grow_up import config, db, pipeline

LEGACY_ENV = {"IMMICH_URL": "https://immich.example.com", "IMMICH_API_KEY": "legacy-key"}


def cfg(raw: dict) -> config.Config:
    return config.Config(raw=raw, root=Path("."))


class TestTheReleasedConfigStillWorks:
    """1.0.0's shape, verbatim, with nothing added."""

    RELEASED = {"immich": {"person_name": "Their Name", "person_id": "person-1"}}

    def test_it_yields_exactly_one_source(self):
        assert len(config.sources(cfg(self.RELEASED))) == 1

    def test_that_source_carries_the_legacy_person_keys(self):
        source, = config.sources(cfg(self.RELEASED))
        assert (source.person_name, source.person_id) == ("Their Name", "person-1")

    def test_it_reads_the_legacy_environment_variables(self, monkeypatch):
        for name, value in LEGACY_ENV.items():
            monkeypatch.setenv(name, value)
        source, = config.sources(cfg(self.RELEASED))
        assert source.credentials() == config.credentials()

    def test_an_empty_immich_section_still_resolves(self):
        """No person configured is a runtime error, not a parse error."""
        source, = config.sources(cfg({"immich": {}}))
        assert source.person_id == "" and source.name == config.LEGACY_SOURCE_NAME

    def test_the_legacy_name_is_fixed(self):
        """It is written into assets.source; changing it orphans every row."""
        assert config.LEGACY_SOURCE_NAME == "default"


class TestDeclaredSources:
    TWO = {"immich": {"sources": [
        {"name": "me", "person_id": "person-a"},
        {"name": "partner", "person_id": "person-b", "key_env": "IMMICH_API_KEY_PARTNER"},
    ]}}

    def test_they_are_returned_in_order(self):
        assert [s.name for s in config.sources(cfg(self.TWO))] == ["me", "partner"]

    def test_each_keeps_its_own_person(self):
        me, partner = config.sources(cfg(self.TWO))
        assert (me.person_id, partner.person_id) == ("person-a", "person-b")

    def test_unset_credentials_fall_back_to_the_defaults(self):
        me, partner = config.sources(cfg(self.TWO))
        assert me.key_env == "IMMICH_API_KEY"
        assert partner.key_env == "IMMICH_API_KEY_PARTNER"
        assert me.url_env == partner.url_env == "IMMICH_URL"

    def test_each_reads_its_own_key(self, monkeypatch):
        for name, value in LEGACY_ENV.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setenv("IMMICH_API_KEY_PARTNER", "partner-key")
        me, partner = config.sources(cfg(self.TWO))
        assert me.credentials().api_key == "legacy-key"
        assert partner.credentials().api_key == "partner-key"

    def test_a_second_server_is_allowed(self, monkeypatch):
        monkeypatch.setenv("OTHER_URL", "https://other.example.com")
        monkeypatch.setenv("OTHER_KEY", "k")
        source, = config.sources(cfg({"immich": {"sources": [
            {"name": "other", "url_env": "OTHER_URL", "key_env": "OTHER_KEY"}]}}))
        assert source.credentials().url == "https://other.example.com/api"

    def test_a_missing_variable_names_the_source(self, monkeypatch):
        monkeypatch.delenv("IMMICH_API_KEY_PARTNER", raising=False)
        _, partner = config.sources(cfg(self.TWO))
        with pytest.raises(RuntimeError, match="partner.*IMMICH_API_KEY_PARTNER"):
            partner.credentials()


class TestConfigMistakesFailLoudly:
    def test_a_nameless_source_is_rejected(self):
        with pytest.raises(RuntimeError, match="no name"):
            config.sources(cfg({"immich": {"sources": [{"person_id": "p"}]}}))

    def test_a_repeated_name_is_rejected(self):
        """Two accounts merged under one identity would look like drift forever."""
        with pytest.raises(RuntimeError, match="same name"):
            config.sources(cfg({"immich": {"sources": [
                {"name": "me", "key_env": "A"}, {"name": "me", "key_env": "B"}]}}))

    def test_reusing_one_key_for_two_sources_is_rejected(self):
        """The copy-paste mistake, and it looks exactly like the bug being fixed.

        Both blocks would index the same account, so the partner's photos are
        still missing -- with nothing on screen to say why.
        """
        with pytest.raises(RuntimeError, match="same credentials"):
            config.sources(cfg({"immich": {"sources": [
                {"name": "me", "person_id": "a"},
                {"name": "partner", "person_id": "b"}]}}))


class TestTheDatabaseUpgrade:
    # assets, exactly as 1.0.0 created it: no source column.
    RELEASED_SCHEMA = """
    CREATE TABLE assets (
        id TEXT PRIMARY KEY, local_datetime TEXT, file_created_at TEXT,
        updated_at TEXT, width INTEGER, height INTEGER, checksum TEXT,
        original_file_name TEXT, indexed_at TEXT NOT NULL
    );
    """

    def released_database(self, path: Path) -> None:
        raw = sqlite3.connect(path)
        raw.executescript(self.RELEASED_SCHEMA)
        raw.executemany(
            "INSERT INTO assets (id, indexed_at) VALUES (?, '2026-01-01T00:00:00.000Z')",
            [("a",), ("b",)])
        raw.commit()
        raw.close()

    def test_migrate_adds_the_column(self, tmp_path):
        path = tmp_path / "released.sqlite"
        self.released_database(path)
        conn = db.connect(path)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(assets)")}
        assert "source" in columns

    def test_existing_rows_survive_the_upgrade(self, tmp_path):
        path = tmp_path / "released.sqlite"
        self.released_database(path)
        assert db.count_assets(db.connect(path)) == 2

    def test_adopting_claims_them_for_the_first_source(self, tmp_path):
        path = tmp_path / "released.sqlite"
        self.released_database(path)
        conn = db.connect(path)

        assert db.adopt_unsourced(conn, "me") == 2
        assert [r["source"] for r in conn.execute("SELECT source FROM assets")] == ["me", "me"]

    def test_adopting_twice_claims_nothing_the_second_time(self, tmp_path):
        path = tmp_path / "released.sqlite"
        self.released_database(path)
        conn = db.connect(path)

        db.adopt_unsourced(conn, "me")
        assert db.adopt_unsourced(conn, "partner") == 0

    def test_it_never_steals_a_row_that_already_has_an_owner(self, tmp_path):
        conn = db.connect(tmp_path / "fresh.sqlite")
        db.upsert_asset(conn, {**BLANK_ASSET, "id": "x", "source": "partner"})

        db.adopt_unsourced(conn, "me")
        assert conn.execute("SELECT source FROM assets").fetchone()["source"] == "partner"


BLANK_ASSET = {
    "id": "", "local_datetime": "2026-01-01T00:00:00.000Z",
    "file_created_at": None, "updated_at": None, "width": None, "height": None,
    "checksum": None, "original_file_name": None,
}


class TestStagesStayInTheirOwnAccount:
    """The defect this guards: asking account A about account B's asset.

    Immich answers 404 for an id the key cannot see, so getting this wrong turns
    into a stage that fails on exactly the other account's half of the library.
    """

    @pytest.fixture()
    def conn(self, tmp_path):
        conn = db.connect(tmp_path / "t.sqlite")
        for asset_id, source in (("a1", "me"), ("a2", "me"), ("b1", "partner")):
            db.upsert_asset(conn, {**BLANK_ASSET, "id": asset_id, "source": source})
        return conn

    def pending_faces(self, conn, source):
        where, params = pipeline._source_clause(source)
        return [r[0] for r in conn.execute(
            "SELECT a.id FROM assets a LEFT JOIN faces f ON f.asset_id = a.id"
            " WHERE f.asset_id IS NULL" + where + " ORDER BY a.id", params)]

    def test_a_source_sees_only_its_own_assets(self, conn):
        assert self.pending_faces(conn, "me") == ["a1", "a2"]
        assert self.pending_faces(conn, "partner") == ["b1"]

    def test_no_source_means_the_whole_library(self, conn):
        assert self.pending_faces(conn, None) == ["a1", "a2", "b1"]

    def test_pending_counts_can_be_narrowed(self, conn):
        assert pipeline.pending_counts(conn, "me")["faces"] == 2
        assert pipeline.pending_counts(conn, "partner")["faces"] == 1
        assert pipeline.pending_counts(conn)["faces"] == 3

    def test_indexing_stamps_the_account(self, tmp_path):
        conn = db.connect(tmp_path / "i.sqlite")

        async def go():
            await pipeline.stage_index(
                FakeSearch(["p1", "p2"]), conn, "person-b",
                pipeline.Watermark(None, "full"), 1000, lambda _: None, "partner")

        asyncio.run(go())
        assert {r["source"] for r in conn.execute("SELECT source FROM assets")} == {"partner"}

    def test_reindexing_moves_an_asset_to_its_current_account(self, tmp_path):
        """Renaming a source re-indexes it; the rows have to follow."""
        conn = db.connect(tmp_path / "i.sqlite")
        db.upsert_asset(conn, {**BLANK_ASSET, "id": "p1", "source": "old-name"})

        async def go():
            await pipeline.stage_index(
                FakeSearch(["p1"]), conn, "person-b",
                pipeline.Watermark(None, "full"), 1000, lambda _: None, "new-name")

        asyncio.run(go())
        assert conn.execute("SELECT source FROM assets").fetchone()["source"] == "new-name"


class FakeSearch:
    """Just enough client for stage_index: it only calls search_assets."""

    def __init__(self, ids: list[str]):
        self.ids = ids

    async def search_assets(self, person_id, updated_after=None, page_size=1000):
        for asset_id in self.ids:
            yield {"id": asset_id, "localDateTime": "2026-01-01T00:00:00.000Z",
                   "fileCreatedAt": "2026-01-01T00:00:00.000Z", "updatedAt": None,
                   "checksum": None, "originalFileName": f"{asset_id}.jpg"}


class TestSplittingASample:
    """`trial -n 100` must measure a hundred photos, not a hundred per account.

    The projection multiplies a measured per-item cost by the whole workload, so
    a sample that is quietly twice the requested size makes the estimate wrong
    with nothing on screen to reveal it.
    """

    def test_the_shares_add_up_to_the_limit(self):
        assert sum(pipeline.split_limit(100, [500, 500])) == 100
        assert sum(pipeline.split_limit(100, [900, 100])) == 100
        assert sum(pipeline.split_limit(7, [10, 10, 10])) == 7

    def test_it_follows_how_much_each_has_pending(self):
        assert pipeline.split_limit(100, [900, 100]) == [90, 10]

    def test_nobody_is_asked_for_more_than_they_have(self):
        shares = pipeline.split_limit(100, [4, 500])
        assert shares[0] <= 4
        assert sum(shares) == 100

    def test_a_limit_larger_than_the_work_asks_only_for_the_work(self):
        assert pipeline.split_limit(500, [4, 6]) == [4, 6]

    def test_an_idle_library_splits_evenly_rather_than_dividing_by_zero(self):
        assert pipeline.split_limit(10, [0, 0]) == [5, 5]

    def test_one_account_gets_the_whole_limit(self):
        assert pipeline.split_limit(100, [900]) == [100]

    def test_no_accounts_is_not_an_error(self):
        assert pipeline.split_limit(100, []) == []


class TestWatermarksAreIndependentPerAccount:
    """Two accounts means two person records, and sync_state is already keyed on
    person id -- so this is an assertion that nothing needs to change, which is
    exactly the kind of claim that rots silently."""

    @pytest.fixture()
    def conn(self, tmp_path):
        return db.connect(tmp_path / "w.sqlite")

    def test_each_person_keeps_its_own_watermark(self, conn):
        started = db.now_utc()
        run_a = db.start_run(conn, "person-a", started, None, "full: first run")
        db.commit_watermark(conn, run_a, "person-a", started, 10, 10)

        assert db.get_sync_state(conn, "person-a") is not None
        assert db.get_sync_state(conn, "person-b") is None

    def test_advancing_one_leaves_the_other_alone(self, conn):
        first = db.now_utc()
        db.commit_watermark(
            conn, db.start_run(conn, "person-a", first, None, "full"), "person-a",
            first, 10, 10)
        before = db.get_sync_state(conn, "person-a").watermark

        later = first + db.SKEW_MARGIN * 100
        db.commit_watermark(
            conn, db.start_run(conn, "person-b", later, None, "full"), "person-b",
            later, 3, 3)

        assert db.get_sync_state(conn, "person-a").watermark == before
        assert db.get_sync_state(conn, "person-b").watermark != before

    def test_drift_is_measured_per_person(self, conn):
        """One account gaining photos must not make the other look drifted."""
        assert pipeline.detect_drift(stored_count=10, current_count=20, newly_indexed=2)
        assert not pipeline.detect_drift(stored_count=10, current_count=12, newly_indexed=2)
