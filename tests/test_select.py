from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta

import pytest

from grow_up import db, select
from grow_up.annotate import months_between

LIMITS = {
    "max_yaw": 20.0, "max_pitch": 18.0, "max_roll": 25.0, "max_gaze": 0.35,
    "max_blink": 0.45, "max_oob_frac": 0.005, "allow_bbox_clipped": False,
    "min_interocular_px": 60.0, "min_sharpness": 12.0,
    "min_exposure_lo": 8.0, "max_exposure_hi": 250.0,
}
WEIGHTS = {"w_pose": 1.0, "w_gaze": 1.0, "w_eyes_open": 1.0,
           "w_sharpness": 1.0, "w_size": 0.5}


@pytest.fixture()
def conn(tmp_path):
    return db.connect(tmp_path / "test.sqlite")


def add(conn, asset_id: str, when: str, *, sharpness: float = 100.0, yaw: float = 2.0,
        gaze: float = 0.0, interocular: float = 150.0) -> None:
    conn.execute(
        "INSERT INTO assets (id, local_datetime, indexed_at) VALUES (?, ?, '2026-01-01T00:00:00.000Z')",
        (asset_id, when),
    )
    conn.execute(
        "INSERT INTO metrics (asset_id, detected, yaw, pitch, roll, gaze_x, gaze_y,"
        " blink_l, blink_r, oob_frac, bbox_clipped, interocular_px, sharpness,"
        " exposure_lo, exposure_hi, analyzed_at)"
        " VALUES (?, 1, ?, 1.0, 1.0, ?, 0.0, 0.05, 0.05, 0.0, 0, ?, ?, 40.0, 200.0,"
        " '2026-01-01T00:00:00.000Z')",
        (asset_id, yaw, gaze, interocular, sharpness),
    )


class TestBucketKey:
    @pytest.mark.parametrize("cadence,expected", [
        ("day", "2026-03-14"),
        ("week", "2026-W11"),
        ("month", "2026-03"),
    ])
    def test_labels(self, cadence, expected):
        assert select.bucket_key(datetime(2026, 3, 14, 9, 30), cadence) == expected

    def test_iso_week_spans_the_new_year(self):
        # 2025-12-29 is ISO week 1 of 2026; a naive %Y-%W would call it week 52 of 2025
        # and split one week across two buckets.
        assert select.bucket_key(datetime(2025, 12, 29), "week") == "2026-W01"

    def test_all_gives_every_photo_its_own_bucket(self):
        a = select.bucket_key(datetime(2026, 3, 14, 9, 30, 0), "all")
        b = select.bucket_key(datetime(2026, 3, 14, 9, 30, 1), "all")
        assert a != b

    def test_unknown_cadence_is_rejected(self):
        with pytest.raises(ValueError):
            select.bucket_key(datetime(2026, 3, 14), "fortnight")


class TestParseWhen:
    @pytest.mark.parametrize("value", [
        "2026-03-14T09:30:00.000Z",
        "2026-03-14T09:30:00+02:00",
        "2026-03-14T09:30:00",
        "2026-03-14",
    ])
    def test_accepts_the_shapes_immich_emits(self, value):
        parsed = select.parse_when(value)
        assert parsed is not None and parsed.year == 2026 and parsed.month == 3

    def test_none_and_garbage_are_survivable(self):
        assert select.parse_when(None) is None
        assert select.parse_when("not a date") is None


class TestApplyFilters:
    def test_recomputes_from_stored_metrics_without_reanalysis(self, conn):
        """Threshold tuning must be a SQL pass, not an ML re-run."""
        add(conn, "a", "2026-01-05T10:00:00.000Z", yaw=30.0)

        kept, total = select.apply_filters(conn, LIMITS, WEIGHTS)
        assert (kept, total) == (0, 1)
        assert conn.execute("SELECT reject_reason FROM metrics").fetchone()[0] == "head_turned"

        loosened = {**LIMITS, "max_yaw": 40.0}
        kept, _ = select.apply_filters(conn, loosened, WEIGHTS)
        assert kept == 1
        row = conn.execute("SELECT reject_reason, score FROM metrics").fetchone()
        assert row["reject_reason"] is None and row["score"] > 0


class TestSelectFrames:
    def test_keeps_the_best_frame_per_bucket(self, conn):
        add(conn, "sharp", "2026-03-10T10:00:00.000Z", sharpness=400.0)
        add(conn, "soft", "2026-03-11T10:00:00.000Z", sharpness=20.0)
        select.apply_filters(conn, LIMITS, WEIGHTS)

        assert select.select_frames(conn, "week", 1) == 1
        assert conn.execute("SELECT asset_id FROM selection").fetchone()[0] == "sharp"

    def test_per_bucket_caps_the_count(self, conn):
        for i in range(5):
            add(conn, f"a{i}", f"2026-03-1{i}T10:00:00.000Z", sharpness=100.0 + i)
        select.apply_filters(conn, LIMITS, WEIGHTS)

        assert select.select_frames(conn, "week", 2) == 2

    def test_evens_out_photo_heavy_periods(self, conn):
        """The reason bucketing exists: a busy holiday must not dominate the video."""
        for i in range(30):  # a burst in one week
            add(conn, f"holiday{i}", f"2026-03-09T{i % 24:02d}:00:00.000Z")
        for week, day in enumerate(("2026-04-06", "2026-04-13", "2026-04-20"), start=1):
            add(conn, f"quiet{week}", f"{day}T10:00:00.000Z")
        select.apply_filters(conn, LIMITS, WEIGHTS)

        select.select_frames(conn, "week", 1)
        rows = conn.execute("SELECT asset_id FROM selection").fetchall()
        chosen = {r["asset_id"] for r in rows}
        assert len(chosen) == 4, "one per week, not 30 from the busy one"
        assert {"quiet1", "quiet2", "quiet3"} <= chosen

    def test_rejected_frames_are_never_selected(self, conn):
        add(conn, "bad", "2026-03-10T10:00:00.000Z", yaw=45.0)
        select.apply_filters(conn, LIMITS, WEIGHTS)
        assert select.select_frames(conn, "week", 1) == 0

    def test_selection_spans_the_whole_corpus_not_just_recent_assets(self, conn):
        """Regression guard for the watermark leaking past the index stage.

        If --since ever reached selection, the timelapse would contain only the
        newest frames -- a failure that produces a plausible-looking video and so
        would not be obvious from the output alone.
        """
        for year in range(2019, 2027):
            add(conn, f"y{year}", f"{year}-06-10T10:00:00.000Z")
        select.apply_filters(conn, LIMITS, WEIGHTS)
        select.select_frames(conn, "month", 1)

        years = {r["asset_id"] for r in conn.execute("SELECT asset_id FROM selection")}
        assert years == {f"y{y}" for y in range(2019, 2027)}

    def test_reselect_replaces_rather_than_accumulates(self, conn):
        for i in range(4):
            add(conn, f"a{i}", f"2026-0{i + 1}-10T10:00:00.000Z")
        select.apply_filters(conn, LIMITS, WEIGHTS)

        select.select_frames(conn, "month", 1)
        first = conn.execute("SELECT count(*) FROM selection").fetchone()[0]
        select.select_frames(conn, "month", 1)
        assert conn.execute("SELECT count(*) FROM selection").fetchone()[0] == first

    def test_ordering_is_chronological_not_by_score(self, conn):
        add(conn, "later_but_better", "2026-05-10T10:00:00.000Z", sharpness=500.0)
        add(conn, "earlier", "2026-01-10T10:00:00.000Z", sharpness=50.0)
        conn.execute("INSERT INTO downloads (asset_id, path, source, fetched_at)"
                     " VALUES ('later_but_better', '/x.jpg', 'original', 'now')")
        conn.execute("INSERT INTO downloads (asset_id, path, source, fetched_at)"
                     " VALUES ('earlier', '/y.jpg', 'original', 'now')")
        select.apply_filters(conn, LIMITS, WEIGHTS)
        select.select_frames(conn, "month", 1)

        order = [r["asset_id"] for r in select.selected_in_order(conn)]
        assert order == ["earlier", "later_but_better"]


def test_reject_summary_counts_by_reason(conn):
    add(conn, "ok", "2026-03-10T10:00:00.000Z")
    add(conn, "turned", "2026-03-11T10:00:00.000Z", yaw=45.0)
    add(conn, "blurry", "2026-03-12T10:00:00.000Z", sharpness=1.0)
    select.apply_filters(conn, LIMITS, WEIGHTS)

    summary = dict(select.reject_summary(conn))
    assert summary == {"accepted": 1, "head_turned": 1, "blurry": 1}


class TestAManualRejectPromotesTheRunnerUp:
    """Rejecting the week's winner must hand the week to the next best photo.

    Before this, `rejects.json` was only applied at encode time, filtering the
    already-selected frame list -- so rejecting a photo deleted its bucket
    outright and the runner-up sitting in the manifest was never considered.
    """

    def week(self, conn) -> None:
        # One ISO week, three candidates, sharpness deciding the order.
        add(conn, "best", "2026-03-02T10:00:00", sharpness=400.0)
        add(conn, "second", "2026-03-03T10:00:00", sharpness=200.0)
        add(conn, "third", "2026-03-04T10:00:00", sharpness=100.0)

    def chosen(self, conn, manual=frozenset(), per_bucket=1) -> list[str]:
        select.apply_filters(conn, LIMITS, WEIGHTS, manual)
        select.select_frames(conn, "week", per_bucket)
        return [r["asset_id"] for r in conn.execute(
            "SELECT asset_id FROM selection ORDER BY rank")]

    def test_without_rejects_the_best_wins(self, conn):
        self.week(conn)
        assert self.chosen(conn) == ["best"]

    def test_rejecting_the_best_promotes_the_second(self, conn):
        self.week(conn)
        assert self.chosen(conn, {"best"}) == ["second"]

    def test_rejecting_two_promotes_the_third(self, conn):
        self.week(conn)
        assert self.chosen(conn, {"best", "second"}) == ["third"]

    def test_rejecting_everything_leaves_the_bucket_empty(self, conn):
        """Not a fallback to a rejected photo -- you said no to all of them."""
        self.week(conn)
        assert self.chosen(conn, {"best", "second", "third"}) == []

    def test_per_bucket_two_takes_the_next_two_down(self, conn):
        self.week(conn)
        assert self.chosen(conn, {"best"}, per_bucket=2) == ["second", "third"]

    def test_other_weeks_are_untouched(self, conn):
        self.week(conn)
        add(conn, "elsewhere", "2026-04-06T10:00:00", sharpness=50.0)
        assert set(self.chosen(conn, {"best"})) == {"second", "elsewhere"}

    def test_un_rejecting_brings_the_photo_back(self, conn):
        """Removing an id from the file has to be enough; nothing is sticky."""
        self.week(conn)
        assert self.chosen(conn, {"best"}) == ["second"]
        assert self.chosen(conn, frozenset()) == ["best"]


class TestHowAManualRejectIsReported:
    def test_it_shows_as_its_own_reason(self, conn):
        add(conn, "a", "2026-03-02T10:00:00")
        select.apply_filters(conn, LIMITS, WEIGHTS, {"a"})
        assert dict(select.reject_summary(conn)) == {"manual": 1}

    def test_a_hard_reason_wins_over_it(self, conn):
        """Keeps the tuner's numbers honest: a blurry photo still reads blurry.

        So the `manual` count means the useful thing -- photos dropped by hand
        that the filters would have kept.
        """
        add(conn, "blurry", "2026-03-02T10:00:00", sharpness=1.0)
        select.apply_filters(conn, LIMITS, WEIGHTS, {"blurry"})
        assert dict(select.reject_summary(conn)) == {"blurry": 1}

    def test_a_rejected_photo_is_not_counted_as_accepted(self, conn):
        add(conn, "a", "2026-03-02T10:00:00")
        add(conn, "b", "2026-03-03T10:00:00")
        kept, scored = select.apply_filters(conn, LIMITS, WEIGHTS, {"a"})
        assert (kept, scored) == (1, 2)

    def test_it_carries_no_score(self, conn):
        add(conn, "a", "2026-03-02T10:00:00")
        select.apply_filters(conn, LIMITS, WEIGHTS, {"a"})
        assert conn.execute("SELECT score FROM metrics").fetchone()["score"] is None


class TestAlternatesAreKeptButNotEncoded:
    """Runner-ups are warped so the contact sheet can show what a rejection
    would promote. They are candidates, not frames: one reaching the video would
    duplicate a week without ever looking wrong."""

    def bucket(self, conn) -> None:
        add(conn, "best", "2026-03-02T10:00:00", sharpness=400.0)
        add(conn, "second", "2026-03-03T10:00:00", sharpness=200.0)
        add(conn, "third", "2026-03-04T10:00:00", sharpness=100.0)
        add(conn, "fourth", "2026-03-05T10:00:00", sharpness=50.0)
        select.apply_filters(conn, LIMITS, WEIGHTS)

    def rows(self, conn):
        return conn.execute(
            "SELECT asset_id, rank, alternate FROM selection ORDER BY rank").fetchall()

    def test_none_are_kept_by_default(self, conn):
        """A config that never heard of alternates behaves as it always did."""
        self.bucket(conn)
        select.select_frames(conn, "week", 1)
        assert [r["asset_id"] for r in self.rows(conn)] == ["best"]

    def test_they_are_stored_in_score_order(self, conn):
        self.bucket(conn)
        select.select_frames(conn, "week", 1, alternates=2)
        assert [(r["asset_id"], r["alternate"]) for r in self.rows(conn)] == [
            ("best", 0), ("second", 1), ("third", 1)]

    def test_the_return_value_counts_only_what_gets_encoded(self, conn):
        """It feeds the trial's projection, which must not count runner-ups."""
        self.bucket(conn)
        assert select.select_frames(conn, "week", 1, alternates=2) == 1

    def test_per_bucket_decides_which_are_spare(self, conn):
        self.bucket(conn)
        select.select_frames(conn, "week", 2, alternates=1)
        assert [(r["asset_id"], r["alternate"]) for r in self.rows(conn)] == [
            ("best", 0), ("second", 0), ("third", 1)]

    def test_asking_for_more_than_exist_is_not_an_error(self, conn):
        self.bucket(conn)
        assert select.select_frames(conn, "week", 1, alternates=99) == 1
        assert len(self.rows(conn)) == 4

    def test_a_rejected_pick_promotes_an_alternate_into_the_video(self, conn):
        """The two features meeting: the runner-up was already warped, so this
        costs no second pass."""
        self.bucket(conn)
        select.apply_filters(conn, LIMITS, WEIGHTS, {"best"})
        select.select_frames(conn, "week", 1, alternates=2)

        rows = self.rows(conn)
        assert [(r["asset_id"], r["alternate"]) for r in rows] == [
            ("second", 0), ("third", 1), ("fourth", 1)]


class TestADatabaseFromTheLastReleaseUpgrades:
    """1.2.0 wrote a selection table with no `alternate` column.

    It arrives through db.ADDED_COLUMNS, and every existing row must default to
    0 -- a stored selection is a pick, not a runner-up. Get that backwards and
    the first run after upgrading encodes nothing.
    """

    RELEASED_SCHEMA = """
    CREATE TABLE assets (
        id TEXT PRIMARY KEY, local_datetime TEXT, indexed_at TEXT NOT NULL
    );
    CREATE TABLE selection (
        asset_id    TEXT PRIMARY KEY,
        bucket      TEXT NOT NULL,
        rank        INTEGER NOT NULL,
        selected_at TEXT NOT NULL
    );
    """

    def released_database(self, path):
        raw = sqlite3.connect(path)
        raw.executescript(self.RELEASED_SCHEMA)
        raw.execute("INSERT INTO assets (id, indexed_at) VALUES ('a', '2026-01-01')")
        raw.execute("INSERT INTO selection (asset_id, bucket, rank, selected_at)"
                    " VALUES ('a', '2026-W10', 0, '2026-01-01')")
        raw.commit()
        raw.close()

    def test_the_column_arrives(self, tmp_path):
        path = tmp_path / "released.sqlite"
        self.released_database(path)
        columns = {r["name"] for r in db.connect(path).execute("PRAGMA table_info(selection)")}
        assert "alternate" in columns

    def test_an_existing_selection_is_a_pick_not_a_runner_up(self, tmp_path):
        path = tmp_path / "released.sqlite"
        self.released_database(path)
        rows = db.connect(path).execute("SELECT asset_id, alternate FROM selection").fetchall()
        assert [(r["asset_id"], r["alternate"]) for r in rows] == [("a", 0)]

    def test_such_a_row_still_reaches_the_video(self, tmp_path):
        """The join stage_encode uses, run against the upgraded database."""
        path = tmp_path / "released.sqlite"
        self.released_database(path)
        conn = db.connect(path)
        conn.execute("INSERT INTO frames (asset_id, path, seq, warped_at)"
                     " VALUES ('a', '/tmp/a.jpg', 0, '2026-01-01')")

        kept = conn.execute(
            "SELECT f.asset_id FROM frames f JOIN selection s ON s.asset_id = f.asset_id"
            " WHERE s.alternate = 0").fetchall()
        assert [r["asset_id"] for r in kept] == ["a"]


class TestBirthdayMonths:
    """Buckets that turn over on the birthday rather than on the 1st.

    A calendar month splits a life at an arbitrary point: the photo that opens
    "2023-08" may be a day either side of turning three and a half. Aligning to
    the birthday makes each frame one month of the subject's own age.
    """

    BIRTH = date(2020, 3, 14)

    def bucket(self, when, birth=None):
        return select.bucket_key(when, select.BIRTHDAY_MONTHS, birth or self.BIRTH)

    def test_the_month_turns_over_on_the_birthday_not_the_first(self):
        assert self.bucket(date(2023, 8, 13)) == "2023-07-14"
        assert self.bucket(date(2023, 8, 14)) == "2023-08-14"

    def test_a_calendar_month_boundary_does_not_split_a_bucket(self):
        """The whole point: these two straddle 1 September and belong together."""
        assert self.bucket(date(2023, 8, 31)) == self.bucket(date(2023, 9, 1))

    def test_every_label_carries_the_birthday(self):
        labels = [self.bucket(date(2023, m, 20)) for m in range(1, 13)]
        assert all(label.endswith("-14") for label in labels), labels

    def test_a_datetime_buckets_the_same_as_its_date(self):
        assert self.bucket(datetime(2023, 8, 14, 23, 59)) == self.bucket(date(2023, 8, 14))

    @pytest.mark.parametrize("birth", [date(2020, 3, 14), date(2020, 1, 31),
                                       date(2020, 2, 29), date(2019, 12, 30)])
    def test_the_label_is_the_exact_inverse_of_the_grouping(self, birth):
        """`months_between` decides the bucket and this only names it.

        A label computed by a different rule drifts from the grouping it claims
        to describe, and two buckets sharing a name silently swallow one another.
        """
        for index in range(-24, 400):
            start = select.birthday_month_start(birth, index)
            assert months_between(birth, start) == index
            assert months_between(birth, start - timedelta(days=1)) == index - 1

    def test_a_birthday_on_the_31st_waits_for_a_month_that_has_one(self):
        """February contains no date on or past the 31st, so the month opens
        when March does -- which is what `months_between` already says."""
        starts = [select.birthday_month_start(date(2020, 1, 31), i) for i in range(4)]
        assert [d.isoformat() for d in starts] == [
            "2020-01-31", "2020-03-01", "2020-03-31", "2020-05-01"]

    def test_a_photo_older_than_the_birth_date_still_buckets(self):
        """A wrong birth date, or a scan from before it, must not crash the run.

        Age runs negative and the months keep counting backwards, so the photo
        lands in a bucket of its own rather than being dropped or lumped in with
        the first month of life.
        """
        assert self.bucket(date(2020, 1, 5)) == "2019-12-14"
        assert self.bucket(date(2020, 1, 5)) != self.bucket(date(2020, 3, 20))

    def test_it_is_offered_as_a_cadence(self):
        assert select.BIRTHDAY_MONTHS in select.CADENCES

    def test_a_missing_birth_date_is_refused_rather_than_guessed(self):
        with pytest.raises(select.BirthDateRequired):
            select.bucket_key(date(2023, 8, 14), select.BIRTHDAY_MONTHS)


class TestTheCadenceThatNeedsABirthDate:
    """Without one the cadence cannot mean anything, so the run stops and says
    where to go and fix it."""

    def conn_with(self, tmp_path, birth):
        from grow_up import db

        conn = db.connect(tmp_path / "t.sqlite")
        db.upsert_person(conn, "p1", "me", "Kid", birth)
        stamp = "2026-01-01T00:00:00.000Z"
        for i, when in enumerate(["2023-08-13", "2023-08-14", "2023-09-20"], start=1):
            conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                         " VALUES (?, ?, ?)", (f"a{i}", when, stamp))
            conn.execute("INSERT INTO metrics (asset_id, detected, reject_reason,"
                         " score, analyzed_at, filtered_at)"
                         " VALUES (?, 1, NULL, ?, ?, ?)",
                         (f"a{i}", 1.0 - i / 10, stamp, stamp))
        return conn

    def test_it_buckets_by_age_when_the_birth_date_is_there(self, tmp_path):
        conn = self.conn_with(tmp_path, "2020-03-14")
        select.select_frames(conn, select.BIRTHDAY_MONTHS, 1, 0)

        buckets = [r[0] for r in conn.execute(
            "SELECT bucket FROM selection ORDER BY bucket")]
        assert buckets == ["2023-07-14", "2023-08-14", "2023-09-14"], (
            "13 and 14 August are different months of this child's life")

    def test_a_missing_birth_date_names_the_cadence_and_the_fix(self, tmp_path):
        conn = self.conn_with(tmp_path, None)
        with pytest.raises(select.BirthDateRequired) as caught:
            select.select_frames(conn, select.BIRTHDAY_MONTHS, 1, 0)

        message = str(caught.value)
        assert select.BIRTHDAY_MONTHS in message
        assert "grow-up index" in message, "must say how to get the birth date stored"
        assert "week" in message, "must offer a cadence that works instead"

    def test_the_error_is_catchable_by_the_cli(self):
        """`cli.app` prints RuntimeError without a traceback; a bare Exception
        would reach the user as a crash report for a config mistake."""
        assert issubclass(select.BirthDateRequired, RuntimeError)

    def test_the_previous_selection_survives_the_refusal(self, tmp_path):
        """The check runs before the DELETE. Emptying the table and then failing
        would turn a fixable mistake into a re-run of select, align and encode."""
        conn = self.conn_with(tmp_path, None)
        select.select_frames(conn, "month", 1, 0)
        before = conn.execute("SELECT count(*) FROM selection").fetchone()[0]
        assert before > 0

        with pytest.raises(select.BirthDateRequired):
            select.select_frames(conn, select.BIRTHDAY_MONTHS, 1, 0)
        assert conn.execute("SELECT count(*) FROM selection").fetchone()[0] == before

    def test_other_cadences_do_not_need_one(self, tmp_path):
        conn = self.conn_with(tmp_path, None)
        assert select.select_frames(conn, "month", 1, 0) > 0
