"""The watermark's failure modes are all silent, so they are tested deliberately."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import pytest

from grow_up import db, pipeline

# Copied verbatim from the Immich OpenAPI spec (v3.1.0), the `date-time` pattern
# applied to updatedAfter. A timestamp this rejects is one the API rejects.
IMMICH_DATETIME = re.compile(
    r"^(?:(?:\d\d[2468][048]|\d\d[13579][26]|\d\d0[48]|[02468][048]00|[13579][26]00)-02-29"
    r"|\d{4}-(?:(?:0[13578]|1[02])-(?:0[1-9]|[12]\d|3[01])|(?:0[469]|11)-(?:0[1-9]|[12]\d|30)"
    r"|(?:02)-(?:0[1-9]|1\d|2[0-8])))"
    r"T(?:(?:[01]\d|2[0-3]):[0-5]\d(?::[0-5]\d(?:\.\d+)?)?(?:Z|([+-](?:[01]\d|2[0-3]):[0-5]\d)))$"
)


@pytest.fixture()
def conn(tmp_path):
    return db.connect(tmp_path / "test.sqlite")


def test_iso_z_matches_the_api_pattern():
    for moment in (
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 2, 29, 23, 59, 59, tzinfo=timezone.utc),  # leap day
        datetime(2026, 12, 31, 0, 0, 0, tzinfo=timezone.utc),
    ):
        assert IMMICH_DATETIME.match(db.iso_z(moment)), db.iso_z(moment)


def test_iso_z_normalises_other_zones_to_utc():
    berlin = timezone(timedelta(hours=2))
    assert db.iso_z(datetime(2026, 6, 1, 12, 0, tzinfo=berlin)) == "2026-06-01T10:00:00.000Z"


def test_iso_z_refuses_naive_datetimes():
    # A naive timestamp would be silently interpreted as UTC and shift the
    # watermark by the local offset.
    with pytest.raises(ValueError):
        db.iso_z(datetime(2026, 1, 1))


def test_watermark_subtracts_the_skew_margin():
    started = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert db.watermark_for(started) == "2026-05-01T11:59:00.000Z"


def test_watermark_uses_run_start_not_completion():
    """An asset uploaded mid-run must be re-seen, never skipped."""
    started = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    finished = started + timedelta(minutes=30)
    stored = db.watermark_for(started)
    assert stored < db.iso_z(finished)
    # An upload at 12:15 falls after the stored watermark, so next run re-covers it.
    assert stored < db.iso_z(started + timedelta(minutes=15))


def test_first_run_is_full(conn):
    assert pipeline.resolve_watermark(conn, "person-1", None, False).is_full


def test_bare_run_resumes_from_the_stored_watermark(conn):
    started = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    run_id = db.start_run(conn, "person-1", started, None, "full: first run")
    db.commit_watermark(conn, run_id, "person-1", started, 100, 100)

    resolved = pipeline.resolve_watermark(conn, "person-1", None, False)
    assert resolved.value == "2026-05-01T11:59:00.000Z"
    assert resolved.source == "stored"


def test_flags_override_the_stored_watermark(conn):
    started = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    run_id = db.start_run(conn, "person-1", started, None, "full: first run")
    db.commit_watermark(conn, run_id, "person-1", started, 100, 100)

    assert pipeline.resolve_watermark(conn, "person-1", None, True).is_full
    assert pipeline.resolve_watermark(conn, "person-1", "2020-01-01T00:00:00Z", False).value == (
        "2020-01-01T00:00:00Z"
    )


def test_watermark_is_scoped_per_person(conn):
    started = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    run_id = db.start_run(conn, "daughter", started, None, "full: first run")
    db.commit_watermark(conn, run_id, "daughter", started, 10, 10)

    assert pipeline.resolve_watermark(conn, "daughter", None, False).value is not None
    assert pipeline.resolve_watermark(conn, "sibling", None, False).is_full


def test_failed_run_does_not_advance_the_watermark(conn):
    """A crash mid-index must leave the window uncovered, not skipped."""
    first = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    run_id = db.start_run(conn, "person-1", first, None, "full: first run")
    db.commit_watermark(conn, run_id, "person-1", first, 100, 100)
    before = db.get_sync_state(conn, "person-1").watermark

    later = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    failed_id = db.start_run(conn, "person-1", later, before, "stored")
    db.fail_run(conn, failed_id)

    assert db.get_sync_state(conn, "person-1").watermark == before
    assert pipeline.resolve_watermark(conn, "person-1", None, False).value == before


class TestDriftDetection:
    """Tagging her in an *old* photo need not bump that asset's updatedAt."""

    def test_no_drift_when_new_tags_are_on_new_photos(self):
        # Three new photos indexed, count up by exactly three.
        assert not pipeline.detect_drift(stored_count=100, current_count=103, newly_indexed=3)

    def test_drift_when_old_photos_gain_tags(self):
        # Count up by three, but the incremental window saw nothing new.
        assert pipeline.detect_drift(stored_count=100, current_count=103, newly_indexed=0)

    def test_no_drift_when_nothing_changed(self):
        assert not pipeline.detect_drift(stored_count=100, current_count=100, newly_indexed=0)

    def test_no_drift_when_assets_were_removed(self):
        assert not pipeline.detect_drift(stored_count=100, current_count=95, newly_indexed=0)

    def test_constant_offset_from_videos_cancels_out(self):
        # The person count includes videos this image-only pipeline skips.
        # Comparing deltas rather than absolute totals keeps that offset harmless.
        assert not pipeline.detect_drift(stored_count=140, current_count=142, newly_indexed=2)

    def test_unknown_counts_do_not_force_a_reindex(self):
        assert not pipeline.detect_drift(None, 103, 0)
        assert not pipeline.detect_drift(100, None, 0)
