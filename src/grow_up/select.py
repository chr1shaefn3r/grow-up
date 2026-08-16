"""Frame selection: hard filters, scoring, and temporal bucketing."""

from __future__ import annotations

import calendar
import sqlite3
from datetime import date, datetime

# The day-of-month correction lives in one place. Re-deriving it here would give
# the footer and the bucketing two different opinions about when someone born on
# the 31st turns a month older, and the video would disagree with its own caption.
from .annotate import months_between
from .db import birth_date, iso_z, now_utc

BIRTHDAY_MONTHS = "birthday-months"

CADENCES = ("day", "week", "month", BIRTHDAY_MONTHS, "all")


class BirthDateRequired(RuntimeError):
    """`birthday-months` was asked for and Immich has no birth date.

    A RuntimeError because `cli.app` catches that and prints it without a
    traceback: this is a thing for the user to go and fix in Immich, not a bug
    for them to report.
    """


def parse_when(value: str | None) -> datetime | None:
    """Parse Immich's localDateTime, which may or may not carry a zone."""
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(text[:26], fmt)
            except ValueError:
                continue
    return None


def require_birth_date(conn: sqlite3.Connection) -> date:
    """The subject's birth date, or a message saying how to get one.

    Written once and raised from both `select_frames` and the early check in
    `run`, so the two cannot drift into giving different advice.
    """
    born = _as_date(birth_date(conn))
    if born is None:
        raise BirthDateRequired(
            f'cadence "{BIRTHDAY_MONTHS}" needs the subject\'s birth date, and '
            "Immich has none for this person.\n"
            "  Set it under the person in Immich, then re-run `grow-up index` "
            "to store it.\n"
            "  Or pick a cadence that does not need one: "
            f"{', '.join(c for c in CADENCES if c != BIRTHDAY_MONTHS)}."
        )
    return born


def _as_date(value: str | None) -> date | None:
    """Immich stores the birth date as `YYYY-MM-DD`, but tolerate a timestamp."""
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def birthday_month_start(birth: date, index: int) -> date:
    """The first date that falls in the `index`-th month of someone's life.

    Exactly the inverse of `months_between`, and it has to be: the bucket a photo
    lands in is decided there, and this only names it. A label computed by some
    other rule would drift from the grouping it claims to describe, which shows
    up as two buckets sharing a name -- one week of the timelapse silently
    swallowing another.

    The awkward case is a birthday on a day the target month does not have. Born
    on 31 January, no February date is on or past the birthday, so the month does
    not begin until March does. `months_between` already says so by treating all
    of February as still belonging to the previous month; this follows it rather
    than inventing a clamp of its own.
    """
    total = birth.year * 12 + (birth.month - 1) + index
    year, month = divmod(total, 12)
    if birth.day <= calendar.monthrange(year, month + 1)[1]:
        return date(year, month + 1, birth.day)
    year, month = divmod(total + 1, 12)
    return date(year, month + 1, 1)


def bucket_key(when: datetime | date, cadence: str, birth: date | None = None) -> str:
    """Temporal bucket label.

    Bucketing is what makes the video read as growing up rather than as a diary
    of when the camera happened to come out: without it a photo-heavy holiday
    dominates the timeline and quiet months flash past in a second.

    `birth` is required by `birthday-months` and ignored by everything else.
    """
    if cadence == "day":
        return when.strftime("%Y-%m-%d")
    if cadence == "week":
        year, week, _ = when.isocalendar()
        return f"{year:04d}-W{week:02d}"
    if cadence == "month":
        return when.strftime("%Y-%m")
    if cadence == BIRTHDAY_MONTHS:
        if birth is None:
            raise BirthDateRequired(
                f'cadence "{BIRTHDAY_MONTHS}" needs a birth date to bucket against'
            )
        # Labelled by the date the month opens, so every label carries the
        # birthday's own day and a glance at the contact sheet says whether the
        # alignment is doing what was asked.
        as_date = when.date() if isinstance(when, datetime) else when
        return birthday_month_start(birth, months_between(birth, as_date)).isoformat()
    if cadence == "all":
        return when.strftime("%Y-%m-%dT%H:%M:%S")
    raise ValueError(f"unknown cadence {cadence!r}; expected one of {CADENCES}")


MANUAL = "manual"


def apply_filters(conn: sqlite3.Connection, limits: dict, weights: dict,
                  manual: frozenset[str] | set[str] = frozenset()) -> tuple[int, int]:
    """Re-evaluate hard filters and scores from stored metrics.

    Runs entirely off the manifest -- no image decoding, no inference -- so
    threshold tuning is a sub-second loop rather than a re-analysis.

    `manual` is what the contact sheet dropped by hand. Applying it here rather
    than at encode time is what lets the runner-up take the bucket: `select_frames`
    picks the best photo that still has no reject reason, so removing one simply
    hands the week to the next best instead of deleting the week.

    A hard reason wins over `manual`, so a blurry photo you also rejected still
    counts as blurry. That keeps the tuner's numbers meaning what they say, and
    leaves `manual` counting the useful quantity: photos you dropped that the
    filters would have kept.
    """
    from .metrics import FaceMetrics, composite_score, hard_reject

    rows = conn.execute("SELECT * FROM metrics").fetchall()
    stamp = iso_z(now_utc())
    kept = 0
    updates = []
    for row in rows:
        data = {k: row[k] for k in row.keys()
                if k in FaceMetrics.__dataclass_fields__}
        m = FaceMetrics(**data)
        m.reject_reason = hard_reject(m, limits)
        if m.reject_reason is None and row["asset_id"] in manual:
            m.reject_reason = MANUAL
        m.score = None if m.reject_reason else composite_score(m, limits, weights)
        kept += int(m.reject_reason is None)
        updates.append((m.reject_reason, m.score, stamp, row["asset_id"]))

    conn.executemany(
        "UPDATE metrics SET reject_reason = ?, score = ?, filtered_at = ?"
        " WHERE asset_id = ?", updates
    )
    return kept, len(rows)


def select_frames(conn: sqlite3.Connection, cadence: str, per_bucket: int,
                  alternates: int = 0) -> int:
    """Pick the top-scoring frames per time bucket into the `selection` table.

    Returns how many will be *encoded*. Beyond those, `alternates` runner-ups per
    bucket are recorded with `alternate = 1`: they get warped so the contact
    sheet can show what would take over if you reject the pick, and they are
    excluded from the video. Judging a replacement means seeing it aligned, and
    the alternative is a re-run per rejection.
    """
    if cadence not in CADENCES:
        raise ValueError(f"unknown cadence {cadence!r}; expected one of {CADENCES}")
    # Resolved before the read and well before the DELETE below: a run that
    # cannot bucket must leave the previous selection intact rather than empty
    # the table and then fail.
    born = require_birth_date(conn) if cadence == BIRTHDAY_MONTHS else None

    rows = conn.execute(
        "SELECT m.asset_id, m.score, a.local_datetime"
        "  FROM metrics m JOIN assets a ON a.id = m.asset_id"
        " WHERE m.reject_reason IS NULL AND m.score IS NOT NULL"
        " ORDER BY m.score DESC"
    ).fetchall()

    depth = per_bucket + max(0, alternates)
    buckets: dict[str, int] = {}
    chosen: list[tuple[str, str, int, int, str]] = []
    stamp = iso_z(now_utc())
    for row in rows:
        when = parse_when(row["local_datetime"])
        if when is None:
            continue
        key = bucket_key(when, cadence, born)
        rank = buckets.get(key, 0)
        if rank >= depth:
            continue
        buckets[key] = rank + 1
        chosen.append((row["asset_id"], key, rank, int(rank >= per_bucket), stamp))

    with conn:
        conn.execute("BEGIN")
        conn.execute("DELETE FROM selection")
        conn.executemany(
            "INSERT INTO selection (asset_id, bucket, rank, alternate, selected_at)"
            " VALUES (?, ?, ?, ?, ?)",
            chosen,
        )
    return sum(1 for row in chosen if not row[3])


def selected_in_order(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Selected frames in chronological order -- the order they appear in the video."""
    return conn.execute(
        "SELECT s.asset_id, s.bucket, s.rank, s.alternate, a.local_datetime, d.path,"
        "       m.left_eye_x, m.left_eye_y, m.right_eye_x, m.right_eye_y, m.score,"
        "       m.span_w, m.span_up, m.span_down"
        "  FROM selection s"
        "  JOIN assets a ON a.id = s.asset_id"
        "  JOIN metrics m ON m.asset_id = s.asset_id"
        "  JOIN downloads d ON d.asset_id = s.asset_id"
        " ORDER BY a.local_datetime ASC"
    ).fetchall()


def reject_summary(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    rows = conn.execute(
        "SELECT coalesce(reject_reason, 'accepted') AS reason, count(*) AS n"
        "  FROM metrics GROUP BY reason ORDER BY n DESC"
    ).fetchall()
    return [(r["reason"], r["n"]) for r in rows]


def filters_applied(conn: sqlite3.Connection) -> bool:
    """Whether `apply_filters` has run against the stored metrics.

    Between `analyze` and `select` every surviving face still has
    `reject_reason IS NULL`, which the summary would render as `accepted` --
    reporting photos that were never evaluated as having passed.

    Recorded explicitly rather than inferred from a non-NULL score: rejected
    rows carry no score, so a library where every photo fails the filters --
    entirely reachable while tuning thresholds tight -- would have looked
    unevaluated.
    """
    return bool(conn.execute(
        "SELECT 1 FROM metrics WHERE filtered_at IS NOT NULL LIMIT 1").fetchone())


def format_reject_summary(conn: sqlite3.Connection, indent: str = "  ",
                          label: str = "filter outcome") -> list[str]:
    """Render the filter outcome.

    Shared by the stages and by `status` so the two cannot drift; they differ
    only in indent and in the label, which `status` uses to say that these
    verdicts are from the last `select` rather than from whatever the config
    currently says.
    """
    heading = f"{indent}{label}"
    if not conn.execute("SELECT 1 FROM metrics LIMIT 1").fetchone():
        return [f"{heading}: nothing analyzed yet"]
    if not filters_applied(conn):
        return [f"{heading}: not yet evaluated — run `grow-up select`"]

    rows = reject_summary(conn)
    total = sum(count for _, count in rows)
    lines = [f"{heading}:"]
    for reason, count in rows:
        share = f"{count / total:>6.1%}" if total else ""
        lines.append(f"{indent}  {reason:<26} {count:>6} {share}")
    return lines
