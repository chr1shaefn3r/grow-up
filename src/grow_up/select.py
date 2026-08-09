"""Frame selection: hard filters, scoring, and temporal bucketing."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime

from .db import iso_z, now_utc

CADENCES = ("day", "week", "month", "all")


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


def bucket_key(when: datetime | date, cadence: str) -> str:
    """Temporal bucket label.

    Bucketing is what makes the video read as growing up rather than as a diary
    of when the camera happened to come out: without it a photo-heavy holiday
    dominates the timeline and quiet months flash past in a second.
    """
    if cadence == "day":
        return when.strftime("%Y-%m-%d")
    if cadence == "week":
        year, week, _ = when.isocalendar()
        return f"{year:04d}-W{week:02d}"
    if cadence == "month":
        return when.strftime("%Y-%m")
    if cadence == "all":
        return when.strftime("%Y-%m-%dT%H:%M:%S")
    raise ValueError(f"unknown cadence {cadence!r}; expected one of {CADENCES}")


def apply_filters(conn: sqlite3.Connection, limits: dict, weights: dict) -> tuple[int, int]:
    """Re-evaluate hard filters and scores from stored metrics.

    Runs entirely off the manifest -- no image decoding, no inference -- so
    threshold tuning is a sub-second loop rather than a re-analysis.
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
        m.score = None if m.reject_reason else composite_score(m, limits, weights)
        kept += int(m.reject_reason is None)
        updates.append((m.reject_reason, m.score, stamp, row["asset_id"]))

    conn.executemany(
        "UPDATE metrics SET reject_reason = ?, score = ?, filtered_at = ?"
        " WHERE asset_id = ?", updates
    )
    return kept, len(rows)


def select_frames(conn: sqlite3.Connection, cadence: str, per_bucket: int) -> int:
    """Pick the top-scoring frames per time bucket into the `selection` table."""
    if cadence not in CADENCES:
        raise ValueError(f"unknown cadence {cadence!r}; expected one of {CADENCES}")

    rows = conn.execute(
        "SELECT m.asset_id, m.score, a.local_datetime"
        "  FROM metrics m JOIN assets a ON a.id = m.asset_id"
        " WHERE m.reject_reason IS NULL AND m.score IS NOT NULL"
        " ORDER BY m.score DESC"
    ).fetchall()

    buckets: dict[str, int] = {}
    chosen: list[tuple[str, str, int, str]] = []
    stamp = iso_z(now_utc())
    for row in rows:
        when = parse_when(row["local_datetime"])
        if when is None:
            continue
        key = bucket_key(when, cadence)
        rank = buckets.get(key, 0)
        if rank >= per_bucket:
            continue
        buckets[key] = rank + 1
        chosen.append((row["asset_id"], key, rank, stamp))

    with conn:
        conn.execute("BEGIN")
        conn.execute("DELETE FROM selection")
        conn.executemany(
            "INSERT INTO selection (asset_id, bucket, rank, selected_at) VALUES (?, ?, ?, ?)",
            chosen,
        )
    return len(chosen)


def selected_in_order(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Selected frames in chronological order -- the order they appear in the video."""
    return conn.execute(
        "SELECT s.asset_id, s.bucket, a.local_datetime, d.path,"
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
