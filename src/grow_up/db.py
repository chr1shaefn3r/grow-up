"""SQLite manifest.

Every stage writes here, which is what makes the pipeline resumable and makes
threshold retuning cheap: `select` re-reads stored metrics instead of re-running
the ML.

This module also owns the sync watermark (see `SyncState`), including the
commit-on-success transaction that keeps a crashed index from advancing it.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id                 TEXT PRIMARY KEY,
    local_datetime     TEXT,
    file_created_at    TEXT,
    updated_at         TEXT,
    width              INTEGER,
    height             INTEGER,
    checksum           TEXT,
    original_file_name TEXT,
    source             TEXT,               -- which configured account it came from;
                                           -- only that account's key can download it
    indexed_at         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS assets_local_datetime ON assets (local_datetime);

CREATE TABLE IF NOT EXISTS faces (
    asset_id     TEXT PRIMARY KEY REFERENCES assets (id) ON DELETE CASCADE,
    status       TEXT NOT NULL,          -- ok | no_face | error
    x1           INTEGER,
    y1           INTEGER,
    x2           INTEGER,
    y2           INTEGER,
    image_width  INTEGER,
    image_height INTEGER,
    source_type  TEXT,
    n_candidates INTEGER NOT NULL DEFAULT 0,
    fetched_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS downloads (
    asset_id   TEXT PRIMARY KEY REFERENCES assets (id) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    bytes      INTEGER,
    source     TEXT NOT NULL,            -- original | preview
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS metrics (
    asset_id       TEXT PRIMARY KEY REFERENCES assets (id) ON DELETE CASCADE,
    detected       INTEGER NOT NULL,
    yaw            REAL,
    pitch          REAL,
    roll           REAL,
    gaze_x         REAL,
    gaze_y         REAL,
    blink_l        REAL,
    blink_r        REAL,
    oob_frac       REAL,
    bbox_clipped   INTEGER,
    interocular_px REAL,
    left_eye_x     REAL,
    left_eye_y     REAL,
    right_eye_x    REAL,
    right_eye_y    REAL,
    sharpness      REAL,
    exposure_lo    REAL,
    exposure_hi    REAL,
    span_w         REAL,               -- face extents in interocular units,
    span_up        REAL,               -- so `align` can predict clipping
    span_down      REAL,
    reject_reason  TEXT,                 -- NULL when the frame passes hard filters
    score          REAL,
    analyzed_at    TEXT NOT NULL,
    filtered_at    TEXT                  -- set by select; distinguishes "passed"
                                         -- from "not yet evaluated"
);

CREATE TABLE IF NOT EXISTS selection (
    asset_id    TEXT PRIMARY KEY REFERENCES assets (id) ON DELETE CASCADE,
    bucket      TEXT NOT NULL,
    rank        INTEGER NOT NULL,
    alternate   INTEGER NOT NULL DEFAULT 0,  -- a runner-up: warped so the contact
                                             -- sheet can show it, never encoded
    selected_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS selection_bucket ON selection (bucket);

CREATE TABLE IF NOT EXISTS frames (
    asset_id  TEXT PRIMARY KEY REFERENCES assets (id) ON DELETE CASCADE,
    path      TEXT NOT NULL,
    seq       INTEGER NOT NULL,
    tx        REAL,
    ty        REAL,
    angle     REAL,
    scale     REAL,
    warped_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS people (
    person_id  TEXT PRIMARY KEY,
    source     TEXT,                       -- the account this record came from
    name       TEXT,
    birth_date TEXT,                       -- YYYY-MM-DD, or NULL when Immich has none
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS renders (
    path         TEXT PRIMARY KEY,      -- the video, absolute, as it was written
    fingerprint  TEXT NOT NULL,         -- encode.fingerprint of the inputs behind it
    rendered_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sync_state (
    person_id     TEXT PRIMARY KEY,
    watermark     TEXT NOT NULL,         -- UTC ISO-8601, passed to the API as updatedAfter
    person_assets INTEGER,               -- GET /people/{id}/statistics at watermark time
    last_run_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id         TEXT NOT NULL,
    started_at        TEXT NOT NULL,
    finished_at       TEXT,
    watermark_used    TEXT,
    watermark_source  TEXT,
    assets_indexed    INTEGER,
    status            TEXT NOT NULL      -- running | ok | failed
);
"""

# Absorbs clock skew between this machine and the Immich server. Re-seeing a
# handful of assets is free; missing one is permanent.
SKEW_MARGIN = timedelta(seconds=60)


def iso_z(dt: datetime) -> str:
    """Format as UTC ISO-8601 with an explicit Z.

    The Immich `date-time` schema pattern requires a zone designator, so a naive
    `datetime.isoformat()` is rejected by the API.
    """
    if dt.tzinfo is None:
        raise ValueError("refusing to format a naive datetime; pass an aware one")
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def watermark_for(started_at: datetime, margin: timedelta = SKEW_MARGIN) -> str:
    """Watermark to store for a run that *started* at `started_at`.

    Deliberately the start of the run rather than its end: an asset uploaded
    while the run is in flight would otherwise fall into the gap between the
    query and the write, and be skipped forever. Storing the start time means it
    is merely re-seen next run, which costs nothing because every downstream
    stage is idempotent and keyed on asset id.
    """
    return iso_z(started_at - margin)


@dataclass(frozen=True)
class SyncState:
    person_id: str
    watermark: str
    person_assets: int | None
    last_run_at: str


# Columns added after the first release. CREATE TABLE IF NOT EXISTS leaves an
# existing table alone, so new ones have to be added explicitly or a database
# built by an earlier version keeps the old shape.
ADDED_COLUMNS = {
    "metrics": (("span_w", "REAL"), ("span_up", "REAL"), ("span_down", "REAL"),
                ("filtered_at", "TEXT")),
    "assets": (("source", "TEXT"),),
    "selection": (("alternate", "INTEGER NOT NULL DEFAULT 0"),),
}


def migrate(conn: sqlite3.Connection) -> list[str]:
    """Add any columns missing from an older database. Returns what was added."""
    added = []
    for table, columns in ADDED_COLUMNS.items():
        present = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        for name, kind in columns:
            if name not in present:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")
                added.append(f"{table}.{name}")
    return added


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    migrate(conn)
    return conn


def get_sync_state(conn: sqlite3.Connection, person_id: str) -> SyncState | None:
    row = conn.execute(
        "SELECT person_id, watermark, person_assets, last_run_at"
        " FROM sync_state WHERE person_id = ?",
        (person_id,),
    ).fetchone()
    if row is None:
        return None
    return SyncState(**dict(row))


def start_run(conn: sqlite3.Connection, person_id: str, started_at: datetime,
              watermark_used: str | None, watermark_source: str) -> int:
    cur = conn.execute(
        "INSERT INTO runs (person_id, started_at, watermark_used, watermark_source, status)"
        " VALUES (?, ?, ?, ?, 'running')",
        (person_id, iso_z(started_at), watermark_used, watermark_source),
    )
    return int(cur.lastrowid)


def commit_watermark(conn: sqlite3.Connection, run_id: int, person_id: str,
                     started_at: datetime, person_assets: int | None,
                     assets_indexed: int) -> str:
    """Advance the watermark and close out the run, atomically.

    Called only after the index stage completes. A crash before this point
    leaves the previous watermark in place, so the un-indexed window is covered
    again on the next run.
    """
    watermark = watermark_for(started_at)
    with conn:
        conn.execute("BEGIN")
        conn.execute(
            "INSERT INTO sync_state (person_id, watermark, person_assets, last_run_at)"
            " VALUES (?, ?, ?, ?)"
            " ON CONFLICT (person_id) DO UPDATE SET"
            "   watermark = excluded.watermark,"
            "   person_assets = excluded.person_assets,"
            "   last_run_at = excluded.last_run_at",
            (person_id, watermark, person_assets, iso_z(now_utc())),
        )
        conn.execute(
            "UPDATE runs SET finished_at = ?, assets_indexed = ?, status = 'ok' WHERE id = ?",
            (iso_z(now_utc()), assets_indexed, run_id),
        )
    return watermark


def fail_run(conn: sqlite3.Connection, run_id: int) -> None:
    conn.execute(
        "UPDATE runs SET finished_at = ?, status = 'failed' WHERE id = ?",
        (iso_z(now_utc()), run_id),
    )


def upsert_asset(conn: sqlite3.Connection, asset: dict) -> None:
    conn.execute(
        "INSERT INTO assets (id, local_datetime, file_created_at, updated_at, width, height,"
        "                    checksum, original_file_name, source, indexed_at)"
        " VALUES (:id, :local_datetime, :file_created_at, :updated_at, :width, :height,"
        "         :checksum, :original_file_name, :source, :indexed_at)"
        " ON CONFLICT (id) DO UPDATE SET"
        "   local_datetime = excluded.local_datetime,"
        "   updated_at = excluded.updated_at,"
        "   width = excluded.width,"
        "   height = excluded.height,"
        "   checksum = excluded.checksum,"
        "   source = excluded.source,"
        "   indexed_at = excluded.indexed_at",
        {"source": None, **asset, "indexed_at": iso_z(now_utc())},
    )


def adopt_unsourced(conn: sqlite3.Connection, source: str) -> int:
    """Claim rows indexed before assets.source existed. Returns how many.

    A database written by 1.0.0 has one account's assets in it and no record of
    which, because there was only ever one. Stamping them with the first
    configured source is therefore correct by construction, and it means every
    query downstream is a plain `source = ?` rather than carrying a coalesce for
    the rest of the project's life.
    """
    cur = conn.execute(
        "UPDATE assets SET source = ? WHERE source IS NULL", (source,))
    return int(cur.rowcount or 0)


def upsert_person(conn: sqlite3.Connection, person_id: str, source: str | None,
                  name: str, birth_date: str | None) -> None:
    conn.execute(
        "INSERT INTO people (person_id, source, name, birth_date, fetched_at)"
        " VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT (person_id) DO UPDATE SET"
        "   source = excluded.source,"
        "   name = excluded.name,"
        "   birth_date = excluded.birth_date,"
        "   fetched_at = excluded.fetched_at",
        (person_id, source, name, birth_date, iso_z(now_utc())),
    )


def birth_date(conn: sqlite3.Connection) -> str | None:
    """The subject's birth date, or None if no account has one.

    Several sources mean several person records for the same human, so the first
    non-null wins -- they describe one person, and typically only one of the two
    accounts ever filled the field in.
    """
    row = conn.execute(
        "SELECT birth_date FROM people WHERE birth_date IS NOT NULL"
        " ORDER BY person_id LIMIT 1").fetchone()
    return row["birth_date"] if row else None


def _render_key(path) -> str:
    """Absolute, so the record survives being run from another directory."""
    return str(Path(path).resolve())


def record_render(conn: sqlite3.Connection, path, fingerprint: str) -> None:
    """Remember what produced this video. Called only after ffmpeg succeeded.

    A killed ffmpeg leaves a partial file and no row, so the next run re-renders
    rather than trusting a truncated video.
    """
    conn.execute(
        "INSERT INTO renders (path, fingerprint, rendered_at) VALUES (?, ?, ?)"
        " ON CONFLICT (path) DO UPDATE SET"
        "   fingerprint = excluded.fingerprint,"
        "   rendered_at = excluded.rendered_at",
        (_render_key(path), fingerprint, iso_z(now_utc())),
    )
    conn.commit()


def render_fingerprint(conn: sqlite3.Connection, path) -> str | None:
    """What produced the video at `path` last time, if this manifest knows."""
    row = conn.execute(
        "SELECT fingerprint FROM renders WHERE path = ?", (_render_key(path),)
    ).fetchone()
    return row["fingerprint"] if row else None


def count_assets(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT count(*) FROM assets").fetchone()[0])
