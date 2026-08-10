"""Stage orchestration.

Every stage is independently re-runnable and skips work the manifest already
records, so interrupting any of them costs only the item in flight.
"""

from __future__ import annotations

import asyncio
import math
import sqlite3
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from . import align, analyze, db, images, review, select
from .encode import encode
from .immich import ImmichClient, pick_face
from .progress import Progress

Log = Callable[[str], None]

# Past this many identical-looking failures, further lines add nothing; the
# summary at the end of the stage carries the detail.
MAX_LOGGED_ERRORS = 5


class StageFailed(RuntimeError):
    """A stage failed so comprehensively that continuing would waste the run."""


def _log_error(log: Log, errors: list[str], what: str, exc: BaseException) -> None:
    """Log the real error, not just its class name.

    A stage that prints `HTTPStatusError` 832 times says nothing about whether
    the problem is a permission, content negotiation or a wrong path. The status
    code is the entire diagnosis, so it has to survive to the terminal.
    """
    if len(errors) <= MAX_LOGGED_ERRORS:
        log(f"  ! {what}: {exc}")
        if len(errors) == MAX_LOGGED_ERRORS:
            log(f"  ! (further errors suppressed; {MAX_LOGGED_ERRORS} shown)")


def _abort_if_hopeless(stage: str, succeeded: int, errors: list[str], attempted: int) -> None:
    """Stop the run when a stage achieved nothing.

    Returning 0 and letting `run` continue means `analyze` finds no images,
    `select` picks no frames, and the failure surfaces several minutes later as
    an empty video rather than at its cause.
    """
    if not errors:
        return
    if succeeded == 0:
        raise StageFailed(
            f"every {stage} failed ({len(errors)}/{attempted}). First error:\n"
            f"    {errors[0]}"
        )
    if len(errors) > attempted * 0.1:
        # Not fatal, but a tenth of the corpus silently missing would skew the
        # timelapse without ever looking wrong.
        raise StageFailed(
            f"{len(errors)} of {attempted} {stage} attempts failed. First error:\n"
            f"    {errors[0]}\n"
            "  Re-run to retry; nothing already fetched is lost."
        )


@dataclass(frozen=True)
class Watermark:
    """The `updatedAfter` bound for the index stage, and where it came from."""

    value: str | None
    source: str

    @property
    def is_full(self) -> bool:
        return self.value is None


def resolve_watermark(conn: sqlite3.Connection, person_id: str, since: str | None,
                      full: bool) -> Watermark:
    """Decide the index window before any network call.

    A bare `run` picks up the stored watermark; the flags are escape hatches.
    """
    if full:
        return Watermark(None, "full: --full")
    if since:
        return Watermark(since, "flag: --since")
    state = db.get_sync_state(conn, person_id)
    if state is None:
        return Watermark(None, "full: first run")
    return Watermark(state.watermark, "stored")


def detect_drift(stored_count: int | None, current_count: int | None,
                 newly_indexed: int) -> bool:
    """True when the incremental window provably missed assets.

    Tagging a person in an *old* photo need not bump that asset's `updatedAt` --
    Immich writes the association to the face and person tables -- so an
    `updatedAfter` query can silently skip it. The person's asset count is
    authoritative, and comparing *deltas* rather than absolute totals makes the
    check immune to the constant offset from videos and trashed assets, which
    the count includes but this image-only pipeline does not.

    A video newly tagged with the person can trigger a false positive here. That costs
    one unnecessary full re-index, which is the right way round to be wrong.
    """
    if stored_count is None or current_count is None:
        return False
    return (current_count - stored_count) > newly_indexed


async def stage_index(client: ImmichClient, conn: sqlite3.Connection, person_id: str,
                      watermark: Watermark, page_size: int, log: Log) -> tuple[int, int]:
    """Index assets, returning (seen, newly_added)."""
    known = {row[0] for row in conn.execute("SELECT id FROM assets")}
    seen = new = 0

    async for item in client.search_assets(person_id, watermark.value, page_size):
        seen += 1
        if item["id"] not in known:
            new += 1
        exif = item.get("exifInfo") or {}
        db.upsert_asset(conn, {
            "id": item["id"],
            "local_datetime": item.get("localDateTime") or item.get("fileCreatedAt"),
            "file_created_at": item.get("fileCreatedAt"),
            "updated_at": item.get("updatedAt"),
            "width": exif.get("exifImageWidth"),
            "height": exif.get("exifImageHeight"),
            "checksum": item.get("checksum"),
            "original_file_name": item.get("originalFileName"),
        })
        if seen % 500 == 0:
            log(f"  index: {seen} assets so far…")

    log(f"  index: {seen} assets ({new} new)")
    return seen, new


def _limit_clause(limit: int | None) -> str:
    """Deterministic sample of the pending work.

    Ordering by id is both stable and effectively random with respect to date,
    resolution and file size, because Immich asset ids are random UUIDs. That
    makes a trial reproducible while still being representative of the average
    photo -- which ordering by date would not be, since cameras and file sizes
    change over the years being sampled.
    """
    return f" ORDER BY a.id LIMIT {int(limit)}" if limit else " ORDER BY a.id"


def eventual_workload(conn: sqlite3.Connection) -> dict[str, int]:
    """How many items each stage will process before the library is finished.

    Distinct from `pending_counts`, which answers "what is actionable right
    now". A projection needs the eventual total: at the start of a trial nothing
    has been downloaded yet, so the actionable count for `analyze` is zero, and
    multiplying a measured per-image cost by zero reported a full run as 0ms.

    Downstream stages are therefore sized off the *population* they will see
    rather than the rows that happen to exist. Assets whose face box has not
    been looked up yet are assumed to resolve at the same rate as those already
    checked -- the only available estimate, and exact once `faces` has run.
    """
    def scalar(sql: str) -> int:
        return int(conn.execute(sql).fetchone()[0])

    total = scalar("SELECT count(*) FROM assets")
    checked = scalar("SELECT count(*) FROM faces")
    usable = scalar("SELECT count(*) FROM faces WHERE status = 'ok'")
    downloaded = scalar("SELECT count(*) FROM downloads")
    analyzed = scalar("SELECT count(*) FROM metrics")

    usable_rate = usable / checked if checked else 1.0
    expected = round(usable + (total - checked) * usable_rate)

    return {
        "faces": max(0, total - checked),
        "fetch": max(0, expected - downloaded),
        "analyze": max(0, expected - analyzed),
    }


def pending_counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Work each stage can act on *right now*, for progress-bar denominators.

    Downstream stages are gated on upstream rows existing, so these are zero
    until the stage before has run. Use `eventual_workload` for projections.
    """
    return {
        "faces": conn.execute(
            "SELECT count(*) FROM assets a LEFT JOIN faces f ON f.asset_id = a.id"
            " WHERE f.asset_id IS NULL").fetchone()[0],
        "fetch": conn.execute(
            "SELECT count(*) FROM assets a"
            "  JOIN faces f ON f.asset_id = a.id AND f.status = 'ok'"
            "  LEFT JOIN downloads d ON d.asset_id = a.id"
            " WHERE d.asset_id IS NULL").fetchone()[0],
        "analyze": conn.execute(
            "SELECT count(*) FROM assets a"
            "  JOIN faces f ON f.asset_id = a.id AND f.status = 'ok'"
            "  JOIN downloads d ON d.asset_id = a.id"
            "  LEFT JOIN metrics m ON m.asset_id = a.id"
            " WHERE m.asset_id IS NULL").fetchone()[0],
    }


async def stage_faces(client: ImmichClient, conn: sqlite3.Connection, person_id: str,
                      log: Log, concurrency: int = 16,
                      limit: int | None = None) -> tuple[int, int]:
    """Fetch the subject's face box per asset. Immich associates it, so no recognition."""
    pending = [row[0] for row in conn.execute(
        "SELECT a.id FROM assets a LEFT JOIN faces f ON f.asset_id = a.id"
        " WHERE f.asset_id IS NULL" + _limit_clause(limit)
    )]
    if not pending:
        log("  faces: nothing new to look up")
        return 0, 0

    sem = asyncio.Semaphore(concurrency)
    ok = missing = 0
    errors: list[str] = []
    lock = asyncio.Lock()
    # `pending` is capped by --limit; the summary should still report position
    # in the whole job, so ask how much is outstanding overall.
    total_assets = conn.execute("SELECT count(*) FROM assets").fetchone()[0]
    outstanding = pending_counts(conn)["faces"]
    bar = Progress("faces", len(pending), emit=log,
                   overall=total_assets, already_done=total_assets - outstanding)

    async def one(asset_id: str) -> None:
        nonlocal ok, missing
        async with sem:
            try:
                faces = await client.faces_for_asset(asset_id)
            except Exception as exc:  # noqa: BLE001 - one bad asset must not stop the stage
                async with lock:
                    conn.execute(
                        "INSERT OR REPLACE INTO faces (asset_id, status, n_candidates, fetched_at)"
                        " VALUES (?, 'error', 0, ?)",
                        (asset_id, db.iso_z(db.now_utc())),
                    )
                    missing += 1
                    errors.append(str(exc))
                _log_error(bar.log, errors, f"faces {asset_id}", exc)
                bar.advance(failed=1)
                return

        face, n = pick_face(faces, person_id)
        async with lock:
            if face is None:
                conn.execute(
                    "INSERT OR REPLACE INTO faces (asset_id, status, n_candidates, fetched_at)"
                    " VALUES (?, 'no_face', 0, ?)",
                    (asset_id, db.iso_z(db.now_utc())),
                )
                missing += 1
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO faces (asset_id, status, x1, y1, x2, y2,"
                    " image_width, image_height, source_type, n_candidates, fetched_at)"
                    " VALUES (?, 'ok', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (asset_id, face.x1, face.y1, face.x2, face.y2, face.image_width,
                     face.image_height, face.source_type, n, db.iso_z(db.now_utc())),
                )
                ok += 1
        bar.advance()

    await asyncio.gather(*(one(a) for a in pending))
    bar.close()
    log(f"  faces: {ok} found, {missing} without a usable detection")
    _abort_if_hopeless("face lookup", ok, errors, len(pending))
    return ok, missing


async def stage_fetch(client: ImmichClient, conn: sqlite3.Connection, cache_dir: Path,
                      source: str, log: Log, concurrency: int = 8,
                      limit: int | None = None) -> int:
    """Download originals, skipping anything already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = conn.execute(
        "SELECT a.id, a.original_file_name FROM assets a"
        "  JOIN faces f ON f.asset_id = a.id AND f.status = 'ok'"
        "  LEFT JOIN downloads d ON d.asset_id = a.id"
        " WHERE d.asset_id IS NULL" + _limit_clause(limit)
    ).fetchall()
    if not pending:
        log("  fetch: nothing new to download")
        return 0

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    done = 0
    errors: list[str] = []
    # Originals run to several MB each, so throughput is the number that tells
    # you whether the run is minutes or hours.
    downloadable = conn.execute(
        "SELECT count(*) FROM faces WHERE status = 'ok'").fetchone()[0]
    outstanding = pending_counts(conn)["fetch"]
    bar = Progress("fetch", len(pending), emit=log, show_bytes=True,
                   overall=downloadable, already_done=downloadable - outstanding)

    async def one(asset_id: str, filename: str | None) -> None:
        nonlocal done
        suffix = Path(filename or "").suffix.lower() or ".jpg"
        target = cache_dir / f"{asset_id}{suffix}"
        async with sem:
            if not target.exists():
                try:
                    await client.download_to(asset_id, target, source)
                except Exception as exc:  # noqa: BLE001
                    async with lock:
                        errors.append(str(exc))
                    _log_error(bar.log, errors, f"download {asset_id}", exc)
                    bar.advance(failed=1)
                    return
        size = target.stat().st_size
        async with lock:
            conn.execute(
                "INSERT OR REPLACE INTO downloads (asset_id, path, bytes, source, fetched_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (asset_id, str(target), size, source, db.iso_z(db.now_utc())),
            )
            done += 1
        bar.advance(nbytes=size)

    await asyncio.gather(*(one(r["id"], r["original_file_name"]) for r in pending))
    bar.close()
    _abort_if_hopeless("download", done, errors, len(pending))
    return done


def stage_analyze(conn: sqlite3.Connection, opts: analyze.AnalyzeOptions, workers: int,
                  log: Log, reanalyze: bool = False, limit: int | None = None,
                  persist: bool = True,
                  collect: list[tuple[str, object]] | None = None) -> int:
    """Landmark every downloaded face and store its metrics."""
    where = "" if reanalyze else " LEFT JOIN metrics m ON m.asset_id = a.id WHERE m.asset_id IS NULL"
    rows = conn.execute(
        "SELECT a.id, d.path, f.x1, f.y1, f.x2, f.y2, f.image_width, f.image_height,"
        "       f.source_type"
        "  FROM assets a"
        "  JOIN faces f ON f.asset_id = a.id AND f.status = 'ok'"
        "  JOIN downloads d ON d.asset_id = a.id"
        f"{where}" + _limit_clause(limit)
    ).fetchall()
    if not rows:
        log("  analyze: nothing new to analyze")
        return 0

    jobs = [(r["id"], r["path"], {
        "x1": r["x1"], "y1": r["y1"], "x2": r["x2"], "y2": r["y2"],
        "image_width": r["image_width"], "image_height": r["image_height"],
        "source_type": r["source_type"],
    }) for r in rows]

    if workers:
        log(f"  analyze: {len(jobs)} images across {workers} workers (from config), "
            f"effort={opts.effort}")
    else:
        workers = analyze.physical_cores()
        log(f"  analyze: {len(jobs)} images across {workers} workers "
            f"({analyze.available_cpus()} CPUs visible), effort={opts.effort}")

    # Chunk finely enough that the tail stays short. Fixed coarse chunks leave
    # stragglers on heterogeneous cores -- an M1's efficiency cores run maybe a
    # third the speed of its performance cores, so a chunk landing on one at the
    # end holds up the whole stage.
    chunksize = max(1, min(16, len(jobs) // (workers * 4) or 1))

    done = 0
    stamp = db.iso_z(db.now_utc())
    analyzable = conn.execute(
        "SELECT count(*) FROM downloads d"
        "  JOIN faces f ON f.asset_id = d.asset_id AND f.status = 'ok'").fetchone()[0]
    outstanding = pending_counts(conn)["analyze"]
    bar = Progress("analyze", len(jobs), emit=log,
                   overall=analyzable, already_done=analyzable - outstanding)
    with ProcessPoolExecutor(max_workers=workers, initializer=analyze.init_worker,
                             initargs=(opts,)) as pool:
        for asset_id, m in pool.map(analyze.analyze_one, jobs, chunksize=chunksize):
            # `persist=False` lets `trial --compare` measure an effort level
            # without overwriting the metrics the pipeline is actually using.
            if persist:
                _store_metrics(conn, asset_id, m, stamp)
            if collect is not None:
                collect.append((asset_id, m))
            done += 1
            bar.advance()

    bar.close()
    return done


def _store_metrics(conn: sqlite3.Connection, asset_id: str, m, stamp: str) -> None:
    payload = m.as_dict()
    payload["asset_id"] = asset_id
    payload["analyzed_at"] = stamp
    columns = ", ".join(payload)
    placeholders = ", ".join(f":{k}" for k in payload)
    conn.execute(
        f"INSERT OR REPLACE INTO metrics ({columns}) VALUES ({placeholders})", payload
    )


def _report_fit(rows: Iterable, eye_distance: float, eye_level: float,
                aspect: float, margin: float, log: Log) -> None:
    """Say which frames clip, and what setting would fit them.

    A default framing is a guess about someone else's photos, so rather than
    leave the user to spot a cropped forehead in the contact sheet, work it out
    from the spans recorded during analysis.
    """
    clipped = 0
    unjudged = 0
    tightest = None

    for row in rows:
        spans = (row["span_w"], row["span_up"], row["span_down"])
        if any(span is None for span in spans):
            unjudged += 1
            continue
        span_w, span_up, span_down = (float(s) for s in spans)
        if not align.head_fits(eye_distance, eye_level, span_w, span_up, span_down,
                               aspect, margin):
            clipped += 1
        limit = align.fitting_eye_distance(span_w, span_up, span_down, eye_level,
                                           aspect, margin)
        tightest = limit if tightest is None else min(tightest, limit)

    if clipped and tightest is not None:
        # Round down, so the suggestion lands inside the limit rather than on it.
        suggestion = math.floor(tightest * 100) / 100
        log(f"  align: {clipped} frames clip the face at eye_distance="
            f"{eye_distance:g}; {suggestion:g} would fit them all")
    if unjudged:
        log(f"  align: {unjudged} frames analyzed before face spans were recorded, "
            "so their framing was not checked — `grow-up analyze --reanalyze` fills "
            "them in")


def stage_align(conn: sqlite3.Connection, frames_dir: Path, output: dict,
                log: Log, workers: int = 0, framing: dict | None = None) -> int:
    """Warp selected frames onto canonical eye positions."""
    import cv2

    framing = dict(framing or {})

    rows = select.selected_in_order(conn)
    if not rows:
        log("  align: nothing selected")
        return 0

    frames_dir.mkdir(parents=True, exist_ok=True)
    width, height = int(output["width"]), int(output["height"])
    eye_distance = float(framing.get("eye_distance", 0.20))
    eye_level = float(framing.get("eye_level", 0.42))
    fill = str(framing.get("fill", "blur"))
    dst_left, dst_right = align.target_eyes_from(width, height, eye_distance, eye_level)

    _report_fit(rows, eye_distance, eye_level, width / height,
                float(framing.get("fit_margin", 1.5)), log)

    eye_pairs = [(np.array([row["left_eye_x"], row["left_eye_y"]]),
                  np.array([row["right_eye_x"], row["right_eye_y"]]))
                 for row in rows]
    matrices = align.transforms_for(eye_pairs, dst_left, dst_right)
    smoothed = [align.decompose_affine(m) for m in matrices]

    workers = workers or max(4, analyze.physical_cores())

    def one(index_row_param: tuple[int, sqlite3.Row, align.TransformParams]) -> tuple:
        seq, row, p = index_row_param
        bgr = images.load_bgr(row["path"])
        warped = align.warp(bgr, align.build_affine(p), width, height, fill=fill)
        out_path = frames_dir / f"frame_{seq:06d}.jpg"
        cv2.imwrite(str(out_path), warped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        luma = float(np.median(cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)[:, :, 0]))
        return row["asset_id"], str(out_path), seq, p, luma

    results = []
    bar = Progress("align", len(rows), emit=log)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(one, [
            (i, row, p) for i, (row, p) in enumerate(zip(rows, smoothed), start=1)
        ]):
            results.append(result)
            bar.advance()
    bar.close()

    results.sort(key=lambda r: r[2])
    stamp = db.iso_z(db.now_utc())
    with conn:
        conn.execute("BEGIN")
        conn.execute("DELETE FROM frames")
        conn.executemany(
            "INSERT INTO frames (asset_id, path, seq, tx, ty, angle, scale, warped_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(a, p, s, tp.tx, tp.ty, tp.angle, tp.scale, stamp)
             for a, p, s, tp, _ in results],
        )

    if output.get("flicker_match", False):
        _damp_flicker(results, log)

    log(f"  align: wrote {len(results)} frames to {frames_dir}")
    return len(results)


def _damp_flicker(results: list[tuple], log: Log, window: int = 9) -> None:
    """Second pass: pull each frame's brightness toward its neighbours' median.

    Done as a separate pass rather than in the warp loop because the reference is
    a rolling window over frames that have not been written yet at that point.
    """
    import cv2

    luma = np.array([r[4] for r in results], dtype=np.float64)
    if len(luma) < 3:
        return
    half = max(1, window // 2)
    adjusted = 0
    for i, (_, path, _, _, _) in enumerate(results):
        lo, hi = max(0, i - half), min(len(luma), i + half + 1)
        reference = float(np.median(luma[lo:hi]))
        if abs(reference - luma[i]) < 1.0:
            continue
        frame = cv2.imread(path)
        if frame is None:
            continue
        cv2.imwrite(path, align.match_luma(frame, reference),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        adjusted += 1
    log(f"  flicker: adjusted {adjusted} frames")


def stage_encode(conn: sqlite3.Connection, out_dir: Path, encode_cfg: dict,
                 log: Log) -> Path:
    """Encode the aligned frames, honouring manual rejects from the contact sheet."""
    rejects = review.load_manual_rejects(out_dir / "rejects.json")
    rows = conn.execute("SELECT asset_id, path FROM frames ORDER BY seq ASC").fetchall()
    frames = [Path(r["path"]) for r in rows if r["asset_id"] not in rejects]

    if rejects:
        log(f"  encode: honouring {len(rejects)} manual rejects from rejects.json")
    if not frames:
        raise RuntimeError("no frames left to encode")

    out_path = out_dir / str(encode_cfg.get("filename", "timelapse.mp4"))
    log(f"  encode: {len(frames)} frames at {encode_cfg.get('fps', 10)} fps")
    return encode(
        frames, out_path,
        fps=float(encode_cfg.get("fps", 10)),
        codec=str(encode_cfg.get("codec", "libx264")),
        crf=int(encode_cfg.get("crf", 18)),
        interpolate=bool(encode_cfg.get("interpolate", False)),
    )


def report_rejects(conn: sqlite3.Connection, log: Log) -> None:
    for line in select.format_reject_summary(conn):
        log(line)
