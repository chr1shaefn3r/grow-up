"""Stage orchestration.

Every stage is independently re-runnable and skips work the manifest already
records, so interrupting any of them costs only the item in flight.
"""

from __future__ import annotations

import asyncio
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

Log = Callable[[str], None]


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

    A video newly tagged with her can trigger a false positive here. That costs
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
            log(f"  indexed {seen}…")

    log(f"  indexed {seen} assets ({new} new)")
    return seen, new


async def stage_faces(client: ImmichClient, conn: sqlite3.Connection, person_id: str,
                      log: Log, concurrency: int = 16) -> tuple[int, int]:
    """Fetch her face box per asset. Immich's association means no recognition here."""
    pending = [row[0] for row in conn.execute(
        "SELECT a.id FROM assets a LEFT JOIN faces f ON f.asset_id = a.id"
        " WHERE f.asset_id IS NULL"
    )]
    if not pending:
        log("  no new assets need face boxes")
        return 0, 0

    sem = asyncio.Semaphore(concurrency)
    ok = missing = 0
    lock = asyncio.Lock()

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
                log(f"  ! {asset_id}: {type(exc).__name__}")
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

    await asyncio.gather(*(one(a) for a in pending))
    log(f"  face boxes: {ok} found, {missing} without a usable detection")
    return ok, missing


async def stage_fetch(client: ImmichClient, conn: sqlite3.Connection, cache_dir: Path,
                      source: str, log: Log, concurrency: int = 8) -> int:
    """Download originals, skipping anything already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    pending = conn.execute(
        "SELECT a.id, a.original_file_name FROM assets a"
        "  JOIN faces f ON f.asset_id = a.id AND f.status = 'ok'"
        "  LEFT JOIN downloads d ON d.asset_id = a.id"
        " WHERE d.asset_id IS NULL"
    ).fetchall()
    if not pending:
        log("  nothing new to download")
        return 0

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    done = 0

    async def one(asset_id: str, filename: str | None) -> None:
        nonlocal done
        suffix = Path(filename or "").suffix.lower() or ".jpg"
        target = cache_dir / f"{asset_id}{suffix}"
        async with sem:
            if not target.exists():
                try:
                    data = await client.download(asset_id, source)
                except Exception as exc:  # noqa: BLE001
                    log(f"  ! download {asset_id}: {type(exc).__name__}")
                    return
                target.write_bytes(data)
        async with lock:
            conn.execute(
                "INSERT OR REPLACE INTO downloads (asset_id, path, bytes, source, fetched_at)"
                " VALUES (?, ?, ?, ?, ?)",
                (asset_id, str(target), target.stat().st_size, source, db.iso_z(db.now_utc())),
            )
            done += 1
            if done % 100 == 0:
                log(f"  downloaded {done}/{len(pending)}…")

    await asyncio.gather(*(one(r["id"], r["original_file_name"]) for r in pending))
    log(f"  downloaded {done} files")
    return done


def stage_analyze(conn: sqlite3.Connection, opts: analyze.AnalyzeOptions, workers: int,
                  log: Log, reanalyze: bool = False) -> int:
    """Landmark every downloaded face and store its metrics."""
    where = "" if reanalyze else " LEFT JOIN metrics m ON m.asset_id = a.id WHERE m.asset_id IS NULL"
    rows = conn.execute(
        "SELECT a.id, d.path, f.x1, f.y1, f.x2, f.y2, f.image_width, f.image_height,"
        "       f.source_type"
        "  FROM assets a"
        "  JOIN faces f ON f.asset_id = a.id AND f.status = 'ok'"
        "  JOIN downloads d ON d.asset_id = a.id"
        f"{where}"
    ).fetchall()
    if not rows:
        log("  nothing new to analyze")
        return 0

    jobs = [(r["id"], r["path"], {
        "x1": r["x1"], "y1": r["y1"], "x2": r["x2"], "y2": r["y2"],
        "image_width": r["image_width"], "image_height": r["image_height"],
        "source_type": r["source_type"],
    }) for r in rows]

    workers = workers or analyze.physical_cores()
    log(f"  analyzing {len(jobs)} images across {workers} workers")

    done = 0
    stamp = db.iso_z(db.now_utc())
    with ProcessPoolExecutor(max_workers=workers, initializer=analyze.init_worker,
                             initargs=(opts,)) as pool:
        for asset_id, m in pool.map(analyze.analyze_one, jobs, chunksize=8):
            _store_metrics(conn, asset_id, m, stamp)
            done += 1
            if done % 200 == 0:
                log(f"  analyzed {done}/{len(jobs)}…")

    log(f"  analyzed {done} images")
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


def stage_align(conn: sqlite3.Connection, frames_dir: Path, output: dict,
                log: Log, workers: int = 0) -> int:
    """Warp selected frames onto canonical eye positions, then smooth the sequence."""
    import cv2

    rows = select.selected_in_order(conn)
    if not rows:
        log("  nothing selected to align")
        return 0

    frames_dir.mkdir(parents=True, exist_ok=True)
    width, height = int(output["width"]), int(output["height"])
    dst_left, dst_right = align.target_eyes(
        width, height, tuple(output["left_eye"]), tuple(output["right_eye"])
    )

    params = []
    for row in rows:
        matrix = align.similarity_transform(
            np.array([row["left_eye_x"], row["left_eye_y"]]),
            np.array([row["right_eye_x"], row["right_eye_y"]]),
            dst_left, dst_right,
        )
        params.append(align.decompose_affine(matrix))

    window = int(output.get("smoothing_window", 0))
    smoothed = align.smooth_params(params, window, int(output.get("smoothing_polyorder", 2)))
    if window > 1:
        log(f"  smoothing transforms over a {min(window, len(params))}-frame window")

    workers = workers or max(4, analyze.physical_cores())

    def one(index_row_param: tuple[int, sqlite3.Row, align.TransformParams]) -> tuple:
        seq, row, p = index_row_param
        bgr = images.load_bgr(row["path"])
        warped = align.warp(bgr, align.build_affine(p), width, height)
        out_path = frames_dir / f"frame_{seq:06d}.jpg"
        cv2.imwrite(str(out_path), warped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        luma = float(np.median(cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)[:, :, 0]))
        return row["asset_id"], str(out_path), seq, p, luma

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, result in enumerate(pool.map(one, [
            (i, row, p) for i, (row, p) in enumerate(zip(rows, smoothed), start=1)
        ]), start=1):
            results.append(result)
            if n % 100 == 0:
                log(f"  aligned {n}/{len(rows)}…")

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

    log(f"  wrote {len(results)} frames to {frames_dir}")
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
    log(f"  flicker pass adjusted {adjusted} frames")


def stage_encode(conn: sqlite3.Connection, out_dir: Path, encode_cfg: dict,
                 log: Log) -> Path:
    """Encode the aligned frames, honouring manual rejects from the contact sheet."""
    rejects = review.load_manual_rejects(out_dir / "rejects.json")
    rows = conn.execute("SELECT asset_id, path FROM frames ORDER BY seq ASC").fetchall()
    frames = [Path(r["path"]) for r in rows if r["asset_id"] not in rejects]

    if rejects:
        log(f"  honouring {len(rejects)} manual rejects from rejects.json")
    if not frames:
        raise RuntimeError("no frames left to encode")

    out_path = out_dir / str(encode_cfg.get("filename", "timelapse.mp4"))
    log(f"  encoding {len(frames)} frames at {encode_cfg.get('fps', 10)} fps")
    return encode(
        frames, out_path,
        fps=float(encode_cfg.get("fps", 10)),
        codec=str(encode_cfg.get("codec", "libx264")),
        crf=int(encode_cfg.get("crf", 18)),
        interpolate=bool(encode_cfg.get("interpolate", False)),
    )


def report_rejects(conn: sqlite3.Connection, log: Log) -> None:
    log("  filter outcome:")
    for reason, count in select.reject_summary(conn):
        log(f"    {reason:<26} {count:>6}")
