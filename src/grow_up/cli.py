"""Command line interface.

`--since` is threaded into the index stage and nowhere else. Selection,
alignment and encoding must always see the whole corpus: constraining them to
the incremental window would silently produce a timelapse containing only the
last few days of frames.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import urllib.request
from pathlib import Path

from . import analyze, config, db, pipeline, review, select, timing
from .encode import FFmpegMissing
from .immich import (
    ANY,
    OPTIONAL_PERMISSIONS,
    REQUIRED_PERMISSIONS,
    ImmichClient,
    ImmichHTTPError,
    missing_permissions,
)


def log(message: str) -> None:
    print(message, flush=True)


def _open(args: argparse.Namespace):
    cfg = config.load(args.config)
    conn = db.connect(cfg.path("db"))
    return cfg, conn


def _client(cfg: config.Config) -> ImmichClient:
    """Build a client with the configured concurrency and retry budget.

    Concurrency is shared with the stages' own limiting, so lowering
    fetch.concurrency genuinely reduces simultaneous load on the server.
    """
    return ImmichClient(
        config.credentials(),
        concurrency=int(cfg.get("fetch", "concurrency", 8)),
        retries=int(cfg.get("fetch", "retries", 4)),
    )


async def preflight(client: ImmichClient) -> set[str]:
    """Check connectivity and the key's scopes before doing any real work.

    `/api-keys/me` needs no permission, so this answers "is it the key?"
    definitively in one request rather than after hundreds of failures.
    """
    await client.ping()
    try:
        granted = await client.my_permissions()
    except ImmichHTTPError as exc:
        log(f"  ! could not read the key's permissions ({exc.status}); continuing")
        return set()

    missing = missing_permissions(granted, REQUIRED_PERMISSIONS)
    if missing:
        detail = "\n".join(f"    {name}  ({REQUIRED_PERMISSIONS[name]})" for name in missing)
        raise RuntimeError(
            "this Immich API key is missing required permission(s):\n"
            f"{detail}\n"
            "  Add them in Immich under Account Settings -> API Keys."
        )

    for name, why in OPTIONAL_PERMISSIONS.items():
        if "all" not in granted and name not in granted:
            log(f"  note: key lacks {name!r}, so grow-up cannot {why}")
    return granted


def _person_id(cfg: config.Config, conn) -> str:
    """Resolve the target person, caching the id in config for later runs."""
    person_id = cfg.get("immich", "person_id")
    if person_id:
        return str(person_id)

    name = cfg.get("immich", "person_name")
    if not name:
        raise SystemExit("set immich.person_id or immich.person_name in config.toml")

    async def resolve() -> str:
        async with _client(cfg) as client:
            person = await client.resolve_person(str(name))
            log(f"resolved {person.name!r} -> {person.id}")
            log("  (put this in config.toml as immich.person_id to skip the lookup)")
            return person.id

    return asyncio.run(resolve())


# --------------------------------------------------------------------------- #
# Individual stages
# --------------------------------------------------------------------------- #

def cmd_fetch_model(args: argparse.Namespace) -> None:
    target = Path(args.output or analyze.DEFAULT_MODEL_PATH)
    if target.exists() and not args.force:
        log(f"{target} already present")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    log(f"downloading {analyze.MODEL_URL}")
    urllib.request.urlretrieve(analyze.MODEL_URL, target)  # noqa: S310
    log(f"saved {target} ({target.stat().st_size / 1e6:.1f} MB)")


def cmd_index(args: argparse.Namespace) -> None:
    cfg, conn = _open(args)
    person_id = _person_id(cfg, conn)
    asyncio.run(_index(cfg, conn, person_id, args.since, args.full))


async def _index(cfg, conn, person_id: str, since: str | None, full: bool) -> int:
    """Index stage plus watermark bookkeeping. Returns assets newly added."""
    started_at = db.now_utc()
    watermark = pipeline.resolve_watermark(conn, person_id, since, full)
    log(f"watermark: {watermark.value or '(none — full index)'}  [{watermark.source}]")

    run_id = db.start_run(conn, person_id, started_at, watermark.value, watermark.source)
    page_size = int(cfg.get("index", "page_size", 1000))

    try:
        async with _client(cfg) as client:
            await preflight(client)
            _, new = await pipeline.stage_index(
                client, conn, person_id, watermark, page_size, log
            )

            current_count = await client.person_asset_count(person_id)
            if current_count is None:
                log("  note: person statistics unavailable, so drift detection is off "
                    "for this run")
            state = db.get_sync_state(conn, person_id)
            stored_count = state.person_assets if state else None

            if not watermark.is_full and pipeline.detect_drift(stored_count, current_count, new):
                log(
                    f"drift detected: Immich reports {current_count} assets for this person "
                    f"(was {stored_count}) but the incremental window only found {new} new. "
                    "She was likely tagged in older photos — re-indexing in full."
                )
                _, new = await pipeline.stage_index(
                    client, conn, person_id, pipeline.Watermark(None, "full: drift detected"),
                    page_size, log,
                )
                current_count = await client.person_asset_count(person_id)
    except Exception:
        # Leave the previous watermark alone so the un-indexed window is covered
        # again next run.
        db.fail_run(conn, run_id)
        raise

    stored = db.commit_watermark(conn, run_id, person_id, started_at, current_count, new)
    log(f"watermark advanced to {stored}")
    return new


def cmd_faces(args: argparse.Namespace) -> None:
    cfg, conn = _open(args)
    person_id = _person_id(cfg, conn)

    async def go() -> None:
        async with _client(cfg) as client:
            await pipeline.stage_faces(client, conn, person_id, log,
                                       int(cfg.get("fetch", "concurrency", 16)),
                                       limit=args.limit)

    asyncio.run(go())


def cmd_fetch(args: argparse.Namespace) -> None:
    cfg, conn = _open(args)

    async def go() -> None:
        async with _client(cfg) as client:
            await pipeline.stage_fetch(
                client, conn, cfg.path("cache"),
                str(cfg.get("fetch", "source", "original")), log,
                int(cfg.get("fetch", "concurrency", 8)), limit=args.limit,
            )

    asyncio.run(go())


def _analyze_options(cfg: config.Config,
                     verbose: bool = False) -> analyze.AnalyzeOptions:
    return analyze.AnalyzeOptions(
        model_path=str(cfg.get("analyze", "model_path", analyze.DEFAULT_MODEL_PATH)),
        bbox_margin=float(cfg.get("analyze", "bbox_margin", 0.8)),
        min_face_detection_confidence=float(
            cfg.get("analyze", "min_face_detection_confidence", 0.5)),
        oob_inset=float(cfg.get("analyze", "oob_inset", 0.0)),
        # --verbose only ever turns logging on; it never silences a config that
        # asked for it.
        verbose=bool(verbose or cfg.get("analyze", "verbose", False)),
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    cfg, conn = _open(args)
    pipeline.stage_analyze(conn, _analyze_options(cfg, args.verbose),
                           int(cfg.get("analyze", "workers", 0)), log,
                           reanalyze=args.reanalyze, limit=args.limit)
    cmd_select(args)


def cmd_select(args: argparse.Namespace) -> None:
    cfg, conn = _open(args)
    limits, weights = cfg.section("filter"), cfg.section("score")
    kept, total = select.apply_filters(conn, limits, weights)
    log(f"  {kept}/{total} images pass the hard filters")

    cadence = getattr(args, "cadence", None) or str(cfg.get("select", "cadence", "week"))
    per_bucket = int(cfg.get("select", "per_bucket", 1))
    n = select.select_frames(conn, cadence, per_bucket)
    log(f"  selected {n} frames (cadence={cadence}, {per_bucket} per bucket)")
    pipeline.report_rejects(conn, log)


def cmd_align(args: argparse.Namespace) -> None:
    cfg, conn = _open(args)
    pipeline.stage_align(conn, cfg.path("frames"), cfg.section("output"), log,
                         int(cfg.get("analyze", "workers", 0)))


def cmd_review(args: argparse.Namespace) -> None:
    cfg, conn = _open(args)
    out_dir = cfg.path("out")
    accepted = review.write_contact_sheet(conn, out_dir / "contact-sheet.html")
    rejected = review.write_rejects_gallery(conn, out_dir / "rejects.html")
    log(f"  contact sheet: {accepted} frames -> {out_dir / 'contact-sheet.html'}")
    log(f"  rejects gallery: {rejected} samples -> {out_dir / 'rejects.html'}")


def cmd_encode(args: argparse.Namespace) -> None:
    cfg, conn = _open(args)
    out = pipeline.stage_encode(conn, cfg.path("out"), cfg.section("encode"), log)
    log(f"  wrote {out}")


def cmd_trial(args: argparse.Namespace) -> None:
    """Run the pipeline over a sample and project the full run from it.

    A trial is a *partial real run*, not a simulation: everything it downloads
    and analyzes is written to the manifest and cache, so none of the work is
    repeated later. Running a trial simply gets you that much further along.
    """
    cfg, conn = _open(args)

    total_assets = db.count_assets(conn)
    if not total_assets:
        raise SystemExit("no assets indexed yet — run `grow-up index` first")

    person_id = _person_id(cfg, conn)
    limit = int(args.limit or cfg.get("trial", "limit", 100))
    before = pipeline.pending_counts(conn)
    trial = timing.Trial(sample_size=limit, total_assets=total_assets)
    log(f"== trial: sampling up to {limit} of {total_assets} indexed assets ==")

    bytes_before = conn.execute(
        "SELECT coalesce(sum(bytes), 0) FROM downloads").fetchone()[0]

    async def network_stages() -> None:
        async with _client(cfg) as client:
            await preflight(client)

            with timing.stopwatch() as faces_elapsed:
                found, _ = await pipeline.stage_faces(
                    client, conn, person_id, log,
                    int(cfg.get("fetch", "concurrency", 16)), limit=limit)
            trial.stages.append(timing.StageTiming(
                "faces", found, faces_elapsed(), before["faces"]))

            with timing.stopwatch() as fetch_elapsed:
                fetched = await pipeline.stage_fetch(
                    client, conn, cfg.path("cache"),
                    str(cfg.get("fetch", "source", "original")), log,
                    int(cfg.get("fetch", "concurrency", 8)), limit=limit)
            downloaded = conn.execute(
                "SELECT coalesce(sum(bytes), 0) FROM downloads").fetchone()[0] - bytes_before
            note = ""
            if fetched and downloaded:
                rate = downloaded / max(fetch_elapsed(), 1e-6)
                projected_bytes = downloaded / fetched * before["fetch"]
                note = (f"{timing.format_bytes(downloaded)} at "
                        f"{timing.format_bytes(rate)}/s -> "
                        f"{timing.format_bytes(projected_bytes)} total")
            trial.stages.append(timing.StageTiming(
                "fetch", fetched, fetch_elapsed(), before["fetch"], note=note))

    asyncio.run(network_stages())

    opts = _analyze_options(cfg, args.verbose)
    with timing.stopwatch() as analyze_elapsed:
        analyzed = pipeline.stage_analyze(
            conn, opts, int(cfg.get("analyze", "workers", 0)), log, limit=limit)
    trial.stages.append(timing.StageTiming(
        "analyze", analyzed, analyze_elapsed(), before["analyze"]))

    limits, weights = cfg.section("filter"), cfg.section("score")
    kept, scored = select.apply_filters(conn, limits, weights)
    cadence = args.cadence or str(cfg.get("select", "cadence", "week"))
    frames = select.select_frames(conn, cadence, int(cfg.get("select", "per_bucket", 1)))
    log(f"  {kept}/{scored} pass the filters; {frames} frames selected "
        f"(cadence={cadence})")

    with timing.stopwatch() as align_elapsed:
        aligned = pipeline.stage_align(conn, cfg.path("frames"), cfg.section("output"),
                                       log, int(cfg.get("analyze", "workers", 0)))
    # Alignment scales with *selected frames*, not assets: bucketing means a
    # bigger library yields proportionally more frames, not one per photo.
    projected_frames = round(frames / scored * total_assets) if scored else 0
    trial.stages.append(timing.StageTiming(
        "align", aligned, align_elapsed(), max(0, projected_frames - aligned),
        unit="frame", note=f"~{projected_frames} frames projected for the full set"))

    if aligned:
        out_dir = cfg.path("out")
        with timing.stopwatch() as review_elapsed:
            cmd_review(args)
        trial.stages.append(timing.StageTiming(
            "review", aligned, review_elapsed(), max(0, projected_frames - aligned),
            unit="frame"))

        # Write the trial video under its own name. A trial renders a handful of
        # frames, and silently overwriting a finished full render with a
        # two-second sample would be a poor trade for the convenience.
        encode_cfg = dict(cfg.section("encode"))
        encode_cfg["filename"] = f"trial-{encode_cfg.get('filename', 'timelapse.mp4')}"
        if args.no_encode:
            log("  skipping encode (--no-encode)")
        else:
            try:
                with timing.stopwatch() as encode_elapsed:
                    video = pipeline.stage_encode(conn, out_dir, encode_cfg, log)
                trial.stages.append(timing.StageTiming(
                    "encode", aligned, encode_elapsed(),
                    max(0, projected_frames - aligned),
                    unit="frame", note=str(video)))
            except FFmpegMissing as exc:
                log(f"  ! skipping encode: {exc}")
            except RuntimeError as exc:
                log(f"  ! encode failed: {exc}")
    else:
        log("  no frames survived the filters, so there is nothing to review or encode")

    for line in trial.render():
        log(line)
    pipeline.report_rejects(conn, log)

    if aligned:
        out_dir = cfg.path("out")
        log("")
        log(f"Look at {out_dir / 'contact-sheet.html'} to judge alignment and framing,")
        log(f"and {out_dir / 'rejects.html'} to check the thresholds are not too tight.")


def cmd_doctor(args: argparse.Namespace) -> None:
    """Probe each endpoint once and report exactly what it returns.

    Exists because a stage that fails on every item is the worst place to learn
    what went wrong: one request, fully described, beats hundreds of identical
    error lines.
    """
    cfg, conn = _open(args)

    row = conn.execute(
        "SELECT a.id FROM assets a JOIN faces f ON f.asset_id = a.id AND f.status = 'ok'"
        " LIMIT 1"
    ).fetchone()
    asset_id = args.asset or (row["id"] if row else None)
    person_id = str(cfg.get("immich", "person_id") or "")

    async def go() -> None:
        async with _client(cfg) as client:
            log(f"server:  {config.credentials().url}")

            probes: list[tuple[str, str, str, dict | None]] = [
                ("connectivity", "/server/ping", "application/json", None),
                ("key metadata", "/api-keys/me", "application/json", None),
            ]
            if person_id:
                probes.append(("person stats", f"/people/{person_id}/statistics",
                               "application/json", None))
            if asset_id:
                probes += [
                    ("faces", "/faces", "application/json", {"id": asset_id}),
                    # The failing call, and the two things it could be confused
                    # with: same endpoint asking for JSON, and the view-scoped
                    # rendition. Whichever succeeds localises the fault.
                    ("download (Accept: */*)", f"/assets/{asset_id}/original", ANY, None),
                    ("download (Accept: json)", f"/assets/{asset_id}/original",
                     "application/json", None),
                    ("preview", f"/assets/{asset_id}/thumbnail", ANY, {"size": "preview"}),
                ]

            log("")
            for label, path, accept, params in probes:
                result = await client.probe(path, accept, params)
                mark = "ok  " if result["ok"] else "FAIL"
                status = result["status"] if result["status"] is not None else "---"
                line = f"  [{mark}] {label:<24} {status}"
                if result["ok"]:
                    line += f"  {result.get('content_type', '')} {result.get('length', 0)}B"
                log(line)
                if result["detail"]:
                    log(f"         {result['detail'][:400]}")

            try:
                granted = await client.my_permissions()
            except ImmichHTTPError:
                granted = set()
            if granted:
                missing = missing_permissions(granted, REQUIRED_PERMISSIONS)
                log("")
                log(f"key permissions: {'all (wildcard)' if 'all' in granted else len(granted)}")
                log(f"missing required: {missing or 'none'}")

    if not asset_id:
        log("note: no indexed asset to probe with; run `grow-up index` first "
            "or pass --asset <uuid>")
    asyncio.run(go())


def cmd_status(args: argparse.Namespace) -> None:
    cfg, conn = _open(args)
    person_id = str(cfg.get("immich", "person_id") or "")
    log(f"database: {cfg.path('db')}")
    for table in ("assets", "faces", "downloads", "metrics", "selection", "frames"):
        n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        log(f"  {table:<12} {n:>7}")

    state = db.get_sync_state(conn, person_id) if person_id else None
    if state:
        log(f"\nwatermark   {state.watermark}")
        log(f"last run    {state.last_run_at}")
        log(f"person has  {state.person_assets} assets per Immich at that point")
    else:
        log("\nno watermark stored yet — the next run will be a full index")

    rows = conn.execute(
        "SELECT started_at, watermark_source, assets_indexed, status"
        "  FROM runs ORDER BY id DESC LIMIT 5"
    ).fetchall()
    if rows:
        log("\nrecent runs:")
        for r in rows:
            log(f"  {r['started_at']}  {r['status']:<7} {r['watermark_source']:<22}"
                f" +{r['assets_indexed'] or 0}")


def cmd_run(args: argparse.Namespace) -> None:
    """The whole pipeline. A bare `run` is implicitly incremental."""
    cfg, conn = _open(args)
    person_id = _person_id(cfg, conn)

    async def network_stages() -> None:
        # --since / --full apply here and stop here.
        await _index(cfg, conn, person_id, args.since, args.full)
        async with _client(cfg) as client:
            await pipeline.stage_faces(client, conn, person_id, log,
                                       int(cfg.get("fetch", "concurrency", 16)))
            await pipeline.stage_fetch(
                client, conn, cfg.path("cache"),
                str(cfg.get("fetch", "source", "original")), log,
                int(cfg.get("fetch", "concurrency", 8)),
            )

    log("== index ==")
    asyncio.run(network_stages())

    log("== analyze ==")
    cmd_analyze(args)

    log("== align ==")
    cmd_align(args)

    log("== review ==")
    cmd_review(args)

    if args.no_encode:
        log("\nskipping encode (--no-encode)")
        return

    log("== encode ==")
    cmd_encode(args)
    log("\nReview out/contact-sheet.html, drop rejects.json beside it, "
        "then re-run `grow-up encode` to apply them.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grow-up",
        description="Build an eye-aligned face timelapse from an Immich library.",
    )
    parser.add_argument("--config", default="config.toml", help="path to config.toml")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show MediaPipe/TFLite native logging, hidden by default")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func, since=None, full=False, reanalyze=False,
                       cadence=None, no_encode=False, asset=None, limit=None)
        return p

    model = add("fetch-model", cmd_fetch_model, "download the MediaPipe face landmarker model")
    model.add_argument("-o", "--output", default=None)
    model.add_argument("--force", action="store_true")

    for name, func, help_text in (
        ("index", cmd_index, "enumerate her photos (honours the sync watermark)"),
        ("run", cmd_run, "run every stage; incremental by default"),
    ):
        p = add(name, func, help_text)
        p.add_argument("--since", default=None,
                       help="override the stored watermark (ISO-8601, e.g. 2026-01-01T00:00:00Z)")
        p.add_argument("--full", action="store_true",
                       help="ignore the stored watermark and re-index everything")
        if name == "run":
            p.add_argument("--cadence", default=None, choices=list(select.CADENCES))
            p.add_argument("--no-encode", action="store_true")

    for name, func, help_text in (
        ("faces", cmd_faces, "fetch her face bounding box per asset"),
        ("fetch", cmd_fetch, "download originals"),
    ):
        p = add(name, func, help_text)
        p.add_argument("-n", "--limit", type=int, default=None,
                       help="process at most this many assets")

    p = add("doctor", cmd_doctor, "probe each endpoint once and report what it returns")
    p.add_argument("--asset", default=None, help="asset UUID to probe with")

    p = add("trial", cmd_trial,
            "process a sample, measure it, and project the full run")
    p.add_argument("-n", "--limit", type=int, default=None,
                   help="how many assets to sample (default: trial.limit in config)")
    p.add_argument("--cadence", default=None, choices=list(select.CADENCES))
    p.add_argument("--no-encode", action="store_true",
                   help="stop after the review pages, skipping the sample video")

    p = add("analyze", cmd_analyze, "landmark faces and compute quality metrics")
    p.add_argument("--reanalyze", action="store_true", help="re-run on already-analyzed assets")
    p.add_argument("-n", "--limit", type=int, default=None,
                   help="process at most this many assets")

    p = add("select", cmd_select, "re-apply thresholds and pick frames (no ML re-run)")
    p.add_argument("--cadence", default=None, choices=list(select.CADENCES))

    add("align", cmd_align, "warp selected frames onto canonical eye positions")
    add("review", cmd_review, "write the contact sheet and rejects gallery")
    add("encode", cmd_encode, "encode the video")
    add("status", cmd_status, "show manifest counts and the stored watermark")
    return parser


def app(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (RuntimeError, FileNotFoundError, SystemExit) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(app())
