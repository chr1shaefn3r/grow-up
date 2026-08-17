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

from . import analyze, config, db, metrics, pipeline, review, select, timing
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
    # A database written before assets carried a source has exactly one
    # account's photos in it, so the first configured source owns them. Done
    # here because this is the only place holding both the config and the
    # connection, and it is a no-op from the second run onwards.
    db.adopt_unsourced(conn, config.sources(cfg)[0].name)
    return cfg, conn


def _sources(cfg: config.Config, only: str | None = None) -> list[config.Source]:
    """The configured accounts, optionally narrowed to one by name."""
    found = config.sources(cfg)
    if only is None:
        return found
    chosen = [s for s in found if s.name == only]
    if not chosen:
        known = ", ".join(repr(s.name) for s in found)
        raise SystemExit(f"no source named {only!r}; configured sources are {known}")
    return chosen


def _labelled(stage: str, source: config.Source, sources: list[config.Source]) -> str:
    """Tag progress with the account, but only when there is more than one.

    A single-account run is the overwhelming majority and its output should not
    change just because the plumbing underneath it grew.
    """
    return stage if len(sources) < 2 else f"{stage}[{source.name}]"


def _apportion(limit: int | None, sources: list[config.Source], conn,
               stage: str) -> list[int | None]:
    """Share a `--limit` out across accounts, so the total is what was asked for."""
    if limit is None:
        return [None] * len(sources)
    if len(sources) == 1:
        return [limit]
    weights = [pipeline.pending_counts(conn, s.name)[stage] for s in sources]
    return list(pipeline.split_limit(limit, weights))


def _client(cfg: config.Config, source: config.Source | None = None) -> ImmichClient:
    """Build a client with the configured concurrency and retry budget.

    Concurrency is shared with the stages' own limiting, so lowering
    fetch.concurrency genuinely reduces simultaneous load on the server.
    """
    return ImmichClient(
        source.credentials() if source else config.credentials(),
        concurrency=int(cfg.get("fetch", "concurrency", 8)),
        retries=int(cfg.get("fetch", "retries", 4)),
    )


async def preflight_all(cfg: config.Config, sources: list[config.Source]) -> None:
    """Check every account before any of them does work.

    Ordering matters more than it looks: preflighting lazily would let the first
    account download a gigabyte before a typo in the second key surfaced. One
    failure aborts the run, because a video quietly missing one account's photos
    looks entirely fine.

    Prints nothing of its own. A healthy preflight has always been silent, so a
    per-source heading here labelled output that never arrived.
    """
    # Every account's environment first. Building a client resolves one
    # account's variables, so without this the second account's missing key is
    # only reported once the first account's have been set.
    config.check_credentials(sources)

    for source in sources:
        whose = f" for source {source.name!r}" if len(sources) > 1 else ""
        async with _client(cfg, source) as client:
            await preflight(client, whose)


async def preflight(client: ImmichClient, whose: str = "") -> set[str]:
    """Check connectivity and the key's scopes before doing any real work.

    `/api-keys/me` needs no permission, so this answers "is it the key?"
    definitively in one request rather than after hundreds of failures.

    `whose` names the account in every message this can emit. With two keys
    configured, "the key lacks person.statistics" does not say which one to go
    and fix -- worst of all on the raising path, where being sent to the wrong
    account costs the most.
    """
    await client.ping()
    try:
        granted = await client.my_permissions()
    except ImmichHTTPError as exc:
        log(f"  ! could not read the permissions of the key{whose} "
            f"({exc.status}); continuing")
        return set()

    missing = missing_permissions(granted, REQUIRED_PERMISSIONS)
    if missing:
        detail = "\n".join(f"    {name}  ({REQUIRED_PERMISSIONS[name]})" for name in missing)
        raise RuntimeError(
            f"this Immich API key{whose} is missing required permission(s):\n"
            f"{detail}\n"
            "  Add them in Immich under Account Settings -> API Keys."
        )

    for name, why in OPTIONAL_PERMISSIONS.items():
        if "all" not in granted and name not in granted:
            log(f"  note: the key{whose} lacks {name!r}, so grow-up cannot {why}")
    return granted


async def _resolve_person(cfg: config.Config, source: config.Source) -> str:
    """Resolve the person within one account.

    Each account keeps its own person record for the same human, so the id is a
    property of the source rather than of the subject.

    A coroutine rather than a blocking call because every caller now resolves
    inside the loop that is already iterating accounts, and `asyncio.run` from
    within a running loop raises.
    """
    if source.person_id:
        return source.person_id

    if not source.person_name:
        where = ("immich.person_id or immich.person_name"
                 if source.name == config.LEGACY_SOURCE_NAME
                 else f"person_id or person_name for source {source.name!r}")
        raise SystemExit(f"set {where} in config.toml")

    async with _client(cfg, source) as client:
        person = await client.resolve_person(source.person_name)
        log(f"  person: {person.name!r} -> {person.id}")
        log("  note: set person_id in config.toml to skip this lookup")
        return person.id


# --------------------------------------------------------------------------- #
# Individual stages
# --------------------------------------------------------------------------- #

def cmd_fetch_model(args: argparse.Namespace) -> None:
    target = Path(args.output or analyze.DEFAULT_MODEL_PATH)
    if target.exists() and not args.force:
        log(f"model: already present at {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    log(f"model: downloading {analyze.MODEL_URL}")
    urllib.request.urlretrieve(analyze.MODEL_URL, target)  # noqa: S310
    log(f"model: saved {target} ({target.stat().st_size / 1e6:.1f} MB)")


def cmd_index(args: argparse.Namespace) -> None:
    cfg, conn = _open(args)
    sources = _sources(cfg, args.source)

    async def go() -> None:
        await preflight_all(cfg, sources)
        for source in sources:
            if len(sources) > 1:
                log(f"  -- {source.name} --")
            await _index(cfg, conn, await _resolve_person(cfg, source),
                         args.since, args.full, source)

    asyncio.run(go())


async def _index(cfg, conn, person_id: str, since: str | None, full: bool,
                 source: config.Source | None = None) -> int:
    """Index stage plus watermark bookkeeping. Returns assets newly added.

    The watermark, the run row and the drift check are all keyed on `person_id`,
    which is already per-account -- so two sources keep two independent sync
    states with no extra bookkeeping.
    """
    source = source or config.sources(cfg)[0]
    started_at = db.now_utc()
    watermark = pipeline.resolve_watermark(conn, person_id, since, full)
    log(f"  watermark: {watermark.value or '(none — full index)'}  [{watermark.source}]")

    run_id = db.start_run(conn, person_id, started_at, watermark.value, watermark.source)
    page_size = int(cfg.get("index", "page_size", 1000))

    try:
        async with _client(cfg, source) as client:
            # No preflight here: preflight_all has already checked every account,
            # which is the point -- a bad second key must not surface only after
            # the first account has finished downloading.
            _, new = await pipeline.stage_index(
                client, conn, person_id, watermark, page_size, log, source.name
            )

            # Cached so `encode` can put an age on the footer without a
            # network call, and refreshed here because a birth date filled in
            # after the first run should reach the next video.
            record = await client.person(person_id)
            if record is not None:
                db.upsert_person(conn, person_id, source.name, record.name,
                                 record.birth_date)

            current_count = await client.person_asset_count(person_id)
            if current_count is None:
                log("  note: person statistics unavailable, so drift detection is off "
                    "for this run")
            state = db.get_sync_state(conn, person_id)
            stored_count = state.person_assets if state else None

            if not watermark.is_full and pipeline.detect_drift(stored_count, current_count, new):
                log(
                    f"  drift detected: Immich reports {current_count} assets for this person "
                    f"(was {stored_count}) but the incremental window only found {new} new. "
                    "The person was likely tagged in older photos — re-indexing in full."
                )
                _, new = await pipeline.stage_index(
                    client, conn, person_id, pipeline.Watermark(None, "full: drift detected"),
                    page_size, log, source.name,
                )
                current_count = await client.person_asset_count(person_id)
    except Exception:
        # Leave the previous watermark alone so the un-indexed window is covered
        # again next run.
        db.fail_run(conn, run_id)
        raise

    stored = db.commit_watermark(conn, run_id, person_id, started_at, current_count, new)
    log(f"  watermark advanced: {stored}")
    return new


def cmd_faces(args: argparse.Namespace) -> None:
    cfg, conn = _open(args)
    sources = _sources(cfg, args.source)

    async def go() -> None:
        # Single-account `faces` and `fetch` never preflighted, and bolting a
        # round-trip plus its notes onto a command that worked fine is a change
        # nobody asked for. With two accounts it earns its keep: the second key
        # is otherwise unexercised until half the work is already done.
        if len(sources) > 1:
            await preflight_all(cfg, sources)
        for source, share in zip(sources, _apportion(args.limit, sources, conn, "faces")):
            async with _client(cfg, source) as client:
                await pipeline.stage_faces(
                    client, conn, await _resolve_person(cfg, source), log,
                    int(cfg.get("fetch", "concurrency", 16)),
                    limit=share, source=source.name,
                    label=_labelled("faces", source, sources))

    asyncio.run(go())


def cmd_fetch(args: argparse.Namespace) -> None:
    cfg, conn = _open(args)
    sources = _sources(cfg, args.source)

    async def go() -> None:
        # Single-account `faces` and `fetch` never preflighted, and bolting a
        # round-trip plus its notes onto a command that worked fine is a change
        # nobody asked for. With two accounts it earns its keep: the second key
        # is otherwise unexercised until half the work is already done.
        if len(sources) > 1:
            await preflight_all(cfg, sources)
        for source, share in zip(sources, _apportion(args.limit, sources, conn, "fetch")):
            async with _client(cfg, source) as client:
                await pipeline.stage_fetch(
                    client, conn, cfg.path("cache"),
                    str(cfg.get("fetch", "source", "original")), log,
                    int(cfg.get("fetch", "concurrency", 8)), limit=share,
                    account=source.name,
                    label=_labelled("fetch", source, sources))

    asyncio.run(go())


def _analyze_options(cfg: config.Config, verbose: bool = False,
                     effort: str | None = None) -> analyze.AnalyzeOptions:
    """Build analyze options: preset first, explicit config settings on top.

    So `effort = "thorough"` gives the whole bundle, but a lone
    `ensemble = 5` beside it still wins for that one field.
    """
    level = effort or str(cfg.get("analyze", "effort", "fast"))
    settings = analyze.preset_for(level)

    section = cfg.section("analyze")
    for key in list(settings):
        if key in section:
            settings[key] = section[key]

    return analyze.AnalyzeOptions(
        model_path=str(cfg.get("analyze", "model_path", analyze.DEFAULT_MODEL_PATH)),
        bbox_margin=float(cfg.get("analyze", "bbox_margin", 0.8)),
        min_face_detection_confidence=float(
            cfg.get("analyze", "min_face_detection_confidence", 0.5)),
        min_face_presence_confidence=float(
            cfg.get("analyze", "min_face_presence_confidence", 0.5)),
        oob_inset=float(cfg.get("analyze", "oob_inset", 0.0)),
        # --verbose only ever turns logging on; it never silences a config that
        # asked for it.
        verbose=bool(verbose or cfg.get("analyze", "verbose", False)),
        effort=level,
        retry_margins=tuple(float(x) for x in settings["retry_margins"]),
        retry_rotations=tuple(float(x) for x in settings["retry_rotations"]),
        retry_equalize=bool(settings["retry_equalize"]),
        ensemble=int(settings["ensemble"]),
        max_crop_px=int(settings["max_crop_px"]),
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    cfg, conn = _open(args)
    pipeline.stage_analyze(conn, _analyze_options(cfg, args.verbose, args.effort),
                           int(cfg.get("analyze", "workers", 0)), log,
                           reanalyze=args.reanalyze, limit=args.limit)
    cmd_select(args)


def _manual_rejects(cfg: config.Config) -> set[str]:
    """What the contact sheet dropped by hand, if the config says where to look.

    A config without `paths.out` never needed one before `select` started
    reading this file, and must not begin failing there now.
    """
    try:
        out_dir = cfg.path("out")
    except KeyError:
        return set()
    return review.load_manual_rejects(out_dir / "rejects.json")


def _select_frames(cfg: config.Config, conn, cadence: str | None = None) -> tuple[int, int, int]:
    """Apply thresholds, pick frames, and report the filter outcome.

    Shared by `select`, `run` and `trial` so the outcome block always lands
    directly after selection. Duplicating it once let `trial` print the same
    table adrift at the very end of the run, after the timing report.

    Returns (kept, scored, frames).
    """
    # Before the filter pass, so an unusable cadence is the first thing said
    # rather than a footnote under a line of progress.
    _check_cadence(cfg, conn, cadence)

    manual = _manual_rejects(cfg)
    kept, scored = select.apply_filters(conn, cfg.section("filter"), cfg.section("score"),
                                        manual)
    log(f"  select: {kept}/{scored} pass the hard filters")
    if manual:
        log(f"  select: {len(manual)} rejected by hand in rejects.json; "
            "buckets fall through to the next best")

    cadence = _cadence(cfg, cadence)
    per_bucket = int(cfg.get("select", "per_bucket", 1))
    alternates = int(cfg.get("select", "alternates", 2))
    frames = select.select_frames(conn, cadence, per_bucket, alternates)
    log(f"  select: {frames} frames (cadence={cadence}, {per_bucket} per bucket)")
    if alternates:
        spare = conn.execute(
            "SELECT count(*) FROM selection WHERE alternate = 1").fetchone()[0]
        log(f"  select: {spare} runner-ups warped alongside them, so the contact "
            "sheet can show what a rejection would promote")
    pipeline.report_rejects(conn, log)
    return kept, scored, frames


def _cadence(cfg: config.Config, override: str | None) -> str:
    """The cadence a run will actually use: the flag if given, else the config."""
    return override or str(cfg.get("select", "cadence", "week"))


def _check_cadence(cfg: config.Config, conn, override: str | None) -> None:
    """Fail on an impossible cadence while it is still cheap to say so."""
    if _cadence(cfg, override) == select.BIRTHDAY_MONTHS:
        select.require_birth_date(conn)


def cmd_select(args: argparse.Namespace) -> None:
    cfg, conn = _open(args)
    _select_frames(cfg, conn, getattr(args, "cadence", None))


def cmd_align(args: argparse.Namespace) -> None:
    cfg, conn = _open(args)
    pipeline.stage_align(conn, cfg.path("frames"), cfg.section("output"), log,
                         int(cfg.get("analyze", "workers", 0)),
                         framing=cfg.section("align"))


def cmd_review(args: argparse.Namespace) -> None:
    cfg, conn = _open(args)
    out_dir = cfg.path("out")
    manual = _manual_rejects(cfg)
    accepted = review.write_contact_sheet(conn, out_dir / "contact-sheet.html", manual)
    rejected = review.write_rejects_gallery(conn, out_dir / "rejects.html",
                                            limits=cfg.section("filter"), manual=manual)
    log(f"  contact sheet: {accepted} frames -> {out_dir / 'contact-sheet.html'}")
    log(f"  rejects gallery: {rejected} samples -> {out_dir / 'rejects.html'}")


def cmd_encode(args: argparse.Namespace) -> None:
    cfg, conn = _open(args)
    # The same worker count `align` uses: with a footer enabled there are two
    # independent renders, and ffmpeg cannot spread either across cores itself.
    for out in pipeline.stage_encode(conn, cfg.path("out"), cfg.section("encode"), log,
                                     int(cfg.get("analyze", "workers", 0))):
        log(f"  encode: wrote {out}")


def cmd_trial(args: argparse.Namespace) -> None:
    """Run the pipeline over a sample and project the full run from it.

    A trial is a *partial real run*, not a simulation: everything it downloads
    and analyzes is written to the manifest and cache, so none of the work is
    repeated later. Running a trial simply gets you that much further along.
    """
    if args.compare:
        cmd_compare(args)
        return

    cfg, conn = _open(args)

    total_assets = db.count_assets(conn)
    if not total_assets:
        raise SystemExit("no assets indexed yet — run `grow-up index` first")

    sources = _sources(cfg, args.source)
    limit = int(args.limit or cfg.get("trial", "limit", 100))
    # Sized off the eventual population, not what is actionable right now:
    # nothing is downloaded when a trial starts, so an actionable count would
    # make the analyze projection zero.
    before = pipeline.eventual_workload(conn)
    trial = timing.Trial(sample_size=limit, total_assets=total_assets)
    log(f"== trial: sampling up to {limit} of {total_assets} indexed assets ==")

    bytes_before = conn.execute(
        "SELECT coalesce(sum(bytes), 0) FROM downloads").fetchone()[0]

    async def network_stages() -> None:
        await preflight_all(cfg, sources)

        # The sample is split across accounts rather than repeated per account,
        # so `-n 100` measures a hundred photos and the projection stays honest.
        face_shares = _apportion(limit, sources, conn, "faces")
        found = 0
        with timing.stopwatch() as faces_elapsed:
            for source, share in zip(sources, face_shares):
                async with _client(cfg, source) as client:
                    got, _ = await pipeline.stage_faces(
                        client, conn, await _resolve_person(cfg, source), log,
                        int(cfg.get("fetch", "concurrency", 16)), limit=share,
                        source=source.name,
                        label=_labelled("faces", source, sources))
                found += got
        trial.stages.append(timing.StageTiming(
            "faces", found, faces_elapsed(), before["faces"]))

        fetch_shares = _apportion(limit, sources, conn, "fetch")
        fetched = 0
        with timing.stopwatch() as fetch_elapsed:
            for source, share in zip(sources, fetch_shares):
                async with _client(cfg, source) as client:
                    fetched += await pipeline.stage_fetch(
                        client, conn, cfg.path("cache"),
                        str(cfg.get("fetch", "source", "original")), log,
                        int(cfg.get("fetch", "concurrency", 8)), limit=share,
                        account=source.name,
                        label=_labelled("fetch", source, sources))
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

    opts = _analyze_options(cfg, args.verbose, args.effort)
    with timing.stopwatch() as analyze_elapsed:
        analyzed = pipeline.stage_analyze(
            conn, opts, int(cfg.get("analyze", "workers", 0)), log, limit=limit)
    trial.stages.append(timing.StageTiming(
        "analyze", analyzed, analyze_elapsed(), before["analyze"]))

    _, scored, frames = _select_frames(cfg, conn, args.cadence)

    with timing.stopwatch() as align_elapsed:
        aligned = pipeline.stage_align(conn, cfg.path("frames"), cfg.section("output"),
                                       log, int(cfg.get("analyze", "workers", 0)),
                                       framing=cfg.section("align"))
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
            log("  encode: skipped (--no-encode)")
        else:
            try:
                with timing.stopwatch() as encode_elapsed:
                    videos = pipeline.stage_encode(
                        conn, out_dir, encode_cfg, log,
                        int(cfg.get("analyze", "workers", 0)))
                trial.stages.append(timing.StageTiming(
                    "encode", aligned, encode_elapsed(),
                    max(0, projected_frames - aligned),
                    unit="frame", note=", ".join(str(v) for v in videos)))
            except FFmpegMissing as exc:
                log(f"  ! skipping encode: {exc}")
            except RuntimeError as exc:
                log(f"  ! encode failed: {exc}")
    else:
        log("  align: no frames survived the filters, nothing to review or encode")

    for line in trial.render():
        log(line)

    if aligned:
        out_dir = cfg.path("out")
        log("")
        log(f"Look at {out_dir / 'contact-sheet.html'} to judge alignment and framing,")
        log(f"and {out_dir / 'rejects.html'} to check the thresholds are not too tight.")


def cmd_compare(args: argparse.Namespace) -> None:
    """Measure every effort level over the same sample and report the trade-off.

    Runs with `persist=False` throughout: a comparison must not leave the stored
    metrics at whichever level happened to run last.
    """
    cfg, conn = _open(args)
    limit = int(args.limit or cfg.get("trial", "limit", 100))
    limits, weights = cfg.section("filter"), cfg.section("score")
    workload = pipeline.eventual_workload(conn)["analyze"]

    ready = conn.execute(
        "SELECT count(*) FROM downloads d"
        "  JOIN faces f ON f.asset_id = d.asset_id AND f.status = 'ok'").fetchone()[0]
    if not ready:
        raise SystemExit("nothing downloaded yet — run `grow-up trial` first")

    header = (f"{'effort':<12}{'analyze':>9}{'per item':>11}"
              f"{'detected':>11}{'accepted':>11}{'projected':>12}")
    rows = [header, "-" * len(header)]

    for level in analyze.EFFORT_LEVELS:
        opts = _analyze_options(cfg, args.verbose, level)
        collected: list[tuple[str, object]] = []
        with timing.stopwatch() as elapsed:
            pipeline.stage_analyze(conn, opts, int(cfg.get("analyze", "workers", 0)),
                                   log, reanalyze=True, limit=limit,
                                   persist=False, collect=collected)
        if not collected:
            continue

        detected = sum(1 for _, m in collected if m.detected)
        accepted = sum(1 for _, m in collected
                       if metrics.hard_reject(m, limits) is None)
        stage = timing.StageTiming(level, len(collected), elapsed(), workload)
        rows.append(
            f"{level:<12}{timing.format_duration(stage.elapsed):>9}"
            f"{timing.format_duration(stage.per_item):>11}"
            f"{f'{detected}/{len(collected)}':>11}"
            f"{f'{accepted}/{len(collected)}':>11}"
            f"{timing.format_duration(stage.projected):>12}"
        )

    log("")
    for row in rows:
        log(row)
    log("")
    log("Projected covers the analyze stage only, over "
        f"{workload} assets still to analyze.")
    log("Stored metrics are untouched; set analyze.effort in config.toml, then")
    log("`grow-up analyze --reanalyze` to apply the level you pick.")


def cmd_doctor(args: argparse.Namespace) -> None:
    """Probe each endpoint once and report exactly what it returns.

    Exists because a stage that fails on every item is the worst place to learn
    what went wrong: one request, fully described, beats hundreds of identical
    error lines.
    """
    cfg, conn = _open(args)
    sources = _sources(cfg, args.source)
    # doctor deliberately skips preflight -- probing each endpoint one at a time
    # is the whole point of it -- so it needs this check of its own. Diagnosing
    # one account at a time is exactly what it should not do here.
    config.check_credentials(sources)

    async def probe_source(source: config.Source) -> None:
        # Probe with an asset this account actually owns: another account's id
        # answers 404, which would read as a broken endpoint rather than the
        # wrong key.
        row = conn.execute(
            "SELECT a.id FROM assets a JOIN faces f ON f.asset_id = a.id AND f.status = 'ok'"
            " WHERE a.source = ? LIMIT 1", (source.name,)
        ).fetchone()
        asset_id = args.asset or (row["id"] if row else None)
        person_id = source.person_id

        if not asset_id:
            log("note: no indexed asset to probe with; run `grow-up index` first "
                "or pass --asset <uuid>")
        await _probe(cfg, source, asset_id, person_id)

    async def go() -> None:
        for source in sources:
            if len(sources) > 1:
                log(f"\n== {source.name} ==")
            await probe_source(source)

    asyncio.run(go())


async def _probe(cfg: config.Config, source: config.Source,
                 asset_id: str | None, person_id: str) -> None:
    async with _client(cfg, source) as client:
        log(f"server:  {source.credentials().url}")

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
    sources = _sources(cfg)
    log(f"database: {cfg.path('db')}")
    for table in ("assets", "faces", "downloads", "metrics", "selection", "frames"):
        n = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        log(f"  {table:<12} {n:>7}")

    # Describes the same stored rows as the counts above, so it belongs with
    # them rather than after the sync bookkeeping.
    log("")
    for line in select.format_reject_summary(conn, indent="",
                                             label="filter outcome (last select)"):
        log(line)

    # One block per account, because each keeps its own watermark: a partner's
    # library being months behind is exactly what this is for.
    for source in sources:
        heading = "" if len(sources) < 2 else f" [{source.name}]"
        state = db.get_sync_state(conn, source.person_id) if source.person_id else None
        if state:
            log(f"\nwatermark{heading}   {state.watermark}")
            log(f"last run    {state.last_run_at}")
            log(f"person has  {state.person_assets} assets per Immich at that point")
        else:
            log(f"\nno watermark stored yet{heading} — the next run will be a full index")

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
    sources = _sources(cfg, args.source)

    async def network_stages() -> None:
        # Every key is checked before any of them does work: a typo in the
        # second one should not surface after the first has pulled a gigabyte.
        await preflight_all(cfg, sources)
        for source in sources:
            if len(sources) > 1:
                log(f"  -- {source.name} --")
            person_id = await _resolve_person(cfg, source)
            # --since / --full apply here and stop here.
            await _index(cfg, conn, person_id, args.since, args.full, source)
            async with _client(cfg, source) as client:
                await pipeline.stage_faces(
                    client, conn, person_id, log,
                    int(cfg.get("fetch", "concurrency", 16)),
                    source=source.name, label=_labelled("faces", source, sources))
                await pipeline.stage_fetch(
                    client, conn, cfg.path("cache"),
                    str(cfg.get("fetch", "source", "original")), log,
                    int(cfg.get("fetch", "concurrency", 8)),
                    account=source.name, label=_labelled("fetch", source, sources),
                )

    log("== index ==")
    asyncio.run(network_stages())

    # `select` runs at the end of analyze, so a cadence that cannot bucket would
    # otherwise surface an hour of landmarking later. Checked here rather than
    # before the run because `index` is what fetches the birth date in the first
    # place, and on a first run there is nothing to check until it has.
    _check_cadence(cfg, conn, getattr(args, "cadence", None))

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
    # Naming all three matters, and so does saying why. `grow-up encode` alone
    # reaches only this stage's own filter, which can drop a frame but cannot
    # reconsider a bucket -- so it leaves the week empty instead of handing it
    # to the runner-up, and says nothing about having done so.
    log("\nReview out/contact-sheet.html, drop rejects.json beside it, then re-run\n"
        "`grow-up select && grow-up align && grow-up encode` to apply them --\n"
        "select is what promotes each rejected photo's bucket to its runner-up.")


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
                       cadence=None, no_encode=False, asset=None, limit=None,
                       effort=None, compare=False, source=None)
        return p

    def add_source_flag(p: argparse.ArgumentParser) -> None:
        p.add_argument("--source", default=None, metavar="NAME",
                       help="work on one configured [[immich.sources]] account only")

    model = add("fetch-model", cmd_fetch_model, "download the MediaPipe face landmarker model")
    model.add_argument("-o", "--output", default=None)
    model.add_argument("--force", action="store_true")

    for name, func, help_text in (
        ("index", cmd_index, "enumerate the person's photos (honours the sync watermark)"),
        ("run", cmd_run, "run every stage; incremental by default"),
    ):
        p = add(name, func, help_text)
        p.add_argument("--since", default=None,
                       help="override the stored watermark (ISO-8601, e.g. 2026-01-01T00:00:00Z)")
        p.add_argument("--full", action="store_true",
                       help="ignore the stored watermark and re-index everything")
        add_source_flag(p)
        if name == "run":
            p.add_argument("--cadence", default=None, choices=list(select.CADENCES))
            p.add_argument("--no-encode", action="store_true")
            p.add_argument("--effort", default=None,
                           choices=list(analyze.EFFORT_LEVELS),
                           help="time/accuracy trade-off "
                                "(default: analyze.effort in config)")

    for name, func, help_text in (
        ("faces", cmd_faces, "fetch the person's face bounding box per asset"),
        ("fetch", cmd_fetch, "download originals"),
    ):
        p = add(name, func, help_text)
        p.add_argument("-n", "--limit", type=int, default=None,
                       help="process at most this many assets")
        add_source_flag(p)

    p = add("doctor", cmd_doctor, "probe each endpoint once and report what it returns")
    p.add_argument("--asset", default=None, help="asset UUID to probe with")
    add_source_flag(p)

    p = add("trial", cmd_trial,
            "process a sample, measure it, and project the full run")
    p.add_argument("-n", "--limit", type=int, default=None,
                   help="how many assets to sample (default: trial.limit in config)")
    p.add_argument("--cadence", default=None, choices=list(select.CADENCES))
    p.add_argument("--no-encode", action="store_true",
                   help="stop after the review pages, skipping the sample video")
    p.add_argument("--effort", default=None, choices=list(analyze.EFFORT_LEVELS),
                   help="time/accuracy trade-off (default: analyze.effort in config)")
    p.add_argument("--compare", action="store_true",
                   help="measure every effort level over the same sample instead")
    add_source_flag(p)

    p = add("analyze", cmd_analyze, "landmark faces and compute quality metrics")
    p.add_argument("--reanalyze", action="store_true", help="re-run on already-analyzed assets")
    p.add_argument("-n", "--limit", type=int, default=None,
                   help="process at most this many assets")
    p.add_argument("--effort", default=None, choices=list(analyze.EFFORT_LEVELS),
                   help="time/accuracy trade-off (default: analyze.effort in config)")

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
