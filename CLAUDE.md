# Notes for Claude Code

Project knowledge for an agent working in this repo. `README.md` is the user-facing
documentation and is current — read it first for what the tool does. This file covers
what the code cannot tell you: the invariants, the mistakes already made, and the limits
of what you are able to verify.

## Start here

`grow-up` turns the photos of one tagged person in an [Immich](https://immich.app)
library into an eye-aligned face timelapse. Python, 14 modules, no framework.

Every module opens with a docstring stating its design rationale. Read the one for the
area you are touching before changing it — most "why is it like this?" questions are
answered there.

```bash
pytest                    # the whole suite, a few seconds, no network needed
grow-up --help            # the CLI surface
```

## Ground rules

**Credentials are environment-only.** `IMMICH_URL` and `IMMICH_API_KEY`, read in
`config.credentials()`. They never go in `config.toml`, which is why that file is safe to
commit. You will not be given a host or a key — do not plan work whose only verification
is a live library.

**The test suite must keep running with nothing installed.** No network, no Immich, no
model file, no ffmpeg, and no mediapipe or opencv. `.github/workflows/tests.yml` installs
only `pytest numpy httpx`, and that is the contract: a test that imports the heavy stack
ungated will pass locally and break CI on three Python versions. Gate pixel-touching
tests with `pytest.importorskip("cv2")` (see `tests/test_framing.py`), and node-dependent
ones with `pytest.mark.skipif(shutil.which("node") is None, ...)` (see
`tests/test_tuner.py`).

Keeping maths out of the heavy dependencies is a deliberate design choice, not an
accident: `metrics.py` and the transform maths in `align.py` are plain numpy so they stay
testable in a bare environment. Preserve that when adding to them.

**Since 1.0.0, an existing `config.toml` must never break.** This rule reversed at the
release and the old text is preserved here because an agent that finds only the new rule
will not understand why `REMOVED_KEYS` exists. Before 1.0.0 the repo had one user, so a
renamed setting was simply renamed and the old name went into `config.REMOVED_KEYS` to
fail loudly. That table is now **frozen** at its pre-1.0 contents — everything in it
predates the release, so no published config can contain one. Do not add to it.

New settings arrive with a default and a fallback for their absence. `config.sources()`
is the worked example: a file with no `[[immich.sources]]` yields exactly the single
source 1.0.0 assumed, on the same two environment variables, and
`TestTheReleasedConfigStillWorks` in `tests/test_sources.py` is that promise written
down. Schema changes go through `db.ADDED_COLUMNS` so a database from an earlier version
migrates in place.

Settings must also not rot. Two tests in `tests/test_config_example.py` hold the example
file honest: every `[analyze]` key must be one the code actually reads, and every key in
the file must carry a comment.

## Traps

Each of these is a mistake that was already made here, or one the code is shaped
specifically to avoid. They all look like improvements.

| Do not | Why | Guarded by |
|---|---|---|
| Smooth or average `(tx, ty, angle, scale)` across frames | These live in each source photo's own pixel space; across one real library `tx` ranged −962 to −5458. This shipped, and put faces 848–1045px off target — outside the frame entirely. Every single-transform test passed the whole time. | `test_frames_are_solved_independently`, `test_every_eye_stays_inside_the_output_frame` (`tests/test_align.py`) |
| Thread `--since` past the `index` stage | Selection, alignment and encoding must see the whole corpus. Constraining them yields a timelapse of only the last week — a failure that produces a plausible-looking video. | `test_selection_spans_the_whole_corpus_not_just_recent_assets` (`tests/test_select.py`) |
| Switch the sync watermark to `takenAfter` | Photos imported late but taken early — restores, scans, a phone offline for a month — fall permanently behind it. It is `updatedAfter`, stores the run's *start* minus 60s, and commits only on success. | `tests/test_watermark.py` |
| Re-implement a filter rule in the page's JavaScript | `metrics.RULES` is one serialized table interpreted by both Python and `rejects.html`. Two spellings of a rule drift, and a tuner that disagrees with the pipeline is worse than no tuner. | `tests/test_tuner.py` runs the page's own filter under node against the Python one |
| Assume SMT when counting cores | Apple Silicon has no hyperthreading, so halving the logical count gives an M1 Pro 4 workers instead of 8. `analyze.physical_cores()` detects properly per platform. | `tests/test_cores.py` |
| Log `type(exc).__name__` for a failed request | Discarding the HTTP status cost an entire debugging round on a real library. `ImmichHTTPError` carries status, path and body. | `test_keeps_the_status_code` (`tests/test_client_errors.py`) |
| Clear a progress line by padding with spaces | Leaves trailing whitespace in the terminal buffer. Use the ANSI erase-to-end-of-line already in `progress.py`. | `test_summary_line_has_no_trailing_whitespace` (`tests/test_progress.py`) |
| Ask one Immich account about another's asset | Face lookups and downloads must go to the account that owns the asset — a key that cannot see an id gets 404, so this fails on exactly the other account's half of the library. `assets.source` records the owner. | `TestStagesStayInTheirOwnAccount` (`tests/test_sources.py`) |
| Apply `--limit` per account instead of splitting it | `trial -n 100` would sample 200 across two accounts, and the projection multiplies a per-item cost by the whole workload — so the estimate is wrong with nothing on screen to say so. | `TestSplittingASample` (`tests/test_sources.py`) |

One more, without a test because it is a shape rather than a behaviour: EXIF orientation
is handled in exactly one place, `images.py`. Immich reports face boxes in oriented
coordinates; decoding elsewhere without the same rule silently crops the wrong region.

## Module map

| File | |
|---|---|
| `cli.py` | argparse surface, one function per command. `--since` reaches `index` and nothing else. Iterates sources for every stage that talks to Immich. |
| `config.py` | TOML loading, `REMOVED_KEYS`, credentials from the environment, and `sources()` — one Immich account each, with the pre-1.0 single-account shape as the fallback. |
| `db.py` | The SQLite manifest: schema, additive migrations, watermark and run bookkeeping. |
| `immich.py` | Async API client, written against the Immich OpenAPI spec 3.1.0. Permission preflight, `ImmichHTTPError`, retry policy. |
| `pipeline.py` | Stage orchestration — the only module that knows the stage order. |
| `images.py` | Decoding, and the EXIF-orientation rule in one place. Crop, rotate, equalise. |
| `analyze.py` | MediaPipe FaceLandmarker, one per worker process. Effort presets, retry ladder, ensembling, core detection. |
| `metrics.py` | Pure numpy: pose, gaze, blink, sharpness, exposure — and `RULES`, the single definition of the hard filters. |
| `align.py` | The similarity transform. Maths is pure numpy; opencv is imported lazily, only for the warp. |
| `select.py` | Apply the filters, score the survivors, bucket them by cadence. |
| `review.py` | The two static HTML pages. No CDN — they open over `file://`. The rejects page is a threshold tuner driven by the serialized `RULES`. |
| `encode.py` | The ffmpeg invocation. |
| `progress.py` | The progress bar. Repaints on a terminal, degrades to plain lines when piped. |
| `timing.py` | Stage timing and the full-run projection behind `grow-up trial`. |

## What you can verify, and what you cannot

**You can run the whole test suite**, and should, for any change. It needs nothing but
`pytest`, `numpy` and `httpx`. Two optional installs unlock more:

```bash
pip install opencv-python-headless   # unlocks the warp and framing tests
# node on PATH                       # unlocks the Python/JavaScript filter parity tests
```

**You cannot verify anything that needs real photographs.** mediapipe generally will not
install in a sandbox, there is no Immich instance, and no credentials will be shared.
This means no change to landmarking, framing or filtering has ever been seen working on
an actual face by the agent that wrote it. Say so plainly rather than describing such a
change as verified.

Hand these to the human running a real library:

1. `grow-up index --since <yesterday>` — pagination works, face boxes come back.
2. `grow-up run --cadence month --no-encode`, then open `out/contact-sheet.html`. If a
   crop is not the right face, suspect the bbox coordinate mapping, and EXIF orientation
   first.
3. `grow-up run` twice back to back — the second reports a stored watermark, finds ~0 new
   assets, and produces an identical video.
4. Interrupt `grow-up index` mid-pagination — `grow-up status` shows the watermark
   *unchanged*.
5. Tag the person in an old photo and re-run — drift detection fires and re-indexes in
   full.

## Conventions

Comments say *why*, never what the line already says; several in this codebase exist
only to record why an obvious alternative was rejected. Tests are named as sentences
that state the claim (`test_a_wider_face_is_the_first_to_clip`), and are grouped in
classes by behaviour. Commit messages describe the defect and its consequence rather
than the diff.

Documentation is held honest by tests, not by discipline: `tests/test_config_example.py`
covers the config example, and `tests/test_docs.py` checks that every `grow-up <command>`
named in this file or the README is a real subcommand. If you add a claim that can rot,
consider adding the check with it.
