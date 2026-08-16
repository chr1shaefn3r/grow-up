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
| Set the output `-r` from `encode.fps` when a transition is on | `-r` runs after the filter chain, so a rate left at the hold rate drops every frame the interpolation just synthesised. This shipped: `interpolate = true` paid for mci motion compensation for three releases and handed back the un-smoothed video. `transition_filters` returns the rate the output must carry for exactly this reason. | `TestTheOutputRateMatchesTheFilter` (`tests/test_encode.py`) |
| Leave either filter's scene detection at its default | `framerate` has `scene` (8.2) and `minterpolate` has `scd` (`fdiff`), and *both* stop blending across what they read as a cut — which is nearly every pair of photographs months apart. The filter runs, costs its time, and produces the hard cuts it was added to remove. Needs `scene=100` and `scd=none`. `scene=100` shipped first and `scd` was missed in the same function, so morph paid for mci motion compensation and returned hard cuts. | `TestMorphSceneDetection`, `test_scene_detection_is_switched_off` (`tests/test_encode.py`) |
| Assume a transition's timing reaches every branch | `crossfade` carves its hold from `framerate`'s interp window; `minterpolate` has no hold at all, so morph's comes from `concat_entries` repeating each frame — still, then moving. `transition_seconds` shipped wired to one branch and silently discarded by the other, and every morph test passed the same value and asserted only that `minterpolate` appeared. Assert that changing the value changes the output. | `TestTheTransitionLengthReachesBothBranches` (`tests/test_encode.py`) |
| Bake the footer into the frames when a transition is on | Whatever filter dissolves the picture dissolves the text with it, so the date ghosts between two values and `morph` warps the glyphs. It goes on as a second input, resampled with `fps` (repeats) and never `framerate` (blends). Its list also opens with a half-length entry: photo timestamps fall at the *end* of the dissolve leading into them, so an unshifted footer leaves the next photo on screen still carrying the previous date. | `TestTheFooterDoesNotDissolve`, `TestTheFooterSwitchesMidDissolve` (`tests/test_encode.py`) |
| Reach for `-threads` to speed up a morph | `minterpolate` does not declare slice threading, so `-threads`, `-filter_threads` and `-filter_complex_threads` all leave it on one core; x264 below it is threaded but sits starved. The only parallelism available is more processes, which is why `stage_encode` runs the plain and annotated videos together. Splitting *one* render means chunking the frame list and concatenating, and every chunk boundary is a seam nobody here can watch. | `TestTheTwoVideosRenderTogether` (`tests/test_encode.py`) |
| Let an alternate reach the video | `select` keeps `alternates` runner-ups per bucket and `align` warps them, so they have `frames` rows like anything else. The join to `selection` on `alternate = 0` in `stage_encode` is the only thing keeping them out — without it a week appears three times, and the video looks plausible. | `TestAlternatesNeverReachTheVideo` (`tests/test_encode.py`) |
| Apply `rejects.json` only at `encode` | It filters the already-selected frame list, so rejecting the photo that won a week deletes the week instead of promoting the runner-up. `select.apply_filters` takes the set; `select_frames` then does the promotion for free. And the contact sheet must be *seeded* from the file — its download replaces the whole thing, so an unseeded page silently discards every earlier decision. The same mistake in prose costs the same: `run`'s closing line told users to re-run only `encode`, which reaches the safety-net filter and leaves the week empty. Four places give this advice — keep them saying `select && align && encode`. | `TestAManualRejectPromotesTheRunnerUp` (`tests/test_select.py`), `TestAStaleRejectIsNotSilent` (`tests/test_encode.py`), `TestTheRunTellsYouHowToApplyRejects` (`tests/test_docs.py`) |
| Give the `birthday-months` bucket a label computed its own way | `months_between` decides which bucket a photo lands in; `birthday_month_start` only names it, and must be its exact inverse. Any other rule — `birth.replace(month=…)`, clamping the day, formatting the age — drifts from the grouping it claims to describe, and two buckets sharing a name means one month of the timelapse silently swallows another. The awkward input is a birthday on the 31st: February contains no such date, so the month does not open until March does. | `test_the_label_is_the_exact_inverse_of_the_grouping`, `test_a_birthday_on_the_31st_waits_for_a_month_that_has_one` (`tests/test_select.py`) |
| Take month names or number grouping from the stdlib `locale` module | It depends on which locales the host has generated, so the same config renders a different video on a Mac and on a Linux desktop. `annotate.LANGUAGES` is a table for that reason, and because `mois` is invariable in French — a rule with an `s` on the end is wrong. | `TestLanguages` (`tests/test_annotate.py`) |
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
| `select.py` | Apply the filters, score the survivors, bucket them by cadence. `birthday-months` buckets on age, borrowing `annotate.months_between` rather than restating the day-of-month rule. |
| `review.py` | The two static HTML pages. No CDN — they open over `file://`. The rejects page is a threshold tuner driven by the serialized `RULES`. |
| `annotate.py` | The date/age footer. Age arithmetic, date tokens and the five language tables are plain Python; Pillow is imported lazily, only to draw. |
| `encode.py` | The ffmpeg invocation, and the transition maths. Hold rate and playback rate are separate; the filter strings are pure arithmetic so they stay testable with no ffmpeg. |
| `progress.py` | The progress bar. Repaints on a terminal, degrades to plain lines when piped. |
| `timing.py` | Stage timing and the full-run projection behind `grow-up trial`. |

## What you can verify, and what you cannot

**You can run the whole test suite**, and should, for any change. It needs nothing but
`pytest`, `numpy` and `httpx`. Two optional installs unlock more:

```bash
pip install opencv-python-headless   # unlocks the warp and framing tests
pip install pillow                   # unlocks the footer-drawing tests
# node on PATH                       # unlocks the Python/JavaScript filter parity tests
```

With none of them, 56 tests skip and the rest still assert everything that matters.

**You cannot verify anything that needs real photographs.** mediapipe generally will not
install in a sandbox, there is no Immich instance, and no credentials will be shared.
This means no change to landmarking, framing or filtering has ever been seen working on
an actual face by the agent that wrote it. Say so plainly rather than describing such a
change as verified.

**ffmpeg is not installed either, and the tests are built so it never needs to be.** You
can assert the exact command string and every number in it; you cannot run it or see one
frame of the result. So anything about how a *filter behaves* — `framerate`'s blending,
`scene` suppressing it, `overlay`'s alpha handling, what `minterpolate` does to a face —
is read from the documentation, not observed. Write it down as such. The way to keep this
honest is the one already used: put the arithmetic in a pure function
(`encode.transition_filters`) and test that, rather than reaching for a render you cannot
watch.

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
6. `transition = "crossfade"`, then `grow-up encode`. Pictures dissolve; the footer
   switches cleanly mid-dissolve. Hard cuts throughout means `scene` suppressed the
   blend; a black band behind the text means the overlay lost its alpha; a ghosted date
   means the footer went through the filter instead of over it.

## Releasing

Four things, every time. The first two are the release; the last two are how anyone hears
about it, and they are the ones an agent forgets.

1. **Bump the version in both files** — `pyproject.toml` and `src/grow_up/__init__.py`.
   They are the only two that carry it.
2. **A `Release X.Y.Z` commit, then an annotated tag** `vX.Y.Z`, whose message is the
   project name followed by the version — read an existing one rather than guessing
   (`git tag -l --format='%(contents)' v1.4.0`). The commit body says what changed and
   why it matters, then why it is a minor rather than a major, then the test count.
3. **A GitHub release body**, in Markdown, in the shape the previous ones use: the
   problem, the fix, what could have gone wrong, what does not break for an existing
   setup, the test counts, and a closing note on authorship and on what was *not*
   verified. Hand it over to paste — the releases are authored by the owner.
4. **A short summary for the Reddit thread**, one bullet appended to the EDIT list in the
   original post. Match the bullets already there: `X.Y.Z — <capability>. <why it
   mattered>.` in 35–55 words of plain prose, no config syntax, no backticks, no error
   handling. They lead with what you can now do, not with the changelog.

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
