# Notes for Claude Code

Project knowledge for an agent working in this repo. `README.md` is the user-facing
documentation and is current — read it first for what the tool does. This file covers
what the code cannot tell you: the invariants, the mistakes already made, and the limits
of what you are able to verify.

## Start here

`grow-up` turns the photos of one tagged person in an [Immich](https://immich.app)
library into an eye-aligned face timelapse. Python, 16 modules, no framework.

Every module opens with a docstring stating its design rationale. Read the one for the
area you are touching before changing it — most "why is it like this?" questions are
answered there.

```bash
pytest                    # the whole suite, a few seconds, no network needed
grow-up --help            # the CLI surface
```

## Ground rules

**Commit to `main` and push it there.** No feature branches and no pull requests. This
holds even when the session environment names a working branch of its own: it is the
owner's standing instruction, and this line exists so they no longer have to repeat it
every session. Fetch first, and rebase onto `origin/main` if it has moved. CI runs on
every push to `main`, so the local `pytest` run is the only gate before the change is
live — push once it passes.

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
| Discover a second account's missing environment variables only after the first account's are set | Credentials resolve when a client is built, so a per-source loop reports one account, and the next failure looks like a fresh problem rather than the rest of the same one. `config.check_credentials` reports every account up front, and it runs before `preflight_all`'s key checks because an unset variable is cheaper to find than a rejected key. `status` is deliberately outside it — it reads the manifest and must keep working offline. | `TestEveryMissingVariableIsNamedAtOnce` (`tests/test_sources.py`) |
| Ask one Immich account about another's asset | Face lookups and downloads must go to the account that owns the asset — a key that cannot see an id gets 404, so this fails on exactly the other account's half of the library. `assets.source` records the owner. | `TestStagesStayInTheirOwnAccount` (`tests/test_sources.py`) |
| Set the output `-r` from `encode.fps` when a transition is on | `-r` runs after the filter chain, so a rate left at the hold rate drops every frame the interpolation just synthesised. This shipped: `interpolate = true` paid for mci motion compensation for three releases and handed back the un-smoothed video. `transition_filters` returns the rate the output must carry for exactly this reason. | `TestTheOutputRateMatchesTheFilter` (`tests/test_encode.py`) |
| Leave either filter's scene detection at its default | `framerate` has `scene` (8.2) and `minterpolate` has `scd` (`fdiff`), and *both* stop blending across what they read as a cut — which is nearly every pair of photographs months apart. The filter runs, costs its time, and produces the hard cuts it was added to remove. Needs `scene=100` and `scd=none`. `scene=100` shipped first and `scd` was missed in the same function, so morph paid for mci motion compensation and returned hard cuts. | `TestMorphSceneDetection`, `test_scene_detection_is_switched_off` (`tests/test_encode.py`) |
| Assume a transition's timing reaches every branch | `crossfade` carves its hold from `framerate`'s interp window; `minterpolate` has no hold at all, so morph's comes from `concat_entries` repeating each frame — still, then moving. `transition_seconds` shipped wired to one branch and silently discarded by the other, and every morph test passed the same value and asserted only that `minterpolate` appeared. Assert that changing the value changes the output. | `TestTheTransitionLengthReachesBothBranches` (`tests/test_encode.py`) |
| Bake the footer into the frames when a transition is on | Whatever filter dissolves the picture dissolves the text with it, so the date ghosts between two values and `morph` warps the glyphs. It goes on as a second input, resampled with `fps` (repeats) and never `framerate` (blends). Its list also opens with a half-length entry: photo timestamps fall at the *end* of the dissolve leading into them, so an unshifted footer leaves the next photo on screen still carrying the previous date. | `TestTheFooterDoesNotDissolve`, `TestTheFooterSwitchesMidDissolve` (`tests/test_encode.py`) |
| Reach for `-threads` to speed up a morph | `minterpolate` does not declare slice threading, so `-threads`, `-filter_threads` and `-filter_complex_threads` all leave it on one core; x264 below it is threaded but sits starved. The only parallelism available is more processes, which is why `stage_encode` runs the plain and annotated videos together. Splitting *one* render means chunking the frame list and concatenating, and every chunk boundary is a seam nobody here can watch. | `TestTheTwoVideosRenderTogether` (`tests/test_encode.py`) |
| Let an alternate reach the video | `select` keeps `alternates` runner-ups per bucket and `align` warps them, so they have `frames` rows like anything else. The join to `selection` on `alternate = 0` in `stage_encode` is the only thing keeping them out — without it a week appears three times, and the video looks plausible. | `TestAlternatesNeverReachTheVideo` (`tests/test_encode.py`) |
| Apply `rejects.json` only at `encode` | It filters the already-selected frame list, so rejecting the photo that won a week deletes the week instead of promoting the runner-up. `select.apply_filters` takes the set; `select_frames` then does the promotion for free. And the contact sheet must be *seeded* from the file — its download replaces the whole thing, so an unseeded page silently discards every earlier decision. The same mistake in prose costs the same: `run`'s closing line told users to re-run only `encode`, which reaches the safety-net filter and leaves the week empty. Four places give this advice — keep them saying `select && align && encode`. | `TestAManualRejectPromotesTheRunnerUp` (`tests/test_select.py`), `TestAStaleRejectIsNotSilent` (`tests/test_encode.py`), `TestTheRunTellsYouHowToApplyRejects` (`tests/test_docs.py`) |
| Name a frame file by its place in the timeline | Places shift whenever the selection does — one rejection in a sparse month moves every later frame along by one — and anything holding an old path then shows a different photograph: a contact sheet written before the change, a tab left open on it, the manifest itself after an `align` that failed part-way (the files are written before the rows that point at them). This shipped: re-opened after `select && align && encode`, which never rewrote it, the sheet showed other photos in months nobody had touched, and a reject clicked there would have dropped a photo the user never saw. Frames are named after their asset, `align` deletes only the files it superseded, and `cmd_align` rewrites the sheet. The video was right whenever `align` finished, which is why nothing that checked it noticed. | `tests/test_frames.py` |
| Call a bucket empty because every candidate on the contact sheet is rejected | The page holds the pick and its `alternates` runner-ups; `select` holds the whole bucket and gives it to the best photo with no reject reason. This shipped: a card with all five rejected said `bucket empty` while the next run put the sixth-best — a photo nobody had seen — into the video. `selection.bucket_size` records how many passed the filters, the page says how many it is not showing and names the buckets with what to run to see them, and a selection written before the count is *unknown*, never zero. | `TestThePagePromotesTheRunnerUp`, `TestThePageSaysWhatToRunToSeeTheRest`, `TestAFullyRejectedBucketMeansWhatSelectDoes` (`tests/test_tuner.py`) |
| Fingerprint a render from the frame list and the settings alone | `align` rewrites the pixels without changing which photographs were chosen, so a reframed set has an identical frame list and would be served the old video. `frames.warped_at` is align's own record of the rewrite and is what makes the digest notice. Reading the frames to compare them would work too, at the cost of reading every frame to decide whether to read every frame. | `test_a_re_aligned_frame_invalidates_it` (`tests/test_encode.py`) |
| Let `trial` reuse an existing render | It exists to measure what encoding costs and to project the full run from that measurement. A skipped render measures nothing and projects from it, so the estimate arrives looking like very good news. `cmd_trial` passes `force=True` for that reason alone. | `TestATrialAlwaysRenders` (`tests/test_encode.py`) |
| Give the `birthday-months` bucket a label computed its own way | `months_between` decides which bucket a photo lands in; `birthday_month_start` only names it, and must be its exact inverse. Any other rule — `birth.replace(month=…)`, clamping the day, formatting the age — drifts from the grouping it claims to describe, and two buckets sharing a name means one month of the timelapse silently swallows another. The awkward input is a birthday on the 31st: February contains no such date, so the month does not open until March does. | `test_the_label_is_the_exact_inverse_of_the_grouping`, `test_a_birthday_on_the_31st_waits_for_a_month_that_has_one` (`tests/test_select.py`) |
| Take month names or number grouping from the stdlib `locale` module | It depends on which locales the host has generated, so the same config renders a different video on a Mac and on a Linux desktop. `annotate.LANGUAGES` is a table for that reason, and because `mois` is invariable in French — a rule with an `s` on the end is wrong. | `TestLanguages` (`tests/test_annotate.py`) |
| Draw into an extra screen's window from the sheet, or let that window decide anything | Over `file://` Chrome gives every load of a page an origin of its own. A blank window the sheet opened and drew into keeps the *old* load's origin, so after the reload that follows `select && align` the sheet may no longer touch it. Navigating it to a fresh `about:blank` does not help. This was built first and only a browser run caught it. The windows show `contact-sheet-screen.html` and talk to the sheet by `postMessage`. They say `ready` every second, because a message posted while the sheet unloads did not arrive. They hold no rejections and apply no rule, so nothing can drift. | `tests/test_screens_browser.py` (Chromium via Playwright; skips without it, so not in CI), `TestEveryScreenChoosesTheSameWay` (`tests/test_tuner.py`), `test_the_screen_page_decides_nothing` (`tests/test_review.py`) |
| Take whichever face the landmarker returns | MediaPipe returns faces in its own order, which is neither by size nor by who was tagged, and with `num_faces=1` that first face can be a stranger. This shipped twice, on one correctly tagged photograph with a person behind the subject. First face-morph's second look took that person from the whole aligned frame. That fix was not enough, because `analyze`'s own crop is grown for context, and the context held the stranger: in a test with a stranger overlapping the face box, the old analyze took them in 5 of 12 placements. Every look now asks for `analyze.MAX_FACES`. `analyze` keeps the face whose eyes are inside Immich's box, nearest its centre. face-morph keeps the face whose irises sit where `align` put the subject's. `align` warns, by date, about stored analyses whose eyes fall outside the box, because `analyze` keeps results from before the fix. | `tests/test_tagged_face.py`, `TestOnlyTheSubjectIsMorphed`, `TestTheSecondLookAsksForEveryFace` (`tests/test_morph.py`), `test_the_landmarker_is_told_where_the_subjects_eyes_are` (`tests/test_encode.py`) |
| Leave face-morph's code out of its render's fingerprint | A render is skipped when its inputs are unchanged, and for face-morph the code that invents the in-between frames is one of them. The fix for the background person changed no frame and no setting, so the next `grow-up run` found the old video already rendered and kept it, and the fix looked as if it had done nothing. `morph.REVISION` goes into the face-morph digest; bump it whenever the same inputs would make different frames. The other transitions' digests stay exactly as they were. | `test_a_change_to_the_morph_code_renders_again`, `test_other_transitions_are_not_re_rendered_by_it` (`tests/test_encode.py`) |
| Reuse an in-between frame that is already on disk | face-morph writes thousands of them per render. The fork it came from named them by place (`morph-000123.jpg`) and kept any that existed, which is the frame-naming trap again: a render that failed part-way, or ran before the selection changed, leaves morphs between two other photographs, and they look plausible. They are named after their two photographs, the directory is emptied before every render, and deleted after it, failed or not. The fork's neutral-background cut-outs (`neutral-000123.jpg`) repeated the mistake; here each is named after its frame, in `neutral/` beside `morph/`, under the same rules. Both directories are named before anything is written, so a morph that fails part-way is cleaned up too. | `test_nothing_left_by_an_earlier_run_survives`, `TestCuttingOutTheFrames` (`tests/test_morph.py`), `test_they_are_thrown_away_when_the_render_fails_too`, `test_a_morph_that_fails_part_way_is_cleaned_up_too` (`tests/test_encode.py`) |
| Fall back to another transition when face-morph cannot run | The render is fingerprinted as face-morph, so a fallback video of hard cuts or dissolves is kept as the finished face-morph by every later run. A missing landmarker, or no face in any frame, stops the encode. One frame with no face found dissolves only its two pairs, deterministically, and is named by date. `neutral_background` is held to the same rule, and the fork it came from was not: a missing segmenter, or a cut-out that loses the subject in every frame, stops the encode rather than rendering the photographs as they are, and one frame whose cut-out loses the subject keeps its own background and is named by date. | `test_no_face_in_any_frame_stops_rather_than_storing_dissolves`, `test_without_its_in_betweens_it_refuses`, `test_a_missing_segmenter_stops_the_encode`, `test_losing_the_subject_everywhere_stops_rather_than_storing_photographs` (`tests/test_encode.py`) |
| Cut out only the in-between frames | `neutral_background` exists to take the background away, and the held frames are most of the film. Cutting out only the morphs puts the room back for every hold and takes it away for every transition, a flicker on every photograph. `stage_encode` hands ffmpeg the cut-outs as the held frames too, in both videos, and cuts out even when the rates leave no room for an in-between. | `test_the_held_frames_are_the_cut_outs`, `test_both_videos_hold_the_cut_outs`, `test_they_are_cut_out_even_with_no_room_to_morph` (`tests/test_encode.py`) |
| Put `neutral_background` into `settings` in `stage_encode` | `settings` is fingerprinted whole and handed to ffmpeg as keyword arguments. A new key there changes the digest of every video already made, so the first run after upgrading re-renders all of them, and `encode` refuses the keyword besides. It goes into the digest through `made_by`, and only when it is on, so a face-morph made before it existed and every other transition keep their digests. | `test_off_the_face_morph_digest_is_what_it_always_was`, `test_another_transition_ignores_it_and_says_so` (`tests/test_encode.py`) |
| Build a MediaPipe segmenter in the process that starts a worker pool | In the fork, this segfaulted the parent as the pool was created, at any worker count, spawned or forked, and closing the segmenter first did not help. The symptom was only "a process in the process pool was terminated abruptly", and three renders died before a bisect found it. A landmarker does not do it, which is why landmark-then-pool always worked. It did *not* reproduce here (Linux, mediapipe 0.10.35, a landmarker and a segmenter both live in the parent, then 12 workers, spawn and fork, three runs each), so treat it as reported rather than understood. The segmenter runs in one short-lived spawned child (`morph._in_a_child`) that builds it once per render, and writes the cut-outs to disk. | `test_the_segmenter_runs_in_a_process_of_its_own`, `test_the_work_is_handed_to_a_process_of_its_own` (`tests/test_morph.py`) |
| Stop `fetch-model` at the first file already present | Everyone set up before the segmenter existed has the landmarker. The fork's version returned at "already present" before reaching the segmenter, so the command that `neutral_background`'s error tells you to run would have done nothing. Each file is checked on its own. | `TestFetchModelBringsTheSegmenter` (`tests/test_morph.py`) |
| Round a face-morph slot to whole frames | The footer advances by exactly one hold per photograph. A slot rounded to whole playback frames drifts against it by a fraction of a frame per photograph, which is a frame every few photographs at `fps = 0.7` and a visibly wrong date by the end. The in-betweens are whole frames; the still part takes up the remainder. | `test_every_slot_lasts_exactly_one_hold` (`tests/test_encode.py`) |
| Treat a stored `faces` row as a settled answer | It is a verdict about *one person*, and re-assigning a face in Immich moves neither the asset id nor its `updatedAt`. So `index` reports nothing new and `faces` skips the row, and the photo is masked for good by the answer that said no. In a fork's real library this hid 2,551 newly tagged photos behind "nothing new to fetch". The person search returning an asset outranks any stored verdict: `index` requeues the ones it contradicts (`faces.stale`), and each row records whose face it is (`faces.person_id`). | `TestTheSearchOverrulesAStoredVerdict`, `TestIndexingRequeuesWhatItCannotReportAsNew`, `TestTheHiddenPhotoReachesTheVideo` (`tests/test_retagging.py`) |
| Read a full index only for what it returns | It enumerates the person completely, so what it leaves out is evidence too. Ignored, an untagged or deleted photo, or one indexed while the account was pointed at the wrong person, stays in the video for good. Every `ok` box in that account that the listing omits is withdrawn, with its metrics. Two conditions make this unsafe, and both are guarded: an incremental window omits nearly everything, and an empty result is likelier a wrong person id. A falling person count triggers the full listing, so untagging reaches the next `run` without anyone thinking of `--full`. | `TestAFullListingSettlesWhatItLeavesOut` (`tests/test_retagging.py`), `TestDriftDetection` (`tests/test_watermark.py`) |
| Correct a face box and keep its metrics | `analyze` skips any asset that already has a metrics row, and `select` reads metrics without consulting `faces`. Numbers measured around the wrong face would outlive the correction and reach the video. `faces` drops them when a re-checked box comes back different, and only then, so a re-check that changes nothing costs no re-analysis. | `TestFacesActsOnTheRequeue` (`tests/test_retagging.py`) |
| Apply `--limit` per account instead of splitting it | `trial -n 100` would sample 200 across two accounts, and the projection multiplies a per-item cost by the whole workload — so the estimate is wrong with nothing on screen to say so. | `TestSplittingASample` (`tests/test_sources.py`) |

One more, without a test because it is a shape rather than a behaviour: EXIF orientation
is handled in exactly one place, `images.py`. Immich reports face boxes in oriented
coordinates; decoding elsewhere without the same rule silently crops the wrong region.

## Module map

| File | |
|---|---|
| `cli.py` | argparse surface, one function per command. `--since` reaches `index` and nothing else. Iterates sources for every stage that talks to Immich. |
| `config.py` | TOML loading, `REMOVED_KEYS`, credentials from the environment, and `sources()` — one Immich account each, with the pre-1.0 single-account shape as the fallback. |
| `db.py` | The SQLite manifest: schema, additive migrations, watermark, run and render bookkeeping. |
| `immich.py` | Async API client, written against the Immich OpenAPI spec 3.1.0. Permission preflight, `ImmichHTTPError`, retry policy. |
| `pipeline.py` | Stage orchestration — the only module that knows the stage order. |
| `images.py` | Decoding, and the EXIF-orientation rule in one place. Crop, rotate, equalise. |
| `analyze.py` | MediaPipe FaceLandmarker, one per worker process. Effort presets, retry ladder, ensembling, core detection. |
| `metrics.py` | Pure numpy: pose, gaze, blink, sharpness, exposure — and `RULES`, the single definition of the hard filters. |
| `align.py` | The similarity transform. Maths is pure numpy; opencv is imported lazily, only for the warp. |
| `select.py` | Apply the filters, score the survivors, bucket them by cadence. `birthday-months` buckets on age, borrowing `annotate.months_between` rather than restating the day-of-month rule. |
| `review.py` | The static HTML pages. No CDN — they open over `file://`. The rejects page is a threshold tuner driven by the serialized `RULES`. The contact sheet's compare can spread over extra monitors through `contact-sheet-screen.html`, a page that only draws what the sheet sends it. |
| `annotate.py` | The date/age footer. Age arithmetic, date tokens and the five language tables are plain Python; Pillow is imported lazily, only to draw. |
| `encode.py` | The ffmpeg invocation, and the transition maths. Hold rate and playback rate are separate; the filter strings are pure arithmetic so they stay testable with no ffmpeg. |
| `morph.py` | face-morph's in-between frames: landmark the aligned frames, warp both photographs onto the in-between mesh, dissolve. The interpolation, affines and sampling map are pure numpy; opencv does the Delaunay, the rasterising and one `remap` per photograph. Also `neutral_background`'s cut-out: MediaPipe's selfie segmenter, in a spawned process of its own, and the composite onto a plain field. |
| `progress.py` | The progress bar. Repaints on a terminal, degrades to plain lines when piped. |
| `timing.py` | Stage timing and the full-run projection behind `grow-up trial`. |

## What you can verify, and what you cannot

**You can run the whole test suite**, and should, for any change. It needs nothing but
`pytest`, `numpy` and `httpx`. Two optional installs unlock more:

```bash
pip install opencv-python-headless   # unlocks the warp and framing tests
pip install pillow                   # unlocks the footer-drawing tests
# node on PATH                       # unlocks the Python/JavaScript filter parity tests
# node that can require('playwright')  # unlocks the multi-screen browser test
```

With none of them, 182 tests skip and the rest still assert everything that matters.

**You cannot verify anything on the owner's library.** There is no Immich instance and
no credentials will be shared, so no change to landmarking, framing or filtering has
been seen on the photographs it is for. Say so plainly rather than describing such a
change as verified.

Two real faces *are* within reach, for checks that need a face rather than a library.
MediaPipe's public test images download from the sandbox (Wikimedia does not):
`https://storage.googleapis.com/mediapipe-assets/portrait.jpg` and
`.../face_stylizer_raw_face_demo.png`. mediapipe itself installs into a separate venv.
Install the version `pyproject.toml` pins (`pip install "mediapipe>=0.10.14,<1.0"`), not
the latest: 1.x crashes building the landmarker on macOS, which is why the pin exists. If
it cannot load `libEGL.so.1`, run `apt-get update && apt-get install -y libegl1
libgles2`. Then `grow-up fetch-model` downloads both models. face-morph was checked end
to end this way, on 0.10.35: landmarked, aligned, morphed and encoded. So was
`neutral_background`, through `stage_encode` itself: the cut-out edge is soft around
hair and ears, and a stranger pasted behind the subject was kept, because the segmenter
finds people, not the tagged person. Keep all of it out of the suite, which must still
run with none of it installed.

**ffmpeg may not be installed, and the tests are built so it never needs to be.** You can
assert the exact command string and every number in it. For a manual check,
`pip install imageio-ffmpeg` brings a static ffmpeg binary you can put on `PATH` and
render with, and then extract frames to look at. Without that, you cannot see one frame
of the result. So anything about how a *filter behaves* — `framerate`'s blending,
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
   full. Then untag them from a photo in the video and re-run: drift fires the other
   way, and the photo leaves the video. `grow-up status` shows `(re-check)` while a
   re-tagged photo waits for `faces`.
6. `transition = "crossfade"`, then `grow-up encode`. Pictures dissolve; the footer
   switches cleanly mid-dissolve. Hard cuts throughout means `scene` suppressed the
   blend; a black band behind the text means the overlay lost its alpha; a ghosted date
   means the footer went through the filter instead of over it.
7. Reject one month's pick on the contact sheet, save `rejects.json` beside it, run
   `grow-up select && grow-up align && grow-up encode`, and reload the page. That month
   shows its runner-up and every other month the photo it showed before. Any other
   month changing means a page is pointing at a frame by something other than its
   photograph.
8. Reject all five photos on a card in a well-photographed month. The card turns amber
   with `N more after re-run` and the top of the page names it. Save `rejects.json`,
   run `grow-up select && grow-up align`, reload: that month shows its sixth-best.
9. With `alternates = 12` and three monitors, choose `3` under screens on the contact
   sheet and put each new window on its own monitor. Compare a month: 1–4 on the sheet,
   5–8 and 9–12 beside it. Press `9` on the third monitor, then reload the sheet and
   compare again: the same two windows fill, with no new ones opening. Only headless and
   virtual-display Chromium have run this. Firefox, Safari and real monitors have not.
10. `transition = "face-morph"`, then `grow-up encode`. Between two photographs the face
    should change shape smoothly, with one set of eyes and one mouth throughout and the
    footer switching mid-morph. A face that tears or folds near the edge of the frame
    means landmarks fell outside it. Every pair dissolving instead of morphing means the
    landmarker found no face in the aligned frames. In a photo with someone else in it,
    the morph must follow the subject while the other person only dissolves. If `align`
    names photos as analyzed on someone other than the tagged person, run
    `grow-up analyze --reanalyze` before judging the morph. It has run on two MediaPipe
    test portraits, one of them with the other pasted in as a stranger, never on a
    child's photographs months apart.
11. Add `neutral_background = true` to that, and encode again. Every frame should sit on
    the same light grey, the held ones as well as the morphs. A background that comes
    back on every photograph means the held frames are not the cut-outs. Look at the
    edge of the hair and the ears: a staircase there means the feather is too small for
    your output size. A photo the run names as keeping its own background should be
    one where the face is hard to see. Anyone else in a photo stays in it. It has run
    on the same two portraits, never on a library.

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
