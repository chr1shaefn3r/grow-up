# grow-up

Builds an eye-aligned face timelapse from the photos of one person in a private
[Immich](https://immich.app) library.

The download is the easy part. The work is in **filtering** — dropping frames where
the face is partly out of shot or the eyes aren't on the camera — and in **aligning**
every remaining frame to the same eye positions so the video doesn't judder.

> ### Authorship
>
> **This project was written entirely by Claude**, Anthropic's AI assistant, running as
> [Claude Code](https://claude.com/claude-code). Every line of the source, the tests, the
> configuration and this README was authored by the model.
>
> The repository owner's role was supervision, not implementation: setting the goal and
> the constraints, choosing between the approaches put to them, requesting changes, and
> running the verification against a real Immich library — which the model never had
> access to, and still does not. No credentials were shared with it.
>
> Read the code with that in mind. It is unit-tested and the design decisions are
> deliberate and documented, but it has been reviewed rather than hand-written, and the
> caveats under [Verifying against a real library](#verifying-against-a-real-library)
> apply.

## How it works

Two things make this tractable:

**Immich already knows which face is hers.** `GET /api/faces?id=<assetId>` returns every
detected face on an asset together with its bounding box *and the person it belongs to*.
Even in a group photo, the right box comes back labelled — so there is no face
recognition in this pipeline at all.

**MediaPipe FaceLandmarker gives every signal the filter needs in one pass:**

| Requirement | Signal |
|---|---|
| align to the eyes | 478 landmarks including **iris centres** (idx 468 / 473) |
| head turned away | `facial_transformation_matrixes` → yaw / pitch / roll |
| **not looking at the camera** | where each iris sits between its own eye corners |
| eyes closed / mid-blink | `eyeBlinkLeft` / `eyeBlinkRight` blendshapes |
| face partly out of frame | landmarks that fall outside the image bounds |

That fourth row is why gaze is tracked separately from pose: a face can point straight
at the lens while the eyes look somewhere else, and head pose alone cannot tell.

Everything heavy — JPEG decode, inference, warping, encoding — is native code. The
Python is glue, and a process pool over physical cores keeps every core busy on both
an M1 and a desktop.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp config.example.toml config.toml   # then edit person_name
grow-up fetch-model                  # downloads face_landmarker.task (~3 MB)
```

ffmpeg must be on `PATH` (`brew install ffmpeg` / `apt install ffmpeg`).

Credentials come from the environment, never from `config.toml`, so the config file
stays safe to commit:

```bash
export IMMICH_URL=https://immich.example.com
export IMMICH_API_KEY=…            # Account Settings → API Keys
```

Immich API keys carry 155 granular permissions. `grow-up` needs four, and checks them
up front via `GET /api-keys/me` (which itself requires no permission), so a too-narrow
key fails in one second with the missing names rather than after hundreds of requests:

| Permission | Used for |
|---|---|
| `person.read` | resolving her by name |
| `asset.read` | indexing her photos |
| `face.read` | reading face bounding boxes |
| `asset.download` | downloading originals |

`person.statistics` is optional — without it, watermark drift detection is skipped and
`grow-up` says so rather than disabling it silently.

## Usage

```bash
grow-up run          # everything: index → faces → fetch → analyze → select → align → review → encode
grow-up status       # manifest counts, filter outcome, stored watermark, recent runs
```

Every stage is also runnable on its own and skips work the manifest already records,
so interrupting any of them costs only the item in flight.

`status` includes the filter outcome, so checking how much of the library survives does
not mean re-running a stage:

```
filter outcome (last select):
  accepted                       50  50.0%
  head_turned                    26  26.0%
  head_tilted                    16  16.0%
  looking_away                    7   7.0%
  blurry                          1   1.0%
```

The heading names its vintage deliberately: these verdicts are whatever the last
`grow-up select` decided, not what an edited `config.toml` would now produce. Between
`analyze` and `select` the outcome is reported as *not yet evaluated* rather than
counting unfiltered photos as accepted.

The bulk stages show live progress, with throughput and an ETA where they mean
something:

```
  fetch    [=========               ]   312/832  38%   1.2 GB at 8.5 MB/s   eta 4m 09s
```

When output is not a terminal — redirected to a file, or running under CI — the bar
degrades to occasional plain lines instead of repainting, so logs stay readable.

### Trial runs

Before committing to a few thousand photos, measure a sample:

```bash
grow-up trial -n 100      # or set trial.limit in config.toml
```

```
stage       items   elapsed   per item   projected
--------------------------------------------------
faces           -         -          -           -   (nothing pending)
fetch         100     48.3s      483ms      6m 41s   394.0 MB at 8.2 MB/s -> 3.2 GB total
analyze       100     12.4s      124ms      1m 43s
align          18      4.2s      233ms       30.8s   ~150 frames projected for the full set
review         18      0.1s        6ms        0.8s
encode         18      1.9s      106ms       14.0s   out/trial-timelapse.mp4
--------------------------------------------------
total         100    1m 07s      667ms      9m 10s

Time per picture:  649ms across all stages (100 sampled)
Full set:          8m 55s for all 832 assets
Still to go:       7m 50s (this trial already banked its own work — nothing is repeated)
```

**A trial is a partial real run, not a simulation.** It goes all the way through to a
video, so you can judge alignment and framing rather than only timings:

```
out/contact-sheet.html      the sampled frames, in order
out/rejects.html            what was filtered out, and why
out/trial-timelapse.mp4     a short video from the sampled frames
```

Everything it downloads and analyzes goes into the cache and manifest, so no work is
thrown away — running one simply gets you that much further along.

The video is written as `trial-` + the configured filename, so a two-second sample can
never silently replace a finished full render. `--no-encode` stops after the review
pages if ffmpeg is unavailable or unwanted.

The sample is deterministic and representative: ordering by asset id is stable across
runs, and since Immich ids are random UUIDs it is also uncorrelated with date,
resolution and file size — which ordering by date would not be, over years of changing
cameras.

Two projections are honest rather than convenient. `align` scales with *selected
frames*, not assets, because bucketing means a larger library yields proportionally more
frames rather than one per photo — projecting it on the asset count would overstate it
several-fold. And `fetch` reports throughput and total bytes, usually the figure that
decides whether the run is minutes or hours.

`-n/--limit` also works on `faces`, `fetch` and `analyze` individually.

### Runs are incremental by default

`run` stores a sync watermark in SQLite, so a later bare `run` picks up where the last
one finished — no flag to remember:

```bash
grow-up run                              # first run: full index
grow-up run                              # later: only what changed
grow-up run --since 2026-01-01T00:00:00Z # override
grow-up run --full                       # ignore the watermark entirely
```

Four details in there are load-bearing:

- **It filters on `updatedAfter`, not `takenAfter`.** Photos imported later but *taken*
  earlier — a backup restore, a phone offline for a month, scans of old photos — would
  fall permanently behind a `takenAfter` watermark and never be indexed.
- **It stores the run's *start* time, minus a 60s skew margin, and only on success.**
  An asset uploaded mid-run would otherwise land in the gap between the query and the
  write and be missed forever; storing the start means it is merely re-seen next run,
  which costs nothing. A crash leaves the old watermark in place.
- **It detects the case `updatedAfter` cannot see.** Tagging her in an *old* photo need
  not bump that asset's `updatedAt`. So the person's asset count from
  `/people/{id}/statistics` is stored alongside the watermark; if it climbs by more than
  the incremental query found, `grow-up` re-indexes in full and says why.
- **The watermark constrains indexing only.** Selection, alignment and encoding always
  span the whole corpus. (Threading `--since` further would yield a timelapse of only
  the last week — a failure that produces a plausible-looking video, so
  `test_selection_spans_the_whole_corpus_not_just_recent_assets` guards it.)

### Trading time for accuracy

Analysis is fast — a thousand photos in about a minute on an M1 Pro — which leaves
plenty of headroom to spend once you know what you want. `analyze.effort` spends it:

| level | cost | what it does |
|---|---|---|
| `fast` | 1× | one crop per photo, no retries |
| `balanced` | ~1.3× | retries a failed detection with wider, tighter and contrast-equalised crops |
| `thorough` | ~3.5× | adds rotated retries, and takes the median of 3 crops per face |

```bash
grow-up trial --compare        # measure all three over the same sample
```

```
effort        analyze   per item   detected   accepted   projected
fast             2.0s      203ms     88/100      46/100      2m 48s
balanced         4.8s      484ms     95/100      52/100      6m 43s
thorough        14.1s      1.4s      97/100      55/100     19m 34s
```

It runs with metrics **not** persisted, so comparing never leaves your stored analysis
at whichever level happened to run last. Pick one, set `analyze.effort`, then
`grow-up analyze --reanalyze`.

Two things the higher levels buy, and why:

- **Detection recall.** A photo where MediaPipe finds nothing is simply lost. The retries
  re-frame it — the face may extend past the crop, or background clutter may distract the
  detector, or it may be too dark. BlazeFace is trained on upright faces, so `thorough`
  also retries rotated, which rescues strongly tilted heads.
- **Landmark precision**, which feeds pose, gaze *and* the eye alignment. MediaPipe is
  deterministic on a fixed crop, so varying the framing is the only way to sample its
  error; `ensemble` takes the **median** of several looks, so one bad framing cannot drag
  the result.

Every setting a preset controls (`retry_margins`, `retry_rotations`, `retry_equalize`,
`ensemble`, `max_crop_px`) can be set individually to override just that one.

`fast` is deliberately bit-identical to the original behaviour — including leaving
`max_crop_px` off, which would otherwise speed it up. Switching level should be the only
thing that moves the numbers, which is also what makes the comparison above fair.

Gaze is measured geometrically: the iris centre's offset from the midpoint of its own eye
corners, normalised by the corner separation. That normalisation is what makes it
comparable between a close-up and a distant shot, and it costs no extra inference.

### Tuning the filter

`out/rejects.html` is an interactive threshold tuner, not just a list of what was
dropped. It carries a slider per `[filter]` threshold and, as you move one, shows
**exactly which photos that change would add or drop** — because guessing what
`max_yaw = 25` buys you is not something a static gallery can answer.

Each slider's range comes from the spread of your own library, so the track always
covers the photos that actually exist. When it looks right, copy the generated
`[filter]` block into `config.toml` and:

```bash
grow-up select             # sub-second; re-applies thresholds from stored metrics
grow-up align && grow-up encode
```

Metrics are stored per asset, not just pass/fail, so retuning never re-runs the ML.

The page re-evaluates the filter in JavaScript, which raises the obvious question of
whether the preview matches reality. Both implementations walk the *same* serialized
rule table (`metrics.RULES`) rather than each spelling the rules out, and the test
suite runs the page's own filter under node against the Python one over every rule,
both sides of each boundary, and missing-metric cases. A preview that disagreed with
the pipeline would be worse than no preview.

Read this page **before** trusting the acceptances: if good photos appear under a
rejection reason, that threshold is too tight.

### Manual review

Landmarks cannot catch sunglasses, a hand over the face, or another child mistagged as
her. `out/contact-sheet.html` shows the aligned frames in order; click to reject, save
`rejects.json` next to it, and re-run `grow-up encode`.

## Troubleshooting

```bash
grow-up doctor            # probes each endpoint once and reports exactly what it returns
grow-up -v analyze        # show the MediaPipe/TFLite native logging that is hidden by default
```

One request per endpoint, fully described — status, content type, body, and the key's
permissions:

```
  [ok  ] connectivity             200  application/json 14B
  [ok  ] key metadata             200  application/json 43B
  [ok  ] faces                    200  application/json 2B
  [ok  ] download (Accept: */*)   200  image/jpeg 2048B
  [FAIL] download (Accept: json)  406
         Not Acceptable
  [ok  ] preview                  200  image/jpeg 512B

key permissions: all (wildcard)
missing required: none
```

It deliberately probes the download endpoint **twice**, with different `Accept` headers,
because that isolates a content-negotiation fault from a permission or path fault: if
`*/*` succeeds where `json` fails, the problem was the header. Errors elsewhere carry the
status code too — 403 names the permission that endpoint requires, 406 points at content
negotiation, 404 at a server older than the API this client targets.

**If `doctor` is clean but bulk `fetch` fails**, the problem is load rather than
configuration: originals are frequently several MB each, and a reverse proxy in front of
Immich will shed load long before Immich does. Requests retry transient failures (429,
5xx, dropped connections) with exponential backoff honouring `Retry-After`; if failures
persist, lower `fetch.concurrency` in `config.toml`. The reported status says which —
429 is rate limiting, 502/503/504 is a proxy or server refusing load.

MediaPipe, TFLite and the GL context log heavily from C++ on startup — fiber init,
XNNPACK delegates, feedback managers — once per worker, so eight workers means eight
copies interleaved. None of it is actionable, and it drowns the progress output, so it is
suppressed by default. `-v/--verbose` (or `analyze.verbose` in `config.toml`) brings it
back when the model itself is what needs diagnosing.

Budget disk for the cache: a few thousand originals at ~4 MB each runs to several GB.
Downloads stream to disk and are renamed into place only when complete, so an interrupted
run never leaves a truncated file that would be mistaken for a cached one.

## Notes on the output

Normalising scale by interocular distance holds head size constant, so growing up reads
as changing facial *proportions* rather than a head that inflates. This is deliberate —
absolute scale makes the face drift around the frame.

### Framing

`[align] eye_distance` decides how much of the frame the face occupies, and so how much
room is left around it. A head is roughly 2.4–3.0× the interocular distance across, so:

| `eye_distance` | head fills | |
|---|---|---|
| 0.28 | 67–84% of the width | crops hair and chins, worst on the youngest photos |
| **0.20** | 48–60% | default: headroom, ears, some shoulders |
| 0.15 | 36–45% | half-body |

`align` checks each frame against the face extents recorded during analysis and says so
when the geometry does not fit:

```
  align: 4 frames clip the face at eye_distance=0.28; 0.25 would fit them all
```

An infant's cranium is large relative to eye spacing, so if anything still clips it will
be the earliest photos. `fit_margin` (default 1.5) allows for hair, ears and the cranium,
none of which the landmark mesh reaches — it affects only that advisory, never the
framing.

Framing changes need no re-analysis: `grow-up align && grow-up encode`. The clipping
report does need the face spans, which are recorded from `grow-up analyze` onwards; older
rows are reported as unchecked rather than assumed to fit.

`align.fill` decides what goes where the frame reaches past the edge of the source photo
— more common once the framing is loose. `edge` stretches the outermost pixels into
visible bars; `blur` (default) keeps the same content softened.

### Stabilisation

Each frame is solved independently and exactly: both eyes land on the canonical
positions every time, and that *is* the stabilisation. There is deliberately no
smoothing across frames. An earlier version averaged the `(tx, ty, angle, scale)`
series along the timeline to calm residual wobble, which was unsound — those parameters
live in each source photo's own pixel coordinate system, so across a library of mixed
resolutions and face sizes `tx` alone ranged over thousands of pixels. The average
belonged to no photo and pushed faces clean out of frame.

Nor would a corrected version buy much. What remains after exact eye alignment is
genuine head pose and expression change, which no similarity transform can smooth
away — it can only unpin the eyes while trying.

`[select] cadence` buckets the timeline and keeps the best frame per bucket. Without it,
a photo-heavy holiday dominates the video while quiet months flash past.

## Tests

```bash
pytest        # works straight from a checkout; no install needed
```

Every push to `main` (and every PR targeting it) runs the suite on Python 3.11, 3.12 and
3.13 via [`.github/workflows/tests.yml`](.github/workflows/tests.yml). CI installs only
`pytest`, `numpy` and `httpx` — pulling the full runtime stack would cost minutes per run
and tie the build to mediapipe's wheel availability for each Python version.

The suite needs no network, no Immich instance, no model download, and no ffmpeg — it
covers the transform maths, the metric definitions, bbox coordinate mapping, and the
watermark's failure modes (including that timestamps match the API's `date-time` pattern
verbatim from the OpenAPI spec).

## Verifying against a real library

1. `grow-up index --since <yesterday>` — check pagination and that face boxes come back.
2. `grow-up run --cadence month --no-encode`, then open `out/contact-sheet.html`.
   If a crop is not her face, the bbox coordinate mapping is wrong (most likely EXIF
   orientation).
3. `grow-up run` twice back to back — the second should report a stored watermark, find
   ~0 new assets, and produce an identical video.
4. Interrupt `grow-up index` mid-pagination — `grow-up status` should show the watermark
   *unchanged*.
5. Tag her in an old photo and re-run — the drift check should fire and re-index in full.
