# grow-up

Builds an eye-aligned face timelapse from the photos of one person in a private
[Immich](https://immich.app) library. It picks the frames where the subject is looking
at the camera and warps every one of them onto the same eye positions, so the years pass
without the face jittering around the frame.

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
> deliberate and documented, but it has been reviewed rather than hand-written, and no
> claim here about how it behaves on real photographs comes from the model having seen
> any.

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
| `person.read` | resolving the person by name |
| `asset.read` | indexing their photos |
| `face.read` | reading face bounding boxes |
| `asset.download` | downloading originals |

`person.statistics` is optional — without it, watermark drift detection is skipped and
`grow-up` says so rather than disabling it silently.

Every setting lives in `config.toml`, and every setting in
[`config.example.toml`](config.example.toml) carries a comment saying what turning it
does. Nothing below asks you to guess a value.

## Usage

### Your first run

You do not have to commit to the whole library to find out whether this works on your
photos. Start small, look at what comes out, and only then spend the hours.

**1. See what there is.** Indexing downloads nothing — it just enumerates the tagged
assets, and takes seconds.

```bash
grow-up index
grow-up status            # how many assets that found
```

**2. Do a real run over a sample.** `trial` goes end to end on a few dozen photos —
download, landmark, filter, align, encode — and tells you what the full set would cost.

```bash
grow-up trial -n 50
```

**3. Look at the two pages it wrote**, in this order:

- `out/rejects.html` — what was dropped and why, with a slider per threshold. If good
  photos are sitting under a rejection reason, that threshold is too tight. Move the
  slider until the page shows the set you want, then copy the generated `[filter]` block
  into `config.toml`.
- `out/contact-sheet.html` — the frames that survived, aligned, in order. Judge framing
  here: too tight, and hair and chins are cut off; too loose, and the face is a dot.
  `[align] eye_distance` is the knob.

**4. Change one thing and see it.** Most tweaks do not re-run the machine learning,
because every metric is stored per asset:

| What you changed | What to re-run | Cost |
|---|---|---|
| `[filter]` thresholds, `[score]` weights, `[select]` cadence | `grow-up select` | sub-second |
| a rejection in `rejects.json` | `grow-up select && grow-up align && grow-up encode` | a warp pass |
| `[align]` framing, `[output]` size | `grow-up align && grow-up encode` | a warp pass |
| `analyze.effort` | `grow-up analyze --reanalyze` | a full re-analysis |

After any of them, `grow-up review` rewrites both pages so you can look again.

**5. Decide how much accuracy you want to pay for.** With a sample already downloaded,
this measures itself:

```bash
grow-up trial --compare
```

Set `analyze.effort` to whichever line you like the look of.

**6. Run the whole thing.** The trial's downloads and analyses are already banked, so
this picks up where it left off.

```bash
grow-up run
```

From then on, a bare `grow-up run` is incremental — it indexes only what changed since
the last successful run.

### Everyday commands

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

### Photos in a partner's account

Immich scopes face recognition per account. The link between a detected face and
a person belongs to whoever owns it, so a search with *your* person id never returns
photos sitting in your partner's account — even when the same child is tagged in both
libraries. If half the photos of your kid were taken on someone else's phone, half of
them are invisible to a single-account run.

The fix is to ask twice. Add a `[[immich.sources]]` block per account:

```toml
[[immich.sources]]
name = "me"
person_id = "…"                      # your account's person record

[[immich.sources]]
name = "partner"
person_id = "…"                      # *their* account's record for the same person
key_env = "IMMICH_API_KEY_PARTNER"
```

```bash
export IMMICH_API_KEY=…              # yours
export IMMICH_API_KEY_PARTNER=…      # theirs, from their own Account Settings → API Keys
```

Their key needs the same four permissions as yours. `url_env` can be set per source too,
if the two accounts are on different servers.

Everything up to and including download runs once per account; filtering, alignment and
encoding then see one merged pool of photos, so the timelapse interleaves both libraries
by date. Each account keeps its own watermark, so they stay incremental independently —
`grow-up status` shows a block per account.

Every key is checked before any of them starts work, and a failure anywhere aborts the
run. A video quietly missing one account's photos would look completely fine, which is
the kind of failure this project goes out of its way not to produce. Use
`--source NAME` on `index`, `faces`, `fetch`, `run`, `trial` or `doctor` to work on one
account at a time.

**None of this is required.** With no `[[immich.sources]]` block, `[immich] person_name`
/ `person_id` and the plain `IMMICH_URL` / `IMMICH_API_KEY` keep working exactly as
before — that is simply the one-account case, and existing databases upgrade in place.

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
- **It detects the case `updatedAfter` cannot see.** Tagging someone in an *old* photo need
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

### Date and age on the video

Off by default. Turned on, every frame gets a footer — capture date bottom left, age
bottom right — and **both videos are written**: the plain `timelapse.mp4` and
`timelapse-annotated.mp4`. A date format you end up disliking can never cost you the
clean render.

```toml
[encode.annotate]
enabled = true
age = "year_months"         # days | months | year_months | off
language = "de"             # en | de | fr | es | it
date_format = "DD.MM.YYYY"
```

```
27.08.2023                                          3 Jahre, 5 Monate
```

The age comes from the **birth date on the person in Immich** — the same field the web UI
uses to show an age. It is read during `index` and cached, so `encode` never touches the
network. If the age is switched on and Immich has no birth date, the run says so and
draws the date alone rather than failing:

```
  ! encode.annotate.age is set but Immich has no birth date for this person;
    annotating with the date only.
```

Fill it in under the person in Immich, then `grow-up index && grow-up encode`.

Some details that are deliberate:

- **The footer is readable over anything.** A translucent band darkens the bottom strip
  and the text carries an outline, so white snow and a night shot both work. Nothing
  above the band is touched.
- **`year_months` reads the way a person speaks.** `3 years, 5 months`, dropping a zero
  part — and under one month old it falls back to days, because `0 months` is wrong on
  exactly the frames where a baby changes fastest.
- **Translations are built in, not taken from the system locale**, which depends on which
  locales a machine has generated. `mois` is invariable in French, `1 día` / `2 días` in
  Spanish, `1.261` versus `1 261` for thousands — all from a table, so a Mac and a Linux
  desktop render the same video.
- **Date patterns are tokens** (`YYYY MM DD`, plus `MMMM` and `MMM` for month names), not
  strftime. `D MMMM YYYY` gives `5 August 2026` or `5 août 2026` depending on `language`.

`font` takes a path to a `.ttf` or `.otf`; left empty it picks a system face and falls
back to the one bundled with Pillow. Set it if your language needs glyphs the default
lacks.

### Manual review

Landmarks cannot catch sunglasses, a hand over the face, or someone else mistagged as
the subject. `out/contact-sheet.html` shows the aligned frames in order; click to reject,
save `rejects.json` next to it, then:

```bash
grow-up select && grow-up align && grow-up encode
```

**A rejection hands the bucket to its runner-up rather than deleting it, and the page
shows you that happening.** Each card carries its bucket's next two candidates as
thumbnails; reject the pick and the card immediately promotes the next one, labelled
`promoted #2`. So the grid always shows the video as it would be, and you can settle a
whole pass of rejections before downloading anything — rather than re-running to find out
the replacement was no good either. Reject *every* candidate and the card says
`bucket empty`, which is presumably what you meant.

Clicking a thumbnail rejects that runner-up directly, so a bad alternate you can already
see goes in the same pass.

`[select] alternates` sets how many are prepared (default 2, `0` to switch it off). They
are warped so you can judge them aligned, which is the only way to judge them — the cost
is roughly proportional in `align` time and in the frames directory. They never enter the
video.

The page opens with your existing rejections already marked, and shows them in a strip
below the accepted frames; click one to keep it again. That seeding matters: the download
replaces the whole file, so without it a second visit would quietly discard everything
you decided on the first.

`rejects.json` is the one thing in `out/` that cannot be regenerated — it exists only
because you looked at every frame. Worth keeping if you ever clear that directory.

## How it works

Two things make this tractable:

**Immich already knows which face belongs to whom.** `GET /api/faces?id=<assetId>` returns every
detected face on an asset together with its bounding box *and the person it belongs to*.
Even in a group photo, the right box comes back labelled — so there is no face
recognition in this pipeline at all. Everything downstream depends on the subject being
tagged in Immich; that tagging is the input this tool does not attempt to reproduce.

**MediaPipe FaceLandmarker gives every signal the filter needs in one pass:**

| Requirement | Signal |
|---|---|
| align to the eyes | 478 landmarks including **iris centres** (idx 468 / 473) |
| head turned away | `facial_transformation_matrixes` → yaw / pitch / roll |
| **not looking at the camera** | where each iris sits between its own eye corners |
| eyes closed / mid-blink | `eyeBlinkLeft` / `eyeBlinkRight` blendshapes |
| face partly out of frame | landmarks that fall outside the image bounds |

That fourth row is why gaze is tracked separately from pose: a face can point straight
at the lens while the eyes look somewhere else, and head pose alone cannot tell. Gaze is
measured geometrically — the iris centre's offset from the midpoint of its own eye
corners, normalised by the corner separation. That normalisation is what makes it
comparable between a close-up and a distant shot, and it costs no extra inference.

Everything heavy is native code, and the Python is glue:

| Dependency | Does |
|---|---|
| [Immich](https://immich.app) | the library itself, and the person tagging every stage builds on |
| `mediapipe` | face landmarks, head pose and blendshapes |
| `opencv-python` | affine warping, colour conversion, sharpness |
| `numpy` | the transform maths and every metric |
| `pillow` + `pillow-heif` | decoding, HEIC included |
| `httpx` | the async Immich client |
| ffmpeg (external binary) | encoding |
| SQLite (stdlib) | the manifest — what exists, what is done, what was rejected |

The manifest is what makes every stage resumable and every tweak cheap: metrics are
stored per asset rather than a pass/fail verdict, so re-filtering never re-runs the
model. Inference runs in a process pool sized to *physical* cores — detected properly,
so an M1 Pro uses 8 rather than assuming hyperthreading and halving it — and warping
runs in a thread pool, since OpenCV releases the GIL.

### Stages

A full `grow-up run` executes these in order. Each one is also a command of its own,
takes its input from the manifest, and skips whatever is already recorded there.

| # | Command | What it does |
|---|---|---|
| 1 | `grow-up index` | Enumerates the subject's assets via `POST /search/metadata`, honouring the sync watermark. Records ids, capture dates and dimensions. Downloads nothing. |
| 2 | `grow-up faces` | For each new asset, `GET /faces?id=…` and stores *their* bounding box. This is what makes group photos usable. |
| 3 | `grow-up fetch` | Downloads the originals into `paths.cache`, concurrently, retrying transient failures. Streams to disk and renames on completion, so an interrupted run leaves no truncated file. |
| 4 | `grow-up analyze` | Crops around the face box, runs FaceLandmarker, and stores the metrics: pose, gaze, blink, sharpness, exposure, iris positions, face extents. The expensive stage, and the only one that runs a model. |
| 5 | `grow-up select` | Applies the `[filter]` thresholds to those stored metrics, scores the survivors with the `[score]` weights, and keeps the best per `[select] cadence` bucket. Reads only stored numbers — no image is opened, so it is sub-second and re-runnable as often as you like. |
| 6 | `grow-up align` | Solves a similarity transform per frame that puts both eyes on the canonical positions, warps the image to `[output]` size, and writes the frames. Optionally damps brightness flicker across neighbours. |
| 7 | `grow-up review` | Writes `contact-sheet.html` (the accepted frames, in order) and `rejects.html` (the interactive threshold tuner). |
| 8 | `grow-up encode` | Feeds the frames to ffmpeg and writes the video into `paths.out`. |

`analyze` runs `select` immediately after itself, so the filter outcome is reported
where you would expect it rather than at the end of the run.

Outside that sequence:

| Command | |
|---|---|
| `fetch-model` | Downloads the FaceLandmarker bundle. Run once, at setup. |
| `trial` | Runs stages 2–8 over a sample and projects the full run. `--compare` measures the effort levels instead. |
| `status` | Manifest counts, filter outcome, stored watermark, recent runs. |
| `doctor` | Probes each Immich endpoint once and reports exactly what came back. |

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

**If `analyze` finds no face in photos where you can plainly see one**, raise
`analyze.effort` before touching the confidence thresholds — the retries re-frame the
crop, which is usually the actual problem.

MediaPipe, TFLite and the GL context log heavily from C++ on startup — fiber init,
XNNPACK delegates, feedback managers — once per worker, so eight workers means eight
copies interleaved. None of it is actionable, and it drowns the progress output, so it is
suppressed by default. `-v/--verbose` (or `analyze.verbose` in `config.toml`) brings it
back when the model itself is what needs diagnosing.

Budget disk for the cache: a few thousand originals at ~4 MB each runs to several GB.
Of the directories in `[paths]`, the database is irreplaceable and so is
`out/rejects.json` if you have curated one — everything else is derived. `frames` is
rewritten by every `align`, and `cache` costs only re-downloading.

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

### Pacing

`[select] cadence` buckets the timeline and keeps the best frame per bucket. Without it,
a photo-heavy holiday dominates the video while quiet months flash past. Together with
`encode.fps` it sets the pace: one frame per week at 10 fps covers about a year in five
seconds.

**`birthday-months` counts from the birthday instead of the 1st.** A calendar month cuts
a life at an arbitrary point — the frame that opens `2023-08` may be a day either side of
turning three and a half. With this cadence the buckets turn over on the subject's own
birthday, so each frame is one month of their age:

```toml
[select]
cadence = "birthday-months"    # born on the 14th → 14 Aug to 13 Sep is one bucket
```

It pairs with the `[encode.annotate]` footer, which counts age the same way, so the
captions advance one clean step per frame rather than drifting within the month. It needs
a birth date on the person in Immich; without one the run stops and says where to set it,
rather than quietly falling back to calendar months. A birthday on the 31st is handled the
way the age footer already handles it — February belongs to the previous month, since no
date in it is on or past the 31st.

Two rates, not one. `encode.fps` is how fast *photographs* advance — `0.5` holds each one
for two seconds. `encode.playback_fps` is the video's own frame rate, and only matters
once you ask for a transition, because a dissolve lives in the frames between two
photographs.

```toml
[encode]
fps = 0.5                    # two seconds per photo
transition = "crossfade"     # none | crossfade | morph
playback_fps = 30
transition_seconds = 1.0     # one second moving, one second still
```

**`crossfade`** is the one to reach for. Every frame it invents is a weighted average of
two real photographs, so there is nothing in it that can look wrong. **`morph`** is
motion-compensated interpolation: on eye-aligned faces it can genuinely look like one
face becoming another, but photos a month apart differ in clothing, light and background,
and motion estimation cannot tell that from movement. Expect smearing, worst in the
background, and a render measured in minutes.

`transition_seconds` governs both. Lower it to hold the picture longer: at `fps = 0.5`,
`0.5` gives a second and a half still and half a second moving. The two place it
differently — a crossfade straddles the boundary between two photographs, while a morph
holds and *then* melts into the next, because its hold is made by repeating the frame and
a repeat can only hold from the start of a slot.

`morph` is also the slow one to render, and it is bound to a single core: ffmpeg's
motion interpolation is single-threaded, so no amount of `-threads` helps it. With the
footer enabled the two videos are rendered at the same time, which halves the wait, but
the cost still scales with how many photographs there are — a weekly cadence is roughly
eight times the work of a monthly one. `grow-up encode` reports what each render took.

Note the trade. A shorter transition is a *faster* one, and morph is warping pixels along
estimated motion — half a second of warp between photographs a month apart is more
violent than a full second of it. If more stillness makes the movement uglier, that is
the mechanism, and `crossfade` is the way out: it invents nothing, so it cannot smear.

**The date and age footer does not dissolve.** It is composited after the transition, on
its own layer, and switches cleanly at the midpoint of each dissolve — where the picture
stops being mostly one photograph and starts being mostly the next. Ghosting the text
between two dates looks like a fault even when the pictures melting looks lovely.

Pacing is cheap to try: `encode` re-runs from the frames already on disk, so changing
`fps`, `transition` or `transition_seconds` costs one render and no re-warp. Changing
`[select] cadence` does not — that needs `select`, `align` and `encode`.

Worth doing the arithmetic before dropping to a monthly cadence. Over three years:

| Cadence | `fps` | Photos | Length |
|---|---|---|---|
| week | 4 | ~156 | 39s |
| week | 1 | ~156 | 2m 36s |
| month | 0.5 | ~36 | 1m 12s |
| month, `per_bucket = 2` | 0.5 | ~72 | 2m 24s |

Monthly at 0.5 fps is slower *and* shows a quarter of the photographs. If the complaint
is pace rather than density, lowering `fps` on the weekly cadence gets there without
throwing any away — and a dissolve is what makes a slow video feel deliberate rather than
merely long.

One knock-on: a dissolve puts two differently-lit photos on screen at once, so exposure
differences show far more than across a cut. `[output] flicker_match` already damps this
and defaults to `true`.

## Tests

```bash
pytest        # works straight from a checkout; no install needed
```

Every push to `main` (and every PR targeting it) runs the suite on Python 3.11, 3.12 and
3.13 via [`.github/workflows/tests.yml`](.github/workflows/tests.yml). CI installs only
`pytest`, `numpy` and `httpx` — pulling the full runtime stack would cost minutes per run
and tie the build to mediapipe's wheel availability for each Python version.

The suite needs no network, no Immich instance, no model download, and no ffmpeg — it
covers the transform maths, the metric definitions, bbox coordinate mapping, the
rejects page's filter under node against the Python one, and the watermark's failure
modes (including that timestamps match the API's `date-time` pattern verbatim from the
OpenAPI spec).

[`CLAUDE.md`](CLAUDE.md) carries what this file does not: the invariants, the mistakes
already made and the tests that guard against repeating them. Worth reading before
changing anything, whether you are a person or an agent.
