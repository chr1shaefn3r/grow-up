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
| **not looking at the camera** | `eyeLookIn/Out/Up/Down` blendshapes |
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
grow-up status       # manifest counts, stored watermark, recent runs
```

Every stage is also runnable on its own and skips work the manifest already records,
so interrupting any of them costs only the item in flight.

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

### Tuning the filter

Metrics are stored per asset, not just pass/fail, so retuning never re-runs the ML:

```bash
$EDITOR config.toml        # loosen a threshold under [filter]
grow-up select             # sub-second; re-applies thresholds from stored metrics
grow-up align && grow-up encode
```

Read `out/rejects.html` **before** trusting the acceptances — it samples dropped frames
grouped by reason, so an over-tight threshold is visible immediately.

### Manual review

Landmarks cannot catch sunglasses, a hand over the face, or another child mistagged as
her. `out/contact-sheet.html` shows the aligned frames in order; click to reject, save
`rejects.json` next to it, and re-run `grow-up encode`.

## Troubleshooting

```bash
grow-up doctor        # probes each endpoint once and reports exactly what it returns
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

## Notes on the output

Normalising scale by interocular distance holds head size constant, so growing up reads
as changing facial *proportions* rather than a head that inflates. This is deliberate —
absolute scale makes the face drift around the frame.

After per-frame alignment the eyes are exact but the rest of the head still wobbles
between shots, so the transform series is smoothed along the timeline with a
Savitzky-Golay filter (which preserves the slow drift rather than flattening it).

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
