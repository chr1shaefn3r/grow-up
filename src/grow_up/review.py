"""Static HTML review pages.

No framework and no CDN -- these are opened straight off disk with file://, so
anything fetched over the network would simply fail. Images are referenced by
relative path rather than inlined as data URIs, which keeps the page a few KB
instead of a few hundred MB.

The rejects page is a threshold tuner: it embeds the metrics for every analyzed
photo and re-evaluates the *same rule table* the Python filter uses, so moving a
slider shows exactly which photos that change would add or drop.
"""

from __future__ import annotations

import html
import json
import math
import os
import sqlite3
from pathlib import Path

from .metrics import RULES, FaceMetrics, hard_reject

# A rejects gallery exists to judge whether thresholds are sane, and a sample
# per reason answers that. Rendering several thousand full-size originals would
# not, and would take a minute to paint.
MAX_REJECT_SAMPLES = 48

# Metric fields the rules read, plus what the page needs to draw a tile.
_METRIC_FIELDS = sorted({field for rule in RULES for field in rule.fields} | {"detected"})

_CSS = """
:root { color-scheme: light dark; --bg:#faf9f7; --fg:#1a1a1a; --muted:#6b6b6b;
        --card:#fff; --line:#e2e0dc; --bad:#c0392b; --good:#1e8449; --tile:340px; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16161a; --fg:#ececec; --muted:#9a9a9a; --card:#212127;
          --line:#33333b; --bad:#ff7a6b; --good:#5fd08a; }
}
* { box-sizing: border-box; }
body { margin:0; padding:24px; background:var(--bg); color:var(--fg);
       font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
h1 { font-size:20px; margin:0 0 4px; }
p.lede { color:var(--muted); margin:0 0 20px; max-width:66ch; }
.bar { position:sticky; top:0; z-index:5; background:var(--bg); padding:12px 0;
       border-bottom:1px solid var(--line); margin-bottom:20px; display:flex;
       gap:10px; align-items:center; flex-wrap:wrap; }
button { font:inherit; padding:7px 14px; border:1px solid var(--line);
         border-radius:7px; background:var(--card); color:var(--fg); cursor:pointer; }
button:hover { border-color:var(--muted); }
button[aria-pressed="true"] { border-color:var(--fg); font-weight:600; }
.count { color:var(--muted); }
.spacer { flex:1; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(var(--tile),1fr));
        gap:16px; align-items:start; }
figure { margin:0; background:var(--card); border:1px solid var(--line);
         border-radius:10px; overflow:hidden; position:relative; }
/* Frames are eye-aligned and identically sized, so the whole image is shown --
   never cropped to fill a cell, which would hide the very framing being judged. */
figure img { display:block; width:100%; height:auto; cursor:zoom-in; }
figcaption { padding:7px 9px; font-size:12px; color:var(--muted);
             display:flex; justify-content:space-between; align-items:center; gap:6px; }
figcaption button { padding:3px 9px; font-size:11px; border-radius:5px; }
figure.rejected { outline:2px solid var(--bad); }
figure.rejected img { opacity:.35; }
figure.rejected::after { content:"rejected"; position:absolute; top:8px; left:8px;
  background:var(--bad); color:#fff; font-size:10px; padding:2px 6px; border-radius:4px; }
textarea { width:100%; height:110px; font-family:ui-monospace,monospace; font-size:12px;
           margin-top:14px; background:var(--card); color:var(--fg);
           border:1px solid var(--line); border-radius:8px; padding:10px; }
h2 { font-size:15px; margin:28px 0 10px; border-top:1px solid var(--line); padding-top:18px; }
h2 span { color:var(--muted); font-weight:400; }

/* Threshold tuner */
#tuner { background:var(--card); border:1px solid var(--line); border-radius:12px;
         padding:16px 18px; margin-bottom:8px; }
.sliders { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
           gap:10px 24px; }
.slider { display:grid; grid-template-columns:1fr 150px 62px auto; gap:10px;
          align-items:center; font-size:13px; }
.slider .name { color:var(--muted); }
.slider output { font-variant-numeric:tabular-nums; text-align:right; }
.slider.changed .name { color:var(--fg); font-weight:600; }
.slider.changed output { color:var(--good); font-weight:600; }
.slider button { padding:2px 8px; font-size:11px; border-radius:5px; visibility:hidden; }
.slider.changed button { visibility:visible; }
input[type=range] { width:100%; accent-color:var(--fg); }
#summary { margin-top:16px; padding-top:14px; border-top:1px solid var(--line);
           display:flex; gap:22px; flex-wrap:wrap; font-size:14px; }
#summary b { font-size:17px; font-variant-numeric:tabular-nums; }
.delta-add { color:var(--good); }
.delta-drop { color:var(--bad); }
figure.added { outline:2px solid var(--good); }
figure.added::after { content:"would be added"; position:absolute; top:8px; left:8px;
  background:var(--good); color:#fff; font-size:10px; padding:2px 6px; border-radius:4px; }
figure.removed { outline:2px solid var(--bad); }
figure.removed::after { content:"would be dropped"; position:absolute; top:8px; left:8px;
  background:var(--bad); color:#fff; font-size:10px; padding:2px 6px; border-radius:4px; }
.empty { color:var(--muted); font-style:italic; padding:6px 0; }

/* Runner-ups for a bucket. Shown small: they exist to answer "what takes over if
   I drop this one", which a thumbnail settles, and the card promotes one to full
   size the moment you reject the pick. */
.alts { display:flex; gap:6px; padding:0 9px 9px; flex-wrap:wrap; }
.alt { padding:0; border-radius:6px; overflow:hidden; width:64px; line-height:0;
       border:1px solid var(--line); background:none; cursor:pointer; }
.alt img { width:100%; height:auto; cursor:pointer; }
.alt.gone { opacity:.3; border-color:var(--bad); }
.alt.now { border-color:var(--good); border-width:2px; }
figure .rank { font-variant-numeric:tabular-nums; }
figure.exhausted { outline:2px solid var(--bad); }
figure.exhausted img { opacity:.35; }

/* Full-size viewer. Every frame is the same size, so stepping through with the
   arrow keys holds the image still and turns the sequence into a flipbook --
   which is the only practical way to see alignment jitter. */
#viewer[hidden] { display:none; }
#viewer { position:fixed; inset:0; z-index:50; background:rgba(0,0,0,.92);
          display:flex; flex-direction:column; align-items:center;
          justify-content:center; gap:14px; padding:20px; }
#viewer img { max-width:min(100%,1000px); max-height:82vh; object-fit:contain;
              border-radius:6px; }
#viewer .meta { color:#f0f0f0; font-size:13px; display:flex; gap:16px;
                align-items:center; flex-wrap:wrap; justify-content:center; }
#viewer .meta kbd { background:#ffffff22; border:1px solid #ffffff33;
                    border-radius:4px; padding:1px 6px; font-size:11px; }
#viewer.rejected img { outline:3px solid var(--bad); }
"""

# Viewer plus manual rejection. Uses event delegation throughout so it keeps
# working over grids the tuner re-renders.
_JS = """
// Seeded from rejects.json by the page writer. Without this the set would start
// empty on every regeneration, and the next download -- which replaces the whole
// file -- would silently un-reject everything decided before it.
const seed = document.getElementById('rejected-seed');
const rejected = new Set(seed ? JSON.parse(seed.textContent) : []);
const viewer = document.getElementById('viewer');
const viewerImg = document.getElementById('viewer-img');
const viewerMeta = document.getElementById('viewer-meta');
let order = [];
let current = -1;

function allFigures() { return [...document.querySelectorAll('figure[data-id]')]; }

function toggleId(id) {
  if (rejected.has(id)) { rejected.delete(id); } else { rejected.add(id); }
  syncRejected();
  if (current >= 0) { paintViewer(); }
  render();
}

function toggle(fig) { toggleId(fig.dataset.id); }

// Each bucket's candidates, best first. Rejecting the current pick shows the
// next one immediately, so the grid always reflects the video the pipeline would
// produce -- otherwise judging a replacement costs a full re-run per rejection.
const bucketsEl = document.getElementById('buckets');
const BUCKETS = bucketsEl ? JSON.parse(bucketsEl.textContent) : [];
const byBucket = new Map(BUCKETS.map(b => [b.bucket, b.candidates]));

function promoteBuckets() {
  document.querySelectorAll('figure[data-bucket]').forEach(fig => {
    const candidates = byBucket.get(fig.dataset.bucket) || [];
    const index = candidates.findIndex(c => !rejected.has(c.id));
    const pick = index < 0 ? candidates[0] : candidates[index];
    if (!pick) { return; }

    fig.dataset.id = pick.id;
    fig.dataset.label = pick.date + '  #' + pick.seq;
    const img = fig.querySelector('img');
    if (img.getAttribute('src') !== pick.src) { img.setAttribute('src', pick.src); }
    const when = fig.querySelector('.when');
    if (when) { when.textContent = pick.date; }

    const rank = fig.querySelector('.rank');
    if (rank) {
      rank.textContent = index < 0 ? 'bucket empty'
        : index > 0 ? 'promoted #' + (index + 1) : '#' + pick.seq;
    }
    fig.classList.toggle('exhausted', index < 0);

    // Dim an alternate that is itself rejected, so the tray shows what is left.
    fig.querySelectorAll('.alt').forEach(alt => {
      alt.classList.toggle('gone', rejected.has(alt.dataset.alt));
      alt.classList.toggle('now', alt.dataset.alt === pick.id);
    });
  });
}

function syncRejected() {
  promoteBuckets();
  allFigures().forEach(fig => {
    // A bucket card shows whichever candidate survives, so it is only "rejected"
    // when every one of them is.
    const candidates = byBucket.get(fig.dataset.bucket);
    const on = candidates ? candidates.every(c => rejected.has(c.id))
                          : rejected.has(fig.dataset.id);
    fig.classList.toggle('rejected', on);
    const btn = fig.querySelector('figcaption button');
    if (btn) { btn.textContent = rejected.has(fig.dataset.id) ? 'keep' : 'reject'; }
  });
}

function paintViewer() {
  const fig = order[current];
  if (!fig) { return; }
  viewerImg.src = fig.querySelector('img').getAttribute('src');
  viewer.classList.toggle('rejected', rejected.has(fig.dataset.id));
  viewerMeta.innerHTML = '';
  const label = document.createElement('span');
  label.textContent = (fig.dataset.label || '') + '  ' + (current + 1) + ' / ' +
    order.length + (rejected.has(fig.dataset.id) ? '  — rejected' : '');
  const hint = document.createElement('span');
  hint.innerHTML = '<kbd>&larr;</kbd> <kbd>&rarr;</kbd> step &nbsp; ' +
                   '<kbd>r</kbd> reject &nbsp; <kbd>esc</kbd> close';
  viewerMeta.appendChild(label);
  viewerMeta.appendChild(hint);
}

function openAt(index) {
  if (!order.length) { return; }
  current = (index + order.length) % order.length;
  viewer.hidden = false;
  paintViewer();
}

function closeViewer() { viewer.hidden = true; current = -1; }

document.addEventListener('click', ev => {
  // A thumbnail in the tray toggles that candidate, not the card's current pick,
  // so a runner-up you can already see is bad can be dropped in the same pass.
  const alt = ev.target.closest && ev.target.closest('.alt');
  if (alt) { ev.stopPropagation(); toggleId(alt.dataset.alt); return; }

  const fig = ev.target.closest && ev.target.closest('figure[data-id]');
  if (!fig) { return; }
  if (ev.target.tagName === 'BUTTON') { ev.stopPropagation(); toggle(fig); return; }
  if (ev.target.tagName === 'IMG') {
    order = allFigures();
    openAt(order.indexOf(fig));
  }
});

if (viewer) {
  viewer.addEventListener('click', ev => { if (ev.target === viewer) { closeViewer(); } });
}

document.addEventListener('keydown', ev => {
  if (!viewer || viewer.hidden) { return; }
  if (ev.key === 'Escape') { closeViewer(); }
  else if (ev.key === 'ArrowRight') { openAt(current + 1); }
  else if (ev.key === 'ArrowLeft') { openAt(current - 1); }
  else if (ev.key === 'r' || ev.key === 'R') { toggle(order[current]); }
});

document.querySelectorAll('[data-tile]').forEach(btn => {
  btn.addEventListener('click', () => {
    document.documentElement.style.setProperty('--tile', btn.dataset.tile);
    document.querySelectorAll('[data-tile]').forEach(
      other => other.setAttribute('aria-pressed', String(other === btn)));
  });
});

function payload() { return JSON.stringify({rejected: [...rejected]}, null, 2); }
function render() {
  const counter = document.getElementById('n');
  if (counter) { counter.textContent = rejected.size; }
  const out = document.getElementById('out');
  if (out) { out.value = payload(); }
}

const save = document.getElementById('save');
if (save) {
  save.addEventListener('click', () => {
    const blob = new Blob([payload()], {type: 'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'rejects.json';
    a.click();
  });
}
const clear = document.getElementById('clear');
if (clear) {
  clear.addEventListener('click', () => { rejected.clear(); syncRejected(); render(); });
}
render();
"""

# The filter itself, mirroring `metrics.violates` / `metrics.hard_reject` rule for
# rule. Kept pure -- no DOM, all inputs as arguments -- so the test suite can run
# it under node against the same fixtures as the Python implementation and prove
# the two agree. A preview that disagreed with the pipeline would be worse than
# no preview at all.
_FILTER_JS = """
function violates(rule, m, limits) {
  const values = rule.fields.map(f => m[f]);
  if (rule.op === 'flag') { return Boolean(values[0]) && !limits[rule.limit]; }
  const present = values.filter(v => v !== null && v !== undefined);
  if (!present.length || !(rule.limit in limits)) { return false; }
  const limit = limits[rule.limit];
  if (rule.op === 'gt') { return present[0] > limit; }
  if (rule.op === 'lt') { return present[0] < limit; }
  if (rule.op === 'abs_gt') { return Math.abs(present[0]) > limit; }
  if (rule.op === 'max_gt') { return Math.max.apply(null, present) > limit; }
  return false;
}

function rejectReason(m, limits) {
  if (!m.detected) { return 'no_face_detected'; }
  for (const rule of RULES) { if (violates(rule, m, limits)) { return rule.reason; } }
  return null;
}
"""

# Threshold tuner UI. Interprets RULES, which is the same table
# `metrics.hard_reject` walks.
_TUNER_JS = """
const limits = Object.assign({}, BASE_LIMITS);

function tile(asset, extraClass) {
  if (!asset.path) { return ''; }
  const label = asset.date + (asset.reason ? '  ' + asset.reason : '');
  return '<figure data-id="' + asset.id + '" data-label="' + label + '" class="' +
         extraClass + '"><img loading="lazy" src="' + asset.path + '" alt="">' +
         '<figcaption><span>' + asset.date + '</span><span>' +
         (asset.reason || 'accepted') + '</span></figcaption></figure>';
}

function fill(id, assets, extraClass, emptyText) {
  const host = document.getElementById(id);
  const shown = assets.slice(0, SAMPLE_LIMIT);
  host.innerHTML = shown.length
    ? shown.map(a => tile(a, extraClass)).join('')
    : '<p class="empty">' + emptyText + '</p>';
  const counter = document.getElementById(id + '-count');
  if (counter) {
    counter.textContent = assets.length
      ? '(' + assets.length + (assets.length > shown.length
          ? ', showing ' + shown.length : '') + ')'
      : '';
  }
}

function numberFor(key, value) {
  const spec = RANGES[key];
  if (!spec) { return value; }
  const decimals = spec.step < 0.01 ? 3 : (spec.step < 1 ? 2 : 0);
  return Number(value).toFixed(decimals);
}

function tomlFor() {
  const lines = ['[filter]'];
  Object.keys(BASE_LIMITS).forEach(key => {
    const value = limits[key];
    lines.push(key + ' = ' +
      (typeof value === 'boolean' ? value : numberFor(key, value)));
  });
  return lines.join('\\n');
}

function update() {
  const accepted = [];
  const rejectedNow = [];
  const added = [];
  const dropped = [];

  ASSETS.forEach(asset => {
    const reason = rejectReason(asset, limits);
    asset.reason = reason;
    if (reason) { rejectedNow.push(asset); } else { accepted.push(asset); }
    if (reason && !asset.baseline) { dropped.push(asset); }
    if (!reason && asset.baseline) { added.push(asset); }
  });

  document.getElementById('summary').innerHTML =
    '<span>accepted <b>' + accepted.length + '</b> of ' + ASSETS.length + '</span>' +
    '<span class="delta-add">+' + added.length + ' would be added</span>' +
    '<span class="delta-drop">\\u2212' + dropped.length + ' would be dropped</span>';

  fill('added', added, 'added',
       'Nothing new passes yet — loosen a threshold above.');
  fill('removed', dropped, 'removed',
       'Nothing currently accepted would be lost.');

  const groups = {};
  rejectedNow.forEach(a => { (groups[a.reason] = groups[a.reason] || []).push(a); });
  const host = document.getElementById('reject-groups');
  host.innerHTML = Object.keys(groups).sort(
      (a, b) => groups[b].length - groups[a].length).map(reason => {
    const list = groups[reason].slice(0, SAMPLE_LIMIT);
    const extra = groups[reason].length > list.length
      ? ' \\u2014 showing ' + list.length + ' of ' + groups[reason].length : '';
    return '<h2>' + reason + ' <span>(' + groups[reason].length + extra + ')</span></h2>' +
           '<div class="grid">' + list.map(a => tile(a, '')).join('') + '</div>';
  }).join('') || '<p class="empty">Nothing is being rejected.</p>';

  document.getElementById('toml').value = tomlFor();
  syncRejected();
}

document.querySelectorAll('.slider input').forEach(input => {
  const key = input.dataset.limit;
  const row = input.closest('.slider');
  const out = row.querySelector('output');

  function apply() {
    limits[key] = input.type === 'checkbox' ? input.checked : Number(input.value);
    if (out) { out.textContent = input.type === 'checkbox' ? '' : numberFor(key, limits[key]); }
    row.classList.toggle('changed', limits[key] !== BASE_LIMITS[key]);
    update();
  }

  input.addEventListener('input', apply);
  const reset = row.querySelector('button');
  if (reset) {
    reset.addEventListener('click', () => {
      if (input.type === 'checkbox') { input.checked = BASE_LIMITS[key]; }
      else { input.value = BASE_LIMITS[key]; }
      apply();
    });
  }
});

ASSETS.forEach(a => { a.baseline = Boolean(rejectReason(a, limits)); });
update();
"""

_VIEWER_HTML = ('<div id="viewer" hidden><img id="viewer-img" alt="">'
                '<div class="meta" id="viewer-meta"></div></div>')

_SIZE_CONTROLS = ('<span class="count">size</span>'
                  '<button data-tile="220px">S</button>'
                  '<button data-tile="340px" aria-pressed="true">M</button>'
                  '<button data-tile="520px">L</button>')


def _page(title: str, lede: str, body: str, script: str = "") -> str:
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        f"<h1>{html.escape(title)}</h1>\n<p class=\"lede\">{html.escape(lede)}</p>\n"
        f"{body}\n"
        + (f"<script>{script}</script>\n" if script else "")
        + "</body>\n</html>\n"
    )


def _rel(path: str | Path, start: Path) -> str:
    return html.escape(os.path.relpath(str(path), start).replace(os.sep, "/"))


def _card(asset_id: str, path: str, when: str, label: str, base: Path,
          rejected: bool = False) -> str:
    state = ' class="rejected"' if rejected else ""
    action = "keep" if rejected else "reject"
    return (
        f'<figure{state} data-id="{html.escape(asset_id)}" '
        f'data-label="{html.escape(label)}">'
        f'<img loading="lazy" src="{_rel(path, base)}" alt="">'
        f'<figcaption><span>{html.escape(when)}</span>'
        f'<button type="button">{action}</button>'
        f'<span>{html.escape(label.split()[-1])}</span></figcaption></figure>'
    )


def _bucket_card(bucket: str, candidates: list[dict]) -> str:
    """One bucket: its current pick, with the runner-ups beneath it.

    Everything mutable is left to the page. The server renders the top candidate
    so the sheet is meaningful before any interaction, and the JS repaints from
    the same list as rejections come and go.
    """
    top = candidates[0]
    label = f"{top['date']}  #{top['seq']}"
    spares = "".join(
        f'<button type="button" class="alt" data-alt="{html.escape(spare["id"])}" '
        f'title="{html.escape(spare["date"])}">'
        f'<img loading="lazy" src="{spare["src"]}" alt="">'
        f'</button>'
        for spare in candidates[1:]
    )
    tray = f'<div class="alts" data-count="{len(candidates) - 1}">{spares}</div>' if spares else ""
    return (
        f'<figure data-bucket="{html.escape(bucket)}" '
        f'data-id="{html.escape(top["id"])}" '
        f'data-label="{html.escape(label)}">'
        f'<img loading="lazy" src="{top["src"]}" alt="">'
        f'<figcaption><span class="when">{html.escape(top["date"])}</span>'
        f'<button type="button">reject</button>'
        f'<span class="rank"></span></figcaption>'
        f"{tray}</figure>"
    )


def write_contact_sheet(conn: sqlite3.Connection, out_path: Path,
                        manual: frozenset[str] | set[str] = frozenset()) -> int:
    """Accepted frames in date order, click to toggle rejection.

    Anything already in rejects.json is seeded into the page and shown in its own
    strip below. Without that, `select` dropping those photos would leave them
    with no card, and the next download -- which replaces the whole file -- would
    quietly un-reject every one of them.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = out_path.parent
    rows = conn.execute(
        "SELECT f.asset_id, f.path, f.seq, a.local_datetime, s.bucket, s.rank,"
        "       s.alternate"
        "  FROM frames f"
        "  JOIN assets a ON a.id = f.asset_id"
        "  JOIN selection s ON s.asset_id = f.asset_id"
        " ORDER BY f.seq ASC"
    ).fetchall()

    # One entry per bucket, its candidates best first. The page walks this list
    # when you reject, so the grid always shows the video as it would be -- which
    # is the whole point: judging a replacement used to cost a full re-run.
    order: list[str] = []
    buckets: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda r: (r["bucket"], r["rank"])):
        if row["bucket"] not in buckets:
            buckets[row["bucket"]] = []
            order.append(row["bucket"])
        buckets[row["bucket"]].append({
            "id": row["asset_id"],
            "src": _rel(row["path"], base),
            "date": (row["local_datetime"] or "")[:10],
            "seq": row["seq"],
            "rank": row["rank"],
        })
    order.sort(key=lambda b: buckets[b][0]["seq"])

    cards = [_bucket_card(bucket, buckets[bucket]) for bucket in order]
    picks = [c[0] for c in buckets.values()]

    # No warped frame exists for a rejected photo -- select excluded it, so align
    # never touched it -- so these show the cached original instead.
    dropped = conn.execute(
        "SELECT a.id AS asset_id, d.path, a.local_datetime FROM assets a"
        "  JOIN downloads d ON d.asset_id = a.id"
        " ORDER BY a.local_datetime ASC"
    ).fetchall()
    dropped = [r for r in dropped if r["asset_id"] in manual]
    rejected_cards = [
        _card(row["asset_id"], row["path"], (row["local_datetime"] or "")[:10],
              f"{(row['local_datetime'] or '')[:10]}  dropped", base, rejected=True)
        for row in dropped
    ]

    seed = (f'<script type="application/json" id="rejected-seed">'
            f'{json.dumps(sorted(manual))}</script>\n'
            f'<script type="application/json" id="buckets">'
            f'{json.dumps([{"bucket": b, "candidates": buckets[b]} for b in order])}'
            f'</script>\n')
    aside = ""
    if rejected_cards:
        aside = (
            '<h2 class="dropped">Rejected by hand</h2>\n'
            '<p class="note">Their buckets fell through to the next best photo. '
            'Click one to keep it again, download rejects.json, then re-run '
            '<code>grow-up select &amp;&amp; grow-up align &amp;&amp; grow-up encode</code>.</p>\n'
            f'<div class="grid">{"".join(rejected_cards)}</div>\n'
        )

    body = (
        seed +
        '<div class="bar">'
        '<button id="save">Download rejects.json</button>'
        '<button id="clear">Clear</button>'
        '<span class="count"><b id="n">0</b> rejected</span>'
        '<span class="spacer"></span>'
        f"{_SIZE_CONTROLS}"
        "</div>\n"
        f'<div class="grid">{"".join(cards)}</div>\n'
        f"{aside}"
        '<textarea id="out" readonly></textarea>\n'
        f"{_VIEWER_HTML}"
    )
    lede = (
        "Click a frame to open it full size; the arrow keys step through the sequence "
        "without moving the image, which is how alignment jitter becomes visible. Press r "
        "(or the reject button) to drop a frame, then download rejects.json into this "
        "directory and re-run `grow-up select && grow-up align && grow-up encode` -- the "
        "bucket falls through to its next best photo rather than disappearing. This "
        "catches what landmarks cannot: sunglasses, a hand over the face, or someone "
        "else mistagged as the subject."
    )
    out_path.write_text(_page("grow-up — accepted frames", lede, body, _JS), encoding="utf-8")
    return len(picks)


def _slider_range(values: list[float], current: float, op: str) -> dict:
    """Pick a slider range from the observed data, not from a guess.

    Thresholds only mean something relative to the spread of this library's
    photos, so the track covers what actually occurs -- with the configured
    value always reachable.
    """
    pool = [abs(v) if op == "abs_gt" else v for v in values if v is not None]
    pool.append(current)
    low, high = min(min(pool), 0.0), max(pool)
    if high <= low:
        high = low + 1.0
    span = high - low
    step = 10 ** math.floor(math.log10(span / 100)) if span > 0 else 0.01
    return {"min": round(low - 0 if low >= 0 else low, 6),
            "max": round(high + step, 6),
            "step": max(step, 1e-4)}


def _collect(conn: sqlite3.Connection, out_dir: Path) -> list[dict]:
    rows = conn.execute(
        "SELECT m.*, a.local_datetime, d.path"
        "  FROM metrics m"
        "  JOIN assets a ON a.id = m.asset_id"
        "  LEFT JOIN downloads d ON d.asset_id = m.asset_id"
        " ORDER BY a.local_datetime ASC"
    ).fetchall()

    assets = []
    for row in rows:
        asset = {
            "id": row["asset_id"],
            "date": (row["local_datetime"] or "")[:10],
            "path": _rel(row["path"], out_dir) if row["path"] else "",
            # What the pipeline itself decided, used when no thresholds are
            # supplied to re-evaluate against.
            "stored_reason": row["reject_reason"],
        }
        for field in _METRIC_FIELDS:
            asset[field] = row[field] if field in row.keys() else None
        assets.append(asset)
    return assets


def write_rejects_gallery(conn: sqlite3.Connection, out_path: Path,
                          limit_per_reason: int = MAX_REJECT_SAMPLES,
                          limits: dict | None = None,
                          manual: frozenset[str] | set[str] = frozenset()) -> int:
    """Threshold tuner: sliders over the live filter, with a before/after diff.

    A static list of rejects shows what was dropped but not what a different
    threshold would buy, which is the actual question. Every analyzed photo's
    metrics are embedded, and the page re-runs the real rule table against them
    as the sliders move.

    Hand-rejected photos are left out entirely. The sliders are about thresholds,
    and offering to add back a photo you deliberately dropped would be noise --
    worse, it would read as the tuner disagreeing with the pipeline.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    limits = dict(limits or {})
    assets = [a for a in _collect(conn, out_path.parent) if a["id"] not in manual]

    # Server-render the initial grouping so the page is meaningful before any
    # interaction, and identical to what the pipeline just did.
    groups: dict[str, list[dict]] = {}
    for asset in assets:
        if limits:
            reason = hard_reject(
                FaceMetrics(**{k: v for k, v in asset.items()
                               if k in FaceMetrics.__dataclass_fields__}),
                limits,
            )
        else:
            reason = asset["stored_reason"]
        asset["reason"] = reason
        if reason:
            groups.setdefault(reason, []).append(asset)

    sections = []
    total = 0
    for reason in sorted(groups, key=lambda r: -len(groups[r])):
        sample = [a for a in groups[reason] if a["path"]][:limit_per_reason]
        total += len(sample)
        extra = (f" — showing {len(sample)} of {len(groups[reason])}"
                 if len(groups[reason]) > len(sample) else "")
        cards = "".join(
            f'<figure data-id="{html.escape(a["id"])}" '
            f'data-label="{html.escape(a["date"] + "  " + reason)}">'
            f'<img loading="lazy" src="{a["path"]}" alt="">'
            f'<figcaption><span>{html.escape(a["date"])}</span>'
            f'<span>{html.escape(reason)}</span></figcaption></figure>'
            for a in sample
        )
        sections.append(f"<h2>{html.escape(reason)} "
                        f"<span>({len(groups[reason])}{html.escape(extra)})</span></h2>"
                        f'<div class="grid">{cards}</div>')

    ranges: dict[str, dict] = {}
    rows_html = []
    for rule in RULES:
        if rule.limit in ranges or rule.limit not in limits:
            continue
        current = limits[rule.limit]
        if isinstance(current, bool):
            control = ('<input type="checkbox" data-limit="'
                       f'{rule.limit}"{" checked" if current else ""}>')
            ranges[rule.limit] = {}
        else:
            values = [a[rule.fields[0]] for a in assets]
            spec = _slider_range(values, float(current), rule.op)
            ranges[rule.limit] = spec
            control = (f'<input type="range" data-limit="{rule.limit}" '
                       f'min="{spec["min"]}" max="{spec["max"]}" '
                       f'step="{spec["step"]}" value="{current}">')
        rows_html.append(
            f'<label class="slider"><span class="name">{html.escape(rule.label)}</span>'
            f'{control}<output></output>'
            f'<button type="button">reset</button></label>'
        )

    data = (f"const RULES={json.dumps([r.__dict__ for r in RULES])};"
            f"const BASE_LIMITS={json.dumps(limits)};"
            f"const RANGES={json.dumps(ranges)};"
            f"const SAMPLE_LIMIT={limit_per_reason};"
            f"const ASSETS={json.dumps(assets)};")

    # Without thresholds there is nothing to tune, so the page degrades to the
    # static grouping rather than rendering dead controls.
    tuner_html = (
        f'<section id="tuner"><div class="sliders">{"".join(rows_html)}</div>'
        '<div id="summary"></div>'
        '<textarea id="toml" readonly></textarea></section>\n'
        '<h2>Would be added <span id="added-count"></span></h2>'
        '<div class="grid" id="added"></div>\n'
        '<h2>Would be dropped <span id="removed-count"></span></h2>'
        '<div class="grid" id="removed"></div>\n'
    ) if limits else ""

    body = (
        f'<div class="bar">{_SIZE_CONTROLS}</div>\n'
        f"{tuner_html}"
        f'<div id="reject-groups">{"".join(sections)}</div>\n'
        f"{_VIEWER_HTML}"
    )
    if limits:
        title = "grow-up — threshold tuner"
        lede = (
            "Move a slider to see exactly which photos that threshold would add or drop — "
            "the page re-runs the same filter the pipeline uses, over every analyzed photo. "
            "When it looks right, copy the [filter] block into config.toml and re-run "
            "select; no re-analysis is needed. Click any photo to see it full size."
        )
        script = data + _FILTER_JS + _JS + _TUNER_JS
    else:
        title = "grow-up — rejected frames"
        lede = (
            "A sample of what was filtered out, grouped by reason. Read this before trusting "
            "the acceptances: if good frames appear here, the corresponding threshold in "
            "config.toml is too tight. Click any photo to see it full size."
        )
        script = _JS

    out_path.write_text(_page(title, lede, body, script), encoding="utf-8")
    return total


def load_manual_rejects(path: Path) -> set[str]:
    """Read rejects.json produced by the contact sheet. Absent file means none."""
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("rejected", [])
    return {str(x) for x in data}
