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
        --card:#fff; --line:#e2e0dc; --bad:#c0392b; --good:#1e8449; --wait:#b7791f;
        --tile:340px; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16161a; --fg:#ececec; --muted:#9a9a9a; --card:#212127;
          --line:#33333b; --bad:#ff7a6b; --good:#5fd08a; --wait:#f0b429; }
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
/* Two fixed columns rather than a row that divides itself up. The width of a
   cell is then a property of the card, not of how many runner-ups happen to be
   prepared: four fill the 2x2, three leave the last cell empty, one sits in the
   corner -- and every one of them is the same size it would be if there were
   four. A row of `flex:1 1 0` had the opposite behaviour, growing the pictures
   as alternates were dropped, so the same photo changed size between buckets.
   min-width:0 keeps an image from forcing its cell wider than the track. */
.alts { display:grid; grid-template-columns:repeat(2,1fr); gap:6px;
        padding:0 9px 9px; align-items:start; }
.alt { padding:0; border-radius:6px; overflow:hidden; min-width:0;
       line-height:0; border:1px solid var(--line); background:none; cursor:pointer; }
.alt img { width:100%; height:auto; cursor:pointer; }
.alt.gone { opacity:.3; border-color:var(--bad); }
.alt.now { border-color:var(--good); border-width:2px; }
figure .rank { font-variant-numeric:tabular-nums; }
figure.exhausted { outline:2px solid var(--bad); }
figure.exhausted img { opacity:.35; }
/* Every candidate on the card is rejected, but more photos in its bucket passed
   the filters than the page was given, and the next select promotes the best
   of them. Amber rather than red: the bucket is not lost, it has photos nobody
   has looked at yet. After .rejected, so it wins over it. */
figure.waiting { outline:2px solid var(--wait); }
figure.waiting img { opacity:.35; }
figure.waiting::after { content:"more to see"; position:absolute; top:8px; left:8px;
  background:var(--wait); color:#1a1a1a; font-size:10px; padding:2px 6px;
  border-radius:4px; }
.notice { border:1px solid var(--wait); border-left-width:4px; border-radius:8px;
          padding:10px 12px; margin:0 0 16px; max-width:80ch; }

/* Full-size viewer. Every frame is the same size, so stepping through with the
   arrow keys holds the image still and turns the sequence into a flipbook --
   which is the only practical way to see alignment jitter. */
#viewer[hidden] { display:none; }
#viewer { position:fixed; inset:0; z-index:50; background:rgba(0,0,0,.92);
          display:flex; flex-direction:column; align-items:center;
          justify-content:center; gap:14px; padding:20px; }
#viewer .stage { display:flex; gap:18px; align-items:center; justify-content:center;
                 max-width:100%; min-height:0; }
#viewer img { max-width:min(100%,1000px); max-height:82vh; object-fit:contain;
              border-radius:6px; min-width:0; }
/* Why this photo was filed where it was: every check, its measurement and the
   current threshold. The category alone cannot say which slider to move --
   looking_away is two checks, and a photo can fail several at once. */
#viewer-details[hidden] { display:none; }
#viewer-details { flex:none; width:360px; max-height:82vh; overflow:auto;
                  background:#1d1d22; color:#e8e8e8; border-radius:8px;
                  padding:12px 14px; font-size:12px; }
#viewer-details h3 { margin:0 0 4px; font-size:14px; }
#viewer-details p { margin:0 0 10px; color:#a8a8a8; }
#viewer-details table { width:100%; border-collapse:collapse; }
#viewer-details td { padding:4px 4px; border-top:1px solid #ffffff14;
                     vertical-align:top; font-variant-numeric:tabular-nums; }
#viewer-details td.num { text-align:right; white-space:nowrap; }
#viewer-details .why { display:block; color:#8a8a8a; font-size:11px; }
#viewer-details tr.fail td { color:#ff7a6b; background:#ff7a6b1a; font-weight:600; }
#viewer-details tr.fail .why { color:#ff7a6b; font-weight:400; }
#viewer-details tr.skip td { color:#777; }
@media (max-width:800px) {
  #viewer .stage { flex-direction:column; }
  #viewer img { max-height:38vh; }
  #viewer-details { width:100%; max-height:44vh; }
}
#viewer .meta { color:#f0f0f0; font-size:13px; display:flex; gap:16px;
                align-items:center; flex-wrap:wrap; justify-content:center; }
#viewer .meta kbd { background:#ffffff22; border:1px solid #ffffff33;
                    border-radius:4px; padding:1px 6px; font-size:11px; }
#viewer.rejected img { outline:3px solid var(--bad); }

/* Side-by-side comparison of a bucket's runner-ups. The tray answers "is there
   a better one in here"; past a thumbnail that question needs them large and
   next to each other, because the differences that decide it -- a half-closed
   eye, which way the head is turned -- do not survive 46px. Above the viewer's
   z-index so opening it from there lands on top rather than behind. */
#compare[hidden] { display:none; }
#compare { position:fixed; inset:0; z-index:60; background:rgba(0,0,0,.94);
           display:flex; flex-direction:column; gap:12px; padding:20px; }
#compare .head { color:#f0f0f0; font-size:13px; display:flex; gap:16px;
                 align-items:center; flex-wrap:wrap; justify-content:center; }
#compare .head kbd { background:#ffffff22; border:1px solid #ffffff33;
                     border-radius:4px; padding:1px 6px; font-size:11px; }
/* Two columns, so the default four land as the 2x2 they were asked for and any
   other count still pairs up rather than becoming a strip. min-height:0 on both
   the track and the cell is what lets the images shrink to fit the screen
   instead of pushing the grid taller than the viewport. */
#compare .pane { flex:1; min-height:0; display:grid; gap:12px;
                 grid-template-columns:repeat(2,1fr); grid-auto-rows:1fr; }
#compare .choice { padding:0; min-height:0; display:flex; flex-direction:column;
                   gap:6px; background:none; cursor:pointer; border-radius:8px;
                   border:2px solid #ffffff22; overflow:hidden; color:#f0f0f0;
                   font:inherit; }
#compare .choice:hover, #compare .choice:focus-visible { border-color:#fff; }
#compare .choice img { width:100%; flex:1; min-height:0; object-fit:contain; }
#compare .choice .cap { padding:0 8px 6px; font-size:12px; display:flex;
                        gap:8px; justify-content:center; align-items:center; }
#compare .choice.gone { opacity:.4; }
#compare .choice.gone .cap::after { content:"rejected"; color:var(--bad); }
#compare .choice.now { border-color:var(--good); }
#compare .choice.now .cap::after { content:"current pick"; color:var(--good); }
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
const viewerDetails = document.getElementById('viewer-details');
// Rejecting and comparing only exist on the contact sheet. The rejects page
// shares this viewer, and advertising keys that do nothing there -- or that
// mark a photo in a set it has no way to save -- is worse than no hint.
const curating = Boolean(viewer && viewer.dataset.curate === 'true');
// Set by the tuner to explain a photo; the contact sheet leaves it unset.
let describe = null;
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
// Photos in each bucket that passed the filters but are not on this page; null
// when the selection predates `select` counting them.
const unseen = new Map(BUCKETS.map(b => [b.bucket, b.more]));

// What a bucket's card shows. The page holds the pick and its runner-ups;
// `select` holds the whole bucket and gives it to the best photo with no
// reject reason. So rejecting every candidate here empties the bucket only
// when nothing else passed the filters -- otherwise the next encode puts in a
// photo this page never showed, and calling the bucket empty hid exactly that.
// Pure, so the test suite can run it under node.
function bucketState(bucket) {
  const candidates = candidatesFor(bucket);
  const more = unseen.get(bucket);
  // Not `more >= 0`: null passes that in JavaScript.
  const known = typeof more === 'number';
  const index = candidates.findIndex(c => !rejected.has(c.id));
  const pick = index < 0 ? candidates[0] : candidates[index];
  const out = index < 0;
  return {
    bucket: bucket, pick: pick, index: index, more: more,
    empty: out && known && more === 0,
    waiting: out && known && more > 0,
    unknown: out && !known,
    label: index > 0 ? 'promoted #' + (index + 1)
      : !out ? '#' + pick.seq
      : !known ? 'all rejected'
      : more === 0 ? 'bucket empty'
      : more + ' more after re-run',
  };
}

// The advice for those buckets, said once at the top and naming them: whether
// to look before encoding is a decision about the run, and the cards it is
// about may be a long scroll away.
function moreNotice(states) {
  const flagged = states.filter(s => s.waiting || s.unknown);
  if (!flagged.length) { return ''; }
  const one = flagged.length === 1;
  const names = flagged.map(s => s.bucket).join(', ');
  const subject = (one ? names + ' has' : flagged.length + ' buckets (' + names + ') have') +
    ' every photo on this page rejected';
  const advice = ' download rejects.json into this directory, run ' +
    '`grow-up select && grow-up align`, and reload this page.';
  if (flagged.every(s => s.waiting)) {
    return subject + ', but more passed the filters. The next `grow-up select` hands ' +
      (one ? 'it' : 'each') + ' to the best of those, and `grow-up encode` would put ' +
      'that photo in the video without your having seen it. To see them here first,' +
      advice;
  }
  return subject + '. This page was written before select counted what else passed ' +
    'the filters, so the next `grow-up select` may hand ' + (one ? 'it' : 'them') +
    ' to a photo you have not seen, and `grow-up encode` would put that in the video. ' +
    'To see what is left first,' + advice;
}

function promoteBuckets() {
  const states = [];
  document.querySelectorAll('figure[data-bucket]').forEach(fig => {
    const state = bucketState(fig.dataset.bucket);
    const pick = state.pick;
    if (!pick) { return; }
    states.push(state);

    fig.dataset.id = pick.id;
    fig.dataset.label = pick.date + '  #' + pick.seq;
    const img = fig.querySelector('img');
    if (img.getAttribute('src') !== pick.src) { img.setAttribute('src', pick.src); }
    const when = fig.querySelector('.when');
    if (when) { when.textContent = pick.date; }

    const rank = fig.querySelector('.rank');
    if (rank) { rank.textContent = state.label; }
    fig.classList.toggle('exhausted', state.empty);
    fig.classList.toggle('waiting', state.waiting);

    // Dim an alternate that is itself rejected, so the tray shows what is left.
    fig.querySelectorAll('.alt').forEach(alt => {
      alt.classList.toggle('gone', rejected.has(alt.dataset.alt));
      alt.classList.toggle('now', alt.dataset.alt === pick.id);
    });
  });

  const note = document.getElementById('more-note');
  if (note) {
    note.textContent = moreNotice(states);
    note.hidden = !note.textContent;
  }
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
    // By class, not position: a card with runner-ups has its compare button
    // first, and relabelling "the first button" turned it into a second reject.
    const btn = fig.querySelector('figcaption button.reject');
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
    (curating ? '<kbd>r</kbd> reject &nbsp; ' : '') +
    (curating && candidatesFor(fig.dataset.bucket).length > 1
      ? '<kbd>c</kbd> compare &nbsp; ' : '') +
    '<kbd>esc</kbd> close';
  viewerMeta.appendChild(label);
  viewerMeta.appendChild(hint);
  if (viewerDetails) {
    viewerDetails.innerHTML = '';
    const panel = describe ? describe(fig.dataset.id) : null;
    if (panel) { viewerDetails.appendChild(panel); }
    viewerDetails.hidden = !panel;
  }
}

function openAt(index) {
  if (!order.length) { return; }
  current = (index + order.length) % order.length;
  viewer.hidden = false;
  paintViewer();
}

function closeViewer() { viewer.hidden = true; current = -1; }

// -- comparing a bucket's runner-ups -------------------------------------- //

const compare = document.getElementById('compare');
let comparing = null;

function candidatesFor(bucket) { return byBucket.get(bucket) || []; }

// Choosing a runner-up means dropping everything ranked ahead of it. That is
// exactly what `select` will do on the next run -- it takes the best candidate
// with no reject reason -- so the page must say the same thing rather than
// inventing a "chosen" state the pipeline has no column for.
function chooseCandidate(bucket, id) {
  const candidates = candidatesFor(bucket);
  const target = candidates.findIndex(c => c.id === id);
  if (target < 0) { return; }
  // The bucket ends up with exactly the photos ahead of the choice rejected:
  // choosing runner-up n rejects n, however many picks came before it. Only
  // adding left an earlier pick's rejections behind, so choosing #3 and then
  // #1 kept #2 rejected too. Clearing everything from the choice on is also
  // how picking a photo that was itself rejected takes it back.
  candidates.forEach((c, i) => {
    if (i < target) { rejected.add(c.id); } else { rejected.delete(c.id); }
  });
  syncRejected();
  render();
  paintCompare();
}

function paintCompare() {
  if (!compare || comparing === null) { return; }
  const candidates = candidatesFor(comparing);
  const spares = candidates.slice(1);
  const state = bucketState(comparing);
  const pick = state.index < 0 ? null : state.pick;

  const head = document.createElement('div');
  head.className = 'head';
  const label = document.createElement('span');
  label.textContent = spares.length + ' runner-up' + (spares.length === 1 ? '' : 's')
    + ' for ' + comparing + (pick ? '  —  showing ' + pick.date : '  —  ' + state.label);
  const hint = document.createElement('span');
  hint.innerHTML = '<kbd>1</kbd>…<kbd>' + Math.min(spares.length, 9) + '</kbd> choose'
    + ' &nbsp; <kbd>esc</kbd> close';
  head.appendChild(label);
  head.appendChild(hint);

  const pane = document.createElement('div');
  pane.className = 'pane';
  spares.forEach((spare, n) => {
    const cell = document.createElement('button');
    cell.type = 'button';
    cell.className = 'choice';
    cell.dataset.choose = spare.id;
    if (rejected.has(spare.id)) { cell.classList.add('gone'); }
    if (pick && spare.id === pick.id) { cell.classList.add('now'); }
    const img = document.createElement('img');
    img.src = spare.src;
    img.alt = '';
    const cap = document.createElement('span');
    cap.className = 'cap';
    cap.textContent = (n + 1) + '.  ' + spare.date;
    cell.appendChild(img);
    cell.appendChild(cap);
    pane.appendChild(cell);
  });

  compare.innerHTML = '';
  compare.appendChild(head);
  compare.appendChild(pane);
}

function openCompare(bucket) {
  if (!compare || candidatesFor(bucket).length < 2) { return; }
  comparing = bucket;
  compare.hidden = false;
  paintCompare();
}

function closeCompare() {
  if (!compare) { return; }
  compare.hidden = true;
  comparing = null;
}

// A choice is the end of the decision, so it goes straight back to the sheet.
// Compare can be opened from the full-size viewer (c), and closing only the
// overlay left that viewer standing, one Escape short of where you started.
function choose(id) {
  chooseCandidate(comparing, id);
  closeCompare();
  closeViewer();
}

document.addEventListener('click', ev => {
  const choice = ev.target.closest && ev.target.closest('[data-choose]');
  if (choice && comparing !== null) { choose(choice.dataset.choose); return; }

  // A thumbnail in the tray toggles that candidate, not the card's current pick,
  // so a runner-up you can already see is bad can be dropped in the same pass.
  const alt = ev.target.closest && ev.target.closest('.alt');
  if (alt) { ev.stopPropagation(); toggleId(alt.dataset.alt); return; }

  const fig = ev.target.closest && ev.target.closest('figure[data-id]');
  if (!fig) { return; }
  if (ev.target.classList && ev.target.classList.contains('compare-open')) {
    ev.stopPropagation();
    openCompare(fig.dataset.bucket);
    return;
  }
  if (ev.target.tagName === 'BUTTON') { ev.stopPropagation(); toggle(fig); return; }
  if (ev.target.tagName === 'IMG') {
    order = allFigures();
    openAt(order.indexOf(fig));
  }
});

if (viewer) {
  viewer.addEventListener('click', ev => { if (ev.target === viewer) { closeViewer(); } });
}
if (compare) {
  compare.addEventListener('click', ev => { if (ev.target === compare) { closeCompare(); } });
}

document.addEventListener('keydown', ev => {
  // Compare sits above the viewer, so it answers for the keyboard first --
  // otherwise Escape would close the thing underneath the one you are looking at.
  if (compare && !compare.hidden) {
    if (ev.key === 'Escape') { closeCompare(); return; }
    const n = Number(ev.key);
    if (n >= 1 && n <= 9) {
      const spares = candidatesFor(comparing).slice(1);
      if (spares[n - 1]) { choose(spares[n - 1].id); }
    }
    return;
  }
  if (!viewer || viewer.hidden) { return; }
  if (ev.key === 'Escape') { closeViewer(); }
  else if (ev.key === 'ArrowRight') { openAt(current + 1); }
  else if (ev.key === 'ArrowLeft') { openAt(current - 1); }
  else if (!curating) { return; }
  else if (ev.key === 'c' || ev.key === 'C') {
    const fig = order[current];
    if (fig) { openCompare(fig.dataset.bucket); }
  }
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
// Painted once on load, not only on the first click. The server draws every
// card at its top candidate, which is wrong as soon as rejects.json names one
// that `select` has not yet seen -- and then neither the promotion nor the
// warning about a bucket's unseen photos appeared until something was clicked.
syncRejected();
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

// Every check, not just the first failure, for the viewer's explanation. The
// verdict on each row is violates() itself, so the panel cannot disagree with
// the grouping -- only the first failing row names the category, and a photo
// stays rejected until every failing row is relaxed.
function breakdown(m, limits) {
  let decided = false;
  return RULES.map(rule => {
    const values = rule.fields.map(f => m[f]);
    const failed = Boolean(m.detected) && violates(rule, m, limits);
    const decides = failed && !decided;
    if (failed) { decided = true; }
    return {reason: rule.reason, label: rule.label, op: rule.op, limit: rule.limit,
            fields: rule.fields, values: values, threshold: limits[rule.limit],
            measured: values.some(v => v !== null && v !== undefined),
            failed: failed, decides: decides};
  });
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

// -- explaining one photo in the viewer ----------------------------------- //

const assetById = new Map(ASSETS.map(a => [a.id, a]));

// At least one decimal more than the slider shows, so a value just past its
// limit does not print as equal to it.
function valueFor(key, value) {
  if (value === null || value === undefined) { return '\u2014'; }
  const spec = RANGES[key];
  const decimals = !spec || spec.step === undefined || spec.step < 1 ? 3 : 1;
  return Number(value).toFixed(decimals);
}

function passesWhen(row) {
  const t = numberFor(row.limit, row.threshold);
  if (row.op === 'flag') { return row.threshold ? 'allowed' : 'not allowed'; }
  if (row.op === 'abs_gt') { return '|x| \u2264 ' + t; }
  if (row.op === 'gt') { return '\u2264 ' + t; }
  if (row.op === 'lt') { return '\u2265 ' + t; }
  if (row.op === 'max_gt') { return 'each \u2264 ' + t; }
  return t;
}

function shownValue(row) {
  if (row.op === 'flag') { return row.values[0] ? 'yes' : 'no'; }
  return row.values.map(v => valueFor(row.limit, v)).join(' / ');
}

function describeAsset(id) {
  const asset = assetById.get(id);
  if (!asset) { return null; }
  const box = document.createElement('div');
  const title = document.createElement('h3');
  const note = document.createElement('p');
  box.appendChild(title);
  box.appendChild(note);

  if (!asset.detected) {
    title.textContent = 'no_face_detected';
    note.textContent = 'No face was found, so there is nothing to measure. ' +
                       'No threshold can bring this photo back.';
    return box;
  }

  const rows = breakdown(asset, limits);
  const failing = rows.filter(r => r.failed);
  title.textContent = failing.length ? 'Rejected: ' + failing[0].reason
                                     : 'Accepted at these thresholds';
  note.textContent = failing.length > 1
    ? 'Fails ' + failing.length + ' checks. It is filed under the first; ' +
      'every red row has to be relaxed to keep it.'
    : failing.length ? 'Fails one check, shown in red.'
    : 'Passes every check at the current slider positions.';

  // Failures first: they are the answer to "which slider", and on a phone the
  // rest of the table is below the fold.
  const table = document.createElement('table');
  failing.concat(rows.filter(r => !r.failed)).forEach(row => {
    const tr = document.createElement('tr');
    if (row.failed) { tr.className = 'fail'; }
    else if (!row.measured && row.op !== 'flag') { tr.className = 'skip'; }

    const name = document.createElement('td');
    name.textContent = row.label;
    const why = document.createElement('span');
    why.className = 'why';
    why.textContent = row.reason + (row.decides ? '  \u00b7 filed here' : '');
    name.appendChild(why);

    const value = document.createElement('td');
    value.className = 'num';
    value.textContent = shownValue(row);
    const limit = document.createElement('td');
    limit.className = 'num';
    limit.textContent = passesWhen(row);

    tr.appendChild(name);
    tr.appendChild(value);
    tr.appendChild(limit);
    table.appendChild(tr);
  });
  box.appendChild(table);
  return box;
}

describe = describeAsset;

ASSETS.forEach(a => { a.baseline = Boolean(rejectReason(a, limits)); });
update();
"""

def _viewer_html(curate: bool) -> str:
    """The full-size viewer. `curate` is the contact sheet: only there can a
    photo be rejected or its bucket compared, so only there does the viewer
    offer the keys for it."""
    flag = ' data-curate="true"' if curate else ""
    return (f'<div id="viewer"{flag} hidden><div class="stage">'
            '<img id="viewer-img" alt=""><div id="viewer-details" hidden></div></div>'
            '<div class="meta" id="viewer-meta"></div></div>'
            '<div id="compare" hidden></div>')

_TILE_SIZES = {"S": "220px", "M": "340px", "L": "520px"}


def _size_controls(active: str = "M") -> str:
    """The tile-size buttons, with `active` pressed.

    Which button is pressed and what the grid actually renders at have to agree,
    so both are derived from `_TILE_SIZES` and the page sets the variable from
    the same name -- a sheet claiming S while drawing at M is simply wrong, and
    nothing on screen would say which of the two to believe.
    """
    buttons = "".join(
        f'<button data-tile="{px}"'
        + (' aria-pressed="true"' if name == active else "")
        + f">{name}</button>"
        for name, px in _TILE_SIZES.items()
    )
    return f'<span class="count">size</span>{buttons}'


def _page(title: str, lede: str, body: str, script: str = "",
          tile: str = "M") -> str:
    # The starting size rides on <html> rather than waiting for the script, so
    # the grid paints at the right size instead of flashing through another.
    # <html> specifically, because that is `documentElement`: the size buttons
    # set the variable there, and an inline style on <body> would outrank them
    # for everything inside it and quietly stop the buttons working.
    style = f' style="--tile:{_TILE_SIZES[tile]}"'
    return (
        f"<!doctype html>\n<html lang=\"en\"{style}>\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{_CSS}</style>\n</head>\n"
        "<body>\n"
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
        f'<button type="button" class="reject">{action}</button>'
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
        + ('<button type="button" class="compare-open">compare</button>'
           if spares else "")
        + f'<button type="button" class="reject">reject</button>'
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
        "       s.alternate, s.bucket_size"
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
    sizes: dict[str, int | None] = {}
    for row in sorted(rows, key=lambda r: (r["bucket"], r["rank"])):
        if row["bucket"] not in buckets:
            buckets[row["bucket"]] = []
            order.append(row["bucket"])
            sizes[row["bucket"]] = row["bucket_size"]
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

    # `more` is how many photos in the bucket passed the filters and are not
    # on this page. Reject every candidate here and `select` still has those to
    # promote, so the card must not call the bucket empty -- that is exactly
    # when the next encode puts in a photo nobody has looked at. None when the
    # selection predates the count: unknown, which the page says rather than
    # guessing either way.
    def more(bucket: str) -> int | None:
        size = sizes[bucket]
        return None if size is None else max(0, size - len(buckets[bucket]))

    listing = [{"bucket": b, "candidates": buckets[b], "more": more(b)} for b in order]
    seed = (f'<script type="application/json" id="rejected-seed">'
            f'{json.dumps(sorted(manual))}</script>\n'
            f'<script type="application/json" id="buckets">'
            f'{json.dumps(listing)}</script>\n')
    aside = ""
    if rejected_cards:
        aside = (
            '<h2 class="dropped">Rejected by hand</h2>\n'
            '<p class="note">Their buckets fell through to the next best photo. '
            'Click one to keep it again, download rejects.json, then re-run '
            '<code>grow-up select &amp;&amp; grow-up align &amp;&amp; grow-up encode</code> '
            'and reload this page.</p>\n'
            f'<div class="grid">{"".join(rejected_cards)}</div>\n'
        )

    body = (
        seed +
        '<div class="bar">'
        '<button id="save">Download rejects.json</button>'
        '<button id="clear">Clear</button>'
        '<span class="count"><b id="n">0</b> rejected</span>'
        '<span class="spacer"></span>'
        f'{_size_controls("S")}'
        "</div>\n"
        '<p class="notice" id="more-note" hidden></p>\n'
        f'<div class="grid">{"".join(cards)}</div>\n'
        f"{aside}"
        '<textarea id="out" readonly></textarea>\n'
        f"{_viewer_html(curate=True)}"
    )
    lede = (
        "Click a frame to open it full size; the arrow keys step through the sequence "
        "without moving the image, which is how alignment jitter becomes visible. Press r "
        "(or the reject button) to drop a frame, then download rejects.json into this "
        "directory, re-run `grow-up select && grow-up align && grow-up encode` and reload "
        "this page -- the bucket falls through to its next best photo rather than "
        "disappearing. This catches what landmarks cannot: sunglasses, a hand over the "
        "face, or someone else mistagged as the subject."
    )
    out_path.write_text(_page("grow-up — accepted frames", lede, body, _JS, tile="S"),
                        encoding="utf-8")
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
        # max_gaze governs both gaze checks. Labelling its slider after only the
        # first, and sizing it from only gaze_x, hid that it also decides gaze
        # up/down -- and could stop short of the photos gaze_y rejected.
        sharing = [r for r in RULES if r.limit == rule.limit]
        label = ", ".join(dict.fromkeys(r.label for r in sharing))
        current = limits[rule.limit]
        if isinstance(current, bool):
            control = ('<input type="checkbox" data-limit="'
                       f'{rule.limit}"{" checked" if current else ""}>')
            ranges[rule.limit] = {}
        else:
            values = [a[f] for r in sharing for f in r.fields for a in assets]
            spec = _slider_range(values, float(current), rule.op)
            ranges[rule.limit] = spec
            control = (f'<input type="range" data-limit="{rule.limit}" '
                       f'min="{spec["min"]}" max="{spec["max"]}" '
                       f'step="{spec["step"]}" value="{current}">')
        rows_html.append(
            f'<label class="slider"><span class="name">{html.escape(label)}</span>'
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
        f'<div class="bar">{_size_controls()}</div>\n'
        f"{tuner_html}"
        f'<div id="reject-groups">{"".join(sections)}</div>\n'
        f"{_viewer_html(curate=False)}"
    )
    if limits:
        title = "grow-up — threshold tuner"
        lede = (
            "Move a slider to see exactly which photos that threshold would add or drop — "
            "the page re-runs the same filter the pipeline uses, over every analyzed photo. "
            "When it looks right, copy the [filter] block into config.toml and re-run "
            "select; no re-analysis is needed. Click any photo to see it full size, "
            "with every check it was measured against -- the ones it fails in red, "
            "so you can tell which slider to move."
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
