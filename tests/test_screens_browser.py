"""The multi-screen comparison in a real browser.

Everything else about the extra screens is tested under node against stubs.
What stubs cannot show is the browser's own rules: over file:// every load of
the sheet is an origin of its own, so the windows it opened must be reached by
message, not by touching their documents -- and the first version, which drew
into them directly, lost every window at the reload that follows
`select && align`. Only a browser shows that, so this drives Chromium through
Playwright when node can load it, and skips otherwise. CI installs neither.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from grow_up import db, review


def _have_playwright() -> bool:
    if shutil.which("node") is None:
        return False
    probe = subprocess.run(["node", "-e", "require('playwright')"], capture_output=True)
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _have_playwright(),
                                reason="needs node with playwright")

# Three screens, a comparison spread over them, a key pressed in the third
# window, a reload, and a second comparison. Waits are on conditions rather
# than on time, so a slow machine is slower rather than wrong.
SCRIPT = r"""
const { chromium } = require('playwright');
(async () => {
  let browser;
  try { browser = await chromium.launch(); }
  catch (e) { console.log(JSON.stringify({skip: String(e).split('\n')[0]})); return; }
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  const errors = [];
  ctx.on('page', p => p.on('pageerror', e => errors.push(String(e))));
  page.on('pageerror', e => errors.push(String(e)));
  await page.goto(process.argv[2]);

  async function until(test) {
    for (let i = 0; i < 100; i++) { if (await test()) { return; } await page.waitForTimeout(50); }
    throw new Error('timed out: ' + test);
  }
  async function screen(n) {
    for (const p of ctx.pages()) { if (await p.title() === 'grow-up — screen ' + n) { return p; } }
  }
  const numbers = p => p.$$eval('#compare .choice .cap',
                                cs => cs.map(c => Number(c.textContent.split('.')[0])));
  const compare = () => page.locator(
    'figure[data-bucket="2026-01"] figcaption .compare-open').click();

  await page.click('[data-screens="3"]');
  await until(async () => ctx.pages().length === 3 && await screen(2) && await screen(3));
  const two = await screen(2), three = await screen(3);
  await until(async () => (await three.textContent('#compare')).startsWith('Screen 3'));

  await compare();
  await until(async () => (await numbers(three)).length === 4);
  const first = {sheet: await numbers(page), two: await numbers(two),
                 three: await numbers(three)};

  await three.keyboard.press('9');
  await until(() => page.$eval('#compare', c => c.hidden));
  const chosen = JSON.parse(await page.inputValue('#out')).rejected.length;

  await page.reload();
  await until(() => page.$eval('[data-screens="3"]',
                               b => b.getAttribute('aria-pressed') === 'true'));
  await compare();
  await until(async () => (await numbers(three)).length === 4);
  const reloaded = {windows: ctx.pages().length, three: await numbers(three)};

  console.log(JSON.stringify({first, chosen, reloaded, errors}));
  await browser.close();
})();
"""


@pytest.fixture()
def sheet(tmp_path):
    conn = db.connect(tmp_path / "t.sqlite")
    stamp = "2026-01-01T00:00:00.000Z"
    for rank in range(13):
        asset_id = f"p{rank}"
        conn.execute("INSERT INTO assets (id, local_datetime, indexed_at) VALUES (?, ?, ?)",
                     (asset_id, f"2026-01-{rank + 1:02d}T10:00:00", stamp))
        conn.execute("INSERT INTO selection (asset_id, bucket, rank, alternate, selected_at,"
                     " bucket_size) VALUES (?, '2026-01', ?, ?, ?, 13)",
                     (asset_id, rank, int(rank > 0), stamp))
        conn.execute("INSERT INTO frames (asset_id, path, seq, warped_at) VALUES (?, ?, ?, ?)",
                     (asset_id, str(tmp_path / "frames" / f"{asset_id}.jpg"), rank + 1, stamp))
    out = tmp_path / "out" / "contact-sheet.html"
    review.write_contact_sheet(conn, out)
    return out


def test_three_screens_survive_a_reload(sheet, tmp_path):
    script = tmp_path / "screens.js"
    script.write_text(SCRIPT, encoding="utf-8")
    result = subprocess.run(["node", str(script), sheet.as_uri()],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout)
    if "skip" in outcome:
        pytest.skip(f"chromium would not start: {outcome['skip']}")

    assert outcome["first"] == {"sheet": [1, 2, 3, 4], "two": [5, 6, 7, 8],
                                "three": [9, 10, 11, 12]}
    assert outcome["chosen"] == 9, "a key in the third window chose runner-up 9"
    assert outcome["reloaded"] == {"windows": 3, "three": [9, 10, 11, 12]}, (
        "the reloaded sheet took its windows back rather than opening new ones")
    assert outcome["errors"] == []
