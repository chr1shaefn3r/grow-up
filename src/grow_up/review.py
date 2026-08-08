"""Static HTML review pages.

No framework and no CDN -- these are opened straight off disk with file://, so
anything fetched over the network would simply fail. Images are referenced by
relative path rather than inlined as data URIs, which keeps the page a few KB
instead of a few hundred MB.
"""

from __future__ import annotations

import html
import json
import os
import sqlite3
from pathlib import Path

# A rejects gallery exists to judge whether thresholds are sane, and a sample
# per reason answers that. Rendering several thousand full-size originals would
# not, and would take a minute to paint.
MAX_REJECT_SAMPLES = 48

_CSS = """
:root { color-scheme: light dark; --bg:#faf9f7; --fg:#1a1a1a; --muted:#6b6b6b;
        --card:#fff; --line:#e2e0dc; --bad:#c0392b; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16161a; --fg:#ececec; --muted:#9a9a9a; --card:#212127;
          --line:#33333b; --bad:#ff7a6b; }
}
* { box-sizing: border-box; }
body { margin:0; padding:24px; background:var(--bg); color:var(--fg);
       font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
h1 { font-size:20px; margin:0 0 4px; }
p.lede { color:var(--muted); margin:0 0 20px; max-width:60ch; }
.bar { position:sticky; top:0; z-index:5; background:var(--bg); padding:12px 0;
       border-bottom:1px solid var(--line); margin-bottom:20px; display:flex;
       gap:12px; align-items:center; flex-wrap:wrap; }
button { font:inherit; padding:7px 14px; border:1px solid var(--line);
         border-radius:7px; background:var(--card); color:var(--fg); cursor:pointer; }
button:hover { border-color:var(--muted); }
.count { color:var(--muted); }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:14px; }
figure { margin:0; background:var(--card); border:1px solid var(--line);
         border-radius:10px; overflow:hidden; cursor:pointer; position:relative; }
figure img { display:block; width:100%; height:auto; }
figcaption { padding:6px 8px; font-size:11px; color:var(--muted);
             display:flex; justify-content:space-between; gap:6px; }
figure.rejected { outline:2px solid var(--bad); opacity:.45; }
figure.rejected::after { content:"rejected"; position:absolute; top:8px; left:8px;
  background:var(--bad); color:#fff; font-size:10px; padding:2px 6px; border-radius:4px; }
textarea { width:100%; height:120px; font-family:ui-monospace,monospace; font-size:12px;
           margin-top:14px; background:var(--card); color:var(--fg);
           border:1px solid var(--line); border-radius:8px; padding:10px; }
h2 { font-size:15px; margin:28px 0 10px; border-top:1px solid var(--line); padding-top:18px; }
h2 span { color:var(--muted); font-weight:400; }
"""

_JS = """
const rejected = new Set();
document.querySelectorAll('figure[data-id]').forEach(fig => {
  fig.addEventListener('click', () => {
    const id = fig.dataset.id;
    if (rejected.has(id)) { rejected.delete(id); fig.classList.remove('rejected'); }
    else { rejected.add(id); fig.classList.add('rejected'); }
    render();
  });
});
function payload() { return JSON.stringify({rejected: [...rejected]}, null, 2); }
function render() {
  document.getElementById('n').textContent = rejected.size;
  document.getElementById('out').value = payload();
}
document.getElementById('save').addEventListener('click', () => {
  const blob = new Blob([payload()], {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'rejects.json';
  a.click();
});
document.getElementById('clear').addEventListener('click', () => {
  rejected.clear();
  document.querySelectorAll('figure.rejected').forEach(f => f.classList.remove('rejected'));
  render();
});
render();
"""


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


def write_contact_sheet(conn: sqlite3.Connection, out_path: Path) -> int:
    """Accepted frames in date order, click to toggle rejection."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = conn.execute(
        "SELECT f.asset_id, f.path, f.seq, a.local_datetime, s.bucket"
        "  FROM frames f"
        "  JOIN assets a ON a.id = f.asset_id"
        "  JOIN selection s ON s.asset_id = f.asset_id"
        " ORDER BY f.seq ASC"
    ).fetchall()

    cards = []
    for row in rows:
        when = (row["local_datetime"] or "")[:10]
        cards.append(
            f'<figure data-id="{html.escape(row["asset_id"])}">'
            f'<img loading="lazy" src="{_rel(row["path"], out_path.parent)}" alt="">'
            f'<figcaption><span>{html.escape(when)}</span>'
            f'<span>#{row["seq"]}</span></figcaption></figure>'
        )

    body = (
        '<div class="bar">'
        '<button id="save">Download rejects.json</button>'
        '<button id="clear">Clear</button>'
        '<span class="count"><b id="n">0</b> rejected</span>'
        "</div>\n"
        f'<div class="grid">{"".join(cards)}</div>\n'
        "<textarea id=\"out\" readonly></textarea>"
    )
    lede = (
        "Click any frame to mark it rejected, then download rejects.json into the output "
        "directory and re-run encode. This catches what landmarks cannot: sunglasses, a hand "
        "over the face, or another child mistagged as her."
    )
    out_path.write_text(_page("grow-up — accepted frames", lede, body, _JS), encoding="utf-8")
    return len(rows)


def write_rejects_gallery(conn: sqlite3.Connection, out_path: Path,
                          limit_per_reason: int = MAX_REJECT_SAMPLES) -> int:
    """A sample of dropped frames grouped by reason, for judging thresholds."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reasons = conn.execute(
        "SELECT reject_reason, count(*) AS n FROM metrics"
        " WHERE reject_reason IS NOT NULL GROUP BY reject_reason ORDER BY n DESC"
    ).fetchall()

    sections = []
    total = 0
    for reason_row in reasons:
        reason = reason_row["reject_reason"]
        rows = conn.execute(
            "SELECT m.asset_id, d.path, a.local_datetime"
            "  FROM metrics m"
            "  JOIN assets a ON a.id = m.asset_id"
            "  LEFT JOIN downloads d ON d.asset_id = m.asset_id"
            " WHERE m.reject_reason = ?"
            " ORDER BY a.local_datetime DESC LIMIT ?",
            (reason, limit_per_reason),
        ).fetchall()
        total += len(rows)

        cards = []
        for row in rows:
            if not row["path"]:
                continue
            when = (row["local_datetime"] or "")[:10]
            cards.append(
                "<figure>"
                f'<img loading="lazy" src="{_rel(row["path"], out_path.parent)}" alt="">'
                f"<figcaption><span>{html.escape(when)}</span></figcaption></figure>"
            )

        shown = f" — showing {len(cards)} of {reason_row['n']}" if reason_row["n"] > len(cards) else ""
        sections.append(
            f"<h2>{html.escape(reason)} <span>({reason_row['n']}{shown})</span></h2>"
            f'<div class="grid">{"".join(cards)}</div>'
        )

    lede = (
        "A sample of what was filtered out, grouped by reason. Read this before trusting the "
        "acceptances: if good frames appear here, the corresponding threshold in config.toml "
        "is too tight, and re-running select alone applies the fix."
    )
    out_path.write_text(
        _page("grow-up — rejected frames", lede, "\n".join(sections)), encoding="utf-8"
    )
    return total


def load_manual_rejects(path: Path) -> set[str]:
    """Read rejects.json produced by the contact sheet. Absent file means none."""
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("rejected", [])
    return {str(x) for x in data}
