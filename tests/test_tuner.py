"""The interactive threshold tuner on the rejects page.

The page re-evaluates the filter in JavaScript so sliders can show what a
threshold change would add or drop. The danger is the two implementations
drifting, so both walk the same serialized `metrics.RULES` table; these tests
guard that arrangement.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from grow_up import db, metrics, review
from grow_up.metrics import RULES, FaceMetrics

LIMITS = {
    "max_yaw": 20.0, "max_pitch": 18.0, "max_roll": 25.0, "max_gaze": 0.35,
    "max_blink": 0.45, "max_oob_frac": 0.005, "allow_bbox_clipped": False,
    "min_interocular_px": 60.0, "min_sharpness": 12.0,
    "min_exposure_lo": 8.0, "max_exposure_hi": 250.0,
}


@pytest.fixture()
def conn(tmp_path):
    conn = db.connect(tmp_path / "t.sqlite")
    stamp = "2026-01-01T00:00:00.000Z"
    rows = [
        ("good", 2.0, 100.0),      # passes
        ("turned", 45.0, 100.0),   # fails on yaw
        ("soft", 2.0, 3.0),        # fails on sharpness
    ]
    for asset_id, yaw, sharpness in rows:
        conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                     " VALUES (?, '2026-03-10T10:00:00.000Z', ?)", (asset_id, stamp))
        conn.execute("INSERT INTO downloads (asset_id, path, source, fetched_at)"
                     " VALUES (?, ?, 'original', ?)",
                     (asset_id, str(tmp_path / "originals" / f"{asset_id}.jpg"), stamp))
        conn.execute(
            "INSERT INTO metrics (asset_id, detected, yaw, pitch, roll, gaze_x, gaze_y,"
            " blink_l, blink_r, oob_frac, bbox_clipped, interocular_px, sharpness,"
            " exposure_lo, exposure_hi, analyzed_at)"
            " VALUES (?, 1, ?, 1, 1, 0.02, 0, 0.05, 0.05, 0, 0, 150, ?, 40, 200, ?)",
            (asset_id, yaw, sharpness, stamp))
    return conn


def payload(html: str, name: str):
    """Pull an embedded `const NAME=<json>;` out of the page.

    Decoded rather than pattern-matched, so nested braces and strings in the
    data cannot fool the extraction.
    """
    marker = f"const {name}="
    start = html.index(marker) + len(marker)
    value, _ = json.JSONDecoder().raw_decode(html, start)
    return value


class TestRuleTable:
    def test_drives_the_python_filter(self):
        """hard_reject must walk RULES, not a second hand-written list."""
        frame = FaceMetrics(detected=1, yaw=45.0, pitch=1.0, roll=1.0,
                            gaze_x=0.0, gaze_y=0.0, blink_l=0.0, blink_r=0.0,
                            oob_frac=0.0, bbox_clipped=0, interocular_px=150.0,
                            sharpness=100.0, exposure_lo=40.0, exposure_hi=200.0)
        assert metrics.hard_reject(frame, LIMITS) == "head_turned"

    def test_every_limit_in_config_is_reachable_from_a_rule(self):
        """A threshold no rule consults would sit in config.toml doing nothing."""
        assert {rule.limit for rule in RULES} == set(LIMITS)

    def test_every_rule_field_exists_on_the_metrics_record(self):
        for rule in RULES:
            for field in rule.fields:
                assert field in FaceMetrics.__dataclass_fields__, field

    def test_ops_are_all_implemented_in_python(self):
        frame = FaceMetrics(detected=1)
        for rule in RULES:
            metrics.violates(rule, frame, LIMITS)  # must not raise

    def test_unknown_op_is_rejected_loudly(self):
        bogus = metrics.Rule("x", ("yaw",), "sideways", "max_yaw", "x")
        with pytest.raises(ValueError, match="unknown rule op"):
            metrics.violates(bogus, FaceMetrics(detected=1, yaw=1.0), LIMITS)


def fixtures() -> list[dict]:
    """One case per rule, plus boundary and missing-metric cases."""
    base = dict(detected=1, yaw=2.0, pitch=1.0, roll=1.0, gaze_x=0.02, gaze_y=0.0,
                blink_l=0.05, blink_r=0.05, oob_frac=0.0, bbox_clipped=0,
                interocular_px=150.0, sharpness=100.0, exposure_lo=40.0,
                exposure_hi=200.0)
    cases = [
        base,
        {**base, "detected": 0},
        {**base, "bbox_clipped": 1},
        {**base, "oob_frac": 0.5},
        {**base, "yaw": 45.0}, {**base, "yaw": -45.0},
        {**base, "pitch": -40.0}, {**base, "roll": 50.0},
        {**base, "gaze_x": 0.9}, {**base, "gaze_y": -0.9},
        {**base, "blink_l": 0.95}, {**base, "blink_r": 0.95},
        {**base, "interocular_px": 20.0},
        {**base, "sharpness": 1.0},
        {**base, "exposure_hi": 254.0}, {**base, "exposure_lo": 1.0},
        # Boundaries: comparisons are strict, so exactly-at-limit must pass.
        {**base, "yaw": 20.0}, {**base, "sharpness": 12.0},
        {**base, "interocular_px": 60.0}, {**base, "max_gaze": 0.35},
        # Missing metrics must not be treated as zero.
        {**base, "yaw": None, "sharpness": None},
        {k: (None if k != "detected" else 1) for k in base},
    ]
    return cases


class TestJavaScriptParity:
    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    def test_page_filter_agrees_with_python_on_every_case(self, tmp_path):
        """Runs the page's own filter under node and compares verdicts.

        Structural checks can only show the rules were serialized; this shows
        the two implementations actually decide the same way, which is what
        makes the preview trustworthy.
        """
        cases = fixtures()
        expected = [
            metrics.hard_reject(
                FaceMetrics(**{k: v for k, v in case.items()
                               if k in FaceMetrics.__dataclass_fields__}),
                LIMITS)
            for case in cases
        ]

        script = tmp_path / "parity.mjs"
        script.write_text(
            f"const RULES={json.dumps([r.__dict__ for r in RULES])};\n"
            f"const LIMITS={json.dumps(LIMITS)};\n"
            f"const CASES={json.dumps(cases)};\n"
            f"{review._FILTER_JS}\n"
            "console.log(JSON.stringify(CASES.map(c => rejectReason(c, LIMITS))));\n",
            encoding="utf-8",
        )
        result = subprocess.run(["node", str(script)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

        assert json.loads(result.stdout) == expected

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    def test_page_filter_tracks_a_threshold_change(self, tmp_path):
        """The point of the sliders: a loosened limit must flip the same photos
        in the page as it would in the pipeline."""
        case = {"detected": 1, "yaw": 30.0, "pitch": 1.0, "roll": 1.0,
                "gaze_x": 0.0, "gaze_y": 0.0, "blink_l": 0.0, "blink_r": 0.0,
                "oob_frac": 0.0, "bbox_clipped": 0, "interocular_px": 150.0,
                "sharpness": 100.0, "exposure_lo": 40.0, "exposure_hi": 200.0}
        loosened = {**LIMITS, "max_yaw": 40.0}

        script = tmp_path / "parity2.mjs"
        script.write_text(
            f"const RULES={json.dumps([r.__dict__ for r in RULES])};\n"
            f"{review._FILTER_JS}\n"
            f"const c={json.dumps(case)};\n"
            f"console.log(JSON.stringify([rejectReason(c, {json.dumps(LIMITS)}),"
            f" rejectReason(c, {json.dumps(loosened)})]));\n",
            encoding="utf-8",
        )
        result = subprocess.run(["node", str(script)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

        frame = FaceMetrics(**case)
        assert json.loads(result.stdout) == [
            metrics.hard_reject(frame, LIMITS),
            metrics.hard_reject(frame, loosened),
        ] == ["head_turned", None]

    def test_the_page_implements_every_op_python_uses(self, conn, tmp_path):
        """The guard against drift: a new op added to RULES without teaching the
        page would silently make the preview lie."""
        out = tmp_path / "out" / "rejects.html"
        review.write_rejects_gallery(conn, out, limits=LIMITS)
        html = out.read_text()

        for op in {rule.op for rule in RULES}:
            assert f"'{op}'" in html, f"op {op!r} not handled in the page"

    def test_page_embeds_the_rules_verbatim(self, conn, tmp_path):
        out = tmp_path / "out" / "rejects.html"
        review.write_rejects_gallery(conn, out, limits=LIMITS)

        embedded = payload(out.read_text(), "RULES")
        assert [r["reason"] for r in embedded] == [r.reason for r in RULES]
        assert [r["op"] for r in embedded] == [r.op for r in RULES]
        assert [r["limit"] for r in embedded] == [r.limit for r in RULES]

    def test_page_reproduces_the_no_face_short_circuit(self, conn, tmp_path):
        out = tmp_path / "out" / "rejects.html"
        review.write_rejects_gallery(conn, out, limits=LIMITS)
        assert "no_face_detected" in out.read_text()


class TestTunerPage:
    def build(self, conn, tmp_path):
        out = tmp_path / "out" / "rejects.html"
        review.write_rejects_gallery(conn, out, limits=LIMITS)
        return out.read_text()

    def test_has_a_slider_per_numeric_threshold(self, conn, tmp_path):
        html = self.build(conn, tmp_path)
        for key, value in LIMITS.items():
            if isinstance(value, bool):
                assert f'type="checkbox" data-limit="{key}"' in html
            else:
                assert f'data-limit="{key}"' in html, key

    def test_embeds_every_analyzed_photo_not_just_rejects(self, conn, tmp_path):
        """Loosening a threshold must be able to surface a photo that currently
        passes the filter, so accepted ones have to be in the payload too."""
        assets = payload(self.build(conn, tmp_path), "ASSETS")
        assert {a["id"] for a in assets} == {"good", "turned", "soft"}

    def test_embedded_assets_carry_the_metrics_the_rules_read(self, conn, tmp_path):
        assets = payload(self.build(conn, tmp_path), "ASSETS")
        needed = {f for rule in RULES for f in rule.fields} | {"detected"}
        for asset in assets:
            assert needed <= set(asset), needed - set(asset)

    def test_baseline_limits_are_the_configured_ones(self, conn, tmp_path):
        assert payload(self.build(conn, tmp_path), "BASE_LIMITS") == LIMITS

    def test_slider_ranges_cover_the_observed_data(self, conn, tmp_path):
        """A track that stops below the outliers cannot express the change that
        would actually admit them."""
        ranges = payload(self.build(conn, tmp_path), "RANGES")
        assert ranges["max_yaw"]["max"] >= 45.0, "the 45-degree photo must be reachable"
        assert ranges["max_yaw"]["min"] <= 20.0
        assert ranges["max_yaw"]["step"] > 0

    def test_reports_added_and_dropped_sets(self, conn, tmp_path):
        html = self.build(conn, tmp_path)
        assert 'id="added"' in html and 'id="removed"' in html
        assert "would be added" in html and "would be dropped" in html

    def test_offers_the_config_snippet(self, conn, tmp_path):
        html = self.build(conn, tmp_path)
        assert 'id="toml"' in html
        assert "[filter]" in html

    def test_initial_grouping_matches_the_pipeline(self, conn, tmp_path):
        """Rendered server-side, so the page is correct before any interaction."""
        html = self.build(conn, tmp_path)
        assert "head_turned" in html and "blurry" in html

    def test_stays_self_contained(self, conn, tmp_path):
        html = self.build(conn, tmp_path)
        assert not re.search(r"""(src|href)\s*=\s*["']https?://""", html)


class TestSliderRange:
    def test_spans_zero_to_beyond_the_maximum(self):
        spec = review._slider_range([1.0, 5.0, 40.0], current=20.0, op="gt")
        assert spec["min"] <= 0 and spec["max"] > 40.0

    def test_uses_magnitude_for_signed_metrics(self):
        """Yaw is compared as an absolute value, so -60 must widen the track."""
        spec = review._slider_range([-60.0, 2.0], current=20.0, op="abs_gt")
        assert spec["max"] > 60.0

    def test_current_value_is_always_reachable(self):
        spec = review._slider_range([0.1, 0.2], current=900.0, op="gt")
        assert spec["max"] >= 900.0

    def test_tolerates_missing_metrics(self):
        spec = review._slider_range([None, None], current=5.0, op="gt")
        assert spec["max"] > 5.0 and spec["step"] > 0

    def test_constant_data_still_yields_a_usable_track(self):
        spec = review._slider_range([3.0, 3.0], current=3.0, op="gt")
        assert spec["max"] > spec["min"] and spec["step"] > 0


class TestTheSeedSurvivesIntoTheDownload:
    """The contact sheet's download replaces the whole file, so anything already
    rejected has to be in the set before the user touches anything.

    Without the seed, `select` dropping a rejected photo would leave it with no
    card, and the next download would silently un-reject it -- destroying
    curation that only exists because someone looked at every frame.
    """

    def seeded_page_script(self, ids: list[str], toggle: str | None = None) -> str:
        """The page's own reject bookkeeping, driven headlessly."""
        return (
            "globalThis.document = {\n"
            f"  getElementById: (id) => id === 'rejected-seed'"
            f" ? {{textContent: {json.dumps(json.dumps(ids))}}} : null,\n"
            "  querySelectorAll: () => [],\n"
            "  addEventListener: () => {},\n"
            "};\n"
            "const seed = document.getElementById('rejected-seed');\n"
            "const rejected = new Set(seed ? JSON.parse(seed.textContent) : []);\n"
            + (f"rejected.add({json.dumps(toggle)});\n" if toggle else "")
            + "console.log(JSON.stringify({rejected: [...rejected]}));\n"
        )

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    def test_existing_rejects_are_in_the_payload_before_any_click(self, tmp_path):
        script = tmp_path / "seed.mjs"
        script.write_text(self.seeded_page_script(["a", "b"]), encoding="utf-8")
        result = subprocess.run(["node", str(script)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

        assert sorted(json.loads(result.stdout)["rejected"]) == ["a", "b"]

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    def test_a_new_rejection_adds_to_them_rather_than_replacing_them(self, tmp_path):
        script = tmp_path / "seed2.mjs"
        script.write_text(self.seeded_page_script(["a", "b"], toggle="c"),
                          encoding="utf-8")
        result = subprocess.run(["node", str(script)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

        assert sorted(json.loads(result.stdout)["rejected"]) == ["a", "b", "c"]

    def test_the_page_really_reads_that_element(self):
        """Ties the script above to the shipped JS, so it cannot drift away."""
        assert "rejected-seed" in review._JS
        assert "new Set(seed ? JSON.parse(seed.textContent) : [])" in review._JS
