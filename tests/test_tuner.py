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


needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def run_page(tmp_path, buckets: list[dict], rejected: list[str], epilogue: str,
             figures: list[str] = ()):
    """The shipped contact-sheet JS under node, and the value of `epilogue` after it.

    The document is just big enough to load it: the seed, the bucket listing,
    the notice, and a stand-in card per name in `figures` -- enough for the
    page's own load-time paint to run, with no browser.
    """
    script = tmp_path / f"page-{len(list(tmp_path.glob('page-*.mjs')))}.mjs"
    script.write_text(
        "const ELEMENTS = {\n"
        f"  'rejected-seed': {{textContent: {json.dumps(json.dumps(rejected))}}},\n"
        f"  'buckets': {{textContent: {json.dumps(json.dumps(buckets))}}},\n"
        "  'more-note': {textContent: '', hidden: true},\n"
        "};\n"
        "function card(bucket) {\n"
        "  const parts = {img: {src: '', getAttribute() { return this.src; },\n"
        "                       setAttribute(_, value) { this.src = value; }},\n"
        "                 '.rank': {textContent: ''}};\n"
        "  return {dataset: {bucket: bucket}, classes: new Set(),\n"
        "          classList: {toggle(name, on) {\n"
        "            if (on) { this.owner.classes.add(name); }\n"
        "            else { this.owner.classes.delete(name); } }},\n"
        "          querySelector: (sel) => parts[sel] || null,\n"
        "          querySelectorAll: () => []};\n"
        "}\n"
        f"const CARDS = {json.dumps(list(figures))}.map(card);\n"
        "CARDS.forEach(c => { c.classList.owner = c; });\n"
        "globalThis.document = {\n"
        "  getElementById: (id) => ELEMENTS[id] || null,\n"
        "  querySelectorAll: (sel) => sel.startsWith('figure') ? CARDS : [],\n"
        "  addEventListener: () => {},\n"
        "  createElement: () => ({classList: {add(){}, toggle(){}}, dataset: {},\n"
        "                        appendChild(){}, setAttribute(){}}),\n"
        "};\n"
        f"{review._JS}\n"
        f"console.log(JSON.stringify({epilogue}));\n",
        encoding="utf-8",
    )
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


class TestThePagePromotesTheRunnerUp:
    """The whole point: reject a pick and see its replacement without a re-run.

    Driven under node against the shipped logic, because "the grid stops matching
    what the pipeline would produce" is a silent failure -- the page would look
    fine and simply be lying about the video.
    """

    CANDIDATES = [
        {"id": "best", "src": "a.jpg", "date": "2026-03-02", "seq": 1, "rank": 0},
        {"id": "second", "src": "b.jpg", "date": "2026-03-03", "seq": 2, "rank": 1},
        {"id": "third", "src": "c.jpg", "date": "2026-03-04", "seq": 3, "rank": 2},
    ]
    ALL = ["best", "second", "third"]

    def card(self, tmp_path, rejected: list[str], more: int | None = 0) -> dict:
        """What the shipped page shows for this bucket, given those rejections.

        `more` is how many photos in the bucket passed the filters beyond the
        three on the page, as `write_contact_sheet` embeds it.
        """
        buckets = [{"bucket": "2026-03", "candidates": self.CANDIDATES, "more": more}]
        return run_page(tmp_path, buckets, rejected, "bucketState('2026-03')")

    @needs_node
    def test_nothing_rejected_shows_the_best(self, tmp_path):
        result = self.card(tmp_path, [])
        assert result["pick"]["id"] == "best"
        assert result["label"] == "#1"

    @needs_node
    def test_rejecting_the_pick_shows_the_second(self, tmp_path):
        result = self.card(tmp_path, ["best"])
        assert result["pick"]["id"] == "second"
        assert result["label"] == "promoted #2"

    @needs_node
    def test_rejecting_two_shows_the_third(self, tmp_path):
        assert self.card(tmp_path, ["best", "second"])["pick"]["id"] == "third"

    @needs_node
    def test_rejecting_out_of_order_still_picks_the_best_survivor(self, tmp_path):
        """Dropping a runner-up you can see is bad must not disturb the pick."""
        assert self.card(tmp_path, ["second"])["pick"]["id"] == "best"

    @needs_node
    def test_rejecting_all_of_them_empties_a_bucket_with_nothing_else(self, tmp_path):
        result = self.card(tmp_path, self.ALL, more=0)
        assert result["empty"] is True and result["waiting"] is False
        assert result["label"] == "bucket empty"

    @needs_node
    def test_rejecting_all_of_them_is_not_empty_while_more_passed(self, tmp_path):
        """What shipped: `bucket empty` on the card, and then a photo nobody had
        seen in the video, because select promoted the next one down."""
        result = self.card(tmp_path, self.ALL, more=2)
        assert result["empty"] is False and result["waiting"] is True
        assert result["label"] == "2 more after re-run"

    @needs_node
    def test_without_a_count_it_claims_neither(self, tmp_path):
        """A selection written before select counted its buckets: the page
        does not know, and must not guess "empty"."""
        result = self.card(tmp_path, self.ALL, more=None)
        assert result["empty"] is False and result["waiting"] is False
        assert result["unknown"] is True
        assert result["label"] == "all rejected"

    @needs_node
    def test_more_photos_change_nothing_while_a_candidate_survives(self, tmp_path):
        assert self.card(tmp_path, ["best"], more=2)["label"] == "promoted #2"

    def test_the_cards_and_the_comparison_are_painted_from_it(self):
        """Ties the tests above to what reaches the screen."""
        assert "const state = bucketState(fig.dataset.bucket);" in review._JS
        assert "rank.textContent = state.label;" in review._JS
        assert "const state = bucketState(comparing);" in review._JS


class TestThePageSaysWhatToRunToSeeTheRest:
    """A fully rejected bucket with photos left is a decision to make before
    encoding: select will use the next one, seen or not. The page names those
    buckets and the commands that bring the next photos onto it."""

    CANDIDATES = TestThePagePromotesTheRunnerUp.CANDIDATES

    def listing(self, *buckets: tuple[str, int | None]) -> list[dict]:
        return [{"bucket": name, "more": more,
                 "candidates": [{**c, "id": f"{name}/{c['id']}"} for c in self.CANDIDATES]}
                for name, more in buckets]

    def notice(self, tmp_path, *buckets: tuple[str, int | None],
               rejected: list[str] | None = None) -> str:
        listing = self.listing(*buckets)
        if rejected is None:
            rejected = [c["id"] for b in listing for c in b["candidates"]]
        return run_page(tmp_path, listing, rejected,
                        "moreNotice([...byBucket.keys()].map(bucketState))")

    @needs_node
    def test_it_names_the_bucket_and_what_to_run(self, tmp_path):
        text = self.notice(tmp_path, ("2026-03", 2))
        assert text.startswith("2026-03 has every photo on this page rejected")
        assert "download rejects.json into this directory" in text
        assert "`grow-up select && grow-up align`" in text
        assert "reload this page" in text

    @needs_node
    def test_it_says_encode_would_use_a_photo_nobody_saw(self, tmp_path):
        text = self.notice(tmp_path, ("2026-03", 2))
        assert "`grow-up encode` would put that photo in the video without your having" in text

    @needs_node
    def test_several_buckets_are_all_named(self, tmp_path):
        text = self.notice(tmp_path, ("2026-03", 2), ("2026-04", 1))
        assert text.startswith("2 buckets (2026-03, 2026-04) have")

    @needs_node
    def test_a_bucket_that_really_is_empty_needs_no_advice(self, tmp_path):
        assert self.notice(tmp_path, ("2026-03", 0)) == ""

    @needs_node
    def test_nothing_fully_rejected_needs_no_advice(self, tmp_path):
        assert self.notice(tmp_path, ("2026-03", 2), rejected=["2026-03/best"]) == ""

    @needs_node
    def test_without_a_count_it_says_may_rather_than_will(self, tmp_path):
        text = self.notice(tmp_path, ("2026-03", None))
        assert "may hand it to a photo you have not seen" in text
        assert "`grow-up select && grow-up align`" in text

    @needs_node
    def test_a_page_opened_with_every_candidate_rejected_warns_before_any_click(
            self, tmp_path):
        """rejects.json can name every candidate of a bucket before select has
        seen it -- `grow-up review` straight after saving it. The page used to
        paint only on a click, so it opened looking as if nothing were wrong."""
        listing = self.listing(("2026-03", 2), ("2026-04", 0))
        rejected = [c["id"] for c in listing[0]["candidates"]]
        result = run_page(
            tmp_path, listing, rejected,
            "{note: ELEMENTS['more-note'], label: CARDS[0].querySelector('.rank').textContent,"
            " classes: [...CARDS[0].classes], other: CARDS[1].querySelector('.rank').textContent}",
            figures=["2026-03", "2026-04"])

        assert result["note"]["hidden"] is False
        assert result["note"]["textContent"].startswith("2026-03 has every photo")
        assert result["label"] == "2 more after re-run"
        assert "waiting" in result["classes"]
        assert result["other"] == "#1"


class TestAFullyRejectedBucketMeansWhatSelectDoes:
    """End to end: the count select stores, what the page makes of it, and what
    select then does with exactly those rejections."""

    def outcome(self, tmp_path, photos: int) -> tuple[dict, list[str]]:
        from grow_up import select

        conn = db.connect(tmp_path / "t.sqlite")
        stamp = "2026-01-01T00:00:00.000Z"
        for i in range(photos):
            # Yaw rising with i, so p0 is the best and p5 the sixth best.
            conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                         " VALUES (?, ?, ?)", (f"p{i}", f"2026-01-{10 + i:02d}T10:00:00", stamp))
            conn.execute(
                "INSERT INTO metrics (asset_id, detected, yaw, pitch, roll, gaze_x,"
                " gaze_y, blink_l, blink_r, oob_frac, bbox_clipped, interocular_px,"
                " sharpness, exposure_lo, exposure_hi, analyzed_at)"
                " VALUES (?, 1, ?, 1, 1, 0.02, 0, 0.05, 0.05, 0, 0, 150, 100, 40, 200, ?)",
                (f"p{i}", float(i), stamp))

        def reselect(manual: set[str]) -> None:
            select.apply_filters(conn, LIMITS, {"w_pose": 1.0}, manual)
            select.select_frames(conn, "month", 1, alternates=4)

        reselect(set())
        # Frames for what was selected, as `align` would leave them.
        for seq, row in enumerate(conn.execute(
                "SELECT asset_id FROM selection ORDER BY rank").fetchall(), 1):
            conn.execute("INSERT INTO frames (asset_id, path, seq, warped_at)"
                         " VALUES (?, ?, ?, ?)",
                         (row["asset_id"], str(tmp_path / f"{row['asset_id']}.jpg"), seq, stamp))
        page = tmp_path / "out" / "contact-sheet.html"
        review.write_contact_sheet(conn, page)
        listing = json.loads(page.read_text().split('id="buckets">')[1].split("</script>")[0])

        shown = [c["id"] for c in listing[0]["candidates"]]
        state = run_page(tmp_path, listing, shown, "bucketState('2026-01')")
        reselect(set(shown))
        picked = [r["asset_id"] for r in conn.execute(
            "SELECT asset_id FROM selection WHERE alternate = 0")]
        return state, picked

    @needs_node
    def test_a_bucket_the_page_calls_empty_is_left_empty(self, tmp_path):
        state, picked = self.outcome(tmp_path, 5)
        assert state["label"] == "bucket empty"
        assert picked == []

    @needs_node
    def test_a_bucket_the_page_says_has_more_goes_to_the_sixth_best(self, tmp_path):
        state, picked = self.outcome(tmp_path, 7)
        assert state["label"] == "2 more after re-run"
        assert picked == ["p5"]


class TestChoosingARunnerUpMeansWhatSelectMeans:
    """The compare overlay lets you pick a runner-up outright. There is no
    "chosen" column in the manifest -- `select` simply takes the best candidate
    with no reject reason -- so choosing has to be expressed as rejecting
    everything ahead of it. If the page and the pipeline ever disagree about
    that, the sheet shows one photo and the video contains another.
    """

    CANDIDATES = [
        {"id": "pick", "date": "2026-01-02", "seq": 1, "src": "a.jpg"},
        {"id": "s1", "date": "2026-01-09", "seq": 2, "src": "b.jpg"},
        {"id": "s2", "date": "2026-01-16", "seq": 3, "src": "c.jpg"},
        {"id": "s3", "date": "2026-01-23", "seq": 4, "src": "d.jpg"},
        {"id": "s4", "date": "2026-01-30", "seq": 5, "src": "e.jpg"},
    ]

    def page_chooses(self, tmp_path, chosen: str | list[str],
                     seed: list[str] | None = None):
        """Run the shipped page JS headlessly and return what it rejected.

        `chosen` may be a sequence, applied in order, for changing your mind.
        """
        buckets = [{"bucket": "2026-01", "candidates": self.CANDIDATES}]
        choices = [chosen] if isinstance(chosen, str) else chosen
        script = tmp_path / f"choose-{'-'.join(choices)}.mjs"
        script.write_text(
            "const ELEMENTS = {\n"
            f"  'rejected-seed': {{textContent: {json.dumps(json.dumps(seed or []))}}},\n"
            f"  'buckets': {{textContent: {json.dumps(json.dumps(buckets))}}},\n"
            "};\n"
            "globalThis.document = {\n"
            "  getElementById: (id) => ELEMENTS[id] || null,\n"
            "  querySelectorAll: () => [],\n"
            "  addEventListener: () => {},\n"
            "  createElement: () => ({classList: {add(){}, toggle(){}}, dataset: {},\n"
            "                        appendChild(){}, setAttribute(){}}),\n"
            "};\n"
            f"{review._JS}\n"
            + "".join(f"chooseCandidate('2026-01', {json.dumps(c)});\n" for c in choices)
            + "console.log(JSON.stringify([...rejected]));\n",
            encoding="utf-8",
        )
        result = subprocess.run(["node", str(script)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return sorted(json.loads(result.stdout))

    def pipeline_picks(self, tmp_path, manual: list[str]) -> str:
        """What `select` actually puts in the video, given those rejections."""
        from grow_up import select

        conn = db.connect(tmp_path / "pipeline.sqlite")
        stamp = "2026-01-01T00:00:00.000Z"
        # Scores descending in candidate order, which is what makes the page's
        # list and the pipeline's ranking the same ordering.
        for n, candidate in enumerate(self.CANDIDATES):
            conn.execute("INSERT INTO assets (id, local_datetime, indexed_at)"
                         " VALUES (?, ?, ?)", (candidate["id"], candidate["date"], stamp))
            conn.execute(
                "INSERT INTO metrics (asset_id, detected, yaw, pitch, roll, gaze_x,"
                " gaze_y, blink_l, blink_r, oob_frac, bbox_clipped, interocular_px,"
                " sharpness, exposure_lo, exposure_hi, analyzed_at)"
                " VALUES (?, 1, ?, 1, 1, 0.02, 0, 0.05, 0.05, 0, 0, 150, 100, 40, 200, ?)",
                (candidate["id"], float(n), stamp))

        select.apply_filters(conn, LIMITS, {}, set(manual))
        select.select_frames(conn, "month", 1, 0)
        row = conn.execute(
            "SELECT asset_id FROM selection WHERE alternate = 0").fetchone()
        return row["asset_id"] if row else ""

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    def test_choosing_rejects_exactly_what_is_ahead_of_it(self, tmp_path):
        assert self.page_chooses(tmp_path, "s3") == ["pick", "s1", "s2"]

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    def test_choosing_the_first_runner_up_drops_only_the_pick(self, tmp_path):
        assert self.page_chooses(tmp_path, "s1") == ["pick"]

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    def test_choosing_one_you_had_rejected_takes_it_back(self, tmp_path):
        """Otherwise the overlay would let you select a photo and then not use
        it, which is the most confusing outcome available."""
        assert self.page_chooses(tmp_path, "s2", seed=["s2"]) == ["pick", "s1"]

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    @pytest.mark.parametrize("chosen", ["s1", "s2", "s3", "s4"])
    def test_the_pipeline_then_picks_the_one_you_chose(self, tmp_path, chosen):
        """The whole claim, end to end: what the page writes into rejects.json
        makes `select` choose the photo you clicked."""
        rejected = self.page_chooses(tmp_path, chosen)
        assert self.pipeline_picks(tmp_path, rejected) == chosen

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_choosing_runner_up_n_rejects_exactly_n(self, tmp_path, n):
        chosen = self.CANDIDATES[n]["id"]
        assert self.page_chooses(tmp_path, chosen) == sorted(
            c["id"] for c in self.CANDIDATES[:n])


class TestChangingYourMindInTheComparison:
    """Each choice sets the bucket afresh. Choices used to only add, so picking
    #3 and then #1 left #2 rejected as well -- two rejections where one choice
    of the first runner-up should make exactly one."""

    page_chooses = TestChoosingARunnerUpMeansWhatSelectMeans.page_chooses
    CANDIDATES = TestChoosingARunnerUpMeansWhatSelectMeans.CANDIDATES

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    @pytest.mark.parametrize("first", ["s1", "s2", "s3", "s4"])
    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_the_last_choice_alone_decides_what_is_rejected(self, tmp_path, first, n):
        chosen = self.CANDIDATES[n]["id"]
        assert self.page_chooses(tmp_path, [first, chosen]) == sorted(
            c["id"] for c in self.CANDIDATES[:n])

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    def test_it_holds_across_a_reload_too(self, tmp_path):
        """After a reload the earlier choice is only rejects.json; the rule
        must not depend on remembering that a choice made it."""
        assert self.page_chooses(tmp_path, "s1", seed=["pick", "s1", "s2"]) == ["pick"]

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    def test_another_bucket_keeps_its_rejections(self, tmp_path):
        assert self.page_chooses(tmp_path, "s1", seed=["elsewhere"]) == [
            "elsewhere", "pick"]


class TestTheViewerExplainsARejection:
    """A category is not an answer to "which slider". looking_away is two checks
    on one slider, and a photo can fail several checks while being filed under
    only the first -- so clicking a photo lists every check, red where it fails.
    """

    def run_breakdown(self, tmp_path, cases):
        script = tmp_path / "breakdown.mjs"
        script.write_text(
            f"const RULES={json.dumps([r.__dict__ for r in RULES])};\n"
            f"const LIMITS={json.dumps(LIMITS)};\n"
            f"const CASES={json.dumps(cases)};\n"
            f"{review._FILTER_JS}\n"
            "console.log(JSON.stringify(CASES.map(c => breakdown(c, LIMITS))));\n",
            encoding="utf-8",
        )
        result = subprocess.run(["node", str(script)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    def test_every_red_row_is_a_check_python_fails(self, tmp_path):
        cases = [c for c in fixtures() if c.get("detected")]
        for case, rows in zip(cases, self.run_breakdown(tmp_path, cases)):
            frame = FaceMetrics(**{k: v for k, v in case.items()
                                   if k in FaceMetrics.__dataclass_fields__})
            assert [r["failed"] for r in rows] == [
                metrics.violates(rule, frame, LIMITS) for rule in RULES], case

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    def test_the_row_marked_as_the_reason_is_the_category_it_is_filed_under(self, tmp_path):
        cases = [c for c in fixtures() if c.get("detected")]
        for case, rows in zip(cases, self.run_breakdown(tmp_path, cases)):
            frame = FaceMetrics(**{k: v for k, v in case.items()
                                   if k in FaceMetrics.__dataclass_fields__})
            deciding = [r["reason"] for r in rows if r["decides"]]
            expected = metrics.hard_reject(frame, LIMITS)
            assert deciding == ([expected] if expected else []), case

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    def test_it_says_which_gaze_check_failed_not_just_looking_away(self, tmp_path):
        base = fixtures()[0]
        [rows] = self.run_breakdown(tmp_path, [{**base, "gaze_y": -0.9}])
        failed = [r["label"] for r in rows if r["failed"]]
        assert failed == ["gaze up/down"]

    @pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
    def test_a_photo_failing_several_checks_shows_them_all(self, tmp_path):
        base = fixtures()[0]
        [rows] = self.run_breakdown(
            tmp_path, [{**base, "gaze_x": 0.9, "blink_l": 0.9, "sharpness": 1.0}])
        assert [r["reason"] for r in rows if r["failed"]] == [
            "looking_away", "eyes_closed", "blurry"]

    def test_the_tuner_wires_the_explanation_into_the_viewer(self, conn, tmp_path):
        out = tmp_path / "out" / "rejects.html"
        review.write_rejects_gallery(conn, out, limits=LIMITS)
        page = out.read_text()
        assert 'id="viewer-details"' in page
        assert "describe = describeAsset;" in page


class TestASharedThresholdIsOneSliderForAllItsChecks:
    """max_gaze decides both gaze checks. Its slider used to be labelled and
    sized from gaze_x alone."""

    def test_the_slider_names_every_check_it_controls(self, conn, tmp_path):
        out = tmp_path / "out" / "rejects.html"
        review.write_rejects_gallery(conn, out, limits=LIMITS)
        assert "gaze sideways, gaze up/down</span>" in out.read_text()

    def test_the_track_reaches_a_photo_only_gaze_y_rejected(self, conn, tmp_path):
        conn.execute("UPDATE metrics SET gaze_y = -0.9 WHERE asset_id = 'good'")
        out = tmp_path / "out" / "rejects.html"
        review.write_rejects_gallery(conn, out, limits=LIMITS)
        assert payload(out.read_text(), "RANGES")["max_gaze"]["max"] >= 0.9


class TestTheViewerOnlyOffersWhatThePageCanDo:
    """Rejecting and comparing exist on the contact sheet alone. The rejects page
    shares the viewer, and used to advertise both keys anyway."""

    def test_the_rejects_page_viewer_does_not_curate(self, conn, tmp_path):
        out = tmp_path / "out" / "rejects.html"
        review.write_rejects_gallery(conn, out, limits=LIMITS)
        assert '<div id="viewer" hidden>' in out.read_text()

    def test_the_contact_sheet_viewer_does(self, tmp_path):
        conn = db.connect(tmp_path / "c.sqlite")
        out = tmp_path / "out" / "contact-sheet.html"
        review.write_contact_sheet(conn, out)
        assert '<div id="viewer" data-curate="true" hidden>' in out.read_text()

    def test_both_hints_and_both_keys_are_gated_on_it(self):
        assert "(curating ? '<kbd>r</kbd> reject" in review._JS
        assert "(curating && candidatesFor(fig.dataset.bucket).length > 1" in review._JS
        assert "else if (!curating) { return; }" in review._JS
        # Nothing advertises the keys outside those guards.
        assert review._JS.count("<kbd>r</kbd>") == 1
        assert review._JS.count("<kbd>c</kbd>") == 1
