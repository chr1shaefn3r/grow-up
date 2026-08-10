"""config.example.toml is documentation that has to stay true.

Three ways it can lie. It can fail to parse -- a repeated key is the usual cause,
and `tomllib` refuses those outright, so this test is what turns "the example is
broken" into a failure here rather than on the user's first run. It can list a
setting nothing reads, which is worse: it looks configured and does nothing. Or a
setting can arrive with no explanation of what turning it would do, which is how
the example stops being worth reading at all.
"""

from __future__ import annotations

import dataclasses
import re
import tomllib
from pathlib import Path

import pytest

from grow_up import analyze, cli, config

EXAMPLE = Path(__file__).resolve().parents[1] / "config.example.toml"

# Everything the analyze stage consumes: the option fields themselves, plus the
# worker count, which the CLI reads straight from the section.
ANALYZE_KEYS = {f.name for f in dataclasses.fields(analyze.AnalyzeOptions)} | {"workers"}


@pytest.fixture(scope="module")
def raw() -> dict:
    with EXAMPLE.open("rb") as fh:
        return tomllib.load(fh)


def test_the_example_parses(raw):
    assert raw["analyze"], "the example lost its [analyze] section"


def test_every_analyze_setting_is_one_the_code_reads(raw):
    orphans = sorted(set(raw["analyze"]) - ANALYZE_KEYS)
    assert not orphans, f"documented but never read: {', '.join(orphans)}"


def test_the_example_uses_no_setting_that_was_removed(raw):
    config.check_removed(raw)


def test_the_documented_values_are_the_ones_that_take_effect(raw):
    """A setting can be read and still be overwritten by a preset."""
    section = raw["analyze"]
    opts = cli._analyze_options(config.Config(raw=raw, root=EXAMPLE.parent))

    assert opts.effort == section["effort"]
    assert opts.model_path == section["model_path"]
    assert opts.bbox_margin == section["bbox_margin"]
    assert opts.min_face_detection_confidence == section["min_face_detection_confidence"]
    assert opts.min_face_presence_confidence == section["min_face_presence_confidence"]
    assert opts.oob_inset == section["oob_inset"]
    assert opts.verbose == section["verbose"]


KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=")


def _says_something(comment: str) -> bool:
    """True for prose, false for a `# -- divider ---`."""
    return len(comment.strip("#-= ").split()) >= 3


def _has_inline_comment(line: str) -> bool:
    return "#" in re.sub(r'"[^"]*"', "", line.split("=", 1)[1])


def undocumented(text: str) -> list[str]:
    """Keys with no comment beside them, above them, or above their group.

    A comment can cover a run of related keys -- the two confidence thresholds
    share one, and so do width and height -- so a key counts as documented when
    the line above it is either prose or another documented key. A blank line or
    a section header ends the run.
    """
    missing, previous = [], None
    for line in text.splitlines():
        text = line.strip()
        if not text or text.startswith("["):
            previous = None
        elif text.startswith("#"):
            previous = "prose" if _says_something(text) else None
        elif match := KEY.match(text):
            covered = _has_inline_comment(text) or previous in ("prose", "documented")
            if not covered:
                missing.append(match.group(1))
            previous = "documented" if covered else None
        else:
            previous = None
    return missing


def test_every_toggle_says_what_it_does():
    bare = undocumented(EXAMPLE.read_text())
    assert not bare, f"settings with no comment: {', '.join(bare)}"


class TestTheDocumentationCheckItself:
    """The check above is only worth having if it can fail."""

    def test_a_bare_key_is_caught(self):
        assert undocumented("[a]\nx = 1\n") == ["x"]

    def test_a_comment_covers_the_run_it_heads_but_not_past_a_blank_line(self):
        text = "[a]\n# says what it does, at length\nx = 1\ny = 2\n\nz = 3\n"
        assert undocumented(text) == ["z"]

    def test_an_inline_comment_is_enough(self):
        assert undocumented('[a]\nx = 1  # what it does\n') == []

    def test_a_hash_inside_a_string_is_not_a_comment(self):
        assert undocumented('[a]\nx = "#ff0000"\n') == ["x"]

    def test_a_divider_documents_nothing(self):
        assert undocumented("[a]\n# -- runtime ----------\nx = 1\n") == ["x"]

    def test_a_section_header_ends_the_run(self):
        text = "[a]\n# says what it does, at length\nx = 1\n[b]\ny = 2\n"
        assert undocumented(text) == ["y"]
