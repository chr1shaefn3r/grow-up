"""config.example.toml is documentation that has to stay true.

Two ways it can lie. It can fail to parse -- a repeated key is the usual cause,
and `tomllib` refuses those outright, so this test is what turns "the example is
broken" into a failure here rather than on the user's first run. Or it can list a
setting nothing reads, which is worse: it looks configured and does nothing.
"""

from __future__ import annotations

import dataclasses
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
