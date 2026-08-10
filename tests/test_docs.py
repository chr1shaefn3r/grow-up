"""The docs name commands about forty times; renaming one would rot them silently.

README.md is what a user follows literally, so a command that no longer exists is not a
stale sentence -- it is an instruction that fails. Tie both documents to the parser
instead of to whoever remembers to grep.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from grow_up import cli

ROOT = Path(__file__).resolve().parents[1]
DOCS = ("README.md", "CLAUDE.md")

# Global options that may sit between `grow-up` and the subcommand. --config takes a
# value, which must be skipped with it.
FLAGS_WITH_VALUE = {"--config"}


def subcommands() -> set[str]:
    for action in cli.build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    raise AssertionError("the CLI parser has no subparsers")


def _command_in(invocation: str) -> str | None:
    """The subcommand from a `grow-up …` line, or None if it names no subcommand.

    None covers three cases: nothing but global flags (`grow-up --help`), a `<command>`
    placeholder standing for any of them, and an empty invocation.
    """
    tokens = invocation.split()[1:]
    while tokens:
        token = tokens.pop(0)
        if token in FLAGS_WITH_VALUE:
            tokens = tokens[1:]
        elif token.startswith("<") and token.endswith(">"):
            return None
        elif not token.startswith("-"):
            return token
    return None


def invocations(text: str):
    """Every `grow-up …` in a code span or a fenced block, with its line number.

    Restricted to code deliberately: prose says things like "grow-up says so rather
    than", and "says" is not a command.
    """
    fenced, spans = False, []
    for number, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            stripped = line.strip()
            if stripped.startswith("grow-up "):
                spans.append((number, stripped.split("#")[0].strip()))
        else:
            spans.extend((number, code) for code in re.findall(r"`([^`]+)`", line)
                         if code.startswith("grow-up "))
    return spans


@pytest.mark.parametrize("doc", DOCS)
def test_every_documented_command_exists(doc):
    known = subcommands()
    wrong = [(number, text, command)
             for number, text in invocations((ROOT / doc).read_text())
             if (command := _command_in(text)) is not None and command not in known]

    detail = "\n".join(f"  {doc}:{n}: {t!r} -> no such command {c!r}" for n, t, c in wrong)
    assert not wrong, f"documented commands that do not exist:\n{detail}"


def test_every_command_is_documented_somewhere():
    """The other direction: a feature nobody can find is not shipped."""
    mentioned = {command
                 for doc in DOCS
                 for _, text in invocations((ROOT / doc).read_text())
                 if (command := _command_in(text)) is not None}

    assert not subcommands() - mentioned


class TestTheDocCheckItself:
    def test_it_catches_a_renamed_command(self):
        assert _command_in("grow-up analyse") == "analyse"

    def test_flags_are_not_mistaken_for_commands(self):
        assert _command_in("grow-up -v analyze") == "analyze"
        assert _command_in("grow-up --config config.example.toml status") == "status"
        assert _command_in("grow-up --help") is None

    def test_a_placeholder_is_not_a_command(self):
        assert _command_in("grow-up <command>") is None

    def test_prose_is_not_scanned(self):
        assert invocations("The `grow-up` tool, which grow-up says is fine.\n") == []

    def test_both_code_spans_and_fenced_blocks_are(self):
        text = "Run `grow-up status` first.\n\n```bash\ngrow-up encode   # then this\n```\n"
        assert invocations(text) == [(1, "grow-up status"), (4, "grow-up encode")]
