"""Config loading. Everything tunable lives in config.toml; secrets live in env.

Since 1.0.0 an existing config.toml must keep working. New settings arrive with
defaults and a fallback for their absence -- see `sources()`, where a file with
no `[[immich.sources]]` yields exactly the single source 1.0.0 assumed.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path("config.toml")


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)
    root: Path = Path(".")

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.raw.get(section, {}).get(key, default)

    def path(self, key: str) -> Path:
        value = self.get("paths", key)
        if value is None:
            raise KeyError(f"paths.{key} is not set in config")
        return self.root / value


# Settings that were replaced rather than kept, and what replaced them. Failing
# on these beats ignoring them: a key that used to control framing, silently
# dropped, reads as "the new setting doesn't work".
#
# Frozen at its pre-1.0 contents. Everything here predates the first release, so
# no published config can contain one. Do not add: a released setting has users,
# and breaking their file to tidy a name is not a trade worth making.
REMOVED_KEYS = {
    ("output", "left_eye"): "align.eye_distance / align.eye_level",
    ("output", "right_eye"): "align.eye_distance / align.eye_level",
    ("output", "smoothing_window"): "nothing — transform smoothing was removed",
    ("output", "smoothing_polyorder"): "nothing — transform smoothing was removed",
    ("analyze", "gaze_method"): "nothing — gaze is always measured geometrically",
}


def check_removed(raw: dict) -> None:
    stale = [(f"{section}.{key}", replacement)
             for (section, key), replacement in REMOVED_KEYS.items()
             if key in raw.get(section, {})]
    if not stale:
        return
    detail = "\n".join(f"    {name}  ->  {replacement}" for name, replacement in stale)
    raise RuntimeError(f"config.toml uses settings that no longer exist:\n{detail}")


def load(path: str | Path | None = None) -> Config:
    p = Path(path or DEFAULT_CONFIG)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Copy config.example.toml to config.toml and edit it."
        )
    with p.open("rb") as fh:
        raw = tomllib.load(fh)
    check_removed(raw)
    return Config(raw=raw, root=p.parent if p.parent != Path("") else Path("."))


@dataclass(frozen=True)
class Credentials:
    url: str
    api_key: str


DEFAULT_URL_ENV = "IMMICH_URL"
DEFAULT_KEY_ENV = "IMMICH_API_KEY"


def _missing_env(url_env: str, key_env: str) -> list[str]:
    """Which of the two variables are unset, in the order they are documented."""
    url = os.environ.get(url_env, "").rstrip("/")
    api_key = os.environ.get(key_env, "")
    return [n for n, v in ((url_env, url), (key_env, api_key)) if not v]


def _credentials_from(url_env: str, key_env: str, whose: str = "") -> Credentials:
    url = os.environ.get(url_env, "").rstrip("/")
    api_key = os.environ.get(key_env, "")
    missing = _missing_env(url_env, key_env)
    if missing:
        raise RuntimeError(
            f"missing environment variable(s){whose}: {', '.join(missing)}"
        )
    if not url.endswith("/api"):
        url = f"{url}/api"
    return Credentials(url=url, api_key=api_key)


def credentials() -> Credentials:
    """Read Immich credentials from the environment.

    Kept out of config.toml so the config file stays committable.
    """
    return _credentials_from(DEFAULT_URL_ENV, DEFAULT_KEY_ENV)


# The name given to the single source synthesised from a pre-1.0 config. It is
# written into assets.source, so changing it would orphan every indexed row.
LEGACY_SOURCE_NAME = "default"


@dataclass(frozen=True)
class Source:
    """One Immich account, and the person record for the subject *within it*.

    Immich scopes face recognition per account: the association between a
    detected face and a person belongs to whoever owns it. A partner's photos
    are therefore invisible to this account's person id no matter how the search
    is phrased, and the only way to reach them is to ask again as them. Hence a
    key, a URL and a person id travelling together.

    `name` is an identity rather than a label -- it is stored on every asset so
    later stages know which key can download it -- so renaming one re-indexes it.
    """

    name: str
    person_name: str = ""
    person_id: str = ""
    url_env: str = DEFAULT_URL_ENV
    key_env: str = DEFAULT_KEY_ENV

    def credentials(self) -> Credentials:
        return _credentials_from(self.url_env, self.key_env, f" for source {self.name!r}")

    def missing_env(self) -> list[str]:
        """The variables this account needs and does not have."""
        return _missing_env(self.url_env, self.key_env)


def check_credentials(sources: list[Source]) -> None:
    """Report every account's missing variables at once, before any work starts.

    Resolving these lazily meant one round trip per account: set the first
    account's two variables, re-run, and be told about the second -- each failure
    looking like a fresh problem rather than the rest of the one already being
    fixed. Setting up two accounts is exactly when the reader is least sure what
    the whole list should be.

    Same reasoning as `cli.preflight_all` checking every key before any of them
    downloads anything, and it runs first because a variable that is not set is
    cheaper to discover than a key that is not accepted.

    One account's worth of gaps keeps the message it has always had; only the
    genuinely new case -- more than one account short -- gets the longer form.
    """
    gaps = [(source.name, missing)
            for source in sources if (missing := source.missing_env())]
    if not gaps:
        return
    if len(gaps) == 1:
        name, missing = gaps[0]
        raise RuntimeError(
            f"missing environment variable(s) for source {name!r}: {', '.join(missing)}"
        )
    detail = "\n".join(f"    source {name!r}: {', '.join(missing)}"
                       for name, missing in gaps)
    raise RuntimeError(f"missing environment variable(s):\n{detail}")


def sources(cfg: Config) -> list[Source]:
    """Every configured account, in order. Never empty.

    A config with no `[[immich.sources]]` yields exactly the one source 1.0.0
    assumed, on the same two environment variables. That is the whole
    compatibility promise for the released config shape, so it is deliberately
    the plain path through this function rather than a special case bolted on.
    """
    declared = cfg.raw.get("immich", {}).get("sources")
    if not declared:
        return [Source(
            name=LEGACY_SOURCE_NAME,
            person_name=str(cfg.get("immich", "person_name") or ""),
            person_id=str(cfg.get("immich", "person_id") or ""),
        )]

    parsed = [
        Source(
            name=str(entry.get("name", "")).strip(),
            person_name=str(entry.get("person_name") or ""),
            person_id=str(entry.get("person_id") or ""),
            url_env=str(entry.get("url_env") or DEFAULT_URL_ENV),
            key_env=str(entry.get("key_env") or DEFAULT_KEY_ENV),
        )
        for entry in declared
    ]
    _reject_ambiguous(parsed)
    return parsed


def _reject_ambiguous(parsed: list[Source]) -> None:
    """Fail on the two config mistakes that would otherwise look like bugs.

    A repeated name silently merges two accounts' assets under one identity. A
    repeated (url_env, key_env) pair is a copy-pasted block still pointing at the
    first account, which indexes it twice and looks like the partner's photos
    simply are not there -- the exact symptom the feature exists to fix.
    """
    for index, source in enumerate(parsed, start=1):
        if not source.name:
            raise RuntimeError(f"[[immich.sources]] #{index} has no name")

    for field, label in (("name", "name"), (("url_env", "key_env"), "credentials")):
        seen: dict = {}
        for source in parsed:
            key = (getattr(source, field) if isinstance(field, str)
                   else tuple(getattr(source, f) for f in field))
            if key in seen:
                raise RuntimeError(
                    f"[[immich.sources]] {seen[key]!r} and {source.name!r} share the same "
                    f"{label}; each source needs its own"
                )
            seen[key] = source.name
