"""Config loading. Everything tunable lives in config.toml; secrets live in env."""

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
REMOVED_KEYS = {
    ("output", "left_eye"): "align.eye_distance / align.eye_level",
    ("output", "right_eye"): "align.eye_distance / align.eye_level",
    ("output", "smoothing_window"): "nothing — transform smoothing was removed",
    ("output", "smoothing_polyorder"): "nothing — transform smoothing was removed",
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


def credentials() -> Credentials:
    """Read Immich credentials from the environment.

    Kept out of config.toml so the config file stays committable.
    """
    url = os.environ.get("IMMICH_URL", "").rstrip("/")
    api_key = os.environ.get("IMMICH_API_KEY", "")
    missing = [n for n, v in (("IMMICH_URL", url), ("IMMICH_API_KEY", api_key)) if not v]
    if missing:
        raise RuntimeError(f"missing environment variable(s): {', '.join(missing)}")
    if not url.endswith("/api"):
        url = f"{url}/api"
    return Credentials(url=url, api_key=api_key)
