"""Environment-based configuration helpers for the research scripts."""

from __future__ import annotations

import os
from pathlib import Path


def env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable with a useful error message."""
    raw = os.getenv(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of true/false, yes/no, on/off, or 1/0; got {raw!r}"
    )


def env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    """Read an integer environment variable and optionally enforce a minimum."""
    raw = os.getenv(name)
    try:
        value = default if raw is None else int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer; got {raw!r}") from exc

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}")
    return value


def env_path(name: str, default: str | Path) -> Path:
    """Read a filesystem path without requiring it to exist yet."""
    return Path(os.getenv(name, str(default))).expanduser()
