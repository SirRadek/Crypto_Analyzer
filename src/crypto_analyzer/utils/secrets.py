"""Utilities for loading environment configuration securely."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

__all__ = ["load_environment", "get_secret"]

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
_ENV_LOADED: bool = False


def load_environment(path: str | Path | None = None, *, override: bool = False) -> None:
    """Load environment variables from ``.env`` without exposing their values."""

    global _ENV_LOADED
    if _ENV_LOADED and path is None:
        return

    env_path = Path(path) if path is not None else _PROJECT_ROOT / ".env"
    load_dotenv(env_path, override=override)
    _ENV_LOADED = True


def get_secret(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    """Return the secret ``name`` without logging the resolved value."""

    value = os.getenv(name)
    if value is not None and value.strip() == "":
        value = None

    if value is None:
        value = default

    if required and value is None:
        raise RuntimeError(f"Required secret {name!r} is missing.")

    return value


load_environment()
