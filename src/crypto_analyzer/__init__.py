"""Lightweight top-level package for Crypto Analyzer.

This module avoids heavy imports at import time. Submodules are lazily
loaded via ``importlib`` when accessed. The public API remains stable and
minimal to keep cold-start times fast.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = ["data", "features", "labeling", "models", "eval", "utils", "legacy"]

_SUBMODULES = {name: f"{__name__}.{name}" for name in __all__}
_ALIAS_MAP = {
    "analysis": _SUBMODULES["features"],
    "ml": _SUBMODULES["models"],
    "prediction": f"{__name__}.models.predictor",
}

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from . import data, eval, features, labeling, legacy, models, utils


def __getattr__(name: str) -> Any:
    target = _SUBMODULES.get(name) or _ALIAS_MAP.get(name)
    if target is not None:
        return import_module(target)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - trivial
    return sorted(set(__all__) | set(_ALIAS_MAP))
