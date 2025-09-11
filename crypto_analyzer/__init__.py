"""Lightweight top-level package for Crypto Analyzer.

This module avoids heavy imports at import time. Submodules are lazily
loaded via ``importlib`` when accessed. The public API remains stable and
minimal to keep cold-start times fast.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["analysis", "ml", "prediction", "legacy", "inference"]

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    import analysis  # type: ignore
    import ml  # type: ignore
    import prediction  # type: ignore
    from . import legacy, inference

def __getattr__(name: str):
    if name in __all__:
        return import_module(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
