"""Machine learning package with lazy imports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["fit_meta_classifier", "fit_meta_regressor", "predict_meta"]

if TYPE_CHECKING:  # pragma: no cover
    from .meta import fit_meta_classifier, fit_meta_regressor, predict_meta


def __getattr__(name: str):
    if name in __all__:
        mod = import_module("ml.meta")
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
