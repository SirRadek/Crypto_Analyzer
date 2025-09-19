"""Lazy wrappers for legacy ensemble helpers."""
from __future__ import annotations

import importlib

__all__ = ["predict_weighted"]


def _lazy_import_ensemble():
    return importlib.import_module("crypto_analyzer.legacy.ensemble_core")


def predict_weighted(*args, **kwargs):
    mod = _lazy_import_ensemble()
    return mod.predict_weighted(*args, **kwargs)
