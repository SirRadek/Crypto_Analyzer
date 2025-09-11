"""Dummy legacy ensemble implementation used via lazy wrappers."""
from __future__ import annotations

from typing import Iterable


def predict_weighted(df, feature_cols: Iterable[str], model_paths, *, usage_path=None):
    """Simple weighted prediction placeholder.

    This function is intentionally small; real logic lives elsewhere in the
    project.  It exists so that the wrapper can lazily import it only when
    needed.
    """
    weights = [1 / len(model_paths)] * len(model_paths)
    preds = [0] * len(df)
    return preds
