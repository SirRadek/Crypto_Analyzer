from __future__ import annotations

from collections.abc import Iterable
from typing import Sequence

import joblib
import numpy as np


def _predict_single(model_path: str, features: np.ndarray) -> np.ndarray:
    model = joblib.load(model_path)
    try:
        preds = model.predict_proba(features)[:, 1]
    except AttributeError:
        preds = model.predict(features)
    return np.asarray(preds, dtype=np.float32)


def predict_weighted(
    df,
    feature_cols: Sequence[str],
    model_paths: Iterable[str],
    *,
    usage_counts_path=None,
    performance_path=None,
) -> np.ndarray:
    """Average predictions from ``model_paths`` using equal weights."""

    cols = list(feature_cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features for ensemble prediction: {missing}")

    features = df[cols].to_numpy(dtype=np.float32)
    paths = list(model_paths)
    if not paths:
        raise ValueError("At least one model path must be provided")

    preds = np.zeros(len(df), dtype=np.float32)
    weight = 1.0 / len(paths)
    for path in paths:
        preds += _predict_single(path, features) * weight
    return preds


__all__ = ["predict_weighted"]
