"""Prediction helpers for classification models."""

from __future__ import annotations

import importlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    pass


def _load_threshold(path: str) -> float:
    try:
        with open(path, encoding="utf-8") as f:
            return float(json.load(f)["threshold"])
    except Exception:
        return 0.5


def _lazy_meta():
    return importlib.import_module("ml.meta")


def predict_ml(
    df,
    feature_cols,
    model_path: str = "ml/meta_model_cls.joblib",
    threshold_path: str = "ml/threshold.json",
):
    """Predict class labels (0/1) using the calibrated meta classifier."""

    probas = _lazy_meta().predict_meta(df, feature_cols, model_path, proba=True)
    threshold = _load_threshold(threshold_path)
    return (probas >= threshold).astype(int)


def predict_ml_proba(df, feature_cols, model_path: str = "ml/meta_model_cls.joblib"):
    """Return calibrated probability of the positive class (price up)."""
    return _lazy_meta().predict_meta(df, feature_cols, model_path, proba=True)


def predict_weighted(
    df,
    feature_cols,
    model_paths,
    usage_path="ml/model_usage.json",
):
    """Backward-compatible wrapper for weighted base-model ensembling."""

    from .ensemble import predict_weighted as _predict_weighted

    return _predict_weighted(
        df,
        feature_cols,
        model_paths,
        usage_counts_path=usage_path,
    )


__all__ = ["predict_ml", "predict_ml_proba", "predict_weighted"]
