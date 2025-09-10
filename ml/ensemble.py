"""Utilities for combining multiple models into meta-models.

This module tracks model usage, loads validation performance metrics,
and provides helpers for stacking base model predictions into a single
meta-classifier or meta-regressor.  During training and inference each
base model's contribution is weighted by its validation accuracy and
historical usage count.  Models whose weights evaluate to zero are
reported so they can be considered for removal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from .train_regressor import train_regressor as _train_regressor

try:
    from numpy.core._exceptions import _ArrayMemoryError
    _MEM_ERRORS = (MemoryError, _ArrayMemoryError)
except Exception:  # pragma: no cover
    _MEM_ERRORS = (MemoryError,)


def _safe_load(path: str):
    try:
        return joblib.load(path)
    except _MEM_ERRORS:
        return joblib.load(path, mmap_mode="r")

# ---------------------------------------------------------------------------
# Paths to auxiliary data
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
USAGE_PATH = BASE_DIR / "model_usage.json"
PERF_PATH = BASE_DIR / "model_performance.json"
META_CLF_PATH = BASE_DIR / "meta_classifier.pkl"
META_REG_PATH = BASE_DIR / "meta_regressor.pkl"


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def load_usage_counts(path: Path = USAGE_PATH) -> Dict[str, int]:
    """Return mapping of model path to usage counts."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_usage_counts(counts: Dict[str, int], path: Path = USAGE_PATH) -> None:
    """Persist usage counts to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)


def increment_usage(model_names: Iterable[str], path: Path = USAGE_PATH) -> None:
    """Increment usage counter for provided model names."""
    counts = load_usage_counts(path)
    for name in model_names:
        counts[name] = counts.get(name, 0) + 1
    save_usage_counts(counts, path)


def load_performance(path: Path = PERF_PATH) -> Dict[str, float]:
    """Load validation accuracy scores for models.

    The file is expected to contain a JSON mapping of model path to its
    validation accuracy (0..1).  Missing entries default to 0.
    """

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------

def compute_weights(
    usage_counts: Dict[str, int],
    performance: Dict[str, float],
    model_names: Iterable[str],
) -> Tuple[np.ndarray, list]:
    """Return normalized weights and list of zero-weight models.

    The weight for each model is defined as

    ``weight = usage_count * validation_accuracy``.

    Parameters
    ----------
    usage_counts:
        Mapping of model name to how many times it has been used.
    performance:
        Mapping of model name to validation accuracy from backtests.
    model_names:
        Iterable of model identifiers to compute weights for.

    Returns
    -------
    weights : np.ndarray
        Normalized weights summing to one.
    zero_weight_models : list[str]
        Models whose weight evaluated to zero (suggest removal).
    """

    names = list(model_names)
    counts = np.array([usage_counts.get(n, 0) for n in names], dtype=float)
    scores = np.array([performance.get(n, 0) for n in names], dtype=float)
    weights = counts * scores

    zero_weight = [n for n, w in zip(names, weights) if w == 0]
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    return weights / weights.sum(), zero_weight


# ---------------------------------------------------------------------------
# Base model ensembling (weighted averaging)
# ---------------------------------------------------------------------------

def predict_weighted(
    df,
    feature_cols,
    model_paths,
    usage_counts_path: Path = USAGE_PATH,
    performance_path: Path = PERF_PATH,
):
    """Predict using multiple models combined by usage/performance weights.

    This is a lightweight ensemble that performs a weighted average of
    the base model outputs without training an additional meta model.
    """

    usage_counts = load_usage_counts(usage_counts_path)
    performance = load_performance(performance_path)
    names = list(model_paths)
    weights, zero_weight = compute_weights(usage_counts, performance, names)
    X = df[feature_cols]
    preds = np.zeros(len(df))
    for path, w in zip(names, weights):
        model = _safe_load(path)
        preds += model.predict(X) * w
        del model
    increment_usage(names, usage_counts_path)
    if zero_weight:
        print(f"Models with zero weight (consider removing): {zero_weight}")
    return preds


# ---------------------------------------------------------------------------
# Meta-model helpers
# ---------------------------------------------------------------------------

def _stack_features(df, feature_cols, horizon_col, model_paths, weights):
    """Construct stacked feature matrix for meta models."""

    X_base = df[feature_cols]
    X_meta = df[feature_cols + [horizon_col]].copy()
    for path, w in zip(model_paths, weights):
        model = _safe_load(path)
        preds = model.predict(X_base)
        X_meta[path] = preds * w
        del model
    return X_meta


def train_meta_classifier(
    df,
    feature_cols,
    horizon_col,
    target_col,
    model_paths,
    meta_model_path: Path = META_CLF_PATH,
    usage_counts_path: Path = USAGE_PATH,
    performance_path: Path = PERF_PATH,
):
    """Train a single meta-classifier using base model predictions."""

    usage_counts = load_usage_counts(usage_counts_path)
    performance = load_performance(performance_path)
    names = list(model_paths)
    weights, zero_weight = compute_weights(usage_counts, performance, names)
    X_meta = _stack_features(df, feature_cols, horizon_col, names, weights)
    y = df[target_col]
    meta = RandomForestClassifier(n_estimators=200, random_state=42)
    meta.fit(X_meta, y)
    joblib.dump(meta, meta_model_path)
    if zero_weight:
        print(f"Models with zero weight (consider removing): {zero_weight}")
    return meta


def train_meta_regressor(
    df,
    feature_cols,
    horizon_col,
    target_col,
    model_paths,
    meta_model_path: Path = META_REG_PATH,
    usage_counts_path: Path = USAGE_PATH,
    performance_path: Path = PERF_PATH,
):
    """Train a single meta-regressor using base model predictions."""

    usage_counts = load_usage_counts(usage_counts_path)
    performance = load_performance(performance_path)
    names = list(model_paths)
    weights, zero_weight = compute_weights(usage_counts, performance, names)
    X_meta = _stack_features(df, feature_cols, horizon_col, names, weights)
    y = df[target_col]

    params = dict(n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=2, verbose=0)
    meta = _train_regressor(X_meta, y, model_path=str(meta_model_path), params=params)
    if zero_weight:
        print(f"Models with zero weight (consider removing): {zero_weight}")
    return meta


def predict_meta_classifier(
    df,
    feature_cols,
    horizon_col,
    model_paths,
    meta_model_path: Path = META_CLF_PATH,
    usage_counts_path: Path = USAGE_PATH,
    performance_path: Path = PERF_PATH,
):
    """Predict using the trained meta-classifier."""

    usage_counts = load_usage_counts(usage_counts_path)
    performance = load_performance(performance_path)
    names = list(model_paths)
    weights, zero_weight = compute_weights(usage_counts, performance, names)
    X_meta = _stack_features(df, feature_cols, horizon_col, names, weights)
    meta = joblib.load(meta_model_path)
    increment_usage(names, usage_counts_path)
    if zero_weight:
        print(f"Models with zero weight (consider removing): {zero_weight}")
    return meta.predict(X_meta)


def predict_meta_regressor(
    df,
    feature_cols,
    horizon_col,
    model_paths,
    meta_model_path: Path = META_REG_PATH,
    usage_counts_path: Path = USAGE_PATH,
    performance_path: Path = PERF_PATH,
):
    """Predict using the trained meta-regressor."""

    usage_counts = load_usage_counts(usage_counts_path)
    performance = load_performance(performance_path)
    names = list(model_paths)
    weights, zero_weight = compute_weights(usage_counts, performance, names)
    X_meta = _stack_features(df, feature_cols, horizon_col, names, weights)
    meta = joblib.load(meta_model_path)
    increment_usage(names, usage_counts_path)
    if zero_weight:
        print(f"Models with zero weight (consider removing): {zero_weight}")
    return meta.predict(X_meta)


__all__ = [
    "load_usage_counts",
    "save_usage_counts",
    "increment_usage",
    "load_performance",
    "compute_weights",
    "predict_weighted",
    "train_meta_classifier",
    "train_meta_regressor",
    "predict_meta_classifier",
    "predict_meta_regressor",
]

