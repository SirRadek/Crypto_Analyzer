"""Helpers for making predictions with regression models."""

import numpy as np

from .train_regressor import load_regressor
from .predict import _load_usage_counts, _save_usage_counts


def predict_prices(df, feature_cols, model_path="ml/model_reg.pkl"):
    """Predict prices using a single regressor model."""
    model = load_regressor(model_path)
    X = df[feature_cols]
    return model.predict(X)


def predict_weighted_prices(
    df,
    feature_cols,
    model_paths,
    usage_path="ml/model_usage.json",
):
    """Predict prices with multiple models combined by usage-based weights.

    Parameters
    ----------
    df : pandas.DataFrame
        Feature data for prediction.
    feature_cols : list[str]
        Columns used as features.
    model_paths : list[str]
        Paths to regressor models.
    usage_path : str, optional
        Location of JSON file storing model usage counts.

    Returns
    -------
    numpy.ndarray
        Weighted average predictions from all models.
    """

    counts = _load_usage_counts(usage_path)
    counts = {path: counts.get(path, 0) for path in model_paths}
    total = sum(counts.values())
    if total == 0:
        weights = {path: 1 / len(model_paths) for path in model_paths}
    else:
        weights = {path: counts[path] / total for path in model_paths}

    models = {path: load_regressor(path) for path in model_paths}
    X = df[feature_cols]
    preds = np.array([models[path].predict(X) for path in model_paths])
    weighted = np.average(
        preds, axis=0, weights=[weights[path] for path in model_paths]
    )

    for path in model_paths:
        counts[path] = counts.get(path, 0) + len(df)
    _save_usage_counts(counts, usage_path)

    return weighted
