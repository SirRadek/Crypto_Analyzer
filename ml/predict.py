import json
import os

import numpy as np

from .train import load_model


def predict_ml(df, feature_cols, model_path="ml/model.pkl"):
    """Predict class labels (0/1) using a trained classifier."""
    model = load_model(model_path)
    X = df[feature_cols]
    preds = model.predict(X)
    return preds


def predict_ml_proba(df, feature_cols, model_path="ml/model.pkl"):
    """Return probability of the positive class (price up)."""
    model = load_model(model_path)
    X = df[feature_cols]
    proba = model.predict_proba(X)[:, 1]
    return proba


def _load_usage_counts(usage_path):
    """Load usage counts from a JSON file."""
    if not os.path.exists(usage_path):
        return {}
    with open(usage_path, "r") as f:
        return json.load(f)


def _save_usage_counts(counts, usage_path):
    """Save usage counts to a JSON file."""
    with open(usage_path, "w") as f:
        json.dump(counts, f, indent=2)


def predict_weighted(df, feature_cols, model_paths, usage_path="ml/model_usage.json"):
    """Predict using multiple models combined with usage-based weights.

    Parameters
    ----------
    df : pandas.DataFrame
        Data containing feature columns.
    feature_cols : list[str]
        Names of the features to use for prediction.
    model_paths : list[str]
        Paths to model files to load.
@@ -50,26 +54,26 @@ def predict_weighted(df, feature_cols, model_paths, usage_path="ml/model_usage.j
    Returns
    -------
    numpy.ndarray
        Final binary predictions after weighting model outputs.
    """

    counts = _load_usage_counts(usage_path)
    counts = {path: counts.get(path, 0) for path in model_paths}
    total = sum(counts.values())
    if total == 0:
        weights = {path: 1 / len(model_paths) for path in model_paths}
    else:
        weights = {path: count / total for path, count in counts.items()}

    models = {path: load_model(path) for path in model_paths}
    X = df[feature_cols]
    preds = np.array([models[path].predict(X) for path in model_paths])
    weight_array = np.array([weights[path] for path in model_paths])
    weighted_pred = np.average(preds, axis=0, weights=weight_array)
    final = (weighted_pred >= 0.5).astype(int)

    for path in model_paths:
        counts[path] = counts.get(path, 0) + 1
    _save_usage_counts(counts, usage_path)

    return final