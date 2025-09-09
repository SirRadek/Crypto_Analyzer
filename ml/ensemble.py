import json
from pathlib import Path

import joblib
import numpy as np

USAGE_PATH = Path(__file__).resolve().parent / "model_usage.json"


def load_usage_counts(path: Path = USAGE_PATH) -> dict:
    """Load a mapping of model path to usage counts."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_usage_counts(usage_counts: dict, path: Path = USAGE_PATH) -> None:
    """Persist usage counts to disk."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(usage_counts, f, indent=2)


def increment_usage(model_names, path: Path = USAGE_PATH) -> None:
    """Increment usage counter for provided model names."""
    counts = load_usage_counts(path)
    for name in model_names:
        counts[name] = counts.get(name, 0) + 1
    save_usage_counts(counts, path)


def compute_weights(usage_counts: dict, model_names):
    """Return normalized weights based on usage counts."""
    counts = np.array([usage_counts.get(name, 0) for name in model_names], dtype=float)
    if counts.sum() == 0:
        counts = np.ones_like(counts)
    return counts / counts.sum()


def load_models(model_paths):
    """Load models given a list of file paths."""
    return {path: joblib.load(path) for path in model_paths}


def predict_weighted(df, feature_cols, model_paths, usage_counts_path: Path = USAGE_PATH):
    """Predict using multiple models combined by usage-based weights."""
    usage_counts = load_usage_counts(usage_counts_path)
    models = load_models(model_paths)
    names = list(models.keys())
    weights = compute_weights(usage_counts, names)
    X = df[feature_cols]
    preds = np.array([models[name].predict(X) for name in names])
    increment_usage(names, usage_counts_path)
    return np.average(preds, axis=0, weights=weights)