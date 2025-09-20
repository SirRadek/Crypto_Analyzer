"""Feature stability utilities used for model diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.inspection import permutation_importance

__all__ = [
    "SelectionConfig",
    "permutation_importance_by_regime",
    "shap_importance_by_regime",
    "drop_correlated_features",
    "export_top_features",
]


@dataclass(slots=True)
class SelectionConfig:
    """Bundle configuration for stability analysis."""

    n_repeats: int = 5
    random_state: int | None = 42
    shap_samples: int = 500
    correlation_threshold: float = 0.95


def permutation_importance_by_regime(
    model: ClassifierMixin,
    X: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
    *,
    config: SelectionConfig | None = None,
) -> pd.DataFrame:
    """Compute permutation importance conditioned on trading regimes."""

    cfg = config or SelectionConfig()
    results: list[pd.DataFrame] = []
    for regime, mask in regimes.groupby(regimes).groups.items():
        if len(mask) < 5:
            continue
        imp = permutation_importance(
            model,
            X.iloc[mask],
            y.iloc[mask],
            n_repeats=cfg.n_repeats,
            random_state=cfg.random_state,
        )
        df = pd.DataFrame(
            {
                "feature": X.columns,
                "importance": imp.importances_mean,
                "std": imp.importances_std,
                "regime": regime,
            }
        )
        results.append(df)
    if not results:
        return pd.DataFrame(columns=["feature", "importance", "std", "regime"])
    return pd.concat(results, ignore_index=True)


def shap_importance_by_regime(
    model: ClassifierMixin,
    X: pd.DataFrame,
    regimes: pd.Series,
    *,
    config: SelectionConfig | None = None,
) -> pd.DataFrame:
    """Compute mean absolute SHAP values for each regime."""

    cfg = config or SelectionConfig()
    try:
        import shap  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install `shap` to compute SHAP importances") from exc

    explainer = shap.Explainer(model, X)
    results: list[pd.DataFrame] = []
    for regime, mask in regimes.groupby(regimes).groups.items():
        if len(mask) == 0:
            continue
        sample = X.iloc[mask].sample(n=min(cfg.shap_samples, len(mask)), random_state=cfg.random_state)
        values = explainer(sample).values
        mean_abs = np.abs(values).mean(axis=0)
        df = pd.DataFrame({"feature": X.columns, "shap": mean_abs, "regime": regime})
        results.append(df)
    if not results:
        return pd.DataFrame(columns=["feature", "shap", "regime"])
    return pd.concat(results, ignore_index=True)


def drop_correlated_features(
    df: pd.DataFrame,
    *,
    threshold: float,
    features: Iterable[str] | None = None,
) -> list[str]:
    """Return a list of features to keep after removing correlated duplicates."""

    cols = list(features) if features is not None else list(df.columns)
    corr = df[cols].corr().abs()
    to_drop: set[str] = set()
    for i, col in enumerate(cols):
        if col in to_drop:
            continue
        for other in cols[i + 1 :]:
            if other in to_drop:
                continue
            if corr.loc[col, other] >= threshold:
                to_drop.add(other)
    kept = [c for c in cols if c not in to_drop]
    return kept


def export_top_features(
    feature_scores: Mapping[str, float],
    *,
    horizon: str,
    top_n: int,
    output_path: str | Path,
) -> None:
    """Persist the top-N ranked features for a given horizon."""

    sorted_features = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)[:top_n]
    data = {
        "horizon": horizon,
        "features": [name for name, _ in sorted_features],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    import json

    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

