import json
from pathlib import Path
from typing import Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


def _save_metadata(
    feature_cols: Iterable[str], version: str, feature_list_path: str, version_path: str
) -> None:
    Path(feature_list_path).parent.mkdir(parents=True, exist_ok=True)
    with open(feature_list_path, "w", encoding="utf-8") as f:
        json.dump(list(feature_cols), f)
    with open(version_path, "w", encoding="utf-8") as f:
        json.dump({"version": version}, f)


def fit_meta_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: Iterable[str],
    *,
    model_path: str = "ml/meta_model_cls.joblib",
    feature_list_path: str = "ml/feature_list.json",
    version_path: str = "ml/meta_version.json",
    version: str = "v1",
    n_splits: int = 5,
    gap: int = 0,
    random_state: int = 42,
    n_estimators: int = 200,
) -> Tuple[RandomForestClassifier, float]:
    """Train and persist the meta classification model.

    Returns the fitted model and mean F1 score from ``TimeSeriesSplit``.
    """

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            oob_score=True,
            warm_start=True,
            n_jobs=-1,
            random_state=random_state,
        )
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = clf.predict(X.iloc[test_idx])
        scores.append(float(f1_score(y.iloc[test_idx], preds)))

    final_model = RandomForestClassifier(
        n_estimators=n_estimators,
        oob_score=True,
        warm_start=True,
        n_jobs=-1,
        random_state=random_state,
    )
    final_model.fit(X, y)
    joblib.dump(final_model, model_path)
    _save_metadata(feature_cols, version, feature_list_path, version_path)
    return final_model, float(np.mean(scores))


def fit_meta_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: Iterable[str],
    *,
    model_path: str = "ml/meta_model_reg.joblib",
    feature_list_path: str = "ml/feature_list.json",
    version_path: str = "ml/meta_version.json",
    version: str = "v1",
    n_splits: int = 5,
    gap: int = 0,
    random_state: int = 42,
    n_estimators: int = 200,
) -> Tuple[RandomForestRegressor, float]:
    """Train and persist the meta regression model.

    Returns the fitted model and mean MAE from ``TimeSeriesSplit``.
    """

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    maes: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        reg = RandomForestRegressor(
            n_estimators=n_estimators,
            oob_score=True,
            warm_start=True,
            n_jobs=-1,
            random_state=random_state,
        )
        reg.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = reg.predict(X.iloc[test_idx])
        maes.append(float(mean_absolute_error(y.iloc[test_idx], pred)))

    final_model = RandomForestRegressor(
        n_estimators=n_estimators,
        oob_score=True,
        warm_start=True,
        n_jobs=-1,
        random_state=random_state,
    )
    final_model.fit(X, y)
    joblib.dump(final_model, model_path)
    _save_metadata(feature_cols, version, feature_list_path, version_path)
    return final_model, float(np.mean(maes))


def predict_meta(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    model_path: str,
    *,
    batch_size: int = 1000,
    proba: bool = False,
) -> np.ndarray:
    """Predict using a stored meta model with batching and memory mapping."""

    model = joblib.load(model_path, mmap_mode="r")
    X = df[list(feature_cols)]
    preds = []
    for start in range(0, len(X), batch_size):
        batch = X.iloc[start : start + batch_size]
        if proba and hasattr(model, "predict_proba"):
            preds.append(model.predict_proba(batch)[:, 1])
        else:
            preds.append(model.predict(batch))
    return np.concatenate(preds)


__all__ = [
    "fit_meta_classifier",
    "fit_meta_regressor",
    "predict_meta",
]
