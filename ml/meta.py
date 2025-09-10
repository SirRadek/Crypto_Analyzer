import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple, Union, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
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
    threshold_path: str = "ml/threshold.json",
    threshold_func: Callable[[np.ndarray, pd.Series], float] | None = None,
) -> Tuple[CalibratedClassifierCV, float]:
    """Train and persist the meta classification model with calibration.

    Returns the fitted calibrated model and mean F1 score from ``TimeSeriesSplit``.
    ``threshold_path`` stores a JSON file with the optimal decision threshold.
    """

    method: Literal["isotonic", "sigmoid"] = (
        "isotonic" if len(y) > 50_000 else "sigmoid"
    )

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores: list[float] = []
    for train_idx, test_idx in tscv.split(X):
        base = RandomForestClassifier(
            n_estimators=n_estimators,
            oob_score=True,
            warm_start=True,
            n_jobs=-1,
            random_state=random_state,
        )
        clf = CalibratedClassifierCV(base, method=method, cv=3)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        probas = clf.predict_proba(X.iloc[test_idx])[:, 1]
        preds = (probas >= 0.5).astype(int)
        scores.append(float(f1_score(y.iloc[test_idx], preds)))

    final_base = RandomForestClassifier(
        n_estimators=n_estimators,
        oob_score=True,
        warm_start=True,
        n_jobs=-1,
        random_state=random_state,
    )
    final_model = CalibratedClassifierCV(final_base, method=method, cv=3)
    final_model.fit(X, y)
    joblib.dump(final_model, model_path)
    _save_metadata(feature_cols, version, feature_list_path, version_path)

    probas = final_model.predict_proba(X)[:, 1]
    if threshold_func is None:
        thresholds = np.linspace(0.0, 1.0, 101)
        f1s = np.array([f1_score(y, probas >= t) for t in thresholds])
        threshold = float(thresholds[int(np.argmax(f1s))])
    else:
        threshold = float(threshold_func(probas, y))
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold}, f)

    return final_model, float(np.mean(scores))


def fit_meta_regressor(
    X: pd.DataFrame,
    y: Union[pd.Series, pd.DataFrame],
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
    multi_output: bool = False,
) -> Tuple[RandomForestRegressor, Union[float, Dict[int, float]]]:
    """Train and persist the meta regression model.

    Parameters
    ----------
    multi_output:
        If ``True``, ``y`` must have shape ``(n_samples, n_horizons)`` and the
        regressor is trained to predict all horizons jointly. The returned
        metric is a ``dict`` mapping horizon index (1-based) to MAE. If
        ``False``, behaves like a standard single-output regressor and returns
        the mean MAE across splits.

    Returns
    -------
    RandomForestRegressor
        The fitted model.
    float or dict
        Mean MAE or per-horizon MAE depending on ``multi_output``.
    """

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    maes: list[Any] = []
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
        if multi_output:
            maes.append(
                mean_absolute_error(y.iloc[test_idx], pred, multioutput="raw_values")
            )
        else:
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

    if multi_output:
        mae_arr = np.mean(maes, axis=0)
        return final_model, {i + 1: float(m) for i, m in enumerate(mae_arr)}
    else:
        return final_model, float(np.mean(maes))


def predict_meta(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    model_path: str,
    *,
    batch_size: int = 1000,
    proba: bool = False,
    multi_output: bool = False,
) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """Predict using a stored meta model with batching and memory mapping.

    Parameters
    ----------
    multi_output:
        If ``True``, the loaded model is expected to output multiple horizons.
        The return value is a dictionary ``{horizon: np.ndarray}`` with
        1-based horizon indices.
    """

    if proba and multi_output:
        raise ValueError("`proba` and `multi_output` cannot both be True")

    model = joblib.load(model_path, mmap_mode="r")
    X = df[list(feature_cols)]
    preds = []
    for start in range(0, len(X), batch_size):
        batch = X.iloc[start : start + batch_size]
        if proba and hasattr(model, "predict_proba"):
            preds.append(model.predict_proba(batch)[:, 1])
        else:
            preds.append(model.predict(batch))

    if multi_output:
        arr = np.concatenate(preds, axis=0)
        return {i + 1: arr[:, i] for i in range(arr.shape[1])}
    return np.concatenate(preds)


__all__ = [
    "fit_meta_classifier",
    "fit_meta_regressor",
    "predict_meta",
]
