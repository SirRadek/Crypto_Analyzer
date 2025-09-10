import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Literal, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    mean_absolute_error,
)
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
    threshold_metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> Tuple[CalibratedClassifierCV, float, float, float]:
    """Train and persist the meta classification model.

    Returns the calibrated classifier, mean F1 score from ``TimeSeriesSplit``,
    and mean Brier scores before and after calibration.
    """

    if threshold_metric is None:

        def _f1(yt: np.ndarray, yp: np.ndarray) -> float:
            return float(f1_score(yt, yp))

        threshold_metric = _f1

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores: list[float] = []
    brier_raw: list[float] = []
    brier_cal: list[float] = []
    all_probas: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    for train_idx, test_idx in tscv.split(X):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        base = RandomForestClassifier(
            n_estimators=n_estimators,
            oob_score=True,
            warm_start=True,
            n_jobs=-1,
            random_state=random_state,
        )
        base.fit(X_train, y_train)
        raw_proba = base.predict_proba(X_test)[:, 1]
        brier_raw.append(float(brier_score_loss(y_test, raw_proba)))

        method: Literal["isotonic", "sigmoid"] = (
            "isotonic" if len(train_idx) > 50000 else "sigmoid"
        )
        calib = CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=n_estimators,
                oob_score=True,
                warm_start=True,
                n_jobs=-1,
                random_state=random_state,
            ),
            cv=3,
            method=method,
        )
        calib.fit(X_train, y_train)
        cal_proba = calib.predict_proba(X_test)[:, 1]
        brier_cal.append(float(brier_score_loss(y_test, cal_proba)))
        preds = (cal_proba >= 0.5).astype(int)
        scores.append(float(f1_score(y_test, preds)))
        all_probas.append(cal_proba)
        all_true.append(y_test.to_numpy())

    # determine optimal threshold across folds
    y_all = np.concatenate(all_true)
    proba_all = np.concatenate(all_probas)
    thresholds = np.linspace(0.0, 1.0, 101)
    metric_scores = [
        threshold_metric(y_all, (proba_all >= t).astype(int)) for t in thresholds
    ]
    best_thr = float(thresholds[int(np.argmax(metric_scores))])
    Path(threshold_path).parent.mkdir(parents=True, exist_ok=True)
    with open(threshold_path, "w", encoding="utf-8") as f:
        json.dump({"threshold": best_thr}, f)

    base_final = RandomForestClassifier(
        n_estimators=n_estimators,
        oob_score=True,
        warm_start=True,
        n_jobs=-1,
        random_state=random_state,
    )
    method_final: Literal["isotonic", "sigmoid"] = (
        "isotonic" if len(X) > 50000 else "sigmoid"
    )
    final_model = CalibratedClassifierCV(
        base_final,
        cv=3,
        method=method_final,
    )
    final_model.fit(X, y)
    joblib.dump(final_model, model_path)
    _save_metadata(feature_cols, version, feature_list_path, version_path)
    return (
        final_model,
        float(np.mean(scores)),
        float(np.mean(brier_raw)),
        float(np.mean(brier_cal)),
    )


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
    threshold_path: str | None = None,
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

    threshold = None
    if threshold_path and Path(threshold_path).exists():
        with open(threshold_path, "r", encoding="utf-8") as f:
            threshold = float(json.load(f)["threshold"])

    preds = []
    for start in range(0, len(X), batch_size):
        batch = X.iloc[start : start + batch_size]
        if proba and hasattr(model, "predict_proba"):
            preds.append(model.predict_proba(batch)[:, 1])
        elif threshold is not None and hasattr(model, "predict_proba"):
            probs = model.predict_proba(batch)[:, 1]
            preds.append((probs >= threshold).astype(int))
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
