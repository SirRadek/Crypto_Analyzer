from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestRegressor


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
) -> tuple[CalibratedClassifierCV, float]:
    """Train and persist the meta classification model with calibration.

    Returns the fitted calibrated model and mean F1 score from ``TimeSeriesSplit``.
    ``threshold_path`` stores a JSON file with the optimal decision threshold.
    """

    import joblib
    import numpy as np
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import TimeSeriesSplit

    method: Literal["isotonic", "sigmoid"] = "isotonic" if len(y) > 50_000 else "sigmoid"

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
    y: pd.Series | pd.DataFrame,
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
) -> tuple[RandomForestRegressor, float | dict[int, float]]:
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

    import joblib
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import TimeSeriesSplit

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
            maes.append(mean_absolute_error(y.iloc[test_idx], pred, multioutput="raw_values"))
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
    return_pi: bool = False,
    quantiles: Iterable[float] = (0.05, 0.95),
    log_pi_width: bool = False,
) -> (
    np.ndarray
    | dict[int, np.ndarray]
    | tuple[np.ndarray, dict[float, np.ndarray]]
    | tuple[dict[int, np.ndarray], dict[float, dict[int, np.ndarray]]]
):
    """Predict using a stored meta model with batching and memory mapping.

    Parameters
    ----------
    multi_output:
        If ``True``, the loaded model is expected to output multiple horizons.
        The return value is a dictionary ``{horizon: np.ndarray}`` with
        1-based horizon indices.
    return_pi:
        If ``True`` (regression only), also return prediction intervals based on
        estimator quantiles. The second return value is a dict mapping each
        quantile to an array shaped like the predictions. When ``multi_output``
        is ``True``, this becomes ``{q: {h: arr}}``.
    log_pi_width:
        If ``True`` and at least two quantiles are provided, log the mean width
        of the extreme interval.
    """

    import joblib
    import numpy as np

    if proba and (multi_output or return_pi):
        raise ValueError("`proba` cannot be combined with `multi_output` or `return_pi`")

    model = joblib.load(model_path, mmap_mode="r")
    X = df[list(feature_cols)]

    if return_pi:
        if not hasattr(model, "estimators_"):
            raise ValueError("Model does not support prediction intervals")
        preds: list[np.ndarray] = []
        q_preds: dict[float, list[np.ndarray]] = {q: [] for q in quantiles}
        ests = model.estimators_
        qs = list(quantiles)
        for start in range(0, len(X), batch_size):
            batch = X.iloc[start : start + batch_size]
            est_batch = np.array([e.predict(batch) for e in ests])
            preds.append(est_batch.mean(axis=0))
            for q in qs:
                q_preds[q].append(np.quantile(est_batch, q, axis=0))
            if log_pi_width and len(qs) >= 2:
                lower = np.quantile(est_batch, min(qs), axis=0)
                upper = np.quantile(est_batch, max(qs), axis=0)
                width = np.mean(upper - lower)
                print(f"PI width mean={width:.6f}")
        pred_arr = np.concatenate(preds, axis=0)
        q_arrs = {q: np.concatenate(v, axis=0) for q, v in q_preds.items()}
        if multi_output:
            pred_dict = {i + 1: pred_arr[:, i] for i in range(pred_arr.shape[1])}
            q_dict: dict[float, dict[int, np.ndarray]] = {}
            for q, arr in q_arrs.items():
                q_dict[q] = {i + 1: arr[:, i] for i in range(arr.shape[1])}
            return pred_dict, q_dict
        return pred_arr, q_arrs

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
