import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV


def _save_metadata(
    feature_cols: Iterable[str], version: str, feature_list_path: str, version_path: str
) -> None:
    Path(feature_list_path).parent.mkdir(parents=True, exist_ok=True)
    with open(feature_list_path, "w", encoding="utf-8") as f:
        json.dump(list(feature_cols), f)
    with open(version_path, "w", encoding="utf-8") as f:
        json.dump({"version": version}, f)


def fit_meta_classifier(
    X: "pd.DataFrame",
    y: "pd.Series",
    feature_cols: Iterable[str],
    *,
    model_path: str = "artifacts/meta_model_cls.joblib",
    feature_list_path: str = "src/crypto_analyzer/features/feature_list.json",
    version_path: str = "src/crypto_analyzer/models/meta_version.json",
    version: str = "v1",
    n_splits: int = 5,
    gap: int = 0,
    random_state: int = 42,
    n_estimators: int = 200,
    threshold_path: str = "artifacts/threshold.json",
    threshold_func: Callable[["np.ndarray", "pd.Series"], float] | None = None,
) -> tuple["CalibratedClassifierCV", float]:
    """Train and persist the meta classification model with calibration."""

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


def predict_meta(
    df: "pd.DataFrame",
    feature_cols: Iterable[str],
    model_path: str,
    *,
    batch_size: int = 1000,
    proba: bool = False,
) -> "np.ndarray":
    """Predict probabilities or hard labels using the meta classifier."""

    import joblib
    import numpy as np

    model = joblib.load(model_path)
    cols = list(feature_cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features for meta prediction: {missing}")

    probs: list[np.ndarray] = []
    for start in range(0, len(df), batch_size):
        end = start + batch_size
        chunk = df.iloc[start:end][cols]
        probs.append(model.predict_proba(chunk)[:, 1])
    prob_array = np.concatenate(probs) if probs else np.empty(0, dtype=float)
    if proba:
        return prob_array
    return (prob_array >= 0.5).astype(int)


__all__ = ["fit_meta_classifier", "predict_meta"]
