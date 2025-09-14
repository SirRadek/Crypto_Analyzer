from __future__ import annotations

import json
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from utils.splitting import WalkForwardSplit

from ml.model_utils import evaluate_model
from crypto_analyzer.model_manager import atomic_write

MODEL_PATH = "ml/meta_model_cls.joblib"

logger = logging.getLogger(__name__)


def _to_f32(X) -> pd.DataFrame | np.ndarray:
    if isinstance(X, pd.DataFrame):
        return X.astype("float32", copy=False)
    return np.asarray(X, dtype=np.float32)


def _fit_no_es(clf: xgb.XGBClassifier, X_train, y_train, X_val, y_val) -> None:
    """Trénink bez early stoppingu pro verze XGBoostu bez podpory ES ve sklearn wrapperu."""
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])


def train_model(
    X,
    y,
    model_path: str = MODEL_PATH,
    tune: bool = False,              # zachováno kvůli rozhraní, netuníme zde
    test_size: float = 0.2,
    random_state: int = 42,
    use_gpu: bool = True,            # defaultně zapínáme GPU, fallback níž
    oob_tol: float | None = None,    # zachováno kvůli kompatibilitě s voláním
    oob_step: int = 50,              # nevyužito u XGBoost, ponecháno pro signaturu
    max_estimators: int = 400,       # nevyužito u XGBoost, ponecháno pro signaturu
    log_path: str = "ml/oob_cls.json",
    split: str = "holdout",
    wfs_params: dict[str, int] | None = None,
):
    """
    Trénuje meta-klasifikátor pomocí XGBoost (pandas+numpy only).
    - float32 vstupy
    - GPU akcelerace přes tree_method="gpu_hist" a predictor="gpu_predictor" s fallbackem na CPU
    - early stopping (callbacks nebo early_stopping_rounds dle verze)
    """
    base_params: dict[str, Any] = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method=("gpu_hist" if use_gpu else "hist"),
        predictor=("gpu_predictor" if use_gpu else "cpu_predictor"),
        n_jobs=-1,
        eval_metric="logloss",
        random_state=random_state,
        verbosity=0,
    )

    if split == "walkforward":
        if not isinstance(X, pd.DataFrame) or "timestamp" not in X.columns:
            raise ValueError("Walkforward split vyžaduje DataFrame s 'timestamp'")
        df = X.reset_index(drop=True)
        features = df.drop(columns=["timestamp"])  # training features
        y_series = pd.Series(y).reset_index(drop=True)

        params = wfs_params or {
            "train_span_days": 30,
            "test_span_days": 7,
            "step_days": 7,
            "min_train_days": 30,
        }
        splitter = WalkForwardSplit(**params)

        fold_metrics: list[dict[str, float]] = []
        preds_frames: list[pd.DataFrame] = []

        for fold, (train_idx, test_idx) in enumerate(splitter.split(df)):
            X_train = _to_f32(features.iloc[train_idx])
            y_train = y_series.iloc[train_idx]
            X_val = _to_f32(features.iloc[test_idx])
            y_val = y_series.iloc[test_idx]

            clf = xgb.XGBClassifier(**base_params)
            try:
                _fit_no_es(clf, X_train, y_train, X_val, y_val)
            except xgb.core.XGBoostError:
                logger.warning("CUDA not available or failed. Falling back to CPU.")
                clf.set_params(tree_method="hist", predictor="cpu_predictor")
                _fit_no_es(clf, X_train, y_train, X_val, y_val)

            acc, f1 = evaluate_model(clf, X_val, y_val)
            fold_metrics.append({"fold": fold, "accuracy": acc, "f1": f1})

            preds = clf.predict(X_val)
            preds_frames.append(
                pd.DataFrame(
                    {"fold": fold, "y_true": y_val, "y_pred": preds},
                    index=test_idx,
                )
            )

        # uložit metriky
        metrics_data = {"folds": fold_metrics}
        atomic_write(Path(log_path), json.dumps(metrics_data, indent=2).encode("utf-8"))

        # uložit predikce
        if preds_frames:
            preds_df = pd.concat(preds_frames).sort_index()
            pred_path = Path(log_path).with_suffix(".preds.csv")
            preds_df.to_csv(pred_path, index_label="index")

        # finální model natrénujeme na všech datech
        X_full = _to_f32(features)
        y_full = y_series
        clf = xgb.XGBClassifier(**base_params)
        try:
            _fit_no_es(clf, X_full, y_full, X_full, y_full)
        except xgb.core.XGBoostError:
            logger.warning("CUDA not available or failed. Falling back to CPU.")
            clf.set_params(tree_method="hist", predictor="cpu_predictor")
            _fit_no_es(clf, X_full, y_full, X_full, y_full)

        buffer = BytesIO()
        joblib.dump(clf, buffer)
        atomic_write(Path(model_path), buffer.getvalue())
        print(f"Model saved to {model_path}")
        return clf

    # --- defaultní holdout split ---------------------------------------------

    if isinstance(X, pd.DataFrame) and "timestamp" in X.columns:
        X = X.drop(columns=["timestamp"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # float32 a zachování DataFrame kvůli feature_names_in_
    if isinstance(X_train, pd.DataFrame):
        X_train = _to_f32(X_train)
        X_val = _to_f32(X_val)
    else:
        X_train = _to_f32(X_train)
        X_val = _to_f32(X_val)

    clf = xgb.XGBClassifier(**base_params)

    # Primární pokus na GPU → fallback CPU při chybě
    try:
        _fit_no_es(clf, X_train, y_train, X_val, y_val)
    except xgb.core.XGBoostError:
        logger.warning("CUDA not available or failed. Falling back to CPU.")
        clf.set_params(tree_method="hist", predictor="cpu_predictor")
        _fit_no_es(clf, X_train, y_train, X_val, y_val)

    # Vyhodnocení a uložení
    evaluate_model(clf, X_val, y_val)

    buffer = BytesIO()
    joblib.dump(clf, buffer)
    atomic_write(Path(model_path), buffer.getvalue())
    print(f"Model saved to {model_path}")
    return clf


def load_model(model_path=MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")
    return joblib.load(model_path)
