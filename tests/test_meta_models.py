import os
import sys

sys.path.append(os.getcwd())

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss

from analysis.feature_engineering import FEATURE_COLUMNS, create_features
from main import prepare_targets
from ml.meta import fit_meta_classifier, predict_meta


def _synthetic_prices(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n, freq="min")
    base = np.cumsum(rng.normal(size=n)) + 100
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": base,
            "high": base + rng.random(n),
            "low": base - rng.random(n),
            "close": base + rng.normal(scale=0.5, size=n),
            "volume": rng.random(n) + 1,
            "quote_asset_volume": rng.random(n) + 1,
            "taker_buy_base": rng.random(n),
            "taker_buy_quote": rng.random(n),
            "number_of_trades": (rng.random(n) * 10 + 1).astype(int),
        }
    )
    return df


def test_classifier_deterministic_oob(tmp_path: Path) -> None:
    df = create_features(_synthetic_prices())
    train_df = prepare_targets(df, forward_steps=1)
    X = train_df[FEATURE_COLUMNS]
    y = train_df["target_cls"]

    model1, f1_1 = fit_meta_classifier(
        X,
        y,
        FEATURE_COLUMNS,
        model_path=str(tmp_path / "m1.joblib"),
        feature_list_path=str(tmp_path / "features.json"),
        version_path=str(tmp_path / "ver.json"),
        n_splits=3,
        gap=1,
        n_estimators=10,
        threshold_path=str(tmp_path / "thr1.json"),
    )
    model2, f1_2 = fit_meta_classifier(
        X,
        y,
        FEATURE_COLUMNS,
        model_path=str(tmp_path / "m2.joblib"),
        feature_list_path=str(tmp_path / "features.json"),
        version_path=str(tmp_path / "ver.json"),
        n_splits=3,
        gap=1,
        n_estimators=10,
        threshold_path=str(tmp_path / "thr2.json"),
    )
    assert np.isclose(f1_1, f1_2)
    assert all(c.estimator.oob_score_ > 0 for c in model1.calibrated_classifiers_)

    probas = predict_meta(train_df, FEATURE_COLUMNS, model_path=str(tmp_path / "m1.joblib"), proba=True)
    assert isinstance(probas, np.ndarray)
    assert probas.shape[0] == len(train_df)


def test_classifier_predictions_shape(tmp_path: Path) -> None:
    df = create_features(_synthetic_prices(500))
    train_df = prepare_targets(df, forward_steps=1)
    X = train_df[FEATURE_COLUMNS]
    y = train_df["target_cls"]

    fit_meta_classifier(
        X,
        y,
        FEATURE_COLUMNS,
        model_path=str(tmp_path / "meta_cls.joblib"),
        feature_list_path=str(tmp_path / "features.json"),
        version_path=str(tmp_path / "ver.json"),
        n_splits=3,
        gap=1,
        n_estimators=10,
        threshold_path=str(tmp_path / "thr.json"),
    )

    preds = predict_meta(train_df, FEATURE_COLUMNS, model_path=str(tmp_path / "meta_cls.joblib"))
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == len(train_df)
    assert set(np.unique(preds)).issubset({0, 1})


def test_calibration_improves_brier(tmp_path: Path) -> None:
    X, y = make_classification(n_samples=1000, n_features=20, random_state=0)
    X_df = pd.DataFrame(X)
    feature_cols = list(X_df.columns)

    base = RandomForestClassifier(n_estimators=50, random_state=0)
    base.fit(X_df, y)
    base_proba = base.predict_proba(X_df)[:, 1] ** 2  # intentionally miscalibrated
    base_brier = brier_score_loss(y, base_proba)

    fit_meta_classifier(
        X_df,
        pd.Series(y),
        feature_cols,
        model_path=str(tmp_path / "cal.joblib"),
        feature_list_path=str(tmp_path / "features.json"),
        version_path=str(tmp_path / "ver.json"),
        n_splits=3,
        gap=1,
        n_estimators=50,
        threshold_path=str(tmp_path / "thr.json"),
    )
    cal_proba = predict_meta(
        X_df,
        feature_cols,
        model_path=str(tmp_path / "cal.joblib"),
        proba=True,
    )
    cal_brier = brier_score_loss(y, cal_proba)
    assert cal_brier <= base_brier + 1e-6
