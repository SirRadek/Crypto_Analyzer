import os
import sys

sys.path.append(os.getcwd())  # noqa: E402

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.feature_engineering import FEATURE_COLUMNS, create_features  # noqa: E402
from main import prepare_targets  # noqa: E402
from ml.meta import fit_meta_classifier, fit_meta_regressor, predict_meta  # noqa: E402


def _synthetic_prices(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=n, freq="T")
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
    )
    assert np.isclose(f1_1, f1_2)
    assert model1.oob_score_ > 0
    preds = predict_meta(
        train_df, FEATURE_COLUMNS, model_path=str(tmp_path / "m1.joblib")
    )
    assert preds.shape[0] == len(train_df)


def test_regressor_deterministic_oob(tmp_path: Path) -> None:
    df = create_features(_synthetic_prices())
    train_df = prepare_targets(df, forward_steps=1)
    X = train_df[FEATURE_COLUMNS]
    y = train_df["target_reg"]

    model1, mae_1 = fit_meta_regressor(
        X,
        y,
        FEATURE_COLUMNS,
        model_path=str(tmp_path / "r1.joblib"),
        feature_list_path=str(tmp_path / "features.json"),
        version_path=str(tmp_path / "ver.json"),
        n_splits=3,
        gap=1,
        n_estimators=10,
    )
    model2, mae_2 = fit_meta_regressor(
        X,
        y,
        FEATURE_COLUMNS,
        model_path=str(tmp_path / "r2.joblib"),
        feature_list_path=str(tmp_path / "features.json"),
        version_path=str(tmp_path / "ver.json"),
        n_splits=3,
        gap=1,
        n_estimators=10,
    )
    assert np.isclose(mae_1, mae_2)
    assert model1.oob_score_ > 0
    preds = predict_meta(
        train_df, FEATURE_COLUMNS, model_path=str(tmp_path / "r1.joblib")
    )
    assert preds.shape[0] == len(train_df)


def test_integration_small_sample(tmp_path: Path) -> None:
    df = create_features(_synthetic_prices(5000))
    train_df = prepare_targets(df, forward_steps=1)
    X_cls = train_df[FEATURE_COLUMNS]
    y_cls = train_df["target_cls"]
    X_reg = train_df[FEATURE_COLUMNS]
    y_reg = train_df["target_reg"]

    cls_model, f1 = fit_meta_classifier(
        X_cls,
        y_cls,
        FEATURE_COLUMNS,
        model_path=str(tmp_path / "meta_cls.joblib"),
        feature_list_path=str(tmp_path / "features.json"),
        version_path=str(tmp_path / "ver.json"),
        n_splits=3,
        gap=5,
        n_estimators=20,
    )
    reg_model, mae = fit_meta_regressor(
        X_reg,
        y_reg,
        FEATURE_COLUMNS,
        model_path=str(tmp_path / "meta_reg.joblib"),
        feature_list_path=str(tmp_path / "features.json"),
        version_path=str(tmp_path / "ver.json"),
        n_splits=3,
        gap=5,
        n_estimators=20,
    )

    assert cls_model.oob_score_ > 0
    assert reg_model.oob_score_ > 0
    assert f1 > 0
    assert mae >= 0

    probas = predict_meta(
        train_df,
        FEATURE_COLUMNS,
        model_path=str(tmp_path / "meta_cls.joblib"),
        proba=True,
    )
    prices = predict_meta(
        train_df, FEATURE_COLUMNS, model_path=str(tmp_path / "meta_reg.joblib")
    )
    assert probas.shape[0] == prices.shape[0] == len(train_df)
    # metadata files were saved
    assert (tmp_path / "features.json").exists()
    assert json.load(open(tmp_path / "ver.json"))
