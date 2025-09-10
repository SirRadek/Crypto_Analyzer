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
from ml.predict import predict_ml  # noqa: E402


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

    model1, f1_1, _, _ = fit_meta_classifier(
        X,
        y,
        FEATURE_COLUMNS,
        model_path=str(tmp_path / "m1.joblib"),
        feature_list_path=str(tmp_path / "features.json"),
        version_path=str(tmp_path / "ver.json"),
        n_splits=3,
        gap=1,
        n_estimators=10,
        threshold_path=str(tmp_path / "t1.json"),
    )
    model2, f1_2, _, _ = fit_meta_classifier(
        X,
        y,
        FEATURE_COLUMNS,
        model_path=str(tmp_path / "m2.joblib"),
        feature_list_path=str(tmp_path / "features.json"),
        version_path=str(tmp_path / "ver.json"),
        n_splits=3,
        gap=1,
        n_estimators=10,
        threshold_path=str(tmp_path / "t2.json"),
    )
    assert np.isclose(f1_1, f1_2)
    assert model1.calibrated_classifiers_[0].estimator.oob_score_ > 0
    preds = predict_meta(
        train_df, FEATURE_COLUMNS, model_path=str(tmp_path / "m1.joblib")
    )
    assert isinstance(preds, np.ndarray)
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
    assert isinstance(mae_1, float) and isinstance(mae_2, float)
    assert np.isclose(mae_1, mae_2)
    assert model1.oob_score_ > 0
    preds = predict_meta(
        train_df, FEATURE_COLUMNS, model_path=str(tmp_path / "r1.joblib")
    )
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == len(train_df)


def test_integration_small_sample(tmp_path: Path) -> None:
    df = create_features(_synthetic_prices(5000))
    train_df = prepare_targets(df, forward_steps=1)
    X_cls = train_df[FEATURE_COLUMNS]
    y_cls = train_df["target_cls"]
    X_reg = train_df[FEATURE_COLUMNS]
    y_reg = train_df["target_reg"]

    cls_model, f1, _, _ = fit_meta_classifier(
        X_cls,
        y_cls,
        FEATURE_COLUMNS,
        model_path=str(tmp_path / "meta_cls.joblib"),
        feature_list_path=str(tmp_path / "features.json"),
        version_path=str(tmp_path / "ver.json"),
        n_splits=3,
        gap=5,
        n_estimators=20,
        threshold_path=str(tmp_path / "thr.json"),
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

    assert cls_model.calibrated_classifiers_[0].estimator.oob_score_ > 0
    assert reg_model.oob_score_ > 0
    assert f1 > 0
    assert isinstance(mae, float)
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
    assert isinstance(probas, np.ndarray)
    assert isinstance(prices, np.ndarray)
    assert probas.shape[0] == prices.shape[0] == len(train_df)
    # metadata files were saved
    assert (tmp_path / "features.json").exists()
    assert json.load(open(tmp_path / "ver.json"))
    assert json.load(open(tmp_path / "thr.json"))


def test_multi_output_regressor(tmp_path: Path) -> None:
    df = create_features(_synthetic_prices())
    horizons = 3
    target_cols = []
    for h in range(1, horizons + 1):
        t_df = prepare_targets(df, forward_steps=h)
        target_cols.append(t_df["target_reg"].rename(f"h{h}"))
    y = pd.concat(target_cols, axis=1).dropna()
    X = df.loc[y.index, FEATURE_COLUMNS]

    model, maes = fit_meta_regressor(
        X,
        y,
        FEATURE_COLUMNS,
        model_path=str(tmp_path / "multi.joblib"),
        feature_list_path=str(tmp_path / "features.json"),
        version_path=str(tmp_path / "ver.json"),
        n_splits=3,
        gap=1,
        n_estimators=10,
        multi_output=True,
    )
    assert isinstance(model.oob_score_, float)
    assert isinstance(maes, dict)
    assert len(maes) == horizons

    preds = predict_meta(
        X,
        FEATURE_COLUMNS,
        model_path=str(tmp_path / "multi.joblib"),
        multi_output=True,
    )
    assert isinstance(preds, dict)
    assert set(preds.keys()) == set(range(1, horizons + 1))
    arr = np.column_stack([preds[h] for h in sorted(preds)])
    assert arr.shape == (len(X), horizons)

    for h in [1, 2]:
        _single_model, _ = fit_meta_regressor(
            X,
            y[f"h{h}"],
            FEATURE_COLUMNS,
            model_path=str(tmp_path / f"single{h}.joblib"),
            feature_list_path=str(tmp_path / "features.json"),
            version_path=str(tmp_path / "ver.json"),
            n_splits=3,
            gap=1,
            n_estimators=10,
        )
        single_preds = predict_meta(
            X,
            FEATURE_COLUMNS,
            model_path=str(tmp_path / f"single{h}.joblib"),
        )
        assert isinstance(single_preds, np.ndarray)
        assert np.allclose(single_preds, preds[h], atol=2.0)


def test_calibration_threshold_inference(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 1000
    x = rng.normal(size=n)
    proba = 1 / (1 + np.exp(-x))
    y = pd.Series((rng.random(n) < proba).astype(int))
    X = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS
    )
    X[FEATURE_COLUMNS[0]] = x

    model, _, b0, b1 = fit_meta_classifier(
        X,
        y,
        FEATURE_COLUMNS,
        model_path=str(tmp_path / "m.joblib"),
        feature_list_path=str(tmp_path / "features.json"),
        version_path=str(tmp_path / "ver.json"),
        n_splits=3,
        gap=0,
        n_estimators=50,
        threshold_path=str(tmp_path / "thr.json"),
    )
    assert b1 < b0
    assert model.calibrated_classifiers_[0].method == "sigmoid"
    assert model.calibrated_classifiers_[0].estimator.oob_score_ > 0

    probas = predict_meta(
        X,
        FEATURE_COLUMNS,
        model_path=str(tmp_path / "m.joblib"),
        proba=True,
    )
    thr = json.load(open(tmp_path / "thr.json"))["threshold"]
    manual = (probas >= thr).astype(int)
    preds = predict_ml(
        X,
        FEATURE_COLUMNS,
        model_path=str(tmp_path / "m.joblib"),
        threshold_path=str(tmp_path / "thr.json"),
    )
    assert np.array_equal(manual, preds)
