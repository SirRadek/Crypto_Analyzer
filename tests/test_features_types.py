import numpy as np
import pandas as pd
import pytest

from analysis.feature_engineering import (
    FEATURE_COLUMNS,
    create_features,
    get_feature_columns,
    validate_feature_inputs,
)
from utils.config import FeatureSettings


def test_features_types():
    rng = np.random.default_rng(0)
    n = 100
    ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 100 + rng.normal(scale=1, size=n).cumsum()
    open_ = close - rng.random(n)
    high = close + rng.random(n)
    low = close - rng.random(n)
    volume = rng.random(n) + 1
    qvol = volume * close
    tbb = volume * 0.5
    tbq = qvol * 0.5
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "close": close,
            "high": high,
            "low": low,
            "volume": volume,
            "quote_asset_volume": qvol,
            "taker_buy_base": tbb,
            "taker_buy_quote": tbq,
        }
    )
    feat_df = create_features(df)
    X = feat_df[FEATURE_COLUMNS]
    assert feat_df.columns.is_unique
    assert X.dtypes.eq("float32").all()
    assert not X.isna().any().any()


def test_feature_toggles_respected():
    rng = np.random.default_rng(1)
    n = 50
    ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 100 + rng.normal(scale=1, size=n).cumsum()
    base = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close + 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": rng.random(n) + 1,
            "quote_asset_volume": rng.random(n) + 1,
            "taker_buy_base": rng.random(n),
            "taker_buy_quote": rng.random(n),
            "number_of_trades": rng.integers(1, 100, size=n),
            "basis_annualized": rng.random(n),
            "open_interest": rng.random(n) * 1000,
            "lob_bid_L1": rng.random(n),
            "lob_ask_L1": rng.random(n),
            "lob_bid_price_1": rng.random(n) + 100,
            "lob_bid_size_1": rng.random(n),
            "lob_ask_price_1": rng.random(n) + 101,
            "lob_ask_size_1": rng.random(n),
            "onch_fee_fast_satvb": rng.random(n),
        }
    )

    settings = FeatureSettings(
        include_onchain=False,
        include_orderbook=False,
        include_derivatives=False,
        forward_fill_limit=0,
        fillna_value=-1.0,
    )
    feat_df = create_features(base, settings=settings)

    assert not any(col.startswith("onch_") for col in feat_df.columns)
    assert not any(col.startswith("lob_") or col.startswith("wall_") for col in feat_df.columns)
    assert {"basis_annualized", "oi_delta_15m"}.isdisjoint(feat_df.columns)

    active_cols = get_feature_columns(settings)
    assert all(col in feat_df.columns for col in active_cols)
    assert "basis_annualized" not in active_cols
    assert "lob_imbalance_L1" not in active_cols
    assert all(not col.startswith("onch_") for col in active_cols)

    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns
    assert not feat_df[numeric_cols].isna().any().any()
    assert np.isclose(float(feat_df["ret3"].iloc[0]), settings.fillna_value)


def test_create_features_rejects_missing_columns():
    ts = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0})
    with pytest.raises(KeyError):
        create_features(df)


def test_validate_feature_inputs_checks_onchain_names():
    ts = pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
            "quote_asset_volume": 1.0,
            "taker_buy_base": 0.5,
            "taker_buy_quote": 0.5,
            "onch_unknown": 1.0,
        }
    )

    settings = FeatureSettings(
        include_onchain=True,
        include_orderbook=False,
        include_derivatives=False,
        forward_fill_limit=1,
        fillna_value=0.0,
    )

    with pytest.raises(ValueError):
        validate_feature_inputs(df, settings)


def test_timestamp_localized_to_utc():
    ts = pd.date_range("2024-01-01", periods=10, freq="5min")  # naive timestamps
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
            "quote_asset_volume": 1.0,
            "taker_buy_base": 0.5,
            "taker_buy_quote": 0.5,
        }
    )

    feat_df = create_features(df)
    assert feat_df["timestamp"].dt.tz is not None
    assert str(feat_df["timestamp"].dt.tz) == "UTC"
