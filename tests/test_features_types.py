import numpy as np
import numpy as np
import pandas as pd
import pytest

from crypto_analyzer.features.engineering import (
    FEATURE_COLUMNS,
    create_features,
    get_feature_columns,
    validate_feature_inputs,
)
from crypto_analyzer.utils.config import CONFIG, FeatureSettings


def test_features_types():
    rng = np.random.default_rng(0)
    n = 100
    ts = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
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
    ts = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
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
            "sent_score": rng.random(n),
        }
    )

    settings = FeatureSettings(
        include_onchain=False,
        include_orderbook=False,
        include_derivatives=False,
        include_sentiment=False,
        forward_fill_limit=0,
        fillna_value=-1.0,
    )
    feat_df = create_features(base, settings=settings)

    assert not any(col.startswith("onch_") for col in feat_df.columns)
    assert not any(col.startswith("lob_") or col.startswith("wall_") for col in feat_df.columns)
    assert {"basis_annualized", "oi_delta_1d"}.isdisjoint(feat_df.columns)
    assert "sent_score" not in feat_df.columns

    active_cols = get_feature_columns(settings)
    assert all(col in feat_df.columns for col in active_cols)
    assert "basis_annualized" not in active_cols
    assert "lob_imbalance_L1" not in active_cols
    assert all(not col.startswith("onch_") for col in active_cols)
    assert all(not col.startswith("sent_") for col in active_cols)

    numeric_cols = feat_df.select_dtypes(include=[np.number]).columns
    assert not feat_df[numeric_cols].isna().any().any()
    assert np.isclose(float(feat_df["ret3"].iloc[0]), settings.fillna_value)


def test_create_features_includes_sentiment_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    ts = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    base = pd.DataFrame(
        {
            "timestamp": ts,
            "open": 1.0,
            "high": 1.5,
            "low": 0.5,
            "close": 1.1,
            "volume": 100.0,
            "quote_asset_volume": 200.0,
            "taker_buy_base": 50.0,
            "taker_buy_quote": 110.0,
            "sent_score": [0.1, np.nan, 0.3, 0.4, 0.5],
        }
    )

    settings = FeatureSettings(
        include_onchain=False,
        include_orderbook=False,
        include_derivatives=False,
        include_sentiment=True,
        forward_fill_limit=0,
        fillna_value=-1.0,
    )

    sentiment_enabled = CONFIG.sentiment.model_copy(update={"use_sentiment": True})
    config_override = CONFIG.model_copy(update={"sentiment": sentiment_enabled})
    monkeypatch.setattr(
        "crypto_analyzer.features.engineering.CONFIG", config_override
    )

    feat_df = create_features(base, settings=settings)
    assert "sent_score" in feat_df.columns
    assert feat_df["sent_score"].dtype == np.float32
    assert not feat_df["sent_score"].isna().any()


def test_create_features_excludes_sentiment_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    ts = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    base = pd.DataFrame(
        {
            "timestamp": ts,
            "open": 1.0,
            "high": 1.5,
            "low": 0.5,
            "close": 1.1,
            "volume": 100.0,
            "quote_asset_volume": 200.0,
            "taker_buy_base": 50.0,
            "taker_buy_quote": 110.0,
            "sent_score": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    settings = FeatureSettings(
        include_onchain=False,
        include_orderbook=False,
        include_derivatives=False,
        include_sentiment=True,
        forward_fill_limit=0,
        fillna_value=-1.0,
    )

    sentiment_disabled = CONFIG.sentiment.model_copy(update={"use_sentiment": False})
    config_override = CONFIG.model_copy(update={"sentiment": sentiment_disabled})
    monkeypatch.setattr(
        "crypto_analyzer.features.engineering.CONFIG", config_override
    )

    feat_df = create_features(base, settings=settings)
    assert "sent_score" not in feat_df.columns


def test_create_features_computes_expected_core_metrics() -> None:
    n = 15
    ts = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    close = pd.Series(100.0 + np.arange(n), index=ts)
    open_ = close - 1.0
    high = close + 2.0
    low = close - 2.0
    volume = pd.Series(10.0 + np.arange(n), index=ts)
    quote_volume = (volume * (close + 0.5)).astype(float)
    taker_buy_base = (volume * 0.6).astype(float)
    taker_buy_quote = (quote_volume * 0.55).astype(float)

    base = pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_.to_numpy(),
            "high": high.to_numpy(),
            "low": low.to_numpy(),
            "close": close.to_numpy(),
            "volume": volume.to_numpy(),
            "quote_asset_volume": quote_volume.to_numpy(),
            "taker_buy_base": taker_buy_base.to_numpy(),
            "taker_buy_quote": taker_buy_quote.to_numpy(),
        }
    )

    settings = FeatureSettings(
        include_onchain=False,
        include_orderbook=False,
        include_derivatives=False,
        include_sentiment=False,
        forward_fill_limit=0,
        fillna_value=-1.0,
    )

    feat_df = create_features(base, settings=settings)

    expected_tbr_base = (taker_buy_base / volume).astype(np.float32).reset_index(drop=True)
    expected_tbr_base.name = "tbr_base"
    pd.testing.assert_series_equal(feat_df["tbr_base"], expected_tbr_base)

    expected_ofi_base = (2.0 * expected_tbr_base - 1.0).astype(np.float32)
    expected_ofi_base.name = "ofi_base"
    pd.testing.assert_series_equal(feat_df["ofi_base"], expected_ofi_base)

    expected_tbr_quote = (taker_buy_quote / quote_volume).astype(np.float32).reset_index(drop=True)
    expected_ofi_quote = (2.0 * expected_tbr_quote - 1.0).astype(np.float32)
    expected_ofi_quote.name = "ofi_quote"
    pd.testing.assert_series_equal(feat_df["ofi_quote"], expected_ofi_quote)

    expected_ret1 = np.log(close).diff().astype(np.float32).reset_index(drop=True)
    expected_ret3 = (
        expected_ret1.rolling(3).sum().astype(np.float32).fillna(np.float32(settings.fillna_value))
    )
    expected_ret3.name = "ret3"
    pd.testing.assert_series_equal(feat_df["ret3"], expected_ret3, rtol=1e-6, atol=1e-6)

    expected_volatility = (
        expected_ret1.rolling(12).std().astype(np.float32).fillna(np.float32(settings.fillna_value))
    )
    expected_volatility.name = "volatility_12d"
    pd.testing.assert_series_equal(
        feat_df["volatility_12d"], expected_volatility, rtol=1e-6, atol=1e-6
    )

    expected_roll_1d = expected_ofi_base.rolling(1).mean().astype(np.float32)
    expected_roll_1d.name = "ofi_base_roll_1d"
    pd.testing.assert_series_equal(feat_df["ofi_base_roll_1d"], expected_roll_1d)

    expected_roll_7d = (
        expected_ofi_base.rolling(7)
        .mean()
        .astype(np.float32)
        .fillna(np.float32(settings.fillna_value))
    )
    expected_roll_7d.name = "ofi_base_roll_7d"
    pd.testing.assert_series_equal(
        feat_df["ofi_base_roll_7d"], expected_roll_7d, rtol=1e-6, atol=1e-6
    )

    expected_ratio = (taker_buy_base / (volume - taker_buy_base)).astype(np.float32).reset_index(drop=True)
    expected_ratio.name = "taker_buy_sell_ratio"
    pd.testing.assert_series_equal(feat_df["taker_buy_sell_ratio"], expected_ratio)

def test_create_features_rejects_missing_columns():
    ts = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0})
    with pytest.raises(KeyError):
        create_features(df)


def test_validate_feature_inputs_checks_onchain_names():
    ts = pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")
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
    ts = pd.date_range("2024-01-01", periods=10, freq="1D")  # naive timestamps
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


def test_feature_generators_match_manual_expectations() -> None:
    timestamps = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    close = pd.Series(np.linspace(100.0, 109.0, num=10), index=timestamps)
    open_ = close - 0.5
    high = close + 1.0
    low = close - 1.0
    volume = pd.Series(np.linspace(10.0, 19.0, num=10), index=timestamps)
    quote_volume = volume * (close + 0.25)
    taker_buy_base = volume * 0.6
    taker_buy_quote = quote_volume * 0.6

    base = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_.to_numpy(),
            "high": high.to_numpy(),
            "low": low.to_numpy(),
            "close": close.to_numpy(),
            "volume": volume.to_numpy(),
            "quote_asset_volume": quote_volume.to_numpy(),
            "taker_buy_base": taker_buy_base.to_numpy(),
            "taker_buy_quote": taker_buy_quote.to_numpy(),
        }
    )

    feat_df = create_features(base)
    fill_value = np.float32(CONFIG.features.fillna_value)

    log_close = np.log(close.replace(0.0, np.nan))
    expected_mom = log_close.diff(1).astype(np.float32).fillna(fill_value)
    np.testing.assert_allclose(
        feat_df["mom_log_ret_1d"].to_numpy(), expected_mom.to_numpy()
    )

    ret1 = log_close.diff().astype(np.float32)
    expected_vol = ret1.rolling(7).std().astype(np.float32).fillna(fill_value)
    np.testing.assert_allclose(
        feat_df["vol_realized_7d"].to_numpy(), expected_vol.to_numpy()
    )
