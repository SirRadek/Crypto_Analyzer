import numpy as np
import pandas as pd

from .indicators import calculate_ema, calculate_rsi, calculate_sma


def create_features(df):
    """Add technical, order-flow and time features to ``df``."""
    df = df.copy()

    # --- Order-flow & volume -------------------------------------------------
    vol = df["volume"].replace(0, np.nan)
    qvol = df["quote_asset_volume"].replace(0, np.nan)

    df["tbr_base"] = df["taker_buy_base"] / vol
    df["tbr_quote"] = df["taker_buy_quote"] / qvol
    df["ofi_base"] = 2 * df["tbr_base"] - 1
    df["ofi_quote"] = 2 * df["tbr_quote"] - 1
    df["d_tbr_base"] = df["tbr_base"].diff()
    df["ema12_tbr_base"] = df["tbr_base"].ewm(span=12, adjust=False).mean()
    df["ema36_tbr_base"] = df["tbr_base"].ewm(span=36, adjust=False).mean()

    trades = df["number_of_trades"].replace(0, np.nan)
    df["avg_trade_size"] = vol / trades
    df["d_volume"] = vol.diff()
    df["z_trades"] = (df["number_of_trades"] - df["number_of_trades"].rolling(36).mean()) / df[
        "number_of_trades"
    ].rolling(36).std()
    df["z_volume"] = (vol - vol.rolling(36).mean()) / vol.rolling(36).std()

    # --- VWAP & price relationship -------------------------------------------
    df["vwap"] = qvol / vol
    df["close_minus_vwap"] = df["close"] - df["vwap"]
    df["rel_close_vwap"] = df["close_minus_vwap"].abs() / df["vwap"]

    # --- Returns & momentum ---------------------------------------------------
    df["return_1d"] = df["close"].pct_change()
    df["ret1"] = np.log(df["close"] / df["close"].shift(1))
    df["ret3"] = df["ret1"].rolling(3).sum()
    df["ret6"] = df["ret1"].rolling(6).sum()
    df["ret12"] = df["ret1"].rolling(12).sum()

    # --- Volatility -----------------------------------------------------------
    df["rv12"] = df["ret1"].rolling(12).std()
    df["rv36"] = df["ret1"].rolling(36).std()

    hl_log = np.log(df["high"] / df["low"])
    parkinson = (hl_log**2) / (4 * np.log(2))
    df["parkinson12"] = np.sqrt(parkinson.rolling(12).mean())

    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    price_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["range_rel"] = price_range / df["close"]
    df["body_rel"] = (df["close"] - df["open"]).abs() / df["open"]
    df["wick_up_rel"] = (df["high"] - np.maximum(df["open"], df["close"])) / price_range
    df["wick_dn_rel"] = (np.minimum(df["open"], df["close"]) - df["low"]) / price_range

    # --- Time features --------------------------------------------------------
    minute = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute
    df["tod_sin"] = np.sin(2 * np.pi * minute / 1440)
    df["tod_cos"] = np.cos(2 * np.pi * minute / 1440)
    df["dow"] = df["timestamp"].dt.dayofweek

    # --- Classic indicators ---------------------------------------------------
    df["sma_7"] = calculate_sma(df["close"], window=7)
    df["sma_14"] = calculate_sma(df["close"], window=14)
    df["ema_7"] = calculate_ema(df["close"], window=7)
    df["ema_14"] = calculate_ema(df["close"], window=14)
    df["rsi_14"] = calculate_rsi(df["close"], window=14)

    # --- Additional market & macro features ----------------------------------
    if "funding_now" in df.columns:
        df["funding_delta_1h"] = df["funding_now"] - df["funding_now"].shift(12)
    else:
        df["funding_now"] = 0.0
        df["funding_delta_1h"] = 0.0

    if "basis_annualized" not in df.columns:
        df["basis_annualized"] = 0.0

    if "open_interest" in df.columns:
        df["oi_delta_15m"] = df["open_interest"].diff(3)
        df["oi_delta_1h"] = df["open_interest"].diff(12)
    else:
        df["oi_delta_15m"] = 0.0
        df["oi_delta_1h"] = 0.0

    if {"liq_long_usd", "liq_short_usd"}.issubset(df.columns):
        denom = df["liq_short_usd"].replace(0, np.nan)
        df["liq_long_short_ratio"] = df["liq_long_usd"] / denom
    else:
        df["liq_long_short_ratio"] = 0.0

    denom = (df["volume"] - df["taker_buy_base"]).replace(0, np.nan)
    df["taker_buy_sell_ratio"] = df["taker_buy_base"] / denom

    for level in range(1, 6):
        bid_col = f"lob_bid_L{level}"
        ask_col = f"lob_ask_L{level}"
        imb_col = f"lob_imbalance_L{level}"
        if bid_col in df.columns and ask_col in df.columns:
            df[imb_col] = (df[bid_col] - df[ask_col]) / (df[bid_col] + df[ask_col])
        else:
            df[imb_col] = 0.0

    df["rv_5m"] = df["ret1"].rolling(1).std()
    df["rv_30m"] = df["ret1"].rolling(6).std()

    hour = df["timestamp"].dt.hour
    df["session_dummy_EU"] = hour.between(7, 15).astype(int)
    df["session_dummy_US"] = hour.between(13, 21).astype(int)
    df["session_dummy_Asia"] = ((hour >= 23) | (hour <= 7)).astype(int)

    if "google_trends_btc" in df.columns:
        df["google_trends_btc_delta"] = df["google_trends_btc"].diff()
    else:
        df["google_trends_btc_delta"] = 0.0

    if "btc_dominance" in df.columns:
        df["btc_dominance_delta"] = df["btc_dominance"].diff()
    else:
        df["btc_dominance_delta"] = 0.0

    for col in ["mvrv_z", "sopr", "fees_per_tx"]:
        if col not in df.columns:
            df[col] = 0.0

    for sym in ["eth", "sol", "bnb"]:
        base_col = f"{sym}_ret"
        lag_col = f"{base_col}_lagged"
        if base_col in df.columns:
            df[lag_col] = df[base_col].shift(1)
        else:
            df[lag_col] = 0.0

    df = df.fillna(0)

    return df


FEATURE_COLUMNS = [
    "tbr_base",
    "tbr_quote",
    "ofi_base",
    "ofi_quote",
    "d_tbr_base",
    "ema12_tbr_base",
    "ema36_tbr_base",
    "avg_trade_size",
    "z_trades",
    "z_volume",
    "d_volume",
    "vwap",
    "close_minus_vwap",
    "rel_close_vwap",
    "return_1d",
    "ret1",
    "ret3",
    "ret6",
    "ret12",
    "rv12",
    "rv36",
    "parkinson12",
    "atr14",
    "range_rel",
    "body_rel",
    "wick_up_rel",
    "wick_dn_rel",
    "tod_sin",
    "tod_cos",
    "dow",
    "sma_7",
    "sma_14",
    "ema_7",
    "ema_14",
    "rsi_14",
    "funding_now",
    "funding_delta_1h",
    "basis_annualized",
    "oi_delta_15m",
    "oi_delta_1h",
    "liq_long_short_ratio",
    "taker_buy_sell_ratio",
    "lob_imbalance_L1",
    "lob_imbalance_L2",
    "lob_imbalance_L3",
    "lob_imbalance_L4",
    "lob_imbalance_L5",
    "rv_5m",
    "rv_30m",
    "session_dummy_EU",
    "session_dummy_US",
    "session_dummy_Asia",
    "google_trends_btc_delta",
    "mvrv_z",
    "sopr",
    "fees_per_tx",
    "eth_ret_lagged",
    "sol_ret_lagged",
    "bnb_ret_lagged",
    "btc_dominance_delta",
]
