import numpy as np
import pandas as pd

from .indicators import calculate_sma, calculate_ema, calculate_rsi


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
    df["z_trades"] = (df["number_of_trades"] - df["number_of_trades"].rolling(36).mean()) / df["number_of_trades"].rolling(36).std()
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
    parkinson = (hl_log ** 2) / (4 * np.log(2))
    df["parkinson12"] = np.sqrt(parkinson.rolling(12).mean())

    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
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

    df = df.dropna()
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
]


