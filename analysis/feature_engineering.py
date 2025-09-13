from __future__ import annotations

import numpy as np
import pandas as pd

# Lehký FE laděný pro 2h BTCUSDT. Bez těžkých závislostí, vše float32.

H = 24  # 120min horizon in 5m candles


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Přidej technické, order-flow a časové rysy. Vše jako float32."""
    df = df.copy(deep=False)

    # --- základní numerika ---------------------------------------------------
    # nech timestamp beze změny, ostatní numerické sloupce konvertuj na float32
    numeric = df.select_dtypes(include=["number"]).columns.drop(["timestamp"], errors="ignore")
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce").astype(np.float32)

    # --- order-flow & volume --------------------------------------------------
    vol = df["volume"].replace(0.0, np.nan)
    qvol = df["quote_asset_volume"].replace(0.0, np.nan)

    df["tbr_base"] = (df["taker_buy_base"] / vol).astype(np.float32)
    tbr_quote = (df["taker_buy_quote"] / qvol).astype(np.float32)

    df["ofi_base"] = (2.0 * df["tbr_base"] - 1.0).astype(np.float32)
    df["ofi_quote"] = (2.0 * tbr_quote - 1.0).astype(np.float32)
    df["d_tbr_base"] = df["tbr_base"].diff().astype(np.float32)
    df["ema12_tbr_base"] = df["tbr_base"].ewm(span=12, adjust=False).mean().astype(np.float32)
    df["z_volume"] = (
        (vol - vol.rolling(36).mean()) / vol.rolling(36).std()
    ).astype(np.float32)

    # --- VWAP vztah ----------------------------------------------------------
    vwap = (qvol / vol).astype(np.float32)
    df["rel_close_vwap"] = ((df["close"] - vwap).abs() / vwap).astype(np.float32)

    # --- výnosy & momentum ----------------------------------------------------
    ret1 = np.log(df["close"] / df["close"].shift(1)).astype(np.float32)
    df["ret3"] = ret1.rolling(3).sum().astype(np.float32)
    df["ret12"] = ret1.rolling(12).sum().astype(np.float32)

    # --- volatilita -----------------------------------------------------------
    df["rv_5m"] = ret1.rolling(1).std().astype(np.float32)
    df["volatility_60m"] = ret1.rolling(12).std().astype(np.float32)

    hl_log = np.log(df["high"] / df["low"]).astype(np.float32)
    parkinson = (hl_log * hl_log) / (4.0 * np.log(2.0))
    df["parkinson12"] = np.sqrt(parkinson.rolling(12).mean()).astype(np.float32)

    tr = pd.concat(
        [
            (df["high"] - df["low"]).astype(np.float32),
            (df["high"] - df["close"].shift(1)).abs().astype(np.float32),
            (df["low"] - df["close"].shift(1)).abs().astype(np.float32),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.rolling(14).mean().astype(np.float32)

    # --- další tržní signály --------------------------------------------------
    denom = (df["volume"] - df["taker_buy_base"]).replace(0.0, np.nan)
    df["taker_buy_sell_ratio"] = (df["taker_buy_base"] / denom).astype(np.float32)

    if "basis_annualized" not in df.columns:
        df["basis_annualized"] = np.float32(0.0)
    else:
        df["basis_annualized"] = df["basis_annualized"].astype(np.float32)

    if "open_interest" in df.columns:
        df["oi_delta_15m"] = df["open_interest"].diff(3).astype(np.float32)
    else:
        df["oi_delta_15m"] = np.float32(0.0)

    for level in range(1, 3):
        bid_col = f"lob_bid_L{level}"
        ask_col = f"lob_ask_L{level}"
        imb_col = f"lob_imbalance_L{level}"
        if bid_col in df.columns and ask_col in df.columns:
            num = (df[bid_col] - df[ask_col]).astype(np.float32)
            den = (df[bid_col] + df[ask_col]).replace(0.0, np.nan).astype(np.float32)
            df[imb_col] = (num / den).astype(np.float32)
        else:
            df[imb_col] = np.float32(0.0)

    # --- časové rysy ----------------------------------------------------------
    ts = df["timestamp"]
    # očekává se tz-aware UTC; pokud ne, nechte jak je
    minute = ts.dt.hour * 60 + ts.dt.minute
    df["tod_sin"] = np.sin(2.0 * np.pi * minute / 1440.0).astype(np.float32)
    df["tod_cos"] = np.cos(2.0 * np.pi * minute / 1440.0).astype(np.float32)
    is_day = ((ts.dt.hour >= 8) & (ts.dt.hour < 20)).astype(np.float32)
    df["is_day"] = is_day
    df["is_night"] = (1.0 - is_day).astype(np.float32)
    is_weekend = (ts.dt.dayofweek >= 5).astype(np.float32)
    df["is_weekend"] = is_weekend
    df["is_weekday"] = (1.0 - is_weekend).astype(np.float32)

    # --- NaN handling před cíli ----------------------------------------------
    df = df.fillna(0.0)

    # --- budoucí extrémy ------------------------------------------------------
    fut_low = df["low"].shift(-H + 1).rolling(H).min().astype(np.float32)
    fut_high = df["high"].shift(-H + 1).rolling(H).max().astype(np.float32)
    df["delta_low_log_120m"] = np.log(fut_low / df["close"]).astype(np.float32)
    df["delta_low_lin_120m"] = (fut_low - df["close"]).astype(np.float32)
    df["delta_high_log_120m"] = np.log(fut_high / df["close"]).astype(np.float32)
    df["delta_high_lin_120m"] = (fut_high - df["close"]).astype(np.float32)

    # --- cíle (60/120/240 min) -----------------------------------------------
    horizon = 24  # 120 min při 5m svících
    df["delta_log_120m"] = np.log(df["close"].shift(-horizon) / df["close"]).astype(np.float32)
    df["delta_lin_120m"] = (df["close"].shift(-horizon) - df["close"]).astype(np.float32)
    future_lows = pd.concat([df["low"].shift(-i) for i in range(1, horizon + 1)], axis=1)
    future_highs = pd.concat([df["high"].shift(-i) for i in range(1, horizon + 1)], axis=1)
    fmin = future_lows.min(axis=1)
    fmax = future_highs.max(axis=1)
    df["delta_low_log_120m"] = np.log(fmin / df["close"]).astype(np.float32)
    df["delta_low_lin_120m"] = (fmin - df["close"]).astype(np.float32)
    df["delta_high_log_120m"] = np.log(fmax / df["close"]).astype(np.float32)
    df["delta_high_lin_120m"] = (fmax - df["close"]).astype(np.float32)

    df["delta_log_60m"] = np.log(df["close"].shift(-12) / df["close"]).astype(np.float32)
    df["delta_lin_60m"] = (df["close"].shift(-12) - df["close"]).astype(np.float32)

    df["delta_log_240m"] = np.log(df["close"].shift(-48) / df["close"]).astype(np.float32)
    df["delta_lin_240m"] = (df["close"].shift(-48) - df["close"]).astype(np.float32)

    # --- validace typů --------------------------------------------------------
    feature_only = df.drop(
        columns=[
            "delta_log_120m",
            "delta_lin_120m",
            "delta_low_log_120m",
            "delta_low_lin_120m",
            "delta_high_log_120m",
            "delta_high_lin_120m",
            "delta_log_60m",
            "delta_lin_60m",
            "delta_log_240m",
            "delta_lin_240m",
        ],
        errors="ignore",
    )
    if feature_only.isna().any().any():
        raise ValueError("NaN values present after feature engineering")
    obj_cols = feature_only.select_dtypes(include="object").columns
    if len(obj_cols) > 0:
        raise TypeError(f"Object dtypes present after feature merge: {list(obj_cols)}")

    return df


FEATURE_COLUMNS: list[str] = [
    "ofi_base",
    "ofi_quote",
    "tbr_base",
    "d_tbr_base",
    "ema12_tbr_base",
    "taker_buy_sell_ratio",
    "z_volume",
    "rel_close_vwap",
    "ret3",
    "ret12",
    "rv_5m",
    "volatility_60m",
    "parkinson12",
    "atr14",
    "lob_imbalance_L1",
    "lob_imbalance_L2",
    "oi_delta_15m",
    "basis_annualized",
    "tod_sin",
    "tod_cos",
    "is_day",
    "is_night",
    "is_weekend",
    "is_weekday",
]
