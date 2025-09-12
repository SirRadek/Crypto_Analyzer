from __future__ import annotations

import numpy as np

# Only a minimal subset of indicators is required for the Core-18 feature set.
# Keeping the imports explicit makes the dependency surface obvious and helps
# speed up feature engineering.


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a lightweight set of technical, order-flow and time features.

    A shallow copy is used to avoid doubling memory usage while keeping the
    original ``df`` unmodified."""
    df = df.copy(deep=False)

    # --- Order-flow & volume -------------------------------------------------
    vol = df["volume"].replace(0, np.nan)
    qvol = df["quote_asset_volume"].replace(0, np.nan)

    df["tbr_base"] = df["taker_buy_base"] / vol
    tbr_quote = df["taker_buy_quote"] / qvol
    df["ofi_base"] = 2 * df["tbr_base"] - 1
    df["ofi_quote"] = 2 * tbr_quote - 1
    df["d_tbr_base"] = df["tbr_base"].diff()
    df["ema12_tbr_base"] = df["tbr_base"].ewm(span=12, adjust=False).mean()
    df["z_volume"] = (vol - vol.rolling(36).mean()) / vol.rolling(36).std()

    # --- VWAP & price relationship -------------------------------------------
    vwap = qvol / vol
    df["rel_close_vwap"] = (df["close"] - vwap).abs() / vwap

    # --- Returns & momentum ---------------------------------------------------
    ret1 = np.log(df["close"] / df["close"].shift(1))
    df["ret3"] = ret1.rolling(3).sum()
    df["ret12"] = ret1.rolling(12).sum()

    # --- Volatility -----------------------------------------------------------
    df["rv_5m"] = ret1.rolling(1).std()
    df["volatility_60m"] = ret1.rolling(12).std()
    hl_log = np.log(df["high"] / df["low"])
    parkinson = (hl_log**2) / (4 * np.log(2))
    df["parkinson12"] = np.sqrt(parkinson.rolling(12).mean())
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    # --- Additional market & order-book features -----------------------------
    denom = (df["volume"] - df["taker_buy_base"]).replace(0, np.nan)
    df["taker_buy_sell_ratio"] = df["taker_buy_base"] / denom

    if "basis_annualized" not in df.columns:
        df["basis_annualized"] = 0.0

    if "open_interest" in df.columns:
        df["oi_delta_15m"] = df["open_interest"].diff(3)
    else:
        df["oi_delta_15m"] = 0.0

    for level in range(1, 3):
        bid_col = f"lob_bid_L{level}"
        ask_col = f"lob_ask_L{level}"
        imb_col = f"lob_imbalance_L{level}"
        if bid_col in df.columns and ask_col in df.columns:
            df[imb_col] = (df[bid_col] - df[ask_col]) / (df[bid_col] + df[ask_col])
        else:
            df[imb_col] = 0.0

    # --- Time features --------------------------------------------------------
    ts = df["timestamp"]
    minute = ts.dt.hour * 60 + ts.dt.minute
    df["tod_sin"] = np.sin(2 * np.pi * minute / 1440)
    df["tod_cos"] = np.cos(2 * np.pi * minute / 1440)
    is_day = ((ts.dt.hour >= 8) & (ts.dt.hour < 20)).astype(np.float32)
    df["is_day"] = is_day
    df["is_night"] = 1.0 - is_day
    is_weekend = (ts.dt.dayofweek >= 5).astype(np.float32)
    df["is_weekend"] = is_weekend
    df["is_weekday"] = 1.0 - is_weekend

    df = df.fillna(0)

    # --- Targets -------------------------------------------------------------
    horizon = 24  # 120 minutes at 5-minute bars
    df["delta_log_120m"] = np.log(df["close"].shift(-horizon) / df["close"])
    df["delta_lin_120m"] = df["close"].shift(-horizon) - df["close"]
    df["delta_log_60m"] = np.log(df["close"].shift(-12) / df["close"])
    df["delta_lin_60m"] = df["close"].shift(-12) - df["close"]
    df["delta_log_240m"] = np.log(df["close"].shift(-48) / df["close"])
    df["delta_lin_240m"] = df["close"].shift(-48) - df["close"]

    # Downcast all float columns to float32 to save memory
    float_cols = df.select_dtypes(include="float64").columns
    df[float_cols] = df[float_cols].astype(np.float32)

    target_cols = [
        "delta_log_120m",
        "delta_lin_120m",
        "delta_log_60m",
        "delta_lin_60m",
        "delta_log_240m",
        "delta_lin_240m",
    ]
    feature_only = df.drop(columns=target_cols, errors="ignore")
    if feature_only.isna().any().any():
        raise ValueError("NaN values present after feature engineering")

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
