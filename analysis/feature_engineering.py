import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from .indicators import calculate_sma, calculate_ema, calculate_rsi


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score with given ``window`` size."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std


@dataclass(frozen=True)
class OnChainConfig:
    """Container for on-chain time series used in feature engineering."""

    data: pd.DataFrame


def _add_onchain_features(df: pd.DataFrame, oc: pd.DataFrame) -> pd.DataFrame:
    """Merge on-chain data ``oc`` into ``df`` and compute derived metrics."""

    if "timestamp" not in oc.columns:
        return df

    oc = oc.copy()
    oc["timestamp"] = pd.to_datetime(oc["timestamp"])
    oc = oc.sort_values("timestamp")
    df = df.sort_values("timestamp")
    df = pd.merge_asof(df, oc, on="timestamp", direction="backward")

    # -- 5m metrics --------------------------------------------------------
    for col in ["feerate_median_5m", "mempool_txcount_5m"]:
        base = f"onch_{col}"
        if col in df.columns:
            df[base] = df[col]
            df[f"{base}_delta_15m"] = df[base].diff(3)
            df[f"{base}_z_30d"] = _zscore(df[base], 30 * 24 * 12)
        else:
            df[base] = df[f"{base}_delta_15m"] = df[f"{base}_z_30d"] = 0.0

    # -- 1h exchange flows -------------------------------------------------
    exch_cols = ["exch_inflow_1h", "exch_outflow_1h", "exch_netflow_1h"]
    for col in exch_cols:
        base = f"onch_{col}"
        if col in df.columns:
            df[base] = df[col]
            df[f"{base}_delta_1h"] = df[base].diff(12)
            df[f"{base}_z_7d"] = _zscore(df[base], 7 * 24 * 12)
        else:
            df[base] = df[f"{base}_delta_1h"] = df[f"{base}_z_7d"] = 0.0

    # -- USDT metrics ------------------------------------------------------
    if "usdt_events_30m" in df.columns:
        df["onch_usdt_events_30m"] = df["usdt_events_30m"]
    else:
        df["onch_usdt_events_30m"] = 0.0

    if "usdt_amount_usd_60m" in df.columns:
        df["onch_usdt_amount_usd_60m"] = df["usdt_amount_usd_60m"]
    else:
        df["onch_usdt_amount_usd_60m"] = 0.0

    if "usdt_large_mint_60m" in df.columns:
        series = df["usdt_large_mint_60m"]
        thresh = series.rolling(90 * 24 * 12).quantile(0.95)
        df["onch_usdt_large_mint_60m"] = (series > thresh).astype(int)
    else:
        df["onch_usdt_large_mint_60m"] = 0.0

    # Drop original columns if present
    drop_cols = [
        "feerate_median_5m",
        "mempool_txcount_5m",
        "exch_inflow_1h",
        "exch_outflow_1h",
        "exch_netflow_1h",
        "usdt_events_30m",
        "usdt_amount_usd_60m",
        "usdt_large_mint_60m",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return df


def create_features(df: pd.DataFrame, onchain: Optional[OnChainConfig] = None) -> pd.DataFrame:
    """Add technical, order-flow, on-chain and time features to ``df``."""
    df = df.copy()
    if onchain is not None:
        df = _add_onchain_features(df, onchain.data)

    # --- Order-flow & volume -------------------------------------------------
    vol = df["volume"].replace(0, np.nan)
    qvol = df["quote_asset_volume"].replace(0, np.nan)

    df["tbr_base"] = df["taker_buy_base"] / vol
    df["tbr_quote"] = df["taker_buy_quote"] / qvol
    df["ofi_base"] = 2 * df["tbr_base"] - 1
    df["ofi_quote"] = 2 * df["tbr_quote"] - 1
    df["ofi_z_24h"] = _zscore(df["ofi_base"], 24 * 12)
    df["d_tbr_base"] = df["tbr_base"].diff()
    df["ema12_tbr_base"] = df["tbr_base"].ewm(span=12, adjust=False).mean()
    df["ema36_tbr_base"] = df["tbr_base"].ewm(span=36, adjust=False).mean()

    trades = df["number_of_trades"].replace(0, np.nan)
    df["avg_trade_size"] = vol / trades
    df["d_volume"] = vol.diff()
    df["z_trades"] = (df["number_of_trades"] - df["number_of_trades"].rolling(36).mean()) / df["number_of_trades"].rolling(36).std()
    df["z_volume"] = (vol - vol.rolling(36).mean()) / vol.rolling(36).std()

    vol_delta = 2 * df["taker_buy_base"] - vol
    df["cvd_5m"] = vol_delta.cumsum()
    df["cvd_5m_ema15"] = df["cvd_5m"].ewm(span=15, adjust=False).mean()
    df["cvd_5m_ema60"] = df["cvd_5m"].ewm(span=60, adjust=False).mean()

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
    df["mom_ret_skew_15m"] = df["ret1"].rolling(3).skew()
    df["mom_ret_kurt_15m"] = df["ret1"].rolling(3).kurt()

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

    ts_local = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("Europe/Prague")
    hour_local = ts_local.dt.hour
    df["time_session_eu"] = hour_local.between(7, 15).astype(int)
    df["time_session_us"] = hour_local.between(13, 21).astype(int)
    df["time_session_asia"] = ((hour_local >= 23) | (hour_local <= 7)).astype(int)
    df["funding_cycle_idx"] = (
        (df["timestamp"].view("int64") // (3600 * 10 ** 9)) % 8
    ).astype(int)

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

    steps_per_cycle = 8 * 60 // 5
    phase = (df["timestamp"].view("int64") // (5 * 60 * 10 ** 9)) % steps_per_cycle
    angle = 2 * np.pi * phase / steps_per_cycle
    df["deriv_funding_phase_sin"] = np.sin(angle)
    df["deriv_funding_phase_cos"] = np.cos(angle)

    if "basis_annualized" not in df.columns:
        df["basis_annualized"] = 0.0
    df["deriv_basis_z_7d"] = _zscore(df["basis_annualized"], 7 * 24 * 12)

    if "open_interest" in df.columns:
        df["deriv_oi_delta_5m"] = df["open_interest"].diff()
        df["deriv_oi_delta_15m"] = df["open_interest"].diff(3)
        df["deriv_oi_delta_60m"] = df["open_interest"].diff(12)
        df["deriv_oi_z_7d"] = _zscore(df["open_interest"], 7 * 24 * 12)
    else:
        df["deriv_oi_delta_5m"] = 0.0
        df["deriv_oi_delta_15m"] = 0.0
        df["deriv_oi_delta_60m"] = 0.0
        df["deriv_oi_z_7d"] = 0.0

    if {"liq_long_usd", "liq_short_usd"}.issubset(df.columns):
        denom = df["liq_short_usd"].replace(0, np.nan)
        df["liq_long_short_ratio"] = df["liq_long_usd"] / denom
    else:
        df["liq_long_short_ratio"] = 0.0

    denom = (df["volume"] - df["taker_buy_base"]).replace(0, np.nan)
    df["taker_buy_sell_ratio"] = df["taker_buy_base"] / denom
    df["tbr_ratio_z_24h"] = _zscore(df["taker_buy_sell_ratio"], 24 * 12)

    for level in range(1, 6):
        bid_col = f"lob_bid_L{level}"
        ask_col = f"lob_ask_L{level}"
        imb_col = f"lob_imbalance_L{level}"
        if bid_col in df.columns and ask_col in df.columns:
            df[imb_col] = (df[bid_col] - df[ask_col]) / (df[bid_col] + df[ask_col])
        else:
            df[imb_col] = 0.0

    for level in [1, 2]:
        bid_col = f"lob_bid_L{level}"
        ask_col = f"lob_ask_L{level}"
        name = f"lob_queue_imbalance_l{level}"
        if bid_col in df.columns and ask_col in df.columns:
            df[name] = (df[bid_col] - df[ask_col]) / (df[bid_col] + df[ask_col])
        else:
            df[name] = 0.0

    bid_cols = [f"lob_bid_L{i}" for i in range(1, 6) if f"lob_bid_L{i}" in df.columns]
    ask_cols = [f"lob_ask_L{i}" for i in range(1, 6) if f"lob_ask_L{i}" in df.columns]
    if len(bid_cols) >= 2:
        x = np.arange(len(bid_cols))
        xc = x - x.mean()
        denom = (xc ** 2).sum()
        df["lob_book_slope_bid"] = (df[bid_cols].values * xc).sum(axis=1) / denom
    else:
        df["lob_book_slope_bid"] = 0.0

    if len(ask_cols) >= 2:
        x = np.arange(len(ask_cols))
        xc = x - x.mean()
        denom = (xc ** 2).sum()
        df["lob_book_slope_ask"] = (df[ask_cols].values * xc).sum(axis=1) / denom
    else:
        df["lob_book_slope_ask"] = 0.0

    for side, cols in {"bid": bid_cols, "ask": ask_cols}.items():
        if not cols:
            df[f"lob_wall_dist_bps_{side}"] = 0.0
            df[f"lob_wall_size_rel_{side}"] = 0.0
            continue
        sizes = df[cols]
        mean = sizes.rolling(288).mean()
        std = sizes.rolling(288).std()
        wall_level = []
        wall_size = []
        for idx in sizes.index:
            level = 0
            size_val = 0.0
            for i, c in enumerate(cols, 1):
                thr = mean.at[idx, c] + 2 * std.at[idx, c]
                if sizes.at[idx, c] > thr:
                    level = i
                    size_val = sizes.at[idx, c]
                    break
            wall_level.append(level)
            wall_size.append(size_val)
        total = sizes.sum(axis=1).replace(0, np.nan)
        df[f"lob_wall_dist_bps_{side}"] = pd.Series(wall_level, index=df.index)
        df[f"lob_wall_size_rel_{side}"] = (
            pd.Series(wall_size, index=df.index) / total
        ).fillna(0)

    df["rv_5m"] = df["ret1"].rolling(1).std()
    df["rv_30m"] = df["ret1"].rolling(6).std()
    df["vol_rv_z_7d"] = _zscore(df["rv_5m"], 7 * 24 * 12)


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
    "ofi_z_24h",
    "d_tbr_base",
    "ema12_tbr_base",
    "ema36_tbr_base",
    "avg_trade_size",
    "z_trades",
    "z_volume",
    "d_volume",
    "cvd_5m",
    "cvd_5m_ema15",
    "cvd_5m_ema60",
    "taker_buy_sell_ratio",
    "tbr_ratio_z_24h",
    "vwap",
    "close_minus_vwap",
    "rel_close_vwap",
    "return_1d",
    "ret1",
    "ret3",
    "ret6",
    "ret12",
    "mom_ret_skew_15m",
    "mom_ret_kurt_15m",
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
    "time_session_eu",
    "time_session_us",
    "time_session_asia",
    "funding_cycle_idx",
    "sma_7",
    "sma_14",
    "ema_7",
    "ema_14",
    "rsi_14",
    "funding_now",
    "funding_delta_1h",
    "deriv_funding_phase_sin",
    "deriv_funding_phase_cos",
    "basis_annualized",
    "deriv_basis_z_7d",
    "deriv_oi_delta_5m",
    "deriv_oi_delta_15m",
    "deriv_oi_delta_60m",
    "deriv_oi_z_7d",
    "liq_long_short_ratio",
    "lob_imbalance_L1",
    "lob_imbalance_L2",
    "lob_imbalance_L3",
    "lob_imbalance_L4",
    "lob_imbalance_L5",
    "lob_queue_imbalance_l1",
    "lob_queue_imbalance_l2",
    "lob_book_slope_bid",
    "lob_book_slope_ask",
    "lob_wall_dist_bps_bid",
    "lob_wall_size_rel_bid",
    "lob_wall_dist_bps_ask",
    "lob_wall_size_rel_ask",
    "rv_5m",
    "rv_30m",
    "vol_rv_z_7d",
    "google_trends_btc_delta",
    "mvrv_z",
    "sopr",
    "fees_per_tx",
    "eth_ret_lagged",
    "sol_ret_lagged",
    "bnb_ret_lagged",
    "btc_dominance_delta",
    "onch_feerate_median_5m",
    "onch_feerate_median_5m_delta_15m",
    "onch_feerate_median_5m_z_30d",
    "onch_mempool_txcount_5m",
    "onch_mempool_txcount_5m_delta_15m",
    "onch_mempool_txcount_5m_z_30d",
    "onch_exch_inflow_1h",
    "onch_exch_inflow_1h_delta_1h",
    "onch_exch_inflow_1h_z_7d",
    "onch_exch_outflow_1h",
    "onch_exch_outflow_1h_delta_1h",
    "onch_exch_outflow_1h_z_7d",
    "onch_exch_netflow_1h",
    "onch_exch_netflow_1h_delta_1h",
    "onch_exch_netflow_1h_z_7d",
    "onch_usdt_events_30m",
    "onch_usdt_amount_usd_60m",
    "onch_usdt_large_mint_60m",
]
