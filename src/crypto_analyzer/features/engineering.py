from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype

from crypto_analyzer.utils.config import CONFIG, FeatureSettings

# Lehký FE laděný pro 2h BTCUSDT. Bez těžkých závislostí, vše float32.

H = 24  # 120min horizon in 5m candles

MULTI_TF_MINUTES: tuple[int, ...] = (15, 60, 240, 1440)
PARKINSON_CONST = 1.0 / (4.0 * np.log(2.0))

CROSS_ASSET_KEYWORDS: dict[str, Sequence[str]] = {
    "ethbtc": ("ethbtc",),
    "btcd": ("btc.d", "btc_dom", "btcdominance", "btc_d"),
    "dxy": ("dxy",),
    "es": ("es1", "es_", "sp500", "spx"),
    "nq": ("nq1", "nasdaq", "nq_"),
    "gold": ("xau", "gold"),
}

FUNDING_COLUMN_CANDIDATES: tuple[str, ...] = (
    "deriv_funding_rate",
    "funding_rate",
    "funding_8h",
    "perp_funding_rate",
)

LIQUIDATION_LONG_CANDIDATES: tuple[str, ...] = (
    "deriv_liq_long",
    "liquidations_long",
    "liquidations_buy",
    "liq_long",
)

LIQUIDATION_SHORT_CANDIDATES: tuple[str, ...] = (
    "deriv_liq_short",
    "liquidations_short",
    "liquidations_sell",
    "liq_short",
)

ONCHAIN_FEATURES = (
    "onch_fee_fast_satvb",
    "onch_fee_30m_satvb",
    "onch_fee_60m_satvb",
    "onch_fee_min_satvb",
    "onch_mempool_count",
    "onch_mempool_vsize_vB",
    "onch_mempool_total_fee_sat",
    "onch_fee_wavg_satvb",
    "onch_fee_p50_satvb",
    "onch_fee_p90_satvb",
    "onch_difficulty",
    "onch_height",
    "onch_diff_change_pct",
)


@dataclass(frozen=True)
class FeatureColumnRegistry:
    """Immutable catalogue of engineered feature column names."""

    base: tuple[str, ...]
    orderbook: tuple[str, ...]
    derivatives: tuple[str, ...]
    onchain: tuple[str, ...]

    def active(self, settings: FeatureSettings) -> list[str]:
        """Return feature names enabled under *settings*."""

        columns: list[str] = list(self.base)
        if settings.include_orderbook:
            columns.extend(self.orderbook)
        if settings.include_derivatives:
            columns.extend(self.derivatives)
        if settings.include_onchain:
            columns.extend(self.onchain)
        return columns


REGISTRY = FeatureColumnRegistry(
    base=(
        "ofi_base",
        "ofi_quote",
        "tbr_base",
        "d_tbr_base",
        "ema12_tbr_base",
        "taker_buy_sell_ratio",
        "z_volume",
        "rel_close_vwap",
        "ret3",
        "ofi_base_roll_15m",
        "ofi_base_roll_1h",
        "ofi_quote_roll_15m",
        "ofi_quote_roll_1h",
        "tbr_base_roll_15m",
        "tbr_base_roll_1h",
        "mom_log_ret_15m",
        "mom_log_ret_1h",
        "mom_log_ret_4h",
        "mom_log_ret_1d",
        "volatility_60m",
        "atr14",
        "vol_realized_15m",
        "vol_realized_1h",
        "vol_realized_4h",
        "vol_realized_1d",
        "vol_of_vol_15m",
        "vol_of_vol_1h",
        "vol_of_vol_4h",
        "vol_of_vol_1d",
        "vol_range_parkinson_15m",
        "vol_range_parkinson_1h",
        "vol_range_parkinson_4h",
        "vol_range_parkinson_1d",
        "mom_microtrend_ema_ratio",
        "tod_sin",
        "tod_cos",
        "is_day",
        "is_night",
        "is_weekend",
        "is_weekday",
        "time_hour_sin",
        "time_hour_cos",
        "time_dow_sin",
        "time_dow_cos",
        "cross_ethbtc_ret",
        "cross_ethbtc_divergence",
        "cross_ethbtc_corr_4h",
        "cross_btcd_ret",
        "cross_btcd_divergence",
        "cross_btcd_corr_4h",
        "cross_dxy_ret",
        "cross_dxy_divergence",
        "cross_dxy_corr_4h",
        "cross_es_ret",
        "cross_es_divergence",
        "cross_es_corr_4h",
        "cross_nq_ret",
        "cross_nq_divergence",
        "cross_nq_corr_4h",
        "cross_gold_ret",
        "cross_gold_divergence",
        "cross_gold_corr_4h",
    ),
    orderbook=(
        "lob_imbalance_L1",
        "lob_imbalance_L2",
        "wall_bid_dist_bps",
        "wall_bid_size_rel",
        "wall_ask_dist_bps",
        "wall_ask_size_rel",
        "lob_spread_bps",
        "lob_depth_imbalance",
        "lob_bid_slope_bps",
        "lob_ask_slope_bps",
    ),
    derivatives=(
        "oi_delta_15m",
        "basis_annualized",
        "deriv_funding_rate",
        "deriv_funding_rate_change",
        "deriv_funding_rate_smooth",
        "deriv_basis_trend",
        "deriv_basis_slope",
        "deriv_open_interest",
        "deriv_oi_change_pct",
        "deriv_oi_velocity",
        "deriv_liq_total",
        "deriv_liq_net",
        "deriv_liq_to_oi",
    ),
    onchain=ONCHAIN_FEATURES,
)


_REQUIRED_BASE_INPUTS: tuple[str, ...] = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "taker_buy_base",
    "taker_buy_quote",
)


def _infer_step_minutes(ts: pd.Series) -> int:
    """Infer the base candle interval expressed in minutes."""

    deltas = ts.diff().dropna()
    if deltas.empty:
        return 1
    median_delta = deltas.median()
    if isinstance(median_delta, pd.Timedelta):
        minutes = median_delta.total_seconds() / 60.0
    else:  # pragma: no cover - defensive for unexpected inputs
        minutes = float(median_delta)
    return max(1, int(round(minutes)))


def _format_minutes_label(minutes: int) -> str:
    """Human-readable suffix for timeframe-derived features."""

    if minutes % 1440 == 0:
        days = minutes // 1440
        return f"{days}d"
    if minutes % 60 == 0:
        hours = minutes // 60
        return f"{hours}h"
    return f"{minutes}m"


def _build_timeframe_info(ts: pd.Series, durations: Sequence[int]) -> dict[int, tuple[int, str]]:
    """Map timeframe duration (minutes) to (window, label)."""

    step_minutes = _infer_step_minutes(ts)
    info: dict[int, tuple[int, str]] = {}
    for minutes in durations:
        if minutes <= 0:
            continue
        if minutes < step_minutes:
            continue
        window = max(1, int(round(minutes / step_minutes)))
        info[minutes] = (window, _format_minutes_label(minutes))
    return info


def _match_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Return the first column containing any of the ``candidates`` substrings."""

    lowered = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        cand = candidate.lower()
        for col_lower, original in lowered.items():
            if cand in col_lower:
                return original
    return None


def _ensure_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Validate that *columns* can be interpreted as numeric."""

    bad: list[str] = []
    for col in columns:
        if col not in df.columns:
            continue
        if is_numeric_dtype(df[col]):
            continue
        try:
            pd.to_numeric(df[col], errors="raise")
        except (TypeError, ValueError):
            bad.append(col)
    if bad:
        joined = ", ".join(bad)
        raise TypeError(f"Non-numeric values detected in input columns: {joined}")


def _normalize_timestamp(df: pd.DataFrame) -> None:
    """Ensure the ``timestamp`` column is timezone-aware UTC."""

    if "timestamp" not in df.columns:
        raise KeyError("Input frame must contain a 'timestamp' column")

    ts = df["timestamp"]
    if not is_datetime64_any_dtype(ts):
        try:
            df["timestamp"] = pd.to_datetime(ts, utc=True, errors="raise")
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise TypeError("Timestamp column could not be parsed as datetime") from exc
        ts = df["timestamp"]
    if ts.dt.tz is None:
        df["timestamp"] = ts.dt.tz_localize("UTC")


def validate_feature_inputs(df: pd.DataFrame, settings: FeatureSettings) -> None:
    """Validate the bare minimum structure required for feature engineering."""

    missing = [col for col in _REQUIRED_BASE_INPUTS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required input columns: {', '.join(sorted(missing))}")

    _ensure_numeric(df, _REQUIRED_BASE_INPUTS[1:])
    _normalize_timestamp(df)

    if not settings.include_onchain:
        return

    prefixed = [col for col in df.columns if col.startswith("onch_")]
    extras = sorted(set(prefixed) - set(ONCHAIN_FEATURES))
    if extras:
        raise ValueError(
            "Unexpected on-chain columns present: " + ", ".join(extras)
        )


def _resolve_feature_settings(settings: FeatureSettings | None) -> FeatureSettings:
    return CONFIG.features if settings is None else settings


def _build_feature_columns(settings: FeatureSettings) -> list[str]:
    return REGISTRY.active(settings)


def get_feature_columns(settings: FeatureSettings | None = None) -> list[str]:
    """Return feature column names honoring *settings* toggles."""

    return _build_feature_columns(_resolve_feature_settings(settings))


def create_features(
    df: pd.DataFrame, settings: FeatureSettings | None = None
) -> pd.DataFrame:
    """Add technical and time-series features to ``df``.

    Parameters
    ----------
    df:
        Input data frame containing OHLCV and optional ``onch_`` columns.
    settings:
        Optional feature settings overriding :data:`CONFIG.features`.

    Returns
    -------
    pandas.DataFrame
        Data frame with additional float32 features.
    """

    settings = _resolve_feature_settings(settings)
    df = df.copy()
    validate_feature_inputs(df, settings)
    fill_value = np.float32(settings.fillna_value)
    ffill_limit = settings.forward_fill_limit
    if ffill_limit < 0:
        ffill_limit = None

    if settings.include_onchain:
        for col in ONCHAIN_FEATURES:
            if col not in df.columns:
                df[col] = fill_value
    else:
        drop_cols = [c for c in df.columns if c.startswith("onch_")]
        if drop_cols:
            df = df.drop(columns=drop_cols)

    ts = pd.to_datetime(df["timestamp"], utc=True)
    timeframe_info = _build_timeframe_info(ts, MULTI_TF_MINUTES)
    timeframe_windows = {minutes: window for minutes, (window, _) in timeframe_info.items()}
    timeframe_labels = {minutes: label for minutes, (_, label) in timeframe_info.items()}
    supported_minutes = set(timeframe_windows)

    # --- základní numerika ---------------------------------------------------
    # nech timestamp beze změny, ostatní numerické sloupce konvertuj na float32
    numeric = df.select_dtypes(include=["number"]).columns.drop(["timestamp"], errors="ignore")
    if len(numeric) > 0:
        df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce").astype(np.float32)

    # forward-fill on-chain metrics only within the same hour to avoid leakage
    if settings.include_onchain:
        onch_cols = [c for c in df.columns if c.startswith("onch_")]
        if onch_cols:
            hour = pd.to_datetime(df["timestamp"]).dt.floor("h")
            grouped = df[onch_cols].groupby(hour)
            if ffill_limit is None:
                filled = grouped.ffill()
            else:
                filled = grouped.ffill(limit=ffill_limit)
            df[onch_cols] = filled.fillna(fill_value).astype(np.float32)

    # --- order-flow & volume --------------------------------------------------
    vol = df["volume"].replace(0.0, np.nan)
    qvol = df["quote_asset_volume"].replace(0.0, np.nan)

    df["tbr_base"] = (df["taker_buy_base"] / vol).astype(np.float32)
    tbr_quote = (df["taker_buy_quote"] / qvol).astype(np.float32)

    df["ofi_base"] = (2.0 * df["tbr_base"] - 1.0).astype(np.float32)
    df["ofi_quote"] = (2.0 * tbr_quote - 1.0).astype(np.float32)
    df["d_tbr_base"] = df["tbr_base"].diff().astype(np.float32)
    df["ema12_tbr_base"] = df["tbr_base"].ewm(span=12, adjust=False).mean().astype(np.float32)
    df["z_volume"] = ((vol - vol.rolling(36).mean()) / vol.rolling(36).std()).astype(np.float32)

    for minutes in MULTI_TF_MINUTES:
        label = timeframe_labels.get(minutes, _format_minutes_label(minutes))
        if minutes not in supported_minutes:
            df[f"ofi_base_roll_{label}"] = np.float32(fill_value)
            df[f"ofi_quote_roll_{label}"] = np.float32(fill_value)
            df[f"tbr_base_roll_{label}"] = np.float32(fill_value)
            continue
        window = timeframe_windows[minutes]
        df[f"ofi_base_roll_{label}"] = df["ofi_base"].rolling(window).mean().astype(np.float32)
        df[f"ofi_quote_roll_{label}"] = df["ofi_quote"].rolling(window).mean().astype(np.float32)
        df[f"tbr_base_roll_{label}"] = df["tbr_base"].rolling(window).mean().astype(np.float32)

    # --- VWAP vztah ----------------------------------------------------------
    vwap = (qvol / vol).astype(np.float32)
    df["rel_close_vwap"] = ((df["close"] - vwap).abs() / vwap).astype(np.float32)

    # --- výnosy & momentum ----------------------------------------------------
    safe_close = df["close"].replace(0.0, np.nan)
    log_close = np.log(safe_close)
    ret1 = log_close.diff().astype(np.float32)
    df["ret3"] = ret1.rolling(3).sum().astype(np.float32)

    for minutes in MULTI_TF_MINUTES:
        label = timeframe_labels.get(minutes, _format_minutes_label(minutes))
        if minutes not in supported_minutes:
            df[f"mom_log_ret_{label}"] = np.float32(fill_value)
            continue
        window = timeframe_windows[minutes]
        df[f"mom_log_ret_{label}"] = log_close.diff(window).astype(np.float32)

    for minutes in MULTI_TF_MINUTES:
        label = timeframe_labels.get(minutes, _format_minutes_label(minutes))
        if minutes not in supported_minutes:
            df[f"vol_realized_{label}"] = np.float32(fill_value)
            df[f"vol_of_vol_{label}"] = np.float32(fill_value)
            continue
        window = timeframe_windows[minutes]
        realized = ret1.rolling(window).std()
        df[f"vol_realized_{label}"] = realized.astype(np.float32)
        vol_of_vol = realized.rolling(window).std()
        df[f"vol_of_vol_{label}"] = vol_of_vol.astype(np.float32)

    high = df["high"].replace(0.0, np.nan)
    low = df["low"].replace(0.0, np.nan)
    log_range = np.log(high / low).pow(2)
    for minutes in MULTI_TF_MINUTES:
        label = timeframe_labels.get(minutes, _format_minutes_label(minutes))
        if minutes not in supported_minutes:
            df[f"vol_range_parkinson_{label}"] = np.float32(fill_value)
            continue
        window = timeframe_windows[minutes]
        pk_var = log_range.rolling(window).mean() * PARKINSON_CONST
        df[f"vol_range_parkinson_{label}"] = np.sqrt(pk_var).astype(np.float32)

    fast_span = max(2, timeframe_windows.get(60, 12))
    slow_span = max(fast_span + 1, timeframe_windows.get(240, fast_span * 4))
    fast_ema = df["close"].ewm(span=fast_span, adjust=False).mean()
    slow_ema = df["close"].ewm(span=slow_span, adjust=False).mean()
    denom = df["close"].replace(0.0, np.nan)
    df["mom_microtrend_ema_ratio"] = ((fast_ema - slow_ema) / denom).astype(np.float32)

    # --- begin minimal anti-fragmentation patch ---
    # původní logika přidávala rozsáhlé z-score a delta kopie téměř všech
    # sloupců. Pro novou minimalistickou sadu rysů je držíme vypnuté, aby
    # nedublovaly momentum/mean-reversion metriky.
    # --- end patch ---

    # --- volatilita -----------------------------------------------------------
    df["volatility_60m"] = ret1.rolling(12).std().astype(np.float32)

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

    if settings.include_derivatives:
        if "basis_annualized" not in df.columns:
            df["basis_annualized"] = fill_value
        else:
            df["basis_annualized"] = df["basis_annualized"].astype(np.float32)

        if "open_interest" in df.columns:
            df["oi_delta_15m"] = df["open_interest"].diff(3).astype(np.float32)
        elif "oi_delta_15m" not in df.columns:
            df["oi_delta_15m"] = fill_value

        funding_col = _match_column(df, FUNDING_COLUMN_CANDIDATES)
        if funding_col is not None:
            funding = pd.to_numeric(df[funding_col], errors="coerce")
            df["deriv_funding_rate"] = funding.astype(np.float32)
        elif "deriv_funding_rate" not in df.columns:
            df["deriv_funding_rate"] = fill_value

        df["deriv_funding_rate_change"] = (
            df["deriv_funding_rate"].diff().astype(np.float32)
        )
        smooth_window = max(1, timeframe_windows.get(240, 12))
        df["deriv_funding_rate_smooth"] = (
            df["deriv_funding_rate"].rolling(smooth_window).mean().astype(np.float32)
        )

        basis = df["basis_annualized"].astype(np.float32)
        basis_window = max(1, timeframe_windows.get(240, 12))
        df["deriv_basis_trend"] = (
            (basis - basis.rolling(basis_window).mean()).astype(np.float32)
        )
        df["deriv_basis_slope"] = basis.diff(basis_window).astype(np.float32)

        if "open_interest" in df.columns:
            oi = pd.to_numeric(df["open_interest"], errors="coerce").astype(np.float32)
            df["deriv_open_interest"] = oi
            pct = oi.pct_change().replace([np.inf, -np.inf], np.nan)
            df["deriv_oi_change_pct"] = pct.astype(np.float32)
            vel_window = max(1, timeframe_windows.get(60, 3))
            df["deriv_oi_velocity"] = oi.diff(vel_window).astype(np.float32)
        else:
            if "deriv_open_interest" not in df.columns:
                df["deriv_open_interest"] = fill_value
            df["deriv_oi_change_pct"] = fill_value
            df["deriv_oi_velocity"] = fill_value

        long_col = _match_column(df, LIQUIDATION_LONG_CANDIDATES)
        short_col = _match_column(df, LIQUIDATION_SHORT_CANDIDATES)
        if long_col is not None or short_col is not None:
            long_vals = (
                pd.to_numeric(df.get(long_col, fill_value), errors="coerce")
                if long_col is not None
                else pd.Series(fill_value, index=df.index, dtype=np.float32)
            )
            short_vals = (
                pd.to_numeric(df.get(short_col, fill_value), errors="coerce")
                if short_col is not None
                else pd.Series(fill_value, index=df.index, dtype=np.float32)
            )
            long_vals = long_vals.astype(np.float32)
            short_vals = short_vals.astype(np.float32)
            df["deriv_liq_total"] = (long_vals + short_vals).astype(np.float32)
            df["deriv_liq_net"] = (long_vals - short_vals).astype(np.float32)
            if "open_interest" in df.columns:
                oi_base = df["open_interest"].replace(0.0, np.nan)
                df["deriv_liq_to_oi"] = (
                    ((long_vals + short_vals) / oi_base).astype(np.float32)
                )
            else:
                df["deriv_liq_to_oi"] = fill_value
        else:
            df["deriv_liq_total"] = fill_value
            df["deriv_liq_net"] = fill_value
            df["deriv_liq_to_oi"] = fill_value
    else:
        df = df.drop(columns=list(REGISTRY.derivatives), errors="ignore")
        df = df.drop(columns=[c for c in df.columns if c.startswith("deriv_")], errors="ignore")

    if settings.include_orderbook:
        for level in range(1, 3):
            bid_col = f"lob_bid_L{level}"
            ask_col = f"lob_ask_L{level}"
            imb_col = f"lob_imbalance_L{level}"
            if bid_col in df.columns and ask_col in df.columns:
                num = (df[bid_col] - df[ask_col]).astype(np.float32)
                den = (df[bid_col] + df[ask_col]).replace(0.0, np.nan).astype(np.float32)
                df[imb_col] = (num / den).astype(np.float32)
            elif imb_col not in df.columns:
                df[imb_col] = fill_value

        # --- LOB walls -------------------------------------------------------
        bid_px_cols = sorted([c for c in df.columns if c.startswith("lob_bid_price_")])
        bid_sz_cols = sorted([c for c in df.columns if c.startswith("lob_bid_size_")])
        ask_px_cols = sorted([c for c in df.columns if c.startswith("lob_ask_price_")])
        ask_sz_cols = sorted([c for c in df.columns if c.startswith("lob_ask_size_")])

        if bid_px_cols and bid_sz_cols:
            bid_px = df[bid_px_cols].to_numpy(dtype=np.float32)
            bid_sz = df[bid_sz_cols].to_numpy(dtype=np.float32)
            mu = np.nanmean(bid_sz, axis=1, keepdims=True)
            sig = np.nanstd(bid_sz, axis=1, keepdims=True)
            mask = bid_sz > mu + 2 * sig
            idx = np.argmax(np.where(mask, bid_sz, -np.inf), axis=1)
            mid = df["close"].to_numpy(dtype=np.float32)
            dist = np.where(
                mask[np.arange(len(df)), idx],
                ((mid - bid_px[np.arange(len(df)), idx]) / mid) * 1e4,
                fill_value,
            )
            rel = np.where(
                mask[np.arange(len(df)), idx],
                bid_sz[np.arange(len(df)), idx] / np.nansum(bid_sz, axis=1),
                fill_value,
            )
            df["wall_bid_dist_bps"] = dist.astype(np.float32)
            df["wall_bid_size_rel"] = rel.astype(np.float32)
        else:
            df["wall_bid_dist_bps"] = fill_value
            df["wall_bid_size_rel"] = fill_value

        if ask_px_cols and ask_sz_cols:
            ask_px = df[ask_px_cols].to_numpy(dtype=np.float32)
            ask_sz = df[ask_sz_cols].to_numpy(dtype=np.float32)
            mu = np.nanmean(ask_sz, axis=1, keepdims=True)
            sig = np.nanstd(ask_sz, axis=1, keepdims=True)
            mask = ask_sz > mu + 2 * sig
            idx = np.argmax(np.where(mask, ask_sz, -np.inf), axis=1)
            mid = df["close"].to_numpy(dtype=np.float32)
            dist = np.where(
                mask[np.arange(len(df)), idx],
                ((ask_px[np.arange(len(df)), idx] - mid) / mid) * 1e4,
                fill_value,
            )
            rel = np.where(
                mask[np.arange(len(df)), idx],
                ask_sz[np.arange(len(df)), idx] / np.nansum(ask_sz, axis=1),
                fill_value,
            )
            df["wall_ask_dist_bps"] = dist.astype(np.float32)
            df["wall_ask_size_rel"] = rel.astype(np.float32)
        else:
            df["wall_ask_dist_bps"] = fill_value
            df["wall_ask_size_rel"] = fill_value

        if bid_px_cols and ask_px_cols:
            best_bid = df[bid_px_cols[0]].astype(np.float32)
            best_ask = df[ask_px_cols[0]].astype(np.float32)
            mid = ((best_bid + best_ask) / 2.0).replace(0.0, np.nan)
            df["lob_spread_bps"] = ((best_ask - best_bid) / mid * 1e4).astype(np.float32)
        else:
            df["lob_spread_bps"] = fill_value

        if bid_sz_cols and ask_sz_cols:
            total_bid = df[bid_sz_cols].sum(axis=1).astype(np.float32)
            total_ask = df[ask_sz_cols].sum(axis=1).astype(np.float32)
            depth_denom = (total_bid + total_ask).replace(0.0, np.nan)
            df["lob_depth_imbalance"] = (
                ((total_bid - total_ask) / depth_denom).astype(np.float32)
            )
        else:
            df["lob_depth_imbalance"] = fill_value

        if bid_px_cols and bid_sz_cols:
            bid_px = df[bid_px_cols].to_numpy(dtype=np.float32)
            bid_sz = df[bid_sz_cols].to_numpy(dtype=np.float32)
            total_bid = np.nansum(bid_sz, axis=1)
            weighted_bid = np.divide(
                np.nansum(bid_px * bid_sz, axis=1),
                total_bid,
                out=np.full(len(df), np.nan, dtype=np.float32),
                where=total_bid != 0,
            )
            weighted_bid = pd.Series(weighted_bid, index=df.index)
            best_bid = df[bid_px_cols[0]].astype(np.float32)
            df["lob_bid_slope_bps"] = (
                ((weighted_bid - best_bid) / best_bid.replace(0.0, np.nan)) * 1e4
            ).astype(np.float32)
        else:
            df["lob_bid_slope_bps"] = fill_value

        if ask_px_cols and ask_sz_cols:
            ask_px = df[ask_px_cols].to_numpy(dtype=np.float32)
            ask_sz = df[ask_sz_cols].to_numpy(dtype=np.float32)
            total_ask = np.nansum(ask_sz, axis=1)
            weighted_ask = np.divide(
                np.nansum(ask_px * ask_sz, axis=1),
                total_ask,
                out=np.full(len(df), np.nan, dtype=np.float32),
                where=total_ask != 0,
            )
            weighted_ask = pd.Series(weighted_ask, index=df.index)
            best_ask = df[ask_px_cols[0]].astype(np.float32)
            df["lob_ask_slope_bps"] = (
                ((weighted_ask - best_ask) / best_ask.replace(0.0, np.nan)) * 1e4
            ).astype(np.float32)
        else:
            df["lob_ask_slope_bps"] = fill_value
    else:
        drop_lob = [c for c in df.columns if c.startswith("lob_") or c.startswith("wall_")]
        if drop_lob:
            df = df.drop(columns=drop_lob, errors="ignore")

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
    df["time_hour_sin"] = np.sin(2.0 * np.pi * ts.dt.hour / 24.0).astype(np.float32)
    df["time_hour_cos"] = np.cos(2.0 * np.pi * ts.dt.hour / 24.0).astype(np.float32)
    df["time_dow_sin"] = np.sin(2.0 * np.pi * ts.dt.dayofweek / 7.0).astype(np.float32)
    df["time_dow_cos"] = np.cos(2.0 * np.pi * ts.dt.dayofweek / 7.0).astype(np.float32)

    # --- cross-asset features -------------------------------------------------
    corr_minutes = 240
    corr_window = timeframe_windows.get(corr_minutes, max(1, timeframe_windows.get(60, 1)))
    corr_label = timeframe_labels.get(corr_minutes, _format_minutes_label(corr_minutes))
    for asset, keywords in CROSS_ASSET_KEYWORDS.items():
        col_name = _match_column(df, keywords)
        ret_col = f"cross_{asset}_ret"
        div_col = f"cross_{asset}_divergence"
        corr_col = f"cross_{asset}_corr_{corr_label}"
        if col_name is None:
            df[ret_col] = np.float32(fill_value)
            df[div_col] = np.float32(fill_value)
            df[corr_col] = np.float32(fill_value)
            continue
        series = pd.to_numeric(df[col_name], errors="coerce")
        series = series.astype(np.float32)
        log_series = np.log(series.replace(0.0, np.nan))
        asset_ret = log_series.diff()
        df[ret_col] = asset_ret.astype(np.float32)
        df[div_col] = (ret1 - asset_ret).astype(np.float32)
        if corr_window > 1:
            corr = ret1.rolling(corr_window).corr(asset_ret)
            df[corr_col] = corr.astype(np.float32)
        else:
            df[corr_col] = np.float32(fill_value)

    # --- prefixed helper copies ----------------------------------------------
    # Duplicate selected features using explicit prefixes so that downstream
    # grouping (e.g. Group-SHAP) can rely on simple name patterns.  Keeping the
    # original column names preserves back-compatibility with existing models
    # and tests.
    copy_map = {
        "basis_annualized": "deriv_basis_annualized",
        "oi_delta_15m": "deriv_oi_delta_15m",
        "lob_imbalance_L1": "lob_imbalance_L1",
        "lob_imbalance_L2": "lob_imbalance_L2",
        "ret3": "mom_ret3",
        "volatility_60m": "vol_volatility_60m",
        "atr14": "vol_atr14",
        "tod_sin": "time_tod_sin",
        "tod_cos": "time_tod_cos",
        "is_day": "time_is_day",
        "is_night": "time_is_night",
        "is_weekend": "time_is_weekend",
        "is_weekday": "time_is_weekday",
    }
    if not settings.include_derivatives:
        copy_map.pop("basis_annualized", None)
        copy_map.pop("oi_delta_15m", None)
    if not settings.include_orderbook:
        copy_map.pop("lob_imbalance_L1", None)
        copy_map.pop("lob_imbalance_L2", None)
    for src, dst in copy_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src].astype(np.float32)

    # --- z-scores and deltas for prefixed features ---------------------------
    prefixes = ["mom_", "vol_", "time_", "cross_"]
    if settings.include_onchain:
        prefixes.insert(0, "onch_")
    if settings.include_derivatives:
        prefixes.append("deriv_")
    if settings.include_orderbook:
        prefixes.append("lob_")
    for prefix in prefixes:
        cols = [c for c in df.columns if c.startswith(prefix)]
        for col in cols:
            if f"z_{col}" not in df.columns:
                z = (df[col] - df[col].rolling(36).mean()) / df[col].rolling(36).std()
                df[f"z_{col}"] = z.astype(np.float32)
            if f"d_{col}" not in df.columns:
                df[f"d_{col}"] = df[col].diff().astype(np.float32)

    # --- NaN handling před cíli ----------------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].astype(np.float32)
        df[numeric_cols] = df[numeric_cols].fillna(fill_value)

    # --- validace typů --------------------------------------------------------
    feature_only = df
    onch_cols = [c for c in feature_only.columns if c.startswith("onch_")]
    if feature_only.drop(columns=onch_cols, errors="ignore").isna().any().any():
        raise ValueError("NaN values present after feature engineering")
    obj_cols = feature_only.select_dtypes(include="object").columns
    if len(obj_cols) > 0:
        raise TypeError(f"Object dtypes present after feature merge: {list(obj_cols)}")

    return df


FEATURE_COLUMNS: list[str] = get_feature_columns()


# Feature groups for Group-SHAP -------------------------------------------------
# Mapping of human readable feature groups to regex patterns used to match
# feature names.  The patterns are intentionally simple and favour readability
# over full regex expressiveness.
FEATURE_GROUPS: dict[str, list[str]] = {
    "orderflow": [r"^tbr_", r"^ofi_", r"^cvd_", r"^taker_", r"rel_close_vwap", r"z_volume"],
    "derivatives": [r"^oi_", r"^basis_", r"^funding_", r"^deriv_"],
    "lob": [r"^lob_", r"^queue_", r"book_slope", r"^wall_"],
    "momentum_vol": [
        r"ret",
        r"^volatility_",
        r"^atr_",
        r"skew",
        r"kurt",
    ],
    "time": [r"^tod_", r"^session_", r"funding_cycle", r"^dow_", r"^time_"],
    "onchain": [r"^onch_"],
    "cross_asset": [r"^cross_"],
    # exact matches to avoid mapping e.g. "open_interest" to this group
    "price_ref": [r"^open$", r"^high$", r"^low$", r"^close$", r"^mid$"],
}


def assign_feature_groups(columns: list[str]) -> dict[str, str]:
    """Assign each feature name in *columns* to a predefined group.

    Parameters
    ----------
    columns:
        Iterable of feature names.

    Returns
    -------
    dict[str, str]
        Mapping from feature name to its group. Features that do not match any
        pattern are assigned to the ``"other"`` group.
    """

    groups: dict[str, str] = {}
    for col in columns:
        assigned = False
        for group, patterns in FEATURE_GROUPS.items():
            if any(re.search(pat, col) for pat in patterns):
                groups[col] = group
                assigned = True
                break
        if not assigned:
            groups[col] = "other"
    return groups


# -- Targets -----------------------------------------------------------------
from crypto_analyzer.labeling.targets import make_targets as _make_targets  # noqa: E402


def make_targets(df: pd.DataFrame, horizon: int = 120) -> pd.DataFrame:
    """Convenience wrapper around :func:`crypto_analyzer.labeling.targets.make_targets`.

    Generates classification targets for the given ``horizon`` (in minutes).
    By default a 120 minute horizon is used, matching the feature engineering
    assumptions in this module.
    """

    return _make_targets(df, horizons_min=[horizon])
