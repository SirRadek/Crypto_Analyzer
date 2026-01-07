"""Feature helpers for computing basic order book statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["depth_imbalance", "spread", "order_flow_imbalance"]


_BUY_SIDES: frozenset[str] = frozenset({"buy", "bid", "b"})
_SELL_SIDES: frozenset[str] = frozenset({"sell", "ask", "s"})


def _nan_series_like(frame: pd.DataFrame | None) -> pd.Series:
    """Return a ``Series`` aligned to ``frame`` filled with ``NaN`` values."""

    if frame is None:
        return pd.Series([np.nan], dtype=float)

    if frame.empty:
        return pd.Series(dtype=float)

    index: pd.Index = frame.index
    if "timestamp" in frame.columns:
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if ts.notna().any():
            index = pd.Index(ts)

    return pd.Series(np.nan, index=index, dtype=float)


def _sum_with_nan(values: pd.DataFrame | pd.Series) -> pd.Series:
    """Column-wise sum that preserves ``NaN`` when all entries are missing."""

    if isinstance(values, pd.DataFrame):
        return values.sum(axis=1, min_count=1)
    return values.sum(min_count=1)


def depth_imbalance(orderbook: pd.DataFrame | None) -> pd.Series:
    """Compute bid/ask depth imbalance.

    The metric is defined as ``(bid_depth - ask_depth) / (bid_depth + ask_depth)``.
    Missing inputs degrade gracefully by returning ``NaN`` for the affected rows.
    """

    if orderbook is None or orderbook.empty:
        return pd.Series(dtype=float)

    bid_cols = [col for col in orderbook.columns if col.lower().startswith("bid_size")]
    ask_cols = [col for col in orderbook.columns if col.lower().startswith("ask_size")]
    if not bid_cols or not ask_cols:
        return _nan_series_like(orderbook)

    bid_sizes = _sum_with_nan(orderbook[bid_cols])
    ask_sizes = _sum_with_nan(orderbook[ask_cols])

    denom = (bid_sizes + ask_sizes).replace(0, np.nan)
    imbalance = (bid_sizes - ask_sizes) / denom
    return imbalance.astype(float)


def spread(orderbook: pd.DataFrame | None) -> pd.Series:
    """Return the quoted spread in basis points."""

    if orderbook is None or orderbook.empty:
        return pd.Series(dtype=float)

    bid_cols = [col for col in orderbook.columns if col.lower().startswith("bid_price")]
    ask_cols = [col for col in orderbook.columns if col.lower().startswith("ask_price")]
    if not bid_cols or not ask_cols:
        return _nan_series_like(orderbook)

    best_bid = orderbook[bid_cols].max(axis=1, skipna=True)
    best_ask = orderbook[ask_cols].min(axis=1, skipna=True)

    mid = (best_bid + best_ask) / 2.0
    spread_bps = ((best_ask - best_bid) / mid).replace([np.inf, -np.inf], np.nan) * 10_000.0
    spread_bps = spread_bps.where(~(best_bid.isna() | best_ask.isna()))
    return spread_bps.astype(float)


def _prepare_event_index(events: pd.DataFrame) -> pd.Index:
    index: pd.Index = events.index
    if "timestamp" in events.columns:
        ts = pd.to_datetime(events["timestamp"], utc=True, errors="coerce")
        if ts.notna().any():
            index = pd.Index(ts)
    return index


def _aggregate_sizes(frame: pd.DataFrame, mask: pd.Series, key: pd.Series | pd.Index) -> pd.Series:
    if not mask.any():
        return pd.Series(dtype=float)

    keyed = key[mask]
    if isinstance(keyed, pd.Index):
        keyed = pd.Series(keyed, index=frame.index[mask])

    sizes = frame.loc[mask, "size"]
    if isinstance(keyed, pd.Series):
        keyed = keyed.dropna()
        sizes = sizes.loc[keyed.index]

    if keyed.empty:
        return pd.Series(dtype=float)

    grouped = sizes.groupby(keyed)
    return grouped.apply(lambda s: s.sum(min_count=1)).astype(float)


def order_flow_imbalance(events: pd.DataFrame | None) -> pd.Series:
    """Compute order flow imbalance grouped by timestamp.

    ``events`` is expected to contain ``timestamp``, ``side`` and ``size`` columns. When
    one of these inputs is missing the function returns ``NaN`` values instead of
    raising.
    """

    if events is None or events.empty:
        return pd.Series(dtype=float)

    required = {"side", "size"}
    if not required.issubset(events.columns):
        return _nan_series_like(events)

    frame = events.copy()
    frame["size"] = pd.to_numeric(frame["size"], errors="coerce")

    index = _prepare_event_index(frame)
    key = pd.Series(index, index=frame.index)
    valid_key = key.dropna()
    if valid_key.empty:
        return _nan_series_like(frame)

    side = frame["side"].astype(str).str.lower()
    buy_mask = side.isin(_BUY_SIDES)
    sell_mask = side.isin(_SELL_SIDES)

    buy_sum = _aggregate_sizes(frame, buy_mask, key)
    sell_sum = _aggregate_sizes(frame, sell_mask, key)

    timeline = pd.Index(pd.unique(valid_key))
    aligned_buys = buy_sum.reindex(timeline, fill_value=0.0)
    aligned_sells = sell_sum.reindex(timeline, fill_value=0.0)

    denom = (aligned_buys + aligned_sells).replace(0, np.nan)
    imbalance = (aligned_buys - aligned_sells) / denom
    imbalance = imbalance.astype(float)
    imbalance.name = None
    return imbalance
