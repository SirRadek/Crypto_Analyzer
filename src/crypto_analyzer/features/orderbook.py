"""Lightweight utilities for deriving order book statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["depth_imbalance", "spread", "order_flow_imbalance"]


def depth_imbalance(orderbook: pd.DataFrame) -> pd.Series:
    """Compute depth imbalance from aggregated bid/ask sizes."""

    if orderbook is None or len(orderbook) == 0:
        return pd.Series(dtype=float)

    bid_cols = [col for col in orderbook.columns if col.lower().startswith("bid_size")]
    ask_cols = [col for col in orderbook.columns if col.lower().startswith("ask_size")]
    if not bid_cols or not ask_cols:
        return pd.Series(np.nan, index=orderbook.index)

    bid_sizes = orderbook[bid_cols].sum(axis=1)
    ask_sizes = orderbook[ask_cols].sum(axis=1)
    denom = (bid_sizes + ask_sizes).replace(0, np.nan)
    imbalance = (bid_sizes - ask_sizes) / denom
    return imbalance.astype(float)


def spread(orderbook: pd.DataFrame) -> pd.Series:
    """Return bid/ask spread in basis points when available."""

    if orderbook is None or len(orderbook) == 0:
        return pd.Series(dtype=float)

    bid_cols = [col for col in orderbook.columns if col.lower().startswith("bid_price")]
    ask_cols = [col for col in orderbook.columns if col.lower().startswith("ask_price")]
    if not bid_cols or not ask_cols:
        return pd.Series(np.nan, index=orderbook.index)

    best_bid = orderbook[bid_cols].max(axis=1)
    best_ask = orderbook[ask_cols].min(axis=1)
    mid = (best_bid + best_ask) / 2.0
    spread_bps = ((best_ask - best_bid) / mid).replace([np.inf, -np.inf], np.nan) * 10_000.0
    return spread_bps.astype(float)


def order_flow_imbalance(events: pd.DataFrame) -> pd.Series:
    """Compute order flow imbalance from trade events."""

    if events is None or len(events) == 0:
        return pd.Series(dtype=float)

    required = {"side", "size"}
    if not required.issubset(events.columns):
        return pd.Series(np.nan, index=(events["timestamp"] if "timestamp" in events.columns else [0]))

    frame = events.copy()
    frame["size"] = frame["size"].astype(float)
    side = frame["side"].astype(str).str.lower()
    buys = frame.loc[side.isin({"buy", "bid", "b"}), "size"]
    sells = frame.loc[side.isin({"sell", "ask", "s"}), "size"]
    buy_sum = buys.groupby(frame.get("timestamp", pd.Series(index=frame.index))).sum()
    sell_sum = sells.groupby(frame.get("timestamp", pd.Series(index=frame.index))).sum()

    if "timestamp" in frame.columns:
        index = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        if index.isna().all():
            index = frame.index
        else:
            index = index.sort_values().unique()
    else:
        index = frame.index.unique()

    aligned_buys = buy_sum.reindex(index, fill_value=0.0)
    aligned_sells = sell_sum.reindex(index, fill_value=0.0)
    denom = (aligned_buys + aligned_sells).replace(0, np.nan)
    imbalance = (aligned_buys - aligned_sells) / denom
    return imbalance.astype(float)
