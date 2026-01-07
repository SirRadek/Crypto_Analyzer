"""Utilities for fetching and aligning raw market data feeds.

This module expands the data ingestion pipeline with helpers that download
on-chain metrics and derivative market statistics.  The functions operate on
plain :class:`pandas.DataFrame` objects so they can be re-used by notebooks,
tests, or automated jobs without depending on CLI entrypoints.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
import requests

from crypto_analyzer.data.db_connector import get_price_data
from crypto_analyzer.data.onchain_fetcher import (
    fetch_exchange_flows,
    fetch_mempool_stats,
)
from crypto_analyzer.utils.logging import get_logger

try:  # pragma: no cover - exercised indirectly in environments with config deps
    from crypto_analyzer.utils.config import CONFIG, AppConfig
except ModuleNotFoundError:  # pragma: no cover - fallback when optional deps absent
    CONFIG = None  # type: ignore[assignment]
    if TYPE_CHECKING:  # pragma: no cover - type hint preservation
        from crypto_analyzer.utils.config import AppConfig
    else:
        AppConfig = Any  # type: ignore[misc,assignment]

logger = get_logger(__name__)

GLASSNODE_ENDPOINT = "https://api.glassnode.com/v1/metrics/addresses/active_count"
BINANCE_FUNDING_ENDPOINT = "https://fapi.binance.com/fapi/v1/fundingRate"
BINANCE_ORDERBOOK_ENDPOINT = "https://fapi.binance.com/fapi/v1/depth"
BINANCE_SPOT_ORDERBOOK_ENDPOINT = "https://api.binance.com/api/v3/depth"
BINANCE_OPEN_INTEREST_ENDPOINT = "https://fapi.binance.com/futures/data/openInterestHist"
BINANCE_PREMIUM_INDEX_ENDPOINT = "https://fapi.binance.com/fapi/v1/premiumIndex"
BINANCE_SPOT_TICKER_ENDPOINT = "https://api.binance.com/api/v3/ticker/price"


@dataclass(frozen=True)
class _HTTPSettings:
    """Lightweight container for configurable HTTP parameters."""

    timeout: int = 10
    limit: int = 1000


def _as_timestamp(value: datetime | pd.Timestamp | str) -> pd.Timestamp:
    """Return ``value`` as a timezone-aware ``Timestamp`` in UTC."""

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def fetch_glassnode_active_addresses(
    start: datetime,
    end: datetime,
    *,
    api_key: str | None = None,
    asset: str = "BTC",
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch daily active address counts for *asset* from Glassnode.

    Parameters
    ----------
    start, end:
        Datetime bounds for the requested interval.  The Glassnode endpoint
        expects second resolution Unix timestamps so values are normalised to
        UTC internally.  The end of the interval is inclusive.
    api_key:
        API key used to authenticate against the Glassnode REST endpoint.  When
        omitted the key is read from :data:`CONFIG`.
    asset:
        Ticker symbol understood by Glassnode.  ``BTC`` is the default.
    session:
        Optional :class:`requests.Session` for advanced customisation or reuse.

    Returns
    -------
    pandas.DataFrame
        Frame with columns ``timestamp`` (UTC) and ``onch_active_addresses``.
    """

    cfg_key = None
    if CONFIG is not None:
        onchain_cfg = getattr(CONFIG, "onchain", None)
        cfg_key = getattr(onchain_cfg, "glassnode_api_key", None)
    key = api_key or cfg_key
    if not key:
        raise ValueError("Glassnode API key must be provided via config or argument")

    start_ts = _as_timestamp(start)
    end_ts = _as_timestamp(end)
    params = {
        "a": asset,
        "s": int(start_ts.timestamp()),
        "u": int(end_ts.timestamp()),
        "i": "24h",
        "api_key": key,
    }

    sess = session or requests.Session()
    response = sess.get(GLASSNODE_ENDPOINT, params=params, timeout=10)
    response.raise_for_status()
    payload = response.json()

    frame = pd.DataFrame(payload or [])
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "onch_active_addresses"])

    if "t" not in frame.columns or "v" not in frame.columns:
        raise ValueError("Unexpected response schema from Glassnode active addresses endpoint")

    frame["timestamp"] = pd.to_datetime(frame["t"], unit="s", utc=True)
    frame["onch_active_addresses"] = pd.to_numeric(frame["v"], errors="coerce")
    frame = frame.sort_values("timestamp")
    frame = frame.loc[:, ["timestamp", "onch_active_addresses"]]
    return frame.reset_index(drop=True)


def _binance_paginated_get(
    url: str,
    *,
    params: dict[str, object],
    limit: int,
    session: requests.Session | None = None,
) -> list[dict[str, object]]:
    """Fetch paginated Binance data between ``startTime`` and ``endTime``."""

    sess = session or requests.Session()
    records: list[dict[str, object]] = []

    end_time = int(params.get("endTime", 0)) or None

    while True:
        resp = sess.get(url, params=params, timeout=10)
        resp.raise_for_status()
        chunk = resp.json()
        if not chunk:
            break
        records.extend(chunk)

        last_time = max(
            int(
                entry.get("time")
                or entry.get("fundingTime")
                or entry.get("timestamp")
                or entry.get("closeTime")
                or 0
            )
            for entry in chunk
        )
        if end_time and last_time >= end_time:
            break
        if len(chunk) < limit:
            break
        params["startTime"] = last_time + 1

    return records


def fetch_binance_funding_rates(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    session: requests.Session | None = None,
    settings: _HTTPSettings | None = None,
) -> pd.DataFrame:
    """Fetch perpetual futures funding rates for ``symbol`` from Binance."""

    cfg = settings or _HTTPSettings()
    start_ts = _as_timestamp(start)
    end_ts = _as_timestamp(end)

    params = {
        "symbol": symbol,
        "startTime": int(start_ts.timestamp() * 1000),
        "endTime": int(end_ts.timestamp() * 1000),
        "limit": cfg.limit,
    }

    records = _binance_paginated_get(
        BINANCE_FUNDING_ENDPOINT,
        params=params,
        limit=cfg.limit,
        session=session,
    )

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    frame["timestamp"] = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True)
    frame["funding_rate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
    frame = frame.sort_values("timestamp")
    return frame.loc[:, ["timestamp", "funding_rate"]].reset_index(drop=True)


def fetch_binance_order_book(
    symbol: str,
    *,
    depth: int = 20,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch a futures order book snapshot for ``symbol`` from Binance.

    Parameters
    ----------
    symbol:
        Trading pair in Binance notation, e.g. ``"BTCUSDT"``.
    depth:
        Number of price levels to request from each side of the book.  Binance
        accepts a discrete set of values; the closest supported depth greater
        than or equal to the requested value is used.
    session:
        Optional :class:`requests.Session` to reuse across invocations.

    Returns
    -------
    pandas.DataFrame
        Frame with a single row describing top-of-book statistics.  When the
        exchange returns an empty book the function yields an empty frame with
        the expected columns instead of raising an exception.
    """

    allowed_depths = (5, 10, 20, 50, 100, 500, 1000)
    if depth <= 0:
        raise ValueError("Depth must be a positive integer")
    limit = next((value for value in allowed_depths if value >= depth), allowed_depths[-1])

    sess = session or requests.Session()
    params = {"symbol": symbol, "limit": limit}

    def _request(endpoint: str) -> dict[str, object]:
        response = sess.get(endpoint, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json() or {}
        if isinstance(payload, dict) and payload.get("code") is not None:
            raise ValueError(str(payload))
        if not isinstance(payload, dict):
            raise ValueError("Unexpected orderbook payload")
        return payload

    try:
        payload = _request(BINANCE_ORDERBOOK_ENDPOINT)
    except Exception:
        payload = _request(BINANCE_SPOT_ORDERBOOK_ENDPOINT)

    bids = payload.get("bids") or []
    asks = payload.get("asks") or []

    def _parse_side(
        levels: list[list[object]] | list[tuple[object, object]],
    ) -> list[tuple[float, float]]:
        parsed: list[tuple[float, float]] = []
        for level in levels[:limit]:
            if not isinstance(level, (list, tuple)) or len(level) < 2:
                continue
            price_raw, qty_raw = level[0], level[1]
            try:
                price = float(price_raw)
                quantity = float(qty_raw)
            except (TypeError, ValueError):
                continue
            parsed.append((price, quantity))
        return parsed

    parsed_bids = _parse_side(bids)
    parsed_asks = _parse_side(asks)

    columns = [
        "timestamp",
        "bid_price",
        "bid_volume",
        "ask_price",
        "ask_volume",
        "spread",
        "mid_price",
        "bid_volume_total",
        "ask_volume_total",
        "bid_notional_total",
        "ask_notional_total",
        "depth_imbalance",
    ]

    if not parsed_bids or not parsed_asks:
        return pd.DataFrame(columns=columns)

    best_bid_price, best_bid_volume = parsed_bids[0]
    best_ask_price, best_ask_volume = parsed_asks[0]

    bid_volume_total = sum(qty for _, qty in parsed_bids)
    ask_volume_total = sum(qty for _, qty in parsed_asks)
    bid_notional_total = sum(price * qty for price, qty in parsed_bids)
    ask_notional_total = sum(price * qty for price, qty in parsed_asks)

    spread = best_ask_price - best_bid_price
    mid_price = (best_bid_price + best_ask_price) / 2

    depth_sum = bid_volume_total + ask_volume_total
    depth_imbalance = (
        (bid_volume_total - ask_volume_total) / depth_sum if depth_sum else float("nan")
    )

    timestamp = pd.Timestamp.utcnow()
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    frame = pd.DataFrame(
        {
            "timestamp": [timestamp],
            "bid_price": [best_bid_price],
            "bid_volume": [best_bid_volume],
            "ask_price": [best_ask_price],
            "ask_volume": [best_ask_volume],
            "spread": [spread],
            "mid_price": [mid_price],
            "bid_volume_total": [bid_volume_total],
            "ask_volume_total": [ask_volume_total],
            "bid_notional_total": [bid_notional_total],
            "ask_notional_total": [ask_notional_total],
            "depth_imbalance": [depth_imbalance],
        }
    )
    return frame


def fetch_binance_open_interest(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    period: str = "5m",
    session: requests.Session | None = None,
    settings: _HTTPSettings | None = None,
) -> pd.DataFrame:
    """Fetch futures open interest history for ``symbol`` from Binance."""

    cfg = settings or _HTTPSettings(limit=500)
    start_ts = _as_timestamp(start)
    end_ts = _as_timestamp(end)

    params = {
        "symbol": symbol,
        "period": period,
        "startTime": int(start_ts.timestamp() * 1000),
        "endTime": int(end_ts.timestamp() * 1000),
        "limit": cfg.limit,
    }

    records = _binance_paginated_get(
        BINANCE_OPEN_INTEREST_ENDPOINT,
        params=params,
        limit=cfg.limit,
        session=session,
    )

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "open_interest"])

    if "sumOpenInterest" not in frame.columns or "timestamp" not in frame.columns:
        raise ValueError("Unexpected schema returned by Binance open interest endpoint")

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame["open_interest"] = pd.to_numeric(frame["sumOpenInterest"], errors="coerce")
    frame = frame.sort_values("timestamp")
    return frame.loc[:, ["timestamp", "open_interest"]].reset_index(drop=True)


def fetch_binance_basis(
    symbol: str,
    *,
    session: requests.Session | None = None,
    settings: _HTTPSettings | None = None,
) -> pd.DataFrame:
    """Compute the futures basis in basis points for ``symbol``.

    The function requests the perpetual futures mark price from Binance's
    ``/premiumIndex`` endpoint and compares it with the spot ticker price from
    the main exchange REST API.  The resulting basis is expressed in basis
    points and aligned to the timestamp reported by the futures endpoint.

    Parameters
    ----------
    symbol:
        Trading pair identifier understood by Binance, e.g. ``"BTCUSDT"``.
    session:
        Optional :class:`requests.Session` reused across invocations.
    settings:
        Optional :class:`_HTTPSettings` overriding timeout defaults.

    Returns
    -------
    pandas.DataFrame
        Frame with columns ``timestamp``, ``basis`` (basis points),
        ``basis_bp``, ``futures_price`` and ``spot_price``.  When the exchange
        returns incomplete data an empty frame is produced.
    """

    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be provided when fetching Binance basis")

    cfg = settings or _HTTPSettings()
    sess = session or requests.Session()

    params = {"symbol": symbol}
    response = sess.get(BINANCE_PREMIUM_INDEX_ENDPOINT, params=params, timeout=cfg.timeout)
    response.raise_for_status()
    payload = response.json() or {}

    mark_raw = payload.get("markPrice")
    try:
        mark_price = float(mark_raw)
    except (TypeError, ValueError):
        mark_price = float("nan")

    time_raw = payload.get("time") or payload.get("closeTime") or payload.get("updateTime")
    if time_raw is not None:
        try:
            timestamp = pd.to_datetime(int(time_raw), unit="ms", utc=True)
        except (TypeError, ValueError):
            timestamp = pd.Timestamp.now(tz="UTC")
    else:
        timestamp = pd.Timestamp.now(tz="UTC")

    spot_response = sess.get(BINANCE_SPOT_TICKER_ENDPOINT, params=params, timeout=cfg.timeout)
    spot_response.raise_for_status()
    spot_payload = spot_response.json() or {}

    spot_raw = spot_payload.get("price")
    try:
        spot_price = float(spot_raw)
    except (TypeError, ValueError):
        spot_price = float("nan")

    if not pd.notna(mark_price) or not pd.notna(spot_price) or spot_price == 0:
        logger.warning(
            "Unable to compute Binance basis due to missing price data",
            extra={"symbol": symbol, "mark_price": mark_raw, "spot_price": spot_raw},
        )
        return pd.DataFrame(
            columns=["timestamp", "basis", "basis_bp", "futures_price", "spot_price"]
        )

    basis_bp = (mark_price - spot_price) / spot_price * 10_000.0

    frame = pd.DataFrame(
        {
            "timestamp": [timestamp],
            "basis": [basis_bp],
            "basis_bp": [basis_bp],
            "futures_price": [mark_price],
            "spot_price": [spot_price],
        }
    )
    return frame


def _merge_daily_features(base: pd.DataFrame, daily_frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Merge daily features into the base OHLCV frame."""

    merged = base.copy()
    merged["date"] = pd.to_datetime(merged["timestamp"], utc=True).dt.floor("D")

    for frame in daily_frames:
        if frame is None or frame.empty:
            continue
        enriched = frame.copy()
        if "date" not in enriched.columns:
            enriched["date"] = pd.to_datetime(enriched["timestamp"], utc=True).dt.floor("D")
        enriched = enriched.drop(columns=["timestamp"], errors="ignore")
        enriched = enriched.drop_duplicates(subset="date")
        merged = merged.merge(enriched, on="date", how="left")

    return merged.drop(columns=["date"])


def load_enriched_market_data(
    *,
    symbol: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    config: AppConfig | None = None,
) -> pd.DataFrame:
    """Load OHLCV data augmented with on-chain and derivative metrics."""

    cfg = config
    if cfg is None:
        if CONFIG is None:
            raise RuntimeError("Application configuration is unavailable; pass config explicitly")
        cfg = CONFIG
    market_symbol = symbol or cfg.symbol
    onchain_cfg = getattr(cfg, "onchain", None)

    start_ms = int(_as_timestamp(start).timestamp() * 1000) if start else None
    end_ms = int(_as_timestamp(end).timestamp() * 1000) if end else None

    base = get_price_data(market_symbol, start_ts=start_ms, end_ts=end_ms, db_path=cfg.db_path)
    if base.empty:
        logger.warning(
            "Base OHLCV dataset is empty; returning without enrichment",
            extra={"symbol": market_symbol},
        )
        return base

    timestamps = pd.to_datetime(base["timestamp"], utc=True)
    span_start = timestamps.min().floor("D")
    span_end = timestamps.max().ceil("D")

    try:
        glassnode = fetch_glassnode_active_addresses(span_start, span_end)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to fetch Glassnode active addresses", exc_info=exc)
        glassnode = pd.DataFrame(columns=["timestamp", "onch_active_addresses"])

    mempool = pd.DataFrame()
    if getattr(onchain_cfg, "use_mempool", False):
        try:
            mempool = fetch_mempool_stats()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to fetch mempool statistics", exc_info=exc)
            mempool = pd.DataFrame()

    exchange_flows = pd.DataFrame()
    if getattr(onchain_cfg, "use_exchange_flows", False):
        api_key = getattr(onchain_cfg, "glassnode_api_key", None)
        if api_key:
            try:
                exchange_flows = fetch_exchange_flows(api_key, start=span_start, end=span_end)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to fetch Glassnode exchange flows", exc_info=exc)
                exchange_flows = pd.DataFrame()
        else:  # pragma: no cover - configuration guard
            logger.warning("Exchange flow fetching enabled but Glassnode API key is missing")

    try:
        funding = fetch_binance_funding_rates(market_symbol, span_start, span_end)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to fetch Binance funding rates", exc_info=exc)
        funding = pd.DataFrame(columns=["timestamp", "funding_rate"])

    try:
        open_interest = fetch_binance_open_interest(market_symbol, span_start, span_end)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to fetch Binance open interest", exc_info=exc)
        open_interest = pd.DataFrame(columns=["timestamp", "open_interest"])

    # Aggregate higher frequency series to a daily grid before merging.
    if not funding.empty:
        funding = (
            funding.assign(date=lambda df: pd.to_datetime(df["timestamp"], utc=True).dt.floor("D"))
            .groupby("date", as_index=False)["funding_rate"]
            .mean()
        )
    if not open_interest.empty:
        open_interest = (
            open_interest.assign(
                date=lambda df: pd.to_datetime(df["timestamp"], utc=True).dt.floor("D")
            )
            .groupby("date", as_index=False)["open_interest"]
            .last()
        )
    if not glassnode.empty:
        glassnode = glassnode.assign(
            date=lambda df: pd.to_datetime(df["timestamp"], utc=True).dt.floor("D")
        )

    if not mempool.empty:
        mempool = (
            mempool.reset_index()
            .assign(
                timestamp=lambda df: pd.to_datetime(df["timestamp"], utc=True),
                date=lambda df: pd.to_datetime(df["timestamp"], utc=True).dt.floor("D"),
            )
            .sort_values("timestamp")
            .drop_duplicates(subset="date", keep="last")
        )

    if not exchange_flows.empty:
        exchange_flows = (
            exchange_flows.reset_index()
            .assign(
                timestamp=lambda df: pd.to_datetime(df["timestamp"], utc=True),
                date=lambda df: pd.to_datetime(df["timestamp"], utc=True).dt.floor("D"),
            )
            .sort_values("timestamp")
            .drop_duplicates(subset="date", keep="last")
        )

    enriched = _merge_daily_features(
        base,
        (
            glassnode,
            funding,
            open_interest,
            mempool,
            exchange_flows,
        ),
    )
    return enriched


__all__ = [
    "fetch_glassnode_active_addresses",
    "fetch_binance_funding_rates",
    "fetch_binance_order_book",
    "fetch_binance_open_interest",
    "fetch_binance_basis",
    "fetch_exchange_flows",
    "fetch_mempool_stats",
    "load_enriched_market_data",
]
