"""Utilities for fetching and aligning raw market data feeds.

This module expands the data ingestion pipeline with helpers that download
on-chain metrics and derivative market statistics.  The functions operate on
plain :class:`pandas.DataFrame` objects so they can be re-used by notebooks,
tests, or automated jobs without depending on CLI entrypoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, TYPE_CHECKING

import pandas as pd
import requests

from crypto_analyzer.data.db_connector import get_price_data
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
BINANCE_OPEN_INTEREST_ENDPOINT = "https://fapi.binance.com/futures/data/openInterestHist"


@dataclass(frozen=True)
class _HTTPSettings:
    """Lightweight container for configurable HTTP parameters."""

    timeout: int = 10
    limit: int = 1000


def _as_timestamp(value: datetime | pd.Timestamp | str) -> pd.Timestamp:
    """Return ``value`` as a timezone-aware ``Timestamp`` in UTC."""

    ts = pd.Timestamp(value, tz="UTC")
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

    start_time = int(params.get("startTime", 0))
    end_time = int(params.get("endTime", 0)) or None

    while True:
        resp = sess.get(url, params=params, timeout=10)
        resp.raise_for_status()
        chunk = resp.json()
        if not chunk:
            break
        records.extend(chunk)

        last_time = max(int(entry.get("time") or entry.get("fundingTime", 0)) for entry in chunk)
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


def fetch_binance_open_interest(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    period: str = "1d",
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

    start_ms = int(_as_timestamp(start).timestamp() * 1000) if start else None
    end_ms = int(_as_timestamp(end).timestamp() * 1000) if end else None

    base = get_price_data(market_symbol, start_ts=start_ms, end_ts=end_ms, db_path=cfg.db_path)
    if base.empty:
        logger.warning("Base OHLCV dataset is empty; returning without enrichment", extra={"symbol": market_symbol})
        return base

    timestamps = pd.to_datetime(base["timestamp"], utc=True)
    span_start = timestamps.min().floor("D")
    span_end = timestamps.max().ceil("D")

    try:
        glassnode = fetch_glassnode_active_addresses(span_start, span_end)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to fetch Glassnode active addresses", exc_info=exc)
        glassnode = pd.DataFrame(columns=["timestamp", "onch_active_addresses"])

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
            open_interest.assign(date=lambda df: pd.to_datetime(df["timestamp"], utc=True).dt.floor("D"))
            .groupby("date", as_index=False)["open_interest"]
            .last()
        )
    if not glassnode.empty:
        glassnode = glassnode.assign(date=lambda df: pd.to_datetime(df["timestamp"], utc=True).dt.floor("D"))

    enriched = _merge_daily_features(base, (glassnode, funding, open_interest))
    return enriched


__all__ = [
    "fetch_glassnode_active_addresses",
    "fetch_binance_funding_rates",
    "fetch_binance_open_interest",
    "load_enriched_market_data",
]

