"""Helpers for fetching on-chain metrics from external APIs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import requests

from crypto_analyzer.utils.logging import get_logger

logger = get_logger(__name__)

MEMPOOL_STATS_ENDPOINT = "https://mempool.space/api/mempool"
MEMPOOL_FEES_ENDPOINT = "https://mempool.space/api/v1/fees/recommended"
GLASSNODE_EXCHANGE_INFLOW_ENDPOINT = "https://api.glassnode.com/v1/metrics/exchanges/inflow_sum"
GLASSNODE_EXCHANGE_OUTFLOW_ENDPOINT = "https://api.glassnode.com/v1/metrics/exchanges/outflow_sum"
COINMETRICS_ASSET_METRICS_ENDPOINT = (
    "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
)

_COINMETRICS_COLUMN_MAP = {
    "ExchgNetFlow": "onch_exchange_net_flow",
    "ExchgInflowVolume": "onch_exchange_inflow",
    "ExchgOutflowVolume": "onch_exchange_outflow",
}


def _empty_timestamp_frame(columns: list[str]) -> pd.DataFrame:
    """Return an empty frame with a timezone aware ``DatetimeIndex``."""

    index = pd.DatetimeIndex([], name="timestamp", tz="UTC")
    return pd.DataFrame(columns=columns, index=index)


def _ensure_utc_timestamp(value: Any) -> pd.Timestamp:
    """Normalise ``value`` into a timezone-aware UTC ``Timestamp``."""

    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def fetch_mempool_stats(*, session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch latest mempool statistics from mempool.space.

    The returned frame contains a timezone-aware UTC ``DatetimeIndex`` with the
    retrieval timestamp and columns describing the mempool count, virtual size
    and fee estimates.  Column names are prefixed with ``onch_mempool_`` so they
    integrate seamlessly with the feature engineering pipeline.
    """

    sess = session or requests.Session()

    try:
        stats_resp = sess.get(MEMPOOL_STATS_ENDPOINT, timeout=10)
        stats_resp.raise_for_status()
        stats_payload: Mapping[str, Any] = stats_resp.json() or {}

        fees_resp = sess.get(MEMPOOL_FEES_ENDPOINT, timeout=10)
        fees_resp.raise_for_status()
        fees_payload: Mapping[str, Any] = fees_resp.json() or {}
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Failed to fetch mempool statistics", exc_info=exc)
        return _empty_timestamp_frame(
            [
                "onch_mempool_count",
                "onch_mempool_vsize",
                "onch_mempool_total_fee",
                "onch_mempool_fee_fastest",
                "onch_mempool_fee_half_hour",
                "onch_mempool_fee_hour",
                "onch_mempool_fee_economy",
                "onch_mempool_fee_minimum",
            ]
        )

    timestamp = pd.Timestamp.utcnow().tz_localize("UTC")

    frame = pd.DataFrame(
        {
            "onch_mempool_count": [stats_payload.get("count")],
            "onch_mempool_vsize": [stats_payload.get("vsize")],
            "onch_mempool_total_fee": [stats_payload.get("total_fee")],
            "onch_mempool_fee_fastest": [fees_payload.get("fastestFee")],
            "onch_mempool_fee_half_hour": [fees_payload.get("halfHourFee")],
            "onch_mempool_fee_hour": [fees_payload.get("hourFee")],
            "onch_mempool_fee_economy": [fees_payload.get("economyFee")],
            "onch_mempool_fee_minimum": [fees_payload.get("minimumFee")],
        },
        index=pd.DatetimeIndex([timestamp], name="timestamp"),
    )

    numeric_cols = [
        "onch_mempool_count",
        "onch_mempool_vsize",
        "onch_mempool_total_fee",
        "onch_mempool_fee_fastest",
        "onch_mempool_fee_half_hour",
        "onch_mempool_fee_hour",
        "onch_mempool_fee_economy",
        "onch_mempool_fee_minimum",
    ]
    frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")

    return frame


def fetch_exchange_flows(
    api_key: str,
    *,
    asset: str = "BTC",
    session: requests.Session | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch exchange inflow/outflow metrics from Glassnode.

    Parameters
    ----------
    api_key:
        Glassnode API key used for authentication.
    asset:
        Asset ticker understood by Glassnode.  ``BTC`` by default.
    session:
        Optional ``requests.Session`` reused for multiple HTTP calls.
    start, end:
        Optional time range bounds.  When omitted, the last 30 days of daily
        data are requested.

    Returns
    -------
    pandas.DataFrame
        Frame indexed by UTC timestamps with columns ``onch_exchange_inflow``
        and ``onch_exchange_outflow``.
    """

    if not api_key:
        raise ValueError("Glassnode API key is required to fetch exchange flows")

    sess = session or requests.Session()

    end_ts = _ensure_utc_timestamp(end) if end is not None else pd.Timestamp.utcnow().tz_localize("UTC")
    start_ts = _ensure_utc_timestamp(start) if start is not None else end_ts - pd.Timedelta(days=30)
    if start_ts > end_ts:
        start_ts = end_ts

    params = {
        "api_key": api_key,
        "a": asset,
        "i": "24h",
        "s": int(start_ts.timestamp()),
        "u": int(end_ts.timestamp()),
    }

    def _load_series(url: str) -> pd.Series:
        response = sess.get(url, params=params, timeout=10)
        response.raise_for_status()
        payload = response.json() or []
        frame = pd.DataFrame(payload)
        if frame.empty:
            return pd.Series(dtype="float64")
        if "t" not in frame or "v" not in frame:
            raise ValueError("Unexpected response schema from Glassnode exchange flow endpoint")
        frame.index = pd.to_datetime(frame["t"], unit="s", utc=True)
        return pd.to_numeric(frame["v"], errors="coerce").rename("value")

    try:
        inflow = _load_series(GLASSNODE_EXCHANGE_INFLOW_ENDPOINT).rename("onch_exchange_inflow")
        outflow = _load_series(GLASSNODE_EXCHANGE_OUTFLOW_ENDPOINT).rename("onch_exchange_outflow")
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Failed to fetch Glassnode exchange flows", exc_info=exc)
        return _empty_timestamp_frame(["onch_exchange_inflow", "onch_exchange_outflow"])

    if inflow.empty and outflow.empty:
        return _empty_timestamp_frame(["onch_exchange_inflow", "onch_exchange_outflow"])

    frame = pd.concat([inflow, outflow], axis=1)
    frame.index.name = "timestamp"
    frame = frame.sort_index()
    return frame


def _format_coinmetrics_timestamp(value: Any) -> str:
    ts = _ensure_utc_timestamp(value)
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_coinmetrics_exchange_flows(
    *,
    asset: str = "btc",
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    metrics: tuple[str, ...] = tuple(_COINMETRICS_COLUMN_MAP.keys()),
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Fetch exchange flow metrics from the CoinMetrics community API."""

    if not metrics:
        raise ValueError("At least one metric must be requested from CoinMetrics")

    params: dict[str, Any] = {
        "assets": asset.lower(),
        "metrics": ",".join(metrics),
        "frequency": "1d",
    }
    if start is not None:
        params["start_time"] = _format_coinmetrics_timestamp(start)
    if end is not None:
        params["end_time"] = _format_coinmetrics_timestamp(end)

    sess = session or requests.Session()
    collected: list[dict[str, Any]] = []
    next_token: str | None = None

    try:
        for _ in range(16):  # defensive guard against endless pagination loops
            request_params = params.copy()
            if next_token:
                request_params["page_token"] = next_token
            response = sess.get(
                COINMETRICS_ASSET_METRICS_ENDPOINT,
                params=request_params,
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json() or {}
            data_chunk = payload.get("data", [])
            if data_chunk:
                collected.extend(data_chunk)
            next_token = payload.get("next_page_token")
            if not next_token:
                break
        else:
            logger.warning("CoinMetrics pagination exceeded iteration guard; aborting fetch")
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Failed to fetch CoinMetrics exchange flows", exc_info=exc)
        target_cols = [_COINMETRICS_COLUMN_MAP.get(metric, metric) for metric in metrics]
        return _empty_timestamp_frame(target_cols)

    if not collected:
        target_cols = [_COINMETRICS_COLUMN_MAP.get(metric, metric) for metric in metrics]
        return _empty_timestamp_frame(target_cols)

    frame = pd.DataFrame(collected)
    if frame.empty or "time" not in frame.columns:
        target_cols = [_COINMETRICS_COLUMN_MAP.get(metric, metric) for metric in metrics]
        return _empty_timestamp_frame(target_cols)

    frame["timestamp"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"])  # drop rows with invalid timestamps

    rename_map = {metric: _COINMETRICS_COLUMN_MAP.get(metric, metric) for metric in metrics}
    available_metrics = [metric for metric in metrics if metric in frame.columns]
    if not available_metrics:
        target_cols = list(rename_map.values())
        return _empty_timestamp_frame(target_cols)

    frame = frame.rename(columns=rename_map)
    for column in rename_map.values():
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    desired_columns = [rename_map[metric] for metric in available_metrics]
    frame = frame.set_index("timestamp")
    frame = frame.loc[:, desired_columns]
    frame.index.name = "timestamp"
    frame = frame.sort_index()
    return frame


__all__ = [
    "fetch_mempool_stats",
    "fetch_exchange_flows",
    "fetch_coinmetrics_exchange_flows",
]
