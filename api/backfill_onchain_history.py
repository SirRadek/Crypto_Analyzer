#!/usr/bin/env python3
"""Backfill five-minute Bitcoin on-chain aggregates.

The script collects public on-chain metrics from a couple of unauthenticated
APIs and stores them in the ``onchain_5m`` SQLite table.  The upstream
providers are not perfectly consistent in their payloads, therefore most helper
functions focus on validating inputs, coercing numbers and normalising
timestamps.  Whenever a response cannot be parsed the offending rows are simply
skipped – data issues should never crash the backfill.
"""

from __future__ import annotations

import contextlib
import logging
import math
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT = 15.0
FIVE_MINUTES = "5min"

HOENICKE_BASE_URL = "https://mempool.jhoenicke.de/api/v1/fees"
MEMPOOL_SUMMARY_URL = "https://mempool.space/api/mempool"
MEMPOOL_RECOMMENDED_URL = "https://mempool.space/api/v1/fees/recommended"
MEMPOOL_HISTOGRAM_URL = "https://mempool.space/api/v1/fees/histogram"
BLOCKCHAIN_MEMPOOL_COUNT_URL = "https://api.blockchain.info/charts/mempool-count"
BLOCKCHAIN_MEMPOOL_TOTAL_FEE_URL = "https://api.blockchain.info/charts/mempool-total-fee"
HASHRATE_URL = "https://api.blockchain.info/charts/hash-rate"
MINING_DIFFICULTY_URL = "https://api.blockchain.info/charts/difficulty"

COLUMNS = [
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
    "onch_hashrate_ehs",
]

_INT_COLUMNS = {"onch_height", "onch_mempool_count", "onch_mempool_total_fee_sat"}

__all__ = [
    "COLUMNS",
    "fetch_hoenicke_fees",
    "fetch_blockchain_mempool",
    "fetch_mining_difficulty",
    "fetch_hashrate",
    "fetch_current_snapshot",
    "backfill_onchain_history",
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _ensure_timestamp(value: Any) -> pd.Timestamp:
    """Convert *value* to a timezone-aware UTC timestamp."""

    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    else:
        ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    else:
        ts = ts.tz_convert(UTC)
    return ts


def _parse_timestamp(value: Any) -> pd.Timestamp | None:
    """Best-effort conversion helper used when parsing API payloads."""

    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return _ensure_timestamp(value)
    if isinstance(value, datetime):
        return _ensure_timestamp(pd.Timestamp(value))
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return pd.Timestamp(float(value), unit="s", tz=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        with contextlib.suppress(ValueError):
            return pd.Timestamp(text, tz=UTC)
        with contextlib.suppress(ValueError, TypeError):
            return pd.Timestamp(float(text), unit="s", tz=UTC)
    return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        with contextlib.suppress(ValueError):
            return float(text)
    return None


def _empty_frame(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns), index=pd.DatetimeIndex([], tz=UTC))


def _hoenicke_span(start: pd.Timestamp, end: pd.Timestamp) -> str:
    """Return a timespan token understood by the Hoenicke API."""

    duration = end - start
    spans: list[tuple[str, timedelta]] = [
        ("3h", timedelta(hours=3)),
        ("6h", timedelta(hours=6)),
        ("12h", timedelta(hours=12)),
        ("24h", timedelta(days=1)),
        ("3d", timedelta(days=3)),
        ("1w", timedelta(days=7)),
        ("1m", timedelta(days=30)),
        ("3m", timedelta(days=90)),
        ("1y", timedelta(days=365)),
    ]
    for token, delta in spans:
        if duration <= delta:
            return token
    return spans[-1][0]


def _call_json(
    session: Any,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    try:
        response = session.get(url, params=params, timeout=timeout)
    except TypeError:
        # Some very small mock objects used in tests do not accept ``params``.
        if params is not None:
            response = session.get(url, timeout=timeout)
        else:  # pragma: no cover - defensive fallback
            raise
    response.raise_for_status()
    try:
        return response.json()
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid JSON payload from {url}") from exc


def _iterate_series(payload: Any) -> Iterable[Any]:
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        for key in ("data", "values", "series", "entries", "points"):
            inner = payload.get(key)
            if isinstance(inner, Sequence):
                return inner
            if isinstance(inner, Mapping):
                for sub_key in ("data", "values", "series"):
                    nested = inner.get(sub_key)
                    if isinstance(nested, Sequence):
                        return nested
    if isinstance(payload, Sequence) and not isinstance(payload, (bytes, str)):
        return payload
    return []


def _extract_point(item: Any) -> tuple[pd.Timestamp | None, float | None]:
    if isinstance(item, Mapping):
        ts_raw = (
            item.get("time")
            or item.get("timestamp")
            or item.get("ts")
            or item.get("x")
            or item.get("t")
        )
        value_raw = (
            item.get("value")
            or item.get("y")
            or item.get("hashrate")
            or item.get("difficulty")
            or item.get("count")
            or item.get("total_fee")
            or item.get("vsize")
        )
    elif isinstance(item, Sequence) and not isinstance(item, (bytes, str)):
        ts_raw = item[0] if len(item) > 0 else None
        value_raw = item[1] if len(item) > 1 else None
    else:
        return None, None
    ts = _parse_timestamp(ts_raw)
    value = _maybe_float(value_raw)
    return ts, value


def _normalise_histogram(hist: Any) -> list[tuple[float, float]]:
    if hist is None:
        return []
    items: Iterable[Any]
    if isinstance(hist, Mapping):
        bins = hist.get("bins")
        counts = hist.get("counts")
        if isinstance(bins, Sequence) and isinstance(counts, Sequence):
            return [
                (float(b), float(c))
                for b, c in zip(bins, counts, strict=False)
                if _maybe_float(b) is not None and _maybe_float(c) is not None
            ]
        items = hist.values()
    else:
        items = hist
    result: list[tuple[float, float]] = []
    for item in items:
        if isinstance(item, Mapping):
            rate = _maybe_float(
                item.get("fee")
                or item.get("fee_rate")
                or item.get("feerate")
                or item.get("avgFeeRate")
                or item.get("sat_per_vb")
                or item.get("sat/vB")
            )
            size = _maybe_float(
                item.get("vsize") or item.get("size") or item.get("count") or item.get("weight")
            )
        elif isinstance(item, Sequence) and not isinstance(item, (bytes, str)):
            rate = _maybe_float(item[0] if len(item) > 0 else None)
            size = _maybe_float(item[1] if len(item) > 1 else None)
        else:
            continue
        if rate is None or size is None or size <= 0:
            continue
        result.append((rate, size))
    return result


def _weighted_avg(hist: Any) -> float | None:
    pairs = _normalise_histogram(hist)
    if not pairs:
        return None
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return None
    return sum(rate * weight for rate, weight in pairs) / total


def _percentile(hist: Any, q: float) -> float | None:
    if not 0 <= q <= 1:
        raise ValueError("quantile must be within [0, 1]")
    pairs = sorted(_normalise_histogram(hist), key=lambda item: item[0])
    if not pairs:
        return None
    total = sum(weight for _, weight in pairs)
    if total <= 0:
        return None
    cutoff = total * q
    cumulative = 0.0
    for rate, weight in pairs:
        cumulative += weight
        if cumulative >= cutoff:
            return rate
    return pairs[-1][0]


def _build_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    floor = start.floor(FIVE_MINUTES)
    ceil = end.ceil(FIVE_MINUTES)
    if ceil < floor:
        ceil = floor
    return pd.date_range(floor, ceil, freq=FIVE_MINUTES, tz=UTC)


def _chart_params(start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    duration = max(1, int((end - start).total_seconds()))
    days = max(1, math.ceil(duration / 86_400))
    return {
        "start": int(start.timestamp()),
        "timespan": f"{days}days",
        "sampled": "false",
        "format": "json",
    }


def _rows_from_dataframe(df: pd.DataFrame) -> Iterable[list[Any]]:
    for ts, row in df.iterrows():
        timestamp = int(ts.timestamp())
        values: list[Any] = [timestamp]
        for column in COLUMNS:
            value = row.get(column)
            if pd.isna(value):
                values.append(None)
                continue
            if column in _INT_COLUMNS:
                with contextlib.suppress(TypeError, ValueError):
                    values.append(int(round(float(value))))
                    continue
            values.append(float(value))
        yield values


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS onchain_5m (
            ts_utc INTEGER PRIMARY KEY,
            onch_fee_fast_satvb REAL,
            onch_fee_30m_satvb REAL,
            onch_fee_60m_satvb REAL,
            onch_fee_min_satvb REAL,
            onch_mempool_count REAL,
            onch_mempool_vsize_vB REAL,
            onch_mempool_total_fee_sat REAL,
            onch_fee_wavg_satvb REAL,
            onch_fee_p50_satvb REAL,
            onch_fee_p90_satvb REAL,
            onch_difficulty REAL,
            onch_height REAL,
            onch_diff_change_pct REAL,
            onch_hashrate_ehs REAL
        )
        """
    )


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_hoenicke_fees(
    start: pd.Timestamp | str | datetime,
    end: pd.Timestamp | str | datetime,
    session: Any,
) -> pd.DataFrame:
    """Fetch fee histogram aggregates from Jochen Hoenicke's API."""

    start_ts = _ensure_timestamp(start)
    end_ts = _ensure_timestamp(end)
    params = {"from": int(start_ts.timestamp()), "to": int(end_ts.timestamp())}
    span = _hoenicke_span(start_ts, end_ts)
    columns = [
        "onch_fee_wavg_satvb",
        "onch_fee_p50_satvb",
        "onch_fee_p90_satvb",
        "onch_mempool_vsize_vB",
    ]
    try:
        payload = _call_json(session, f"{HOENICKE_BASE_URL}/{span}", params=params)
    except Exception:  # pragma: no cover - network issues are expected sometimes
        logger.debug("Hoenicke request failed", exc_info=True)
        return _empty_frame(columns)

    records: list[tuple[pd.Timestamp, dict[str, float | None]]] = []
    for item in _iterate_series(payload):
        hist: Any | None = None
        if isinstance(item, Mapping):
            ts = item.get("time") or item.get("timestamp") or item.get("ts")
            wavg = item.get("weighted_fee") or item.get("avg") or item.get("mean") or item.get("wavg")
            p50 = item.get("median") or item.get("p50")
            p90 = item.get("p90") or item.get("percentile90") or item.get("p95")
            vsize = item.get("vsize") or item.get("size") or item.get("total_vsize")
            hist = item.get("histogram") or item.get("fee_histogram") or item.get("buckets")
        elif isinstance(item, Sequence) and not isinstance(item, (bytes, str)):
            ts = item[0] if len(item) > 0 else None
            wavg = item[1] if len(item) > 1 else None
            p50 = item[2] if len(item) > 2 else None
            p90 = item[3] if len(item) > 3 else None
            vsize = item[4] if len(item) > 4 else None
        else:
            continue
        ts_parsed = _parse_timestamp(ts)
        if ts_parsed is None or ts_parsed < start_ts or ts_parsed > end_ts:
            continue
        record = {
            "onch_fee_wavg_satvb": _maybe_float(wavg),
            "onch_fee_p50_satvb": _maybe_float(p50),
            "onch_fee_p90_satvb": _maybe_float(p90),
            "onch_mempool_vsize_vB": _maybe_float(vsize),
        }
        if hist:
            record["onch_fee_wavg_satvb"] = record["onch_fee_wavg_satvb"] or _weighted_avg(hist)
            record["onch_fee_p50_satvb"] = record["onch_fee_p50_satvb"] or _percentile(hist, 0.5)
            record["onch_fee_p90_satvb"] = record["onch_fee_p90_satvb"] or _percentile(hist, 0.9)
        records.append((ts_parsed, record))

    if not records:
        return _empty_frame(columns)

    idx = pd.DatetimeIndex([ts for ts, _ in records], tz=UTC)
    df = pd.DataFrame([rec for _, rec in records], index=idx)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _fetch_blockchain_chart(
    session: Any,
    url: str,
    column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[pd.Timestamp, dict[str, float]]:
    try:
        payload = _call_json(session, url, params=_chart_params(start, end))
    except Exception:  # pragma: no cover - network
        logger.debug("Blockchain chart request failed", exc_info=True)
        return {}

    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for item in _iterate_series(payload):
        ts, value = _extract_point(item)
        if ts is None or value is None:
            continue
        if ts < start or ts > end:
            continue
        rows.setdefault(ts, {})[column] = value
    return rows


def _fetch_recommended_fees(session: Any) -> dict[str, float | None]:
    try:
        payload = _call_json(session, MEMPOOL_RECOMMENDED_URL)
    except Exception:  # pragma: no cover - network
        logger.debug("Recommended fee request failed", exc_info=True)
        return {}
    if not isinstance(payload, Mapping):
        return {}
    return {
        "onch_fee_fast_satvb": _maybe_float(payload.get("fastestFee")),
        "onch_fee_30m_satvb": _maybe_float(payload.get("halfHourFee")),
        "onch_fee_60m_satvb": _maybe_float(payload.get("hourFee")),
        "onch_fee_min_satvb": _maybe_float(payload.get("minimumFee") or payload.get("economyFee")),
    }


def _fetch_mempool_snapshot(session: Any) -> dict[str, float | None]:
    data: dict[str, float | None] = {}
    try:
        summary = _call_json(session, MEMPOOL_SUMMARY_URL)
        if isinstance(summary, Mapping):
            data["onch_mempool_count"] = _maybe_float(summary.get("count"))
            data["onch_mempool_vsize_vB"] = _maybe_float(summary.get("vsize"))
            data["onch_mempool_total_fee_sat"] = _maybe_float(summary.get("total_fee"))
    except Exception:  # pragma: no cover - network
        logger.debug("Mempool summary request failed", exc_info=True)

    try:
        hist_payload = _call_json(session, MEMPOOL_HISTOGRAM_URL)
        if hist_payload:
            data.setdefault("onch_fee_wavg_satvb", _weighted_avg(hist_payload))
            data.setdefault("onch_fee_p50_satvb", _percentile(hist_payload, 0.5))
            data.setdefault("onch_fee_p90_satvb", _percentile(hist_payload, 0.9))
    except Exception:  # pragma: no cover - network
        logger.debug("Histogram request failed", exc_info=True)

    data.update({k: v for k, v in _fetch_recommended_fees(session).items() if v is not None})
    return data


def fetch_blockchain_mempool(
    start: pd.Timestamp | str | datetime,
    end: pd.Timestamp | str | datetime,
    session: Any,
) -> pd.DataFrame:
    """Fetch mempool statistics from blockchain.info and mempool.space."""

    start_ts = _ensure_timestamp(start)
    end_ts = _ensure_timestamp(end)
    combined: defaultdict[pd.Timestamp, dict[str, float | None]] = defaultdict(dict)

    for ts, row in _fetch_blockchain_chart(
        session, BLOCKCHAIN_MEMPOOL_COUNT_URL, "onch_mempool_count", start_ts, end_ts
    ).items():
        combined[ts].update(row)

    for ts, row in _fetch_blockchain_chart(
        session, BLOCKCHAIN_MEMPOOL_TOTAL_FEE_URL, "onch_mempool_total_fee_sat", start_ts, end_ts
    ).items():
        combined[ts].update(row)

    if not combined:
        return _empty_frame(
            [
                "onch_fee_fast_satvb",
                "onch_fee_30m_satvb",
                "onch_fee_60m_satvb",
                "onch_fee_min_satvb",
                "onch_mempool_count",
                "onch_mempool_vsize_vB",
                "onch_mempool_total_fee_sat",
            ]
        )

    snapshot = _fetch_mempool_snapshot(session)
    if snapshot:
        combined[end_ts.floor(FIVE_MINUTES)].update(snapshot)

    idx = pd.DatetimeIndex(sorted(combined), tz=UTC)
    df = pd.DataFrame([combined[ts] for ts in idx], index=idx)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_mining_difficulty(
    start: pd.Timestamp | str | datetime,
    end: pd.Timestamp | str | datetime,
    session: Any,
) -> pd.DataFrame:
    """Fetch mining difficulty and estimated next adjustment."""

    start_ts = _ensure_timestamp(start)
    end_ts = _ensure_timestamp(end)
    try:
        payload = _call_json(session, MINING_DIFFICULTY_URL, params=_chart_params(start_ts, end_ts))
    except Exception:  # pragma: no cover - network
        logger.debug("Difficulty request failed", exc_info=True)
        return _empty_frame(["onch_difficulty", "onch_height", "onch_diff_change_pct"])

    records: list[tuple[pd.Timestamp, dict[str, float | None]]] = []
    for item in _iterate_series(payload):
        if isinstance(item, Mapping):
            ts = item.get("time") or item.get("timestamp") or item.get("x")
            height = item.get("height")
            diff = item.get("difficulty") or item.get("y") or item.get("value")
            next_diff = item.get("next_difficulty") or item.get("nextDiff") or item.get("estimated_next")
            change_pct = item.get("change") or item.get("diff_change_pct")
        elif isinstance(item, Sequence) and not isinstance(item, (bytes, str)):
            ts = item[0] if len(item) > 0 else None
            height = item[1] if len(item) > 1 else None
            diff = item[2] if len(item) > 2 else None
            next_diff = item[3] if len(item) > 3 else None
            change_pct = item[4] if len(item) > 4 else None
        else:
            continue
        ts_parsed = _parse_timestamp(ts)
        if ts_parsed is None or ts_parsed < start_ts or ts_parsed > end_ts:
            continue
        diff_val = _maybe_float(diff)
        change_val = _maybe_float(change_pct)
        next_val = _maybe_float(next_diff)
        if change_val is None and diff_val is not None and next_val is not None:
            change_val = (next_val - diff_val) * 100.0
        record = {
            "onch_difficulty": diff_val,
            "onch_height": _maybe_float(height),
            "onch_diff_change_pct": change_val,
        }
        records.append((ts_parsed, record))

    if not records:
        return _empty_frame(["onch_difficulty", "onch_height", "onch_diff_change_pct"])

    idx = pd.DatetimeIndex([ts for ts, _ in records], tz=UTC)
    df = pd.DataFrame([rec for _, rec in records], index=idx)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if "onch_diff_change_pct" in df.columns:
        values = df["onch_diff_change_pct"].to_numpy(copy=True)
        for i in range(1, len(values)):
            if pd.isna(values[i]) and not pd.isna(values[i - 1]) and not pd.isna(df.iloc[i]["onch_difficulty"]):
                prev = df.iloc[i - 1]["onch_difficulty"]
                curr = df.iloc[i]["onch_difficulty"]
                if prev not in (None, 0):
                    values[i] = (curr - prev) / prev * 100.0
        df["onch_diff_change_pct"] = values

    return df


def fetch_hashrate(
    start: pd.Timestamp | str | datetime,
    end: pd.Timestamp | str | datetime,
    session: Any,
) -> pd.DataFrame:
    """Fetch hashrate (EH/s) time series."""

    start_ts = _ensure_timestamp(start)
    end_ts = _ensure_timestamp(end)
    try:
        payload = _call_json(session, HASHRATE_URL, params=_chart_params(start_ts, end_ts))
    except Exception:  # pragma: no cover - network
        logger.debug("Hashrate request failed", exc_info=True)
        payload = None

    source: Any = payload
    if isinstance(source, Mapping):
        source = source.get("hashrate") or source.get("data") or source

    rows: list[tuple[pd.Timestamp, float]] = []
    for item in _iterate_series(source):
        ts, value = _extract_point(item)
        if ts is None or value is None:
            continue
        if ts < start_ts or ts > end_ts:
            continue
        rows.append((ts, value))

    if not rows:
        return _empty_frame(["onch_hashrate_ehs"])

    idx = pd.DatetimeIndex([ts for ts, _ in rows], tz=UTC)
    df = pd.DataFrame({"onch_hashrate_ehs": [val for _, val in rows]}, index=idx)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_current_snapshot(
    timestamp: pd.Timestamp | str | datetime,
    session: Any,
) -> pd.DataFrame:
    """Return the latest mempool snapshot aligned to the five-minute grid."""

    ts = _ensure_timestamp(timestamp).floor(FIVE_MINUTES)
    data = _fetch_mempool_snapshot(session)
    if not data:
        return _empty_frame(COLUMNS)
    idx = pd.DatetimeIndex([ts], tz=UTC)
    df = pd.DataFrame([data], index=idx)
    df = df.reindex(columns=COLUMNS)
    return df.dropna(how="all", axis=1)


# ---------------------------------------------------------------------------
# Backfill orchestrator
# ---------------------------------------------------------------------------

def _normalise_frame(frame: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Return *frame* aligned to ``index`` with a clean column set."""

    df = frame.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    else:
        if df.index.tz is None:
            df.index = df.index.tz_localize(UTC)
        else:
            df.index = df.index.tz_convert(UTC)
    if df.index.hasnans:
        df = df[~df.index.isna()]
    df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
    df = df.reindex(columns=COLUMNS)
    df = df.reindex(index)
    return df


def _combine_frames(frames: list[pd.DataFrame], index: pd.DatetimeIndex) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(index=index, columns=COLUMNS, dtype="float64")

    combined = pd.DataFrame(index=index, columns=COLUMNS, dtype="float64")
    for frame in frames:
        if frame is None or frame.empty:
            continue
        normalised = _normalise_frame(frame, index)
        combined.update(normalised)
    return combined


def _write_dataframe(db_path: Path, df: pd.DataFrame) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        rows = list(_rows_from_dataframe(df))
        if not rows:
            return
        placeholders = ",".join(["?"] * (1 + len(COLUMNS)))
        conn.executemany(
            f"INSERT OR REPLACE INTO onchain_5m (ts_utc,{','.join(COLUMNS)}) VALUES ({placeholders})",
            rows,
        )
        conn.commit()


def backfill_onchain_history(
    start: pd.Timestamp | str | datetime,
    end: pd.Timestamp | str | datetime,
    db_path: str | Path,
    *,
    session: requests.Session | None = None,
) -> None:
    """Backfill the ``onchain_5m`` table for the requested time range."""

    start_ts = _ensure_timestamp(start)
    end_ts = _ensure_timestamp(end)
    if start_ts > end_ts:
        raise ValueError("start must be before end")

    index = _build_index(start_ts, end_ts)
    own_session = False
    if session is None:
        session = requests.Session()
        own_session = True

    frames: list[pd.DataFrame] = []
    try:
        for fetcher in (
            fetch_hoenicke_fees,
            fetch_blockchain_mempool,
            fetch_mining_difficulty,
            fetch_hashrate,
        ):
            df = fetcher(start_ts, end_ts, session)
            if df is not None and not df.empty:
                frames.append(df)

        snapshot = fetch_current_snapshot(end_ts, session)
        if snapshot is not None and not snapshot.empty:
            frames.append(snapshot)

        combined = _combine_frames(frames, index)
        combined.index.name = "ts"
        _write_dataframe(Path(db_path), combined)
    finally:
        if own_session:
            session.close()


# ---------------------------------------------------------------------------
# Command line interface
# ---------------------------------------------------------------------------

def cli(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Backfill five-minute on-chain metrics")
    parser.add_argument("--start", required=True, help="Start timestamp (UTC)")
    parser.add_argument("--end", required=True, help="End timestamp (UTC)")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    args = parser.parse_args(argv)

    backfill_onchain_history(args.start, args.end, Path(args.db))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        raise SystemExit(cli())
    except KeyboardInterrupt:
        raise SystemExit(130)

