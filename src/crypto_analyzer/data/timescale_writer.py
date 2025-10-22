"""Utilities for persisting collected data into TimescaleDB tables.

The ingestion scheduler already stores information in lightweight SQLite
tables, however production deployments rely on a TimescaleDB/PostgreSQL
instance.  The helpers in this module accept records produced by the fetch
functions and perform validated ``INSERT .. ON CONFLICT`` statements using
``psycopg2``.  Each function performs basic schema validation, converts the
incoming values to database friendly representations and protects against
duplicate timestamps by leveraging PostgreSQL's upsert facilities.

All functions follow the same calling convention: data may be provided either
as a :class:`pandas.DataFrame`, an iterable of dictionaries or a single
mapping.  A ``psycopg2`` connection can be passed explicitly or connection
parameters are forwarded to :func:`crypto_analyzer.data.db.connection_scope` to
open a short lived connection for the operation.  The helpers return the number
of rows written to the database which makes them easy to assert during tests
and monitoring.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import math

import pandas as pd
try:  # pragma: no cover - optional dependency during some tests
    from psycopg2.extensions import connection as PGConnection
    from psycopg2.extras import execute_batch
except ModuleNotFoundError:  # pragma: no cover - fallback when psycopg2 missing
    PGConnection = Any  # type: ignore[assignment]

    def execute_batch(*args: Any, **kwargs: Any) -> None:
        raise ModuleNotFoundError("psycopg2 is required to persist data")

from crypto_analyzer.utils.logging import get_logger

LOGGER = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _ensure_records(data: Any) -> list[dict[str, Any]]:
    """Normalise ``data`` into a list of dictionaries."""

    if data is None:
        return []

    if isinstance(data, pd.DataFrame):
        return list(data.to_dict(orient="records"))

    if isinstance(data, Mapping):
        return [dict(data)]

    if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        records: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, Mapping):
                records.append(dict(item))
            elif isinstance(item, pd.Series):
                records.append(item.to_dict())
            else:
                raise TypeError(
                    "Data items must be mapping-like objects; received "
                    f"{type(item)!r}"
                )
        return records

    raise TypeError("Unsupported data container provided for persistence")


def _ensure_timestamp(value: Any, *, field: str = "timestamp") -> datetime:
    """Convert ``value`` to an aware :class:`datetime` in UTC."""

    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"Missing required field: {field}")

    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid timestamp for field {field!r}: {value!r}") from exc

    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    else:
        ts = ts.tz_convert(UTC)

    return ts.to_pydatetime()


def _require_string(value: Any, *, field: str) -> str:
    """Return ``value`` as a stripped string ensuring it is not empty."""

    if value is None:
        raise ValueError(f"Missing required field: {field}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"Field {field!r} must be a non-empty string")
    return text


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_float(value: Any, *, field: str, required: bool = False) -> float | None:
    if value is None:
        if required:
            raise ValueError(f"Missing required numeric field: {field}")
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            if required:
                raise ValueError(f"Missing required numeric field: {field}")
            return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for field {field!r}: {value!r}") from exc
    if math.isnan(result) or math.isinf(result):
        raise ValueError(f"Numeric field {field!r} must be finite")
    return result


def _coerce_int(value: Any, *, field: str, required: bool = False) -> int | None:
    if value is None:
        if required:
            raise ValueError(f"Missing required integer field: {field}")
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            if required:
                raise ValueError(f"Missing required integer field: {field}")
            return None
    if isinstance(value, bool):  # pragma: no cover - defensive guard
        raise ValueError(f"Field {field!r} must not be a boolean")
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"Field {field!r} must be an integer value")
        value = int(value)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer value for field {field!r}: {value!r}") from exc


@contextmanager
def _managed_connection(
    connection: PGConnection | None,
    connect_kwargs: dict[str, Any],
) -> Iterator[PGConnection]:
    """Yield a psycopg2 connection, opening one when required."""

    if connection is not None:
        yield connection
    else:
        from crypto_analyzer.data.db import connection_scope  # local import to avoid optional dependency issues

        with connection_scope(**connect_kwargs) as conn:
            yield conn


def _execute_upsert(
    conn: PGConnection,
    sql: str,
    params: Sequence[Sequence[Any]],
    *,
    page_size: int = 100,
) -> int:
    if not params:
        return 0

    try:
        with conn.cursor() as cursor:
            execute_batch(cursor, sql, params, page_size=page_size)
        if not getattr(conn, "autocommit", False):
            conn.commit()
    except Exception:
        if not getattr(conn, "autocommit", False):
            conn.rollback()
        raise
    return len(params)


# ---------------------------------------------------------------------------
# Public persistence helpers
# ---------------------------------------------------------------------------


def save_market_data(
    data: Any,
    *,
    symbol: str | None = None,
    interval: str | None = None,
    connection: PGConnection | None = None,
    batch_size: int = 200,
    **connect_kwargs: Any,
) -> int:
    """Persist OHLCV market data snapshots into the ``market_data`` table."""

    records = _ensure_records(data)
    if not records:
        LOGGER.debug("No market data records supplied – skipping insert")
        return 0

    payload: list[tuple[Any, ...]] = []
    for item in records:
        record = dict(item)
        if symbol and not record.get("symbol"):
            record["symbol"] = symbol
        if interval and not record.get("interval"):
            record["interval"] = interval

        ts = _ensure_timestamp(record.get("timestamp"))
        sym = _require_string(record.get("symbol"), field="symbol")
        inter = _require_string(record.get("interval"), field="interval")
        open_ = _coerce_float(record.get("open"), field="open", required=True)
        high = _coerce_float(record.get("high"), field="high", required=True)
        low = _coerce_float(record.get("low"), field="low", required=True)
        close = _coerce_float(record.get("close"), field="close", required=True)
        volume = _coerce_float(record.get("volume"), field="volume")
        quote_volume = _coerce_float(record.get("quote_volume"), field="quote_volume")
        trades = _coerce_int(record.get("trades"), field="trades")

        payload.append(
            (
                ts,
                sym,
                inter,
                open_,
                high,
                low,
                close,
                volume,
                quote_volume,
                trades,
            )
        )

    if not payload:
        LOGGER.debug("No valid market data rows remained after validation")
        return 0

    sql = (
        "INSERT INTO market_data (timestamp, symbol, interval, open, high, low, close, "
        "volume, quote_volume, trades) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (timestamp, symbol, interval) DO UPDATE SET "
        "open = EXCLUDED.open, "
        "high = EXCLUDED.high, "
        "low = EXCLUDED.low, "
        "close = EXCLUDED.close, "
        "volume = EXCLUDED.volume, "
        "quote_volume = EXCLUDED.quote_volume, "
        "trades = EXCLUDED.trades"
    )

    with _managed_connection(connection, connect_kwargs) as conn:
        written = _execute_upsert(conn, sql, payload, page_size=batch_size)
        LOGGER.info("Stored market data rows", extra={"rows": written})
        return written


def save_derivatives_data(
    data: Any,
    *,
    symbol: str | None = None,
    connection: PGConnection | None = None,
    batch_size: int = 200,
    **connect_kwargs: Any,
) -> int:
    """Persist funding rate and derivatives metrics into ``derivatives_signals``."""

    records = _ensure_records(data)
    if not records:
        LOGGER.debug("No derivatives records supplied – skipping insert")
        return 0

    payload: list[tuple[Any, ...]] = []
    for item in records:
        record = dict(item)
        if symbol and not record.get("symbol"):
            record["symbol"] = symbol

        ts = _ensure_timestamp(record.get("timestamp"))
        sym = _require_string(record.get("symbol"), field="symbol")
        funding = _coerce_float(record.get("funding_rate"), field="funding_rate")
        open_interest = _coerce_float(record.get("open_interest"), field="open_interest")
        basis_value = record.get("basis")
        if basis_value is None:
            basis_value = record.get("basis_bp")
        basis = _coerce_float(basis_value, field="basis")
        payload.append((ts, sym, funding, open_interest, basis))

    sql = (
        "INSERT INTO derivatives_signals (timestamp, symbol, funding_rate, open_interest, basis) "
        "VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (timestamp, symbol) DO UPDATE SET "
        "funding_rate = EXCLUDED.funding_rate, "
        "open_interest = EXCLUDED.open_interest, "
        "basis = EXCLUDED.basis"
    )

    with _managed_connection(connection, connect_kwargs) as conn:
        written = _execute_upsert(conn, sql, payload, page_size=batch_size)
        LOGGER.info("Stored derivatives rows", extra={"rows": written})
        return written


def save_social_sentiment(
    data: Any,
    *,
    connection: PGConnection | None = None,
    batch_size: int = 100,
    **connect_kwargs: Any,
) -> int:
    """Persist aggregated social sentiment values into ``social_sentiment``."""

    records = _ensure_records(data)
    if not records:
        LOGGER.debug("No social sentiment records supplied – skipping insert")
        return 0

    payload: list[tuple[Any, ...]] = []
    for record in records:
        ts = _ensure_timestamp(record.get("timestamp"))
        reddit_score = _coerce_float(record.get("reddit_score"), field="reddit_score")
        twitter_score = _coerce_float(record.get("twitter_score"), field="twitter_score")
        mentions = _coerce_int(record.get("mentions"), field="mentions")
        if mentions is not None and mentions < 0:
            raise ValueError("Mentions count cannot be negative")
        pos_value = record.get("twitter_positive_count", record.get("positive_count"))
        neg_value = record.get("twitter_negative_count", record.get("negative_count"))
        twitter_positive = _coerce_int(pos_value, field="twitter_positive_count")
        twitter_negative = _coerce_int(neg_value, field="twitter_negative_count")
        if twitter_positive is not None and twitter_positive < 0:
            raise ValueError("Twitter positive count cannot be negative")
        if twitter_negative is not None and twitter_negative < 0:
            raise ValueError("Twitter negative count cannot be negative")
        payload.append((ts, reddit_score, twitter_score, mentions, twitter_positive, twitter_negative))

    sql = (
        "INSERT INTO social_sentiment (timestamp, reddit_score, twitter_score, mentions, "
        "twitter_positive_count, twitter_negative_count) "
        "VALUES (%s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (timestamp) DO UPDATE SET "
        "reddit_score = EXCLUDED.reddit_score, "
        "twitter_score = EXCLUDED.twitter_score, "
        "mentions = EXCLUDED.mentions, "
        "twitter_positive_count = EXCLUDED.twitter_positive_count, "
        "twitter_negative_count = EXCLUDED.twitter_negative_count"
    )

    with _managed_connection(connection, connect_kwargs) as conn:
        written = _execute_upsert(conn, sql, payload, page_size=batch_size)
        LOGGER.info("Stored social sentiment rows", extra={"rows": written})
        return written


def save_sentiment_index(
    data: Any,
    *,
    connection: PGConnection | None = None,
    batch_size: int = 100,
    **connect_kwargs: Any,
) -> int:
    """Persist Fear & Greed index values into ``sentiment_index``."""

    records = _ensure_records(data)
    if not records:
        LOGGER.debug("No sentiment index records supplied – skipping insert")
        return 0

    payload: list[tuple[Any, ...]] = []
    for record in records:
        ts = _ensure_timestamp(record.get("timestamp"))
        value = _coerce_int(record.get("value"), field="value", required=True)
        if value < 0 or value > 100:
            raise ValueError("Fear & Greed index must be between 0 and 100")
        classification = _optional_string(record.get("classification"))
        payload.append((ts, value, classification))

    sql = (
        "INSERT INTO sentiment_index (timestamp, value, classification) "
        "VALUES (%s, %s, %s) "
        "ON CONFLICT (timestamp) DO UPDATE SET "
        "value = EXCLUDED.value, "
        "classification = EXCLUDED.classification"
    )

    with _managed_connection(connection, connect_kwargs) as conn:
        written = _execute_upsert(conn, sql, payload, page_size=batch_size)
        LOGGER.info("Stored sentiment index rows", extra={"rows": written})
        return written


def save_onchain_metrics(
    data: Any,
    *,
    connection: PGConnection | None = None,
    batch_size: int = 200,
    **connect_kwargs: Any,
) -> int:
    """Persist on-chain metrics into the ``onchain_metrics`` table."""

    records = _ensure_records(data)
    if not records:
        LOGGER.debug("No on-chain metrics supplied – skipping insert")
        return 0

    payload: list[tuple[Any, ...]] = []
    for record in records:
        ts = _ensure_timestamp(record.get("timestamp"))
        active = _coerce_int(record.get("active_addresses"), field="active_addresses")
        if active is not None and active < 0:
            raise ValueError("Active address count cannot be negative")
        new_addr = _coerce_int(record.get("new_addresses"), field="new_addresses")
        if new_addr is not None and new_addr < 0:
            raise ValueError("New address count cannot be negative")
        inflow = _coerce_float(record.get("exchange_inflow"), field="exchange_inflow")
        outflow = _coerce_float(record.get("exchange_outflow"), field="exchange_outflow")
        whale = _coerce_float(record.get("whale_transactions"), field="whale_transactions")
        payload.append((ts, active, new_addr, inflow, outflow, whale))

    sql = (
        "INSERT INTO onchain_metrics (timestamp, active_addresses, new_addresses, "
        "exchange_inflow, exchange_outflow, whale_transactions) "
        "VALUES (%s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (timestamp) DO UPDATE SET "
        "active_addresses = EXCLUDED.active_addresses, "
        "new_addresses = EXCLUDED.new_addresses, "
        "exchange_inflow = EXCLUDED.exchange_inflow, "
        "exchange_outflow = EXCLUDED.exchange_outflow, "
        "whale_transactions = EXCLUDED.whale_transactions"
    )

    with _managed_connection(connection, connect_kwargs) as conn:
        written = _execute_upsert(conn, sql, payload, page_size=batch_size)
        LOGGER.info("Stored on-chain metric rows", extra={"rows": written})
        return written


def save_whale_transactions(
    data: Any,
    *,
    connection: PGConnection | None = None,
    batch_size: int = 200,
    **connect_kwargs: Any,
) -> int:
    """Persist Whale Alert transactions into the ``whale_transactions`` table."""

    records = _ensure_records(data)
    if not records:
        LOGGER.debug("No whale transactions supplied – skipping insert")
        return 0

    payload: list[tuple[Any, ...]] = []
    for record in records:
        tx_hash_value = record.get("transaction_hash") or record.get("hash")
        currency_value = record.get("currency") or record.get("symbol")
        blockchain = record.get("blockchain")

        from_info = record.get("from") if isinstance(record.get("from"), Mapping) else None
        to_info = record.get("to") if isinstance(record.get("to"), Mapping) else None

        from_address = record.get("from_address")
        if from_address is None and isinstance(from_info, Mapping):
            from_address = from_info.get("address")
        to_address = record.get("to_address")
        if to_address is None and isinstance(to_info, Mapping):
            to_address = to_info.get("address")

        from_owner = record.get("from_owner")
        if from_owner is None and isinstance(from_info, Mapping):
            from_owner = from_info.get("owner")
        to_owner = record.get("to_owner")
        if to_owner is None and isinstance(to_info, Mapping):
            to_owner = to_info.get("owner")

        ts = _ensure_timestamp(record.get("timestamp"))
        tx_hash = _require_string(tx_hash_value, field="transaction_hash")
        currency = _require_string(currency_value, field="currency").upper()
        amount = _coerce_float(record.get("amount"), field="amount")
        amount_usd = _coerce_float(record.get("amount_usd"), field="amount_usd", required=True)
        payload.append(
            (
                ts,
                tx_hash,
                currency,
                amount,
                amount_usd,
                _optional_string(from_address),
                _optional_string(from_owner),
                _optional_string(to_address),
                _optional_string(to_owner),
                _optional_string(blockchain),
            )
        )

    sql = (
        "INSERT INTO whale_transactions "
        "(timestamp, transaction_hash, currency, amount, amount_usd, from_address, "
        "from_owner, to_address, to_owner, blockchain) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (timestamp, transaction_hash) DO UPDATE SET "
        "currency = EXCLUDED.currency, "
        "amount = EXCLUDED.amount, "
        "amount_usd = EXCLUDED.amount_usd, "
        "from_address = EXCLUDED.from_address, "
        "from_owner = EXCLUDED.from_owner, "
        "to_address = EXCLUDED.to_address, "
        "to_owner = EXCLUDED.to_owner, "
        "blockchain = EXCLUDED.blockchain"
    )

    with _managed_connection(connection, connect_kwargs) as conn:
        written = _execute_upsert(conn, sql, payload, page_size=batch_size)
        LOGGER.info("Stored whale transaction rows", extra={"rows": written})
        return written


def save_news(
    data: Any,
    *,
    connection: PGConnection | None = None,
    batch_size: int = 200,
    **connect_kwargs: Any,
) -> int:
    """Persist curated news items into the ``news`` table."""

    records = _ensure_records(data)
    if not records:
        LOGGER.debug("No news records supplied – skipping insert")
        return 0

    payload: list[tuple[Any, ...]] = []
    for record in records:
        ts = _ensure_timestamp(record.get("timestamp"))
        title = _require_string(record.get("title"), field="title")
        url = _optional_string(record.get("url"))
        source = _optional_string(record.get("source"))
        sentiment = _coerce_float(record.get("sentiment"), field="sentiment")
        payload.append((ts, title, url, source, sentiment))

    sql = (
        "INSERT INTO news (timestamp, title, url, source, sentiment) "
        "VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (timestamp, title) DO UPDATE SET "
        "url = EXCLUDED.url, "
        "source = EXCLUDED.source, "
        "sentiment = EXCLUDED.sentiment"
    )

    with _managed_connection(connection, connect_kwargs) as conn:
        written = _execute_upsert(conn, sql, payload, page_size=batch_size)
        LOGGER.info("Stored news rows", extra={"rows": written})
        return written


__all__ = [
    "save_market_data",
    "save_derivatives_data",
    "save_social_sentiment",
    "save_sentiment_index",
    "save_onchain_metrics",
    "save_news",
    "save_whale_transactions",
]

