"""Persistence helpers for scheduler-managed data feeds."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

from crypto_analyzer.data.db_connector import get_engine

_METADATA = MetaData()

DERIVATIVES_TABLE = Table(
    "derivatives_intraday",
    _METADATA,
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("symbol", String(20), nullable=False),
    Column("funding_rate", Float),
    Column("open_interest", Float),
    Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    UniqueConstraint("timestamp", "symbol", name="ux_derivatives_intraday"),
)

ORDERBOOK_TABLE = Table(
    "orderbook_snapshots",
    _METADATA,
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("symbol", String(20), nullable=False),
    Column("bid_price", Float),
    Column("bid_volume", Float),
    Column("ask_price", Float),
    Column("ask_volume", Float),
    Column("spread", Float),
    Column("mid_price", Float),
    Column("bid_volume_total", Float),
    Column("ask_volume_total", Float),
    Column("bid_notional_total", Float),
    Column("ask_notional_total", Float),
    Column("depth_imbalance", Float),
    Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    UniqueConstraint("timestamp", "symbol", name="ux_orderbook_snapshots"),
)

NEWS_TABLE = Table(
    "news_items",
    _METADATA,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("title", String(512), nullable=False, default=""),
    Column("url", String(512), nullable=False, unique=True),
    Column("source", String(128)),
    Column("sentiment", Float),
    Column("positive_votes", Float),
    Column("negative_votes", Float),
    Column("tags", String(256)),
    Column("currencies", String(128)),
    Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
)

REDDIT_SENTIMENT_TABLE = Table(
    "reddit_sentiment",
    _METADATA,
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("subreddit", String(128), nullable=False, default=""),
    Column("query", String(128), nullable=False, default=""),
    Column("reddit_score", Float),
    Column("mentions", Float),
    Column("reddit_positive_ratio", Float),
    Column("reddit_negative_ratio", Float),
    Column("reddit_positive_count", Float),
    Column("reddit_negative_count", Float),
    Column("reddit_neutral_count", Float),
    Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    UniqueConstraint("timestamp", "subreddit", "query", name="ux_reddit_sentiment"),
)

GLASSNODE_ACTIVE_TABLE = Table(
    "glassnode_active_addresses",
    _METADATA,
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("asset", String(16), nullable=False),
    Column("onch_active_addresses", Float),
    Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    UniqueConstraint("timestamp", "asset", name="ux_glassnode_active_addresses"),
)

COINMETRICS_FLOWS_TABLE = Table(
    "coinmetrics_exchange_flows",
    _METADATA,
    Column("timestamp", DateTime(timezone=True), nullable=False),
    Column("asset", String(16), nullable=False),
    Column("onch_exchange_net_flow", Float),
    Column("onch_exchange_inflow", Float),
    Column("onch_exchange_outflow", Float),
    Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
    UniqueConstraint("timestamp", "asset", name="ux_coinmetrics_exchange_flows"),
)

FEAR_GREED_TABLE = Table(
    "fear_greed_index",
    _METADATA,
    Column("timestamp", DateTime(timezone=True), nullable=False, unique=True),
    Column("value", Float),
    Column("classification", String(64)),
    Column("time_until_update", Float),
    Column("fetched_at", DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")),
)


@dataclass(frozen=True, slots=True)
class StoreResult:
    """Summarise how many rows have been written by a persistence call."""

    inserted: int = 0


def ensure_schema(engine: Engine | None = None, *, db_path: str | Path | None = None) -> Engine:
    """Initialise all scheduler-managed tables and return the active engine."""

    if engine is None:
        engine = get_engine(db_path)
    with engine.begin() as conn:
        _METADATA.create_all(conn)
    return engine


def _ensure_utc(value: object) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _prepare_records(frame: pd.DataFrame, *, extra: dict[str, object] | None = None) -> list[dict[str, object]]:
    if frame.empty:
        return []
    records: list[dict[str, object]] = []
    for raw in frame.to_dict(orient="records"):
        record = dict(extra or {})
        for key, value in raw.items():
            if key == "timestamp":
                if value is None or pd.isna(value):
                    break
                record[key] = _ensure_utc(value)
            else:
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    record[key] = None
                else:
                    record[key] = value
        else:
            records.append(record)
    return records


def _upsert(table: Table, records: Iterable[dict[str, object]], engine: Engine, conflict_cols: tuple[str, ...]) -> int:
    payload = list(records)
    if not payload:
        return 0

    dialect = engine.dialect.name

    def _build_update_cols(insert_stmt):
        return {
            col.name: getattr(insert_stmt.excluded, col.name)
            for col in table.c
            if col.name not in conflict_cols and not col.primary_key
        }

    if dialect == "postgresql":
        insert_stmt = pg_insert(table)
        update_cols = _build_update_cols(insert_stmt)
        insert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=[table.c[name] for name in conflict_cols], set_=update_cols
        )
    elif dialect == "sqlite":
        insert_stmt = sqlite_insert(table)
        update_cols = _build_update_cols(insert_stmt)
        insert_stmt = insert_stmt.on_conflict_do_update(index_elements=list(conflict_cols), set_=update_cols)
    else:
        insert_stmt = table.insert().prefix_with("OR REPLACE")

    with engine.begin() as conn:
        conn.execute(insert_stmt, payload)
    return len(payload)


def _latest_timestamp(table: Table, engine: Engine, *, filters: dict[str, object] | None = None) -> pd.Timestamp | None:
    stmt = select(func.max(table.c.timestamp))
    if filters:
        for key, value in filters.items():
            if value is None or key not in table.c:
                continue
            stmt = stmt.where(table.c[key] == value)
    with engine.connect() as conn:
        result = conn.execute(stmt).scalar_one_or_none()
    if result is None:
        return None
    ts = pd.Timestamp(result)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def latest_derivatives_timestamp(engine: Engine, *, symbol: str) -> pd.Timestamp | None:
    return _latest_timestamp(DERIVATIVES_TABLE, engine, filters={"symbol": symbol})


def latest_glassnode_timestamp(engine: Engine, *, asset: str) -> pd.Timestamp | None:
    return _latest_timestamp(GLASSNODE_ACTIVE_TABLE, engine, filters={"asset": asset})


def latest_coinmetrics_timestamp(engine: Engine, *, asset: str) -> pd.Timestamp | None:
    return _latest_timestamp(COINMETRICS_FLOWS_TABLE, engine, filters={"asset": asset})


def latest_fear_greed_timestamp(engine: Engine) -> pd.Timestamp | None:
    return _latest_timestamp(FEAR_GREED_TABLE, engine)


def store_derivatives(
    funding: pd.DataFrame,
    open_interest: pd.DataFrame,
    *,
    engine: Engine,
    symbol: str,
) -> StoreResult:
    frames: list[pd.DataFrame] = []
    if not funding.empty:
        frame = funding.loc[:, ["timestamp", "funding_rate"]].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frames.append(frame)
    if not open_interest.empty:
        frame = open_interest.loc[:, ["timestamp", "open_interest"]].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frames.append(frame)

    if not frames:
        return StoreResult(0)

    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="timestamp", how="outer")

    merged = merged.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"])
    value_cols = [col for col in merged.columns if col not in {"timestamp"}]
    merged = merged.dropna(subset=value_cols, how="all")
    merged["symbol"] = symbol
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    records = _prepare_records(merged)
    inserted = _upsert(DERIVATIVES_TABLE, records, engine, ("timestamp", "symbol"))
    return StoreResult(inserted)


def store_orderbook(snapshot: pd.DataFrame, *, engine: Engine, symbol: str) -> StoreResult:
    if snapshot.empty:
        return StoreResult(0)
    frame = snapshot.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
    frame["symbol"] = symbol
    records = _prepare_records(frame)
    inserted = _upsert(ORDERBOOK_TABLE, records, engine, ("timestamp", "symbol"))
    return StoreResult(inserted)


def store_news(news: pd.DataFrame, *, engine: Engine) -> StoreResult:
    if news.empty:
        return StoreResult(0)
    frame = news.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "url"]).reset_index(drop=True)
    frame["url"] = frame["url"].astype(str).str.strip()
    frame = frame.loc[frame["url"] != ""]
    records = _prepare_records(frame)
    inserted = _upsert(NEWS_TABLE, records, engine, ("url",))
    return StoreResult(inserted)


def store_reddit_sentiment(
    sentiment: pd.DataFrame,
    *,
    engine: Engine,
    subreddit: str | None = None,
    query: str | None = None,
) -> StoreResult:
    if sentiment.empty:
        return StoreResult(0)
    frame = sentiment.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
    frame["subreddit"] = (subreddit or "").strip()
    frame["query"] = (query or "").strip()
    records = _prepare_records(frame)
    inserted = _upsert(REDDIT_SENTIMENT_TABLE, records, engine, ("timestamp", "subreddit", "query"))
    return StoreResult(inserted)


def store_glassnode_active_addresses(
    addresses: pd.DataFrame,
    *,
    engine: Engine,
    asset: str,
) -> StoreResult:
    if addresses.empty:
        return StoreResult(0)
    frame = addresses.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
    frame["asset"] = asset
    records = _prepare_records(frame)
    inserted = _upsert(GLASSNODE_ACTIVE_TABLE, records, engine, ("timestamp", "asset"))
    return StoreResult(inserted)


def store_coinmetrics_flows(
    flows: pd.DataFrame,
    *,
    engine: Engine,
    asset: str,
) -> StoreResult:
    if flows.empty:
        return StoreResult(0)
    frame = flows.copy()
    if "timestamp" not in frame.columns:
        frame = frame.reset_index()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
    frame["asset"] = asset
    records = _prepare_records(frame)
    inserted = _upsert(COINMETRICS_FLOWS_TABLE, records, engine, ("timestamp", "asset"))
    return StoreResult(inserted)


def store_fear_greed_index(index_frame: pd.DataFrame, *, engine: Engine) -> StoreResult:
    if index_frame.empty:
        return StoreResult(0)
    frame = index_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
    records = _prepare_records(frame)
    inserted = _upsert(FEAR_GREED_TABLE, records, engine, ("timestamp",))
    return StoreResult(inserted)


__all__ = [
    "StoreResult",
    "ensure_schema",
    "latest_coinmetrics_timestamp",
    "latest_derivatives_timestamp",
    "latest_fear_greed_timestamp",
    "latest_glassnode_timestamp",
    "store_coinmetrics_flows",
    "store_derivatives",
    "store_fear_greed_index",
    "store_glassnode_active_addresses",
    "store_news",
    "store_orderbook",
    "store_reddit_sentiment",
]
