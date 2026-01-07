"""Database access helpers backed by SQLAlchemy engines.

The original implementation relied on SQLite ``sqlite3`` connections.  To
support TimescaleDB and other SQL backends efficiently we now delegate all
database interactions to SQLAlchemy.  Engines are cached to avoid repeatedly
opening new connections and both inserts and selects are fully parameterised
to prevent SQL injection.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from itertools import chain
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    func,
    inspect,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.engine import Engine

try:  # pragma: no cover - optional dependency for tests
    from crypto_analyzer.utils.config import CONFIG
except ModuleNotFoundError:  # pragma: no cover - config is optional in tests
    CONFIG = None  # type: ignore[assignment]


DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "crypto_data.sqlite"

_METADATA = MetaData()
PRICES_TABLE = Table(
    "prices",
    _METADATA,
    Column("open_time", BigInteger, primary_key=True),
    Column("symbol", String(20), primary_key=True, nullable=False),
    Column("interval", String(10), primary_key=True, nullable=False),
    Column("open", Float, nullable=False),
    Column("high", Float, nullable=False),
    Column("low", Float, nullable=False),
    Column("close", Float, nullable=False),
    Column("volume", Float, nullable=False),
    Column("close_time", BigInteger, nullable=False),
    Column("quote_asset_volume", Float, nullable=False),
    Column("number_of_trades", Integer, nullable=False),
    Column("taker_buy_base", Float, nullable=False),
    Column("taker_buy_quote", Float, nullable=False),
)

_ENGINES: dict[str, Engine] = {}


def _normalise_timestamp(value: int | None) -> int | None:
    """Return ``value`` coerced to ``int`` when provided."""

    return None if value is None else int(value)


def _path_to_sqlite_url(db_path: Path) -> str:
    return f"sqlite:///{db_path.expanduser().resolve()}"


def _default_config_url() -> str:
    if CONFIG is not None:
        db_cfg = getattr(CONFIG, "database", None)
        if db_cfg is not None and getattr(db_cfg, "url", None):
            return str(db_cfg.url)
        cfg_path = getattr(CONFIG, "db_path", None)
        if cfg_path is not None:
            return _path_to_sqlite_url(Path(cfg_path))
    return _path_to_sqlite_url(DEFAULT_DB_PATH)


def _resolve_db_url(db_path: str | Path | None) -> str:
    if db_path is None:
        return _default_config_url()
    if isinstance(db_path, Path):
        return _path_to_sqlite_url(db_path)
    if "://" in db_path:
        return db_path
    return _path_to_sqlite_url(Path(db_path))


def _pool_kwargs(url: str) -> dict[str, object]:
    if CONFIG is None:
        return {"pool_pre_ping": True}
    db_cfg = getattr(CONFIG, "database", None)
    if db_cfg is None:
        return {"pool_pre_ping": True}
    kwargs: dict[str, object] = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        return kwargs
    kwargs["pool_size"] = getattr(db_cfg, "pool_size", 5)
    kwargs["max_overflow"] = getattr(db_cfg, "max_overflow", 10)
    return kwargs


def get_engine(db_path: str | Path | None = None) -> Engine:
    """Return a cached SQLAlchemy :class:`Engine` for ``db_path``."""

    url = _resolve_db_url(db_path)
    engine = _ENGINES.get(url)
    if engine is None:
        engine = create_engine(url, future=True, **_pool_kwargs(url))
        _ENGINES[url] = engine
    return engine


def init_timescale(db_path: str | Path | None = None) -> None:
    """Create the ``prices`` table and promote it to a hypertable when possible."""

    engine = get_engine(db_path)
    with engine.begin() as conn:
        _METADATA.create_all(conn)
        if engine.dialect.name == "postgresql":
            conn.execute(
                text(
                    "SELECT create_hypertable('prices', 'open_time', if_not_exists => TRUE, "
                    "migrate_data => TRUE)"
                )
            )


def _batched(
    iterable: Sequence[Sequence[object]] | Iterable[Sequence[object]], size: int
) -> Iterator[list[Sequence[object]]]:
    batch: list[Sequence[object]] = []
    for row in iterable:
        batch.append(row)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _coerce_row(row: Sequence[object], symbol: str, interval: str) -> dict[str, object]:
    return {
        "open_time": int(row[0]),
        "symbol": symbol,
        "interval": interval,
        "open": float(row[1]),
        "high": float(row[2]),
        "low": float(row[3]),
        "close": float(row[4]),
        "volume": float(row[5]),
        "close_time": int(row[6]),
        "quote_asset_volume": float(row[7]),
        "number_of_trades": int(row[8]),
        "taker_buy_base": float(row[9]),
        "taker_buy_quote": float(row[10]),
    }


def _build_insert(engine: Engine):
    if engine.dialect.name == "postgresql":
        stmt = pg_insert(PRICES_TABLE)
    elif engine.dialect.name == "sqlite":
        stmt = sqlite_insert(PRICES_TABLE)
    else:
        stmt = PRICES_TABLE.insert()
        return stmt

    key_cols = [PRICES_TABLE.c.open_time, PRICES_TABLE.c.symbol, PRICES_TABLE.c.interval]
    update_cols = {
        col.name: getattr(stmt.excluded, col.name)
        for col in PRICES_TABLE.c
        if col.name not in {"open_time", "symbol", "interval"}
    }
    return stmt.on_conflict_do_update(index_elements=key_cols, set_=update_cols)


def save_to_db(
    rows: Sequence[Sequence[object]] | Iterable[Sequence[object]],
    symbol: str,
    interval: str,
    *,
    db_path: str | Path | None = None,
    batch_size: int = 1000,
) -> None:
    """Insert or update OHLCV rows for ``symbol`` in batches."""

    engine = get_engine(db_path)
    insert_stmt = _build_insert(engine)

    iterator = iter(rows)
    try:
        first = next(iterator)
    except StopIteration:
        return

    with engine.begin() as conn:
        stream = _batched(chain([first], iterator), max(1, batch_size))
        for chunk in stream:
            payload = [_coerce_row(row, symbol, interval) for row in chunk]
            conn.execute(insert_stmt, payload)


def get_price_data(
    symbol: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
    db_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load price (and optional on-chain) data for ``symbol`` using SQLAlchemy."""

    start = _normalise_timestamp(start_ts)
    end = _normalise_timestamp(end_ts)

    engine = get_engine(db_path)
    inspector = inspect(engine)
    table_names = set(inspector.get_table_names())
    available = {col["name"] for col in inspector.get_columns(PRICES_TABLE.name)}
    cols = [col for col in PRICES_TABLE.c if col.name in available]
    if not cols:
        raise RuntimeError("Price table has no readable columns")

    stmt = select(*cols)
    if "symbol" in available:
        stmt = stmt.where(PRICES_TABLE.c.symbol == symbol)
    if start is not None and "open_time" in available:
        stmt = stmt.where(PRICES_TABLE.c.open_time >= start)
    if end is not None and "open_time" in available:
        stmt = stmt.where(PRICES_TABLE.c.open_time <= end)
    if "open_time" in available:
        stmt = stmt.order_by(PRICES_TABLE.c.open_time)

    with engine.connect() as conn:
        df = pd.read_sql(stmt, conn)

    if "open_time" in df.columns:
        df = df.rename(columns={"open_time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    drop_cols = [col for col in ("symbol", "interval", "close_time") if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df = df.sort_values("timestamp").reset_index(drop=True)

    if "derivatives_intraday" in table_names:
        df = _merge_derivatives(df, engine, symbol)
    if "orderbook_snapshots" in table_names:
        df = _merge_orderbook(df, engine, symbol)
    if "coinmetrics_exchange_flows" in table_names:
        df = _merge_coinmetrics(df, engine, symbol)
    if "glassnode_active_addresses" in table_names:
        df = _merge_glassnode_active_addresses(df, engine, symbol)
    return df


def _merge_derivatives(df: pd.DataFrame, engine: Engine, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df
    start = (df["timestamp"].min() - pd.Timedelta(hours=12)).to_pydatetime()
    end = (df["timestamp"].max() + pd.Timedelta(hours=1)).to_pydatetime()
    query = text(
        "SELECT timestamp, funding_rate, open_interest, basis "
        "FROM derivatives_intraday "
        "WHERE symbol = :symbol AND timestamp >= :start AND timestamp <= :end "
        "ORDER BY timestamp"
    )
    with engine.connect() as conn:
        deriv = pd.read_sql(
            query,
            conn,
            params={"symbol": symbol, "start": start, "end": end},
        )
    if deriv.empty:
        return df
    deriv["timestamp"] = pd.to_datetime(deriv["timestamp"], utc=True, errors="coerce")
    deriv = deriv.dropna(subset=["timestamp"]).sort_values("timestamp")
    merged = pd.merge_asof(
        df.sort_values("timestamp"),
        deriv,
        on="timestamp",
        direction="backward",
    )
    if "basis" in merged.columns:
        merged["basis_annualized"] = pd.to_numeric(merged["basis"], errors="coerce") / 10_000.0
    return merged


def _merge_orderbook(df: pd.DataFrame, engine: Engine, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df
    start = (df["timestamp"].min() - pd.Timedelta(hours=1)).to_pydatetime()
    end = (df["timestamp"].max() + pd.Timedelta(hours=1)).to_pydatetime()
    query = text(
        "SELECT timestamp, bid_price, bid_volume, ask_price, ask_volume, "
        "spread, mid_price, bid_volume_total, ask_volume_total, "
        "bid_notional_total, ask_notional_total, depth_imbalance "
        "FROM orderbook_snapshots "
        "WHERE symbol = :symbol AND timestamp >= :start AND timestamp <= :end "
        "ORDER BY timestamp"
    )
    with engine.connect() as conn:
        lob = pd.read_sql(
            query,
            conn,
            params={"symbol": symbol, "start": start, "end": end},
        )
    if lob.empty:
        return df
    lob["timestamp"] = pd.to_datetime(lob["timestamp"], utc=True, errors="coerce")
    lob = lob.dropna(subset=["timestamp"]).sort_values("timestamp")
    merged = pd.merge_asof(
        df.sort_values("timestamp"),
        lob,
        on="timestamp",
        direction="backward",
    )
    return merged


def _coinmetrics_asset(symbol: str) -> str | None:
    base = symbol.replace("USDC", "").replace("USDT", "").replace("USD", "")
    if not base:
        return None
    return base.lower()


def _merge_coinmetrics(df: pd.DataFrame, engine: Engine, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df
    asset = _coinmetrics_asset(symbol)
    if asset is None:
        return df
    start = (df["timestamp"].min() - pd.Timedelta(days=2)).to_pydatetime()
    end = (df["timestamp"].max() + pd.Timedelta(days=1)).to_pydatetime()
    query = text(
        "SELECT timestamp, onch_exchange_net_flow, onch_exchange_inflow, "
        "onch_exchange_outflow "
        "FROM coinmetrics_exchange_flows "
        "WHERE asset = :asset AND timestamp >= :start AND timestamp <= :end "
        "ORDER BY timestamp"
    )
    with engine.connect() as conn:
        flows = pd.read_sql(
            query,
            conn,
            params={"asset": asset, "start": start, "end": end},
        )
    if flows.empty:
        return df
    flows["timestamp"] = pd.to_datetime(flows["timestamp"], utc=True, errors="coerce")
    flows = flows.dropna(subset=["timestamp"]).sort_values("timestamp")
    merged = pd.merge_asof(
        df.sort_values("timestamp"),
        flows,
        on="timestamp",
        direction="backward",
    )
    return merged


def _merge_glassnode_active_addresses(df: pd.DataFrame, engine: Engine, symbol: str) -> pd.DataFrame:
    if df.empty:
        return df
    asset = symbol.replace("USDC", "").replace("USDT", "").replace("USD", "")
    if not asset:
        return df
    start = (df["timestamp"].min() - pd.Timedelta(days=2)).to_pydatetime()
    end = (df["timestamp"].max() + pd.Timedelta(days=1)).to_pydatetime()
    query = text(
        "SELECT timestamp, onch_active_addresses "
        "FROM glassnode_active_addresses "
        "WHERE asset = :asset AND timestamp >= :start AND timestamp <= :end "
        "ORDER BY timestamp"
    )
    with engine.connect() as conn:
        active = pd.read_sql(
            query,
            conn,
            params={"asset": asset, "start": start, "end": end},
        )
    if active.empty:
        return df
    active["timestamp"] = pd.to_datetime(active["timestamp"], utc=True, errors="coerce")
    active = active.dropna(subset=["timestamp"]).sort_values("timestamp")
    merged = pd.merge_asof(
        df.sort_values("timestamp"),
        active,
        on="timestamp",
        direction="backward",
    )
    return merged


def get_latest_open_time(
    *, symbol: str | None = None, interval: str | None = None, db_path: str | Path | None = None
) -> int | None:
    """Return the newest ``open_time`` value present in the prices table."""

    engine = get_engine(db_path)
    stmt = select(func.max(PRICES_TABLE.c.open_time))
    if symbol is not None:
        stmt = stmt.where(PRICES_TABLE.c.symbol == symbol)
    if interval is not None:
        stmt = stmt.where(PRICES_TABLE.c.interval == interval)

    with engine.connect() as conn:
        result = conn.execute(stmt).scalar_one_or_none()
    return int(result) if result is not None else None


__all__ = [
    "DEFAULT_DB_PATH",
    "PRICES_TABLE",
    "get_engine",
    "get_latest_open_time",
    "get_price_data",
    "init_timescale",
    "save_to_db",
]
