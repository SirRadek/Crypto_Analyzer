from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Iterable

import pandas as pd

# Resolve database path relative to the project root so callers don't depend on CWD
DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "data" / "crypto_data.sqlite"


def _normalise_timestamp(value: int | None) -> int | None:
    """Return ``value`` coerced to ``int`` when provided."""

    return None if value is None else int(value)


def _build_price_query(start_ts: int | None, end_ts: int | None) -> tuple[str, list[int]]:
    """Return parametrised SQL query and params for the prices lookup."""

    query = "SELECT * FROM prices WHERE symbol = ?"
    params: list[int] = []
    if start_ts is not None:
        query += " AND open_time >= ?"
        params.append(start_ts)
    if end_ts is not None:
        query += " AND open_time <= ?"
        params.append(end_ts)
    query += " ORDER BY open_time"
    return query, params


def get_price_data(
    symbol: str,
    start_ts: int | None = None,
    end_ts: int | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Load price (and optional on-chain) data for ``symbol`` from SQLite.

    The function selects all columns from the ``prices`` table so that any
    additional features (e.g. on-chain metrics prefixed ``onch_``) are loaded as
    well.  The ``open_time`` column is renamed to ``timestamp`` and converted to
    a timezone-aware ``datetime``.  Non-price helper columns such as ``symbol``
    and ``interval`` are dropped if present.
    """

    start = _normalise_timestamp(start_ts)
    end = _normalise_timestamp(end_ts)
    query, extra_params = _build_price_query(start, end)
    params: list[object] = [symbol, *extra_params]

    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql(query, conn, params=params)

    if "open_time" in df.columns:
        df = df.rename(columns={"open_time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    drop_cols: Iterable[str] = (col for col in ("symbol", "interval") if col in df.columns)
    if drop := list(drop_cols):
        df = df.drop(columns=drop)

    return df
