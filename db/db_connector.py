import os
import sqlite3

import pandas as pd

# Resolve database path relative to this file so callers don't depend on CWD
DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "crypto_data.sqlite")


def get_price_data(
    symbol,
    start_ts: int | None = None,
    end_ts: int | None = None,
    db_path: str = DEFAULT_DB_PATH,
):
    """Load price (and optional on-chain) data for ``symbol`` from SQLite.

    The function selects all columns from the ``prices`` table so that any
    additional features (e.g. on-chain metrics prefixed ``onch_``) are loaded as
    well.  The ``open_time`` column is renamed to ``timestamp`` and converted to
    a timezone-aware ``datetime``.  Non-price helper columns such as ``symbol``
    and ``interval`` are dropped if present.
    """

    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM prices WHERE symbol = ?"
    params = [symbol]
    if start_ts is not None:
        query += " AND open_time >= ?"
        params.append(int(start_ts))
    if end_ts is not None:
        query += " AND open_time <= ?"
        params.append(int(end_ts))
    query += " ORDER BY open_time"

    df = pd.read_sql(query, conn, params=params)
    conn.close()

    if "open_time" in df.columns:
        df.rename(columns={"open_time": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    for col in ["symbol", "interval"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df
