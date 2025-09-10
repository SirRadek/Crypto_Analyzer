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
    conn = sqlite3.connect(db_path)
    query = (
        "SELECT "
        "open_time as timestamp, open, high, low, close, volume, "
        "number_of_trades, quote_asset_volume, taker_buy_base, taker_buy_quote "
        "FROM prices WHERE symbol = ?"
    )
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
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
