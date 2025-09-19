"""Live mempool WebSocket logger.

Connects to mempool.space WebSocket API and appends interval-aligned aggregated
snapshots into the configured on-chain table. This module is intended to be run via
``cron`` after the historical backfill to keep the dataset up to date.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import websocket

from utils.config import CONFIG
from utils.timeframes import interval_to_pandas_freq

from .backfill_onchain_history import (
    COLUMNS,
    _percentile,
    _weighted_avg,
    ensure_onchain_schema,
)

WS_URL = "wss://mempool.space/api/v1/ws"
ONCHAIN_TABLE = CONFIG.database.onchain_table
FLOOR_FREQ = interval_to_pandas_freq(CONFIG.interval)


def log_mempool_ws(db_path: str) -> None:  # pragma: no cover - network
    conn = sqlite3.connect(db_path)
    ensure_onchain_schema(conn)
    conn.commit()

    def on_message(_ws: websocket.WebSocketApp, message: str) -> None:
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return
        mempool: dict[str, Any] | None = msg.get("mempool")
        if not isinstance(mempool, dict):
            return
        ts = pd.Timestamp(datetime.now(UTC)).floor(FLOOR_FREQ)
        hist = mempool.get("fee_histogram", [])
        record = {
            "ts_utc": int(ts.timestamp()),
            "onch_mempool_count": mempool.get("count"),
            "onch_mempool_vsize_vB": mempool.get("vsize"),
            "onch_mempool_total_fee_sat": mempool.get("total_fee"),
        }
        if hist:
            record["onch_fee_wavg_satvb"] = _weighted_avg(hist)
            record["onch_fee_p50_satvb"] = _percentile(hist, 0.5)
            record["onch_fee_p90_satvb"] = _percentile(hist, 0.9)
        cols = ["ts_utc"] + [c for c in COLUMNS if c in record]
        placeholders = ",".join(["?"] * len(cols))
        values = [record.get(c) for c in cols]
        conn.execute(
            f"INSERT OR REPLACE INTO {ONCHAIN_TABLE} ({','.join(cols)}) VALUES ({placeholders})",
            values,
        )
        conn.commit()

    ws = websocket.WebSocketApp(WS_URL, on_message=on_message)
    ws.run_forever()


__all__ = ["log_mempool_ws"]


if __name__ == "__main__":  # pragma: no cover - manual execution
    import argparse

    parser = argparse.ArgumentParser(description="Live mempool WebSocket logger")
    parser.add_argument("--db", required=True, help="SQLite database path")
    args = parser.parse_args()
    log_mempool_ws(args.db)
