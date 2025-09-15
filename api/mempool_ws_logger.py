"""Live mempool WebSocket logger.

Connects to mempool.space WebSocket API and appends five-minute aggregated
snapshots into the ``onchain_5m`` table. This module is intended to be run via
``cron`` after the historical backfill to keep the dataset up to date.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import websocket

from .backfill_onchain_history import COLUMNS, _percentile, _weighted_avg

WS_URL = "wss://mempool.space/api/v1/ws"


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
            onch_diff_progress_pct REAL,
            onch_diff_change_pct REAL,
            onch_blocks_remaining REAL,
            onch_retarget_ts INTEGER
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS ix_onchain_ts ON onchain_5m(ts_utc)")
    conn.commit()


def log_mempool_ws(db_path: str) -> None:  # pragma: no cover - network
    conn = sqlite3.connect(db_path)
    _ensure_schema(conn)

    def on_message(_ws: websocket.WebSocketApp, message: str) -> None:
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return
        mempool: dict[str, Any] | None = msg.get("mempool")
        if not isinstance(mempool, dict):
            return
        ts = pd.Timestamp(datetime.now(UTC)).floor("5min")
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
            f"INSERT OR REPLACE INTO onchain_5m ({','.join(cols)}) VALUES ({placeholders})",
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
