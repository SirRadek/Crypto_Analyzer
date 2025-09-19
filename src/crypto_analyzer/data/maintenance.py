"""Database maintenance utilities."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterable

from crypto_analyzer.utils.config import CONFIG, AppConfig

DEFAULT_PRICE_RETENTION_MS = 365 * 24 * 60 * 60 * 1000
DEFAULT_PREDICTION_RETENTION_MS = 6 * 60 * 60 * 1000

__all__ = ["delete_old_records"]


def _table_columns(cursor: sqlite3.Cursor, table: str) -> set[str]:
    """Return the column names of ``table``."""

    return {row[1] for row in cursor.execute(f"PRAGMA table_info({table})")}


def _delete_prices(cursor: sqlite3.Cursor, threshold_ms: int) -> int:
    cursor.execute("DELETE FROM prices WHERE open_time <= ?", (threshold_ms,))
    return max(cursor.rowcount or 0, 0)


def _delete_predictions(
    cursor: sqlite3.Cursor,
    table: str,
    threshold_ms: int,
) -> int:
    columns = _table_columns(cursor, table)
    if "target_time_ms" in columns:
        query = f"DELETE FROM {table} WHERE target_time_ms <= ? OR y_true_hat IS NULL"
        params: Iterable[int] = (threshold_ms,)
    else:
        query = f"DELETE FROM {table} WHERE y_true_hat IS NULL"
        params = ()
    cursor.execute(query, params)
    return max(cursor.rowcount or 0, 0)


def delete_old_records(
    db_path: str | Path | None = None,
    *,
    price_retention_ms: int = DEFAULT_PRICE_RETENTION_MS,
    prediction_retention_ms: int = DEFAULT_PREDICTION_RETENTION_MS,
    config: AppConfig | None = None,
) -> tuple[int, int]:
    """Delete stale rows from the prices and predictions tables."""

    cfg = config or CONFIG
    db = Path(db_path) if db_path is not None else Path(cfg.db_path)
    now_ms = int(time.time() * 1000)
    prices_before = now_ms - int(price_retention_ms)
    preds_before = now_ms - int(prediction_retention_ms)

    with sqlite3.connect(str(db)) as conn:
        cursor = conn.cursor()

        prices_deleted = _delete_prices(cursor, prices_before)
        preds_deleted = _delete_predictions(cursor, cfg.table_pred, preds_before)

        conn.commit()
        cursor.execute("PRAGMA optimize;")
        conn.commit()

    return prices_deleted, preds_deleted
