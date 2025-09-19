"""Utilities for maintaining the SQLite dataset."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from crypto_analyzer.utils.config import CONFIG

ONE_YEAR_MS = 365 * 24 * 60 * 60 * 1000
SIX_HOURS_MS = 6 * 60 * 60 * 1000


def delete_old_records(db_path: str | Path = CONFIG.db_path) -> tuple[int, int]:
    """Delete stale rows from the prices and prediction tables."""

    now_ms = int(time.time() * 1000)
    prices_before = now_ms - ONE_YEAR_MS
    preds_before = now_ms - SIX_HOURS_MS

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        cur.execute("DELETE FROM prices WHERE open_time <= ?", (prices_before,))
        prices_deleted = cur.rowcount if cur.rowcount is not None else 0

        cur.execute(
            f"DELETE FROM {CONFIG.table_pred} WHERE y_true_hat IS NULL",
            (preds_before,),
        )
        preds_deleted = cur.rowcount if cur.rowcount is not None else 0

        conn.commit()
        cur.execute("PRAGMA optimize;")
        conn.commit()

    return prices_deleted, preds_deleted


__all__ = ["delete_old_records"]
