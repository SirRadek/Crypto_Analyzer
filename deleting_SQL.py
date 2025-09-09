"""Utility for purging old data from the SQLite database."""

from __future__ import annotations

import sqlite3
import time

from utils.config import CONFIG

TWO_YEARS_MS = 2 * 365 * 24 * 60 * 60 * 1000
ONE_YEAR_MS = 1 * 365 * 24 * 60 * 60 * 1000
HALF_YEAR_MS = 0,5 * 365 * 24 * 60 * 60 * 1000
SIX_HOURS_MS = 6 * 60 * 60 * 1000


def delete_old_records(db_path: str = CONFIG.db_path) -> tuple[int, int]:
    """Delete stale rows from the prices and prediction tables.

    Returns:
        Tuple[int, int]: counts of deleted rows from prices and prediction.
    """

    now_ms = int(time.time() * 1000)
    prices_before = now_ms - HALF_YEAR_MS
    preds_before = now_ms - SIX_HOURS_MS

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()

        cur.execute("DELETE FROM prices WHERE open_time <= ?", (prices_before,))
        prices_deleted = cur.rowcount if cur.rowcount is not None else 0

        cur.execute(
            "DELETE FROM prediction WHERE target_time_ms <= ? AND y_true IS NOT NULL",
            (preds_before,),
        )
        preds_deleted = cur.rowcount if cur.rowcount is not None else 0

        conn.commit()
        cur.execute("PRAGMA optimize;")
        conn.commit()

    return prices_deleted, preds_deleted


__all__ = ["delete_old_records"]

