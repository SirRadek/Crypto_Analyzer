from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from crypto_analyzer.data.maintenance import delete_old_records
from crypto_analyzer.utils.config import CONFIG


def _create_tables(db: Path) -> None:
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE prices (
                open_time INTEGER PRIMARY KEY,
                open REAL
            )
            """
        )
        cursor.execute(
            f"""
            CREATE TABLE {CONFIG.table_pred} (
                id INTEGER PRIMARY KEY,
                target_time_ms INTEGER,
                y_true_hat REAL
            )
            """
        )
        conn.commit()


def test_delete_old_records(tmp_path: Path) -> None:
    db_path = tmp_path / "data.sqlite"
    _create_tables(db_path)

    now_ms = int(time.time() * 1000)
    old_price = now_ms - 50_000
    fresh_price = now_ms + 50_000
    old_pred = now_ms - 50_000
    fresh_pred = now_ms + 50_000

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO prices(open_time, open) VALUES (?, ?)",
            [(old_price, 1.0), (fresh_price, 2.0)],
        )
        cursor.executemany(
            f"INSERT INTO {CONFIG.table_pred}(target_time_ms, y_true_hat) VALUES (?, ?)",
            [
                (old_pred, 0.1),
                (fresh_pred, 0.2),
                (now_ms, None),
            ],
        )
        conn.commit()

    deleted_prices, deleted_preds = delete_old_records(
        db_path,
        price_retention_ms=10_000,
        prediction_retention_ms=10_000,
    )
    assert deleted_prices == 1
    assert deleted_preds == 2  # one stale row + one with null y_true_hat

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        remaining_prices = cursor.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        remaining_preds = cursor.execute(
            f"SELECT COUNT(*) FROM {CONFIG.table_pred}"
        ).fetchone()[0]

    assert remaining_prices == 1
    assert remaining_preds == 1
