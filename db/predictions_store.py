from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Iterable, Sequence

DB_PATH = Path("db/data/crypto_data.sqlite")  # same DB like prices
TABLE_NAME = "predictions"


def _ensure_dir(path: Path | str) -> None:
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)


def _table_columns(cursor: sqlite3.Cursor, table_name: str) -> set[str]:
    """Return existing column names for ``table_name``."""

    return {row[1] for row in cursor.execute(f"PRAGMA table_info({table_name})")}


def create_predictions_table(
    db_path: Path | str = DB_PATH, table_name: str = TABLE_NAME
) -> None:
    _ensure_dir(db_path)
    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()
        cols = _table_columns(cursor, table_name)
        current = {
            "symbol",
            "interval",
            "target_time_ms",
            "p_hat",
            "prob_move_ge_05",
            "y_true_hat",
            "abs_error",
        }
        has_legacy_bounds = {"p_low", "p_high", "y_true_low", "y_true_high"} & cols
        if current <= cols and not has_legacy_bounds:
            cursor.execute(
                f"CREATE UNIQUE INDEX IF NOT EXISTS ux_{table_name} "
                f"ON {table_name}(symbol, interval, target_time_ms)"
            )
            conn.commit()
            return

        cursor.executescript(
            f"""
CREATE TABLE IF NOT EXISTS {table_name}_new (
  id INTEGER PRIMARY KEY,
  symbol TEXT NOT NULL,
  interval TEXT NOT NULL,
  target_time_ms INTEGER NOT NULL,
  p_hat REAL,
  prob_move_ge_05 REAL,
  y_true_hat REAL,
  abs_error REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_{table_name} ON {table_name}_new(symbol, interval, target_time_ms);
"""
        )
        if cols:
            select_cols = ["symbol", "interval", "target_time_ms"]
            if "p_hat" in cols:
                select_cols.append("p_hat")
            elif "y_pred" in cols:
                select_cols.append("y_pred AS p_hat")
            else:
                select_cols.append("NULL AS p_hat")
            if "prob_move_ge_05" in cols:
                select_cols.append("prob_move_ge_05")
            else:
                select_cols.append("NULL AS prob_move_ge_05")
            if "y_true_hat" in cols:
                select_cols.append("y_true_hat")
            else:
                select_cols.append("NULL AS y_true_hat")
            if "abs_error" in cols:
                select_cols.append("abs_error")
            else:
                select_cols.append("NULL AS abs_error")
            cursor.execute(
                f"INSERT OR IGNORE INTO {table_name}_new("
                "symbol, interval, target_time_ms, p_hat, prob_move_ge_05, y_true_hat, abs_error"
                ") SELECT "
                + ", ".join(select_cols)
                + f" FROM {table_name}"
            )
            cursor.execute(f"ALTER TABLE {table_name} RENAME TO {table_name}_backup")
        cursor.execute(f"ALTER TABLE {table_name}_new RENAME TO {table_name}")
        conn.commit()


def save_predictions(
    rows: Iterable[Sequence[object]],
    db_path: Path | str = DB_PATH,
    table_name: str = TABLE_NAME,
) -> None:
    values = list(rows)
    if not values:
        return

    with sqlite3.connect(str(db_path)) as conn:
        cursor = conn.cursor()
        cursor.executemany(
            f"""
            INSERT INTO {table_name} (symbol, interval, target_time_ms, p_hat, prob_move_ge_05)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol, interval, target_time_ms)
            DO UPDATE SET
              p_hat=excluded.p_hat,
              prob_move_ge_05=excluded.prob_move_ge_05
            """,
            values,
        )
        conn.commit()
