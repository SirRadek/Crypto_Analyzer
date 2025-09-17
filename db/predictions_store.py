import os
import sqlite3

DB_PATH = "db/data/crypto_data.sqlite"  # same DB like prices
TABLE_NAME = "predictions"


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def create_predictions_table(db_path=DB_PATH, table_name=TABLE_NAME):
    _ensure_dir(db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    cols = {r[1] for r in c.execute(f"PRAGMA table_info({table_name})")}
    if {"p_hat", "p_low", "p_high"} <= cols:
        if "abs_error" not in cols:
            c.execute(f"ALTER TABLE {table_name} ADD COLUMN abs_error REAL")

        c.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS ux_{table_name} "
            f"ON {table_name}(symbol, interval, target_time_ms)"
        )
        conn.commit()
        conn.close()
        return
    c.executescript(
        f"""
CREATE TABLE IF NOT EXISTS {table_name}_new (
  id INTEGER PRIMARY KEY,
  symbol TEXT NOT NULL,
  interval TEXT NOT NULL,
  target_time_ms INTEGER NOT NULL,
  p_hat REAL,
  p_low REAL,
  p_high REAL,
  y_true_hat REAL,
  y_true_low REAL,
  y_true_high REAL,
  abs_error REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_{table_name} ON {table_name}_new(symbol, interval, target_time_ms);
"""
    )
    if cols:
        if {"y_pred", "y_pred_low", "y_pred_high"} <= cols:
            src_extra = ", abs_error" if "abs_error" in cols else ""
            dst_extra = ", abs_error" if "abs_error" in cols else ""
            c.execute(
                f"INSERT OR IGNORE INTO {table_name}_new("
                "symbol, interval, target_time_ms, p_hat, p_low, p_high" + dst_extra + ") "
                "SELECT symbol, interval, target_time_ms, y_pred, y_pred_low, y_pred_high"
                + src_extra
                + f" FROM {table_name}"
            )
        c.execute(f"ALTER TABLE {table_name} RENAME TO {table_name}_backup")
    c.execute(f"ALTER TABLE {table_name}_new RENAME TO {table_name}")
    conn.commit()
    conn.close()


def save_predictions(rows, db_path=DB_PATH, table_name=TABLE_NAME):
    if not rows:
        return
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.executemany(
        f"""
        INSERT INTO {table_name} (symbol, interval, target_time_ms, p_hat, p_low, p_high)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, interval, target_time_ms)
        DO UPDATE SET
          p_hat=excluded.p_hat,
          p_low=excluded.p_low,
          p_high=excluded.p_high
        """,
        rows,
    )
    conn.commit()
    conn.close()


