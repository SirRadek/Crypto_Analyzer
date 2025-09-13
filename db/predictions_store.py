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
  y_true_high REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_{table_name} ON {table_name}_new(symbol, interval, target_time_ms);
"""
    )
    if cols:
        c.execute(
            f"INSERT OR IGNORE INTO {table_name}_new(symbol, interval, target_time_ms, p_hat, p_low, p_high) "
            f"SELECT symbol, interval, target_time_ms, y_pred, y_pred_low, y_pred_high FROM {table_name}"
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


def delete_unmatched_duplicates(db_path: str = DB_PATH, table_name: str = TABLE_NAME) -> int:
    """Remove older duplicate predictions lacking ground truth.

    For entries where ``y_true`` is ``NULL`` we may occasionally store the same
    prediction multiple times (e.g. repeated model runs).  Such duplicates
    cannot be paired with a true value yet and only waste space.  This helper
    keeps the most recent row for each
    ``(symbol, interval, horizon_steps, prediction_time_ms)`` combination and
    deletes the rest.

    Parameters
    ----------
    db_path : str, optional
        Path to the SQLite database.
    table_name : str, optional
        Name of the predictions table.

    Returns
    -------
    int
        Number of deleted rows.
    """

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_pred_time "
        f"ON {table_name}(symbol, interval, target_time_ms)"
    )
    c.execute(
        f"""
        WITH ranked AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY symbol, interval, horizon_steps, prediction_time_ms
                    ORDER BY id DESC
                ) AS rn
            FROM {table_name}
            WHERE y_true_hat IS NULL
        )
        DELETE FROM {table_name}
        WHERE id IN (
            SELECT id FROM ranked WHERE rn > 1
        )
        """
    )
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return deleted
