import os
import sqlite3
from datetime import datetime

DB_PATH = "db/data/crypto_data.sqlite"   # same DB like prices
TABLE_NAME = "prediction"                # new table for predictions

def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def create_predictions_table(db_path=DB_PATH, table_name=TABLE_NAME):
    _ensure_dir(db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        interval TEXT NOT NULL,
        horizon_steps INTEGER NOT NULL,
        prediction_time_ms INTEGER NOT NULL,  -- t (time)
        target_time_ms INTEGER NOT NULL,      -- t+H (time close)
        y_pred REAL NOT NULL,                 -- predict price 
        y_true REAL,                          -- fill up at backfill
        abs_error REAL,                       -- |y_pred - y_true|
        model_name TEXT,
        features_version TEXT,
        created_at TEXT NOT NULL
    );
    """)
    # užitečné indexy
    c.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name}(symbol)")
    c.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_target_time ON {table_name}(target_time_ms)")
    conn.commit()
    conn.close()

def save_predictions(rows, db_path=DB_PATH, table_name=TABLE_NAME):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.executemany(f"""
      INSERT INTO {table_name} (
        symbol, interval, horizon_steps, prediction_time_ms, target_time_ms,
        y_pred, y_true, abs_error, model_name, features_version, created_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()
