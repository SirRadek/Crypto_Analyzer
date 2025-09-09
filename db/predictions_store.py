import os
import sqlite3

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
        prediction_time_cet TEXT,             -- Prague time (human readable)
        target_time_cet TEXT,                 -- Prague time for target
        y_pred REAL NOT NULL,                 -- predict price
        y_pred_low REAL,                      -- min across ensemble
        y_pred_high REAL,                     -- max across ensemble
        y_true REAL,                          -- fill up at backfill
        abs_error REAL,                       -- |y_pred - y_true|
        model_name TEXT,
        features_version TEXT,
        created_at TEXT NOT NULL
    );
    """)
    # Ensure new columns exist if table was created previously without them
    cols = {r[1] for r in c.execute(f"PRAGMA table_info({table_name})")}
    for col, coltype in (
        ("prediction_time_cet", "TEXT"),
        ("target_time_cet", "TEXT"),
        ("y_pred_low", "REAL"),
        ("y_pred_high", "REAL"),
    ):
        if col not in cols:
            c.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {coltype}")
    # useful indexes
    c.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name}(symbol)")
    c.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_target_time ON {table_name}(target_time_ms)")
    conn.commit()
    conn.close()



def save_predictions(rows, db_path=DB_PATH, table_name=TABLE_NAME):
    if not rows:
        return
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    row_len = len(rows[0])
    if row_len == 11:
        c.executemany(f"""
          INSERT INTO {table_name} (
            symbol, interval, horizon_steps, prediction_time_ms, target_time_ms,
            y_pred, y_true, abs_error, model_name, features_version, created_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
    elif row_len == 13 and isinstance(rows[0][5], (int, float)):
        c.executemany(f"""
          INSERT INTO {table_name} (
            symbol, interval, horizon_steps, prediction_time_ms, target_time_ms,
            y_pred, y_pred_low, y_pred_high,
            y_true, abs_error, model_name, features_version, created_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
    elif row_len == 13:
        c.executemany(f"""
          INSERT INTO {table_name} (
            symbol, interval, horizon_steps, prediction_time_ms, target_time_ms,
            prediction_time_cet, target_time_cet,
            y_pred, y_true, abs_error, model_name, features_version, created_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
    elif row_len == 15:
        c.executemany(f"""
          INSERT INTO {table_name} (
            symbol, interval, horizon_steps, prediction_time_ms, target_time_ms,
            prediction_time_cet, target_time_cet,
            y_pred, y_pred_low, y_pred_high,
            y_true, abs_error, model_name, features_version, created_at
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
    else:
        raise ValueError(f"Unexpected row length for save_predictions: {row_len}")
    conn.commit()
    conn.close()