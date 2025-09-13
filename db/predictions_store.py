import os
import sqlite3

DB_PATH = "db/data/crypto_data.sqlite"  # same DB like prices
TABLE_NAME = "prediction"  # new table for predictions


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def create_predictions_table(db_path=DB_PATH, table_name=TABLE_NAME):
    _ensure_dir(db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        interval TEXT NOT NULL,
        horizon_steps INTEGER NOT NULL,
        prediction_time_ms INTEGER NOT NULL,  -- t (time)
        target_time_ms INTEGER NOT NULL,      -- t+H (time close)
        prediction_time_cet TEXT,             -- Prague time (human readable)
        target_time_cet TEXT,                 -- Prague time for target
        p_hat REAL NOT NULL,                  -- central predicted price
        p_low REAL,
        p_high REAL,
        y_true REAL,                          -- fill up at backfill
        abs_error REAL,                       -- |p_hat - y_true|
        model_name TEXT,
        features_version TEXT,
        created_at TEXT NOT NULL
    );
    """
    )
    # Ensure new columns exist or rename legacy ones
    cols = {r[1] for r in c.execute(f"PRAGMA table_info({table_name})")}
    rename_map = {
        "y_pred": "p_hat",
        "y_pred_low": "p_low",
        "y_pred_high": "p_high",
    }
    for old, new in rename_map.items():
        if old in cols and new not in cols:
            c.execute(f"ALTER TABLE {table_name} RENAME COLUMN {old} TO {new}")
    cols = {r[1] for r in c.execute(f"PRAGMA table_info({table_name})")}
    for col, coltype in (
        ("prediction_time_cet", "TEXT"),
        ("target_time_cet", "TEXT"),
        ("p_low", "REAL"),
        ("p_hat", "REAL"),
        ("p_high", "REAL"),
    ):
        if col not in cols:
            c.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {coltype}")
    # useful indexes
    c.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name}(symbol)")
    c.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_target_time ON {table_name}(target_time_ms)"
    )
    c.execute(
        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_pred_time "
        f"ON {table_name}(symbol, interval, horizon_steps, prediction_time_ms)"
    )
    c.execute(
        f"CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_uniq "
        f"ON {table_name}(symbol, interval, target_time_ms)"
    )
    conn.commit()
    conn.close()


def save_predictions(rows, db_path=DB_PATH, table_name=TABLE_NAME):
    if not rows:
        return
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    row_len = len(rows[0])
    prepped: list[tuple] = []
    if row_len == 13:
        for r in rows:
            r = list(r)
            p_hat = r[6]
            y_true = r[8]
            if y_true is not None:
                r[9] = abs(p_hat - y_true)
            prepped.append(tuple(r))
        sql = f'''
        INSERT INTO {table_name} (
            symbol, interval, horizon_steps, prediction_time_ms, target_time_ms,
            p_low, p_hat, p_high, y_true, abs_error, model_name, features_version, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, interval, target_time_ms) DO UPDATE SET
            y_true = COALESCE({table_name}.y_true, excluded.y_true),
            abs_error = CASE
                WHEN {table_name}.y_true IS NULL AND excluded.y_true IS NOT NULL THEN excluded.abs_error
                ELSE {table_name}.abs_error
            END
        '''
    elif row_len == 15:
        for r in rows:
            r = list(r)
            p_hat = r[8]
            y_true = r[10]
            if y_true is not None:
                r[11] = abs(p_hat - y_true)
            prepped.append(tuple(r))
        sql = f'''
        INSERT INTO {table_name} (
            symbol, interval, horizon_steps, prediction_time_ms, target_time_ms,
            prediction_time_cet, target_time_cet,
            p_low, p_hat, p_high, y_true, abs_error, model_name, features_version, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, interval, target_time_ms) DO UPDATE SET
            y_true = COALESCE({table_name}.y_true, excluded.y_true),
            abs_error = CASE
                WHEN {table_name}.y_true IS NULL AND excluded.y_true IS NOT NULL THEN excluded.abs_error
                ELSE {table_name}.abs_error
            END
        '''
    else:
        conn.close()
        raise ValueError(f"Unexpected row length for save_predictions: {row_len}")
    c.executemany(sql, prepped)
    conn.commit()
    conn.close()


def delete_unmatched_duplicates(db_path: str = DB_PATH, table_name: str = TABLE_NAME) -> int:
    """Remove older duplicate predictions lacking ground truth.

    A unique index on ``(symbol, interval, target_time_ms)`` prevents new
    duplicates, but legacy data may still contain them.  This helper keeps the
    most recent row for such combinations where ``y_true`` is still ``NULL``.
    """

    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        f"""
        WITH ranked AS (
            SELECT
                id,
                y_true,
                ROW_NUMBER() OVER (
                    PARTITION BY symbol, interval, target_time_ms
                    ORDER BY id DESC
                ) AS rn
            FROM {table_name}
        )
        DELETE FROM {table_name}
        WHERE id IN (
            SELECT id FROM ranked WHERE rn > 1 AND y_true IS NULL
        )
        """
    )
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return deleted
