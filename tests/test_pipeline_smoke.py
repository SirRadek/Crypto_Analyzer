import json
import os
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


def _create_synth_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE prices (
            open_time INTEGER PRIMARY KEY,
            symbol TEXT,
            interval TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            quote_asset_volume REAL,
            number_of_trades INTEGER,
            taker_buy_base REAL,
            taker_buy_quote REAL,
            onch_dummy REAL
        )
        """
    )
    base_ts = pd.Timestamp("2024-01-01", tz="UTC")
    periods = 10 * 24 * 12  # 10 days of 5m data
    ts = base_ts + pd.to_timedelta(np.arange(periods) * 5, unit="m")
    df = pd.DataFrame(
        {
            "open_time": (ts.view("int64") // 1_000_000).astype(int),
            "symbol": "BTCUSDT",
            "interval": "5m",
            "open": np.random.rand(periods) + 10000,
            "high": np.random.rand(periods) + 10001,
            "low": np.random.rand(periods) + 9999,
            "close": np.random.rand(periods) + 10000,
            "volume": np.random.rand(periods),
            "quote_asset_volume": np.random.rand(periods),
            "number_of_trades": np.random.randint(1, 100, size=periods),
            "taker_buy_base": np.random.rand(periods),
            "taker_buy_quote": np.random.rand(periods),
            "onch_dummy": np.random.rand(periods),
        }
    )
    df.to_sql("prices", conn, if_exists="append", index=False)
    conn.close()


def _run_pipeline(tmp_path: Path, task: str) -> None:
    db_path = tmp_path / "data.sqlite"
    if not db_path.exists():
        _create_synth_db(db_path)
    out_dir = tmp_path / "outputs"
    env = os.environ.copy()
    env["DB_PATH"] = str(db_path)
    cmd = [
        "python",
        "main.py",
        "--task",
        task,
        "--horizon",
        "120",
        "--split_params",
        json.dumps({"test_size": 0.2}),
        "--out_dir",
        str(out_dir),
        "--use_onchain",
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parent.parent, env=env)
    assert (out_dir / "run_config.json").exists()
    assert (out_dir / f"{task}_model.json").exists()


def test_pipeline_smoke(tmp_path):
    _run_pipeline(tmp_path, "clf")
    _run_pipeline(tmp_path, "reg")
