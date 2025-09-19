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
            onch_fee_fast_satvb REAL
        )
        """
    )
    base_ts = pd.Timestamp("2024-01-01", tz="UTC")
    periods = 10 * 24 * 4  # 10 days of 15m data
    ts = base_ts + pd.to_timedelta(np.arange(periods) * 15, unit="m")
    rng = np.random.default_rng(1234)
    df = pd.DataFrame(
        {
            "open_time": (ts.view("int64") // 1_000_000).astype(int),
            "symbol": "BTCUSDT",
            "interval": "15m",
            "open": rng.random(periods) + 10000,
            "high": rng.random(periods) + 10001,
            "low": rng.random(periods) + 9999,
            "close": rng.random(periods) + 10000,
            "volume": rng.random(periods),
            "quote_asset_volume": rng.random(periods),
            "number_of_trades": rng.integers(1, 100, size=periods),
            "taker_buy_base": rng.random(periods),
            "taker_buy_quote": rng.random(periods),
            "onch_fee_fast_satvb": rng.random(periods),
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
    run_dirs = sorted(p for p in out_dir.iterdir() if p.is_dir())
    assert run_dirs, "Expected at least one run directory"
    run_dir = run_dirs[-1]
    assert run_dir.name.startswith("run_id=")

    metadata_path = run_dir / "metadata.json"
    metrics_path = run_dir / "metrics.json"
    model_path = run_dir / "clf_model.json"
    config_snapshot_path = run_dir / "config_snapshot.yaml"

    assert metadata_path.exists()
    assert metrics_path.exists()
    assert model_path.exists()
    assert config_snapshot_path.exists()

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["task"] == task
    assert metadata["output_dir"] == str(run_dir)
    assert "random_seed" in metadata


def test_pipeline_smoke(tmp_path):
    _run_pipeline(tmp_path, "clf")
