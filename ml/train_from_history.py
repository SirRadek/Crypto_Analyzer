"""Iterative training on historical data in fixed steps.

This script retrains the price regressor on progressively larger slices of
historical data.  Starting from ``start`` it advances in ``step_minutes``
increments up to ``end``.  After each step the model is retrained using all
available data up to the current cut-off and optionally evaluated on the next
chunk.  The goal is to simulate learning from past mistakes while marching
towards the present.
"""

from __future__ import annotations

import argparse
import csv
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.feature_engineering import FEATURE_COLUMNS, create_features
from db.db_connector import get_price_data
from ml.train_regressor import train_regressor


def train_from_history(
    symbol: str,
    start: str,
    end: str,
    *,
    step_minutes: int = 120,
    horizon_steps: int = 24,
    use_gpu: bool = False,
) -> None:
    """Retrain the regressor in a loop over historical data.

    Parameters
    ----------
    symbol:
        Trading pair to load from the database.
    start, end:
        Time range (ISO format) to iterate over.
    step_minutes:
        Size of each training/evaluation step in minutes (default 2h).
    horizon_steps:
        Forward steps for the regression target.
    use_gpu:
        Whether to enable GPU training if available.
    """

    df = get_price_data(symbol)
    df = create_features(df)
    df["target"] = df["close"].shift(-horizon_steps)
    df = df.dropna(subset=["target"])  # drop rows without target

    current = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    step = timedelta(minutes=step_minutes)

    log_dir = Path("logs/runs")
    log_dir.mkdir(parents=True, exist_ok=True)
    run_file = log_dir / f"history_{symbol}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with run_file.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cutoff", "mae"])

        while current + step <= end_ts:
            train_df = df[df["timestamp"] <= current]
            if len(train_df) < 10:
                current += step
                continue
            X = train_df[FEATURE_COLUMNS]
            y = train_df["target"]
            try:
                model = train_regressor(X, y, use_gpu=use_gpu)
            except MemoryError:
                model = train_regressor(X, y, use_gpu=False, params={"nthread": 1})
                use_gpu = False

            # Optional evaluation on the next window to gauge progress
            eval_mask = (df["timestamp"] > current) & (df["timestamp"] <= current + step)
            eval_df = df.loc[eval_mask]
            err = None
            if not eval_df.empty:
                preds = model.predict(eval_df[FEATURE_COLUMNS])
                err = float(np.abs(preds - eval_df["target"]).mean())
                print(f"{current} → MAE={err:.4f}")
            writer.writerow([current.isoformat(), "" if err is None else f"{err:.6f}"])
            fh.flush()

            current += step


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Iterative historic training")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--step-minutes", type=int, default=120)
    parser.add_argument("--horizon-steps", type=int, default=24)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args(argv)

    train_from_history(
        args.symbol,
        args.start,
        args.end,
        step_minutes=args.step_minutes,
        horizon_steps=args.horizon_steps,
        use_gpu=args.use_gpu,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
