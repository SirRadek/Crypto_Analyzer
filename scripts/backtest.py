#!/usr/bin/env python
"""CLI helper for running quick equity backtests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from crypto_analyzer.eval.backtest import run_backtest


def _read_predictions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalise_columns(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    prediction_col: str,
    target_col: str,
    price_col: str,
) -> pd.DataFrame:
    missing = [
        column
        for column in (timestamp_col, prediction_col, target_col, price_col)
        if column not in df.columns
    ]
    if missing:
        raise KeyError("Missing required columns: " + ", ".join(sorted(missing)))

    out = df.copy()
    out = out.rename(
        columns={
            timestamp_col: "timestamp",
            prediction_col: "p_hat",
            target_col: "target",
            price_col: "last_price",
        }
    )
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp", "p_hat", "target", "last_price"])
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a simple long/short backtest")
    parser.add_argument("predictions", type=Path, help="CSV/Parquet file with model forecasts.")
    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
        help="Column containing the prediction timestamp.",
    )
    parser.add_argument(
        "--prediction-column",
        default="p_hat",
        help="Column with the model's predicted price or probability.",
    )
    parser.add_argument(
        "--target-column",
        default="target",
        help="Column with the realised target used for P&L computation.",
    )
    parser.add_argument(
        "--price-column",
        default="last_price",
        help="Reference price column used when computing trade returns.",
    )
    parser.add_argument(
        "--fee",
        type=float,
        default=0.0004,
        help="Proportional transaction cost per trade (in decimal form).",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("backtest_metrics.json"),
        help="Destination JSON file for summary metrics.",
    )
    parser.add_argument(
        "--equity-output",
        type=Path,
        default=Path("backtest_equity.csv"),
        help="Destination CSV file storing the equity curve.",
    )
    return parser


def main(argv: list[str] | None = None) -> tuple[Path, Path]:
    parser = _build_parser()
    args = parser.parse_args(argv)

    df = _read_predictions(args.predictions)
    normalised = _normalise_columns(
        df,
        timestamp_col=args.timestamp_column,
        prediction_col=args.prediction_column,
        target_col=args.target_column,
        price_col=args.price_column,
    )

    result = run_backtest(normalised, fee=args.fee)

    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.equity_output.parent.mkdir(parents=True, exist_ok=True)

    equity = result["equity"]
    metrics = result["metrics"]
    equity.to_csv(args.equity_output, index=False)
    args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        "Backtest complete. Final equity: "
        f"{float(equity['equity'].iloc[-1]):.4f}, PnL: {metrics['pnl']:.4f}"
    )
    return args.metrics_output, args.equity_output


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    main()
