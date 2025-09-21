#!/usr/bin/env python
"""CLI helper for running quick equity backtests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from crypto_analyzer.eval.backtest import run_backtest
from crypto_analyzer.eval.threshold_sweep import SweepParams, sweep_threshold_grid
from crypto_analyzer.utils.config import CONFIG


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
        "--prob-column",
        default=None,
        help="Optional probability column for EV-based backtest rules.",
    )
    parser.add_argument(
        "--fee_bps",
        "--fee-bps",
        dest="fee_bps",
        type=float,
        default=4.0,
        help="Proportional transaction cost per trade in basis points.",
    )
    parser.add_argument(
        "--slip_bps",
        "--slip-bps",
        dest="slip_bps",
        type=float,
        default=0.0,
        help="Slippage assumption in basis points added to the fee.",
    )
    parser.add_argument(
        "--latency_min",
        "--latency-min",
        dest="latency_min",
        type=float,
        default=0.0,
        help="Execution latency in minutes applied by shifting the entry candle forward.",
    )
    parser.add_argument(
        "--p_touch_thr",
        "--p-touch-thr",
        dest="p_touch_thr",
        type=float,
        default=None,
        help="Minimum probability of touching the target required to open a trade.",
    )
    parser.add_argument(
        "--p_up_thr",
        "--p-up-thr",
        dest="p_up_thr",
        type=float,
        default=None,
        help=(
            "Directional probability threshold. Values above open longs, below (1-thr) open shorts."
        ),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional identifier used when storing reports. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--optimize-thresholds",
        action="store_true",
        help="Run a quick EV sweep across default threshold grids before executing the backtest.",
    )
    return parser


def _infer_latency_steps(timestamps: pd.Series, latency_minutes: float) -> int:
    if latency_minutes <= 0:
        return 0

    if timestamps.empty:
        return 0

    ordered = timestamps.sort_values().reset_index(drop=True)
    diffs = ordered.diff().dropna()
    if diffs.empty:
        return 0

    step_minutes = diffs.dt.total_seconds().median() / 60.0
    if not np.isfinite(step_minutes) or step_minutes <= 0:
        return 0

    steps = int(round(latency_minutes / step_minutes))
    if steps <= 0:
        steps = 1
    return steps


def _find_horizon_column(df: pd.DataFrame) -> str | None:
    for column in ("horizon", "horizon_min", "horizon_minutes"):
        if column in df.columns:
            return column
    return None


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

    latency_steps = _infer_latency_steps(normalised["timestamp"], args.latency_min)

    prob_col: Optional[str] = args.prob_column or None
    if prob_col is not None and prob_col == args.prediction_column:
        prob_col = "p_hat"
    p_up_col = "p_up" if "p_up" in normalised.columns else None

    if args.optimize_thresholds:
        horizon_col = _find_horizon_column(normalised)
        touch_grid = np.round(np.arange(0.5, 0.8001, 0.05), 4)
        up_grid = np.round(np.arange(0.5, 0.7001, 0.05), 4)
        params = SweepParams(
            p_touch_values=touch_grid,
            p_up_values=up_grid,
            fee_bps=args.fee_bps,
            slippage_bps=args.slip_bps,
            latency_steps=latency_steps,
            p_touch_col="p_hat",
            p_up_col=p_up_col,
        )
        frames: list[pd.DataFrame] = []
        if horizon_col is not None:
            values = normalised[horizon_col].dropna().unique()
            for value in sorted(values):
                subset = normalised[normalised[horizon_col] == value]
                if subset.empty:
                    continue
                frames.append(sweep_threshold_grid(subset, params=params, horizon_label=value))
        if not frames:
            frames.append(sweep_threshold_grid(normalised, params=params, horizon_label=None))
        preview = pd.concat(frames, ignore_index=True)
        top_preview = preview.sort_values("ev", ascending=False).head(5)
        print("Threshold sweep preview (top 5 by EV):")
        print(top_preview[["horizon", "p_touch_thr", "p_up_thr", "ev", "pnl", "sharpe", "trades"]])

    result = run_backtest(
        normalised,
        fee_bps=args.fee_bps,
        slippage_bps=args.slip_bps,
        prob_col=prob_col,
        latency_steps=latency_steps,
        p_touch_col="p_hat" if args.p_touch_thr is not None else None,
        p_touch_threshold=args.p_touch_thr,
        p_up_col=p_up_col,
        p_up_threshold=args.p_up_thr,
    )

    run_id = args.run_id or pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs") / f"run_id={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    equity = result["equity"].assign(run_id=run_id)
    metrics = result["metrics"]

    summary_metrics = {
        "ev": metrics.get("ev"),
        "hit_rate": metrics.get("hit_rate"),
        "sharpe": metrics.get("sharpe"),
        "maxDD": metrics.get("maxDD", metrics.get("max_drawdown")),
    }

    for key, value in summary_metrics.items():
        display: str
        if value is None:
            display = "nan"
        else:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                display = str(value)
            else:
                if not np.isfinite(numeric_value):
                    display = "nan"
                else:
                    display = f"{numeric_value:.6f}"
        print(f"{key}: {display}")

    equity_output = reports_dir / f"equity_{run_id}.csv"
    summary_output = reports_dir / f"summary_{run_id}.json"

    equity.to_csv(equity_output, index=False)
    equity.to_csv(run_dir / "equity.csv", index=False)

    summary = {
        "run_id": run_id,
        "parameters": {
            "fee_bps": args.fee_bps,
            "slip_bps": args.slip_bps,
            "latency_minutes": args.latency_min,
            "p_touch_threshold": args.p_touch_thr,
            "p_up_threshold": args.p_up_thr,
            "prob_column": prob_col,
            "latency_steps": latency_steps,
        },
        "metrics": summary_metrics,
    }
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    config_dump = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "config": CONFIG.config_path.as_posix() if CONFIG.config_path else None,
    }
    (run_dir / "config_dump.json").write_text(json.dumps(config_dump, indent=2), encoding="utf-8")

    print(
        "Backtest complete. Final equity: "
        f"{float(equity['equity'].iloc[-1]):.4f}, PnL: {metrics['pnl']:.4f}, "
        f"Sharpe: {metrics['sharpe']:.4f}, EV: {metrics['ev']:.6f}, "
        f"MaxDD: {metrics['max_drawdown']:.4f}"
    )

    return summary_output, equity_output


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    main()
