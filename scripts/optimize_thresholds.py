#!/usr/bin/env python
"""Grid-search utility for optimising backtest probability thresholds."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

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
    return max(steps, 1)


def _build_grid(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("Step must be positive")
    values = np.arange(start, stop + step * 0.5, step)
    if values.size == 0:
        raise ValueError("Invalid grid range")
    values = np.clip(values, start, stop)
    return np.unique(np.round(values, 4))


def _find_horizon_column(df: pd.DataFrame, preferred: str | None) -> str | None:
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["horizon", "horizon_min", "horizon_minutes"])
    for name in candidates:
        if name and name in df.columns:
            return name
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep p_touch/p_up thresholds and plot EV heatmaps")
    parser.add_argument("predictions", type=Path, help="CSV/Parquet file with model forecasts.")
    parser.add_argument(
        "--timestamp-column",
        default="timestamp",
        help="Column containing the prediction timestamp.",
    )
    parser.add_argument(
        "--prediction-column",
        default="p_hat",
        help="Column with the touch probability or price forecast.",
    )
    parser.add_argument(
        "--target-column",
        default="target",
        help="Column with the realised label used for P&L computation.",
    )
    parser.add_argument(
        "--price-column",
        default="last_price",
        help="Reference price column used when computing trade returns.",
    )
    parser.add_argument(
        "--p-up-column",
        default="p_up",
        help="Directional probability column gating long/short decisions.",
    )
    parser.add_argument(
        "--group-column",
        default="horizon",
        help="Optional column used to split predictions by horizon before sweeping.",
    )
    parser.add_argument(
        "--fee_bps",
        "--fee-bps",
        dest="fee_bps",
        type=float,
        default=getattr(CONFIG.backtest, "fee_bps", 4.0),
        help="Proportional transaction cost per trade in basis points.",
    )
    parser.add_argument(
        "--slip_bps",
        "--slip-bps",
        dest="slip_bps",
        type=float,
        default=getattr(CONFIG.backtest, "slippage_bps", 0.0),
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
        "--p-touch-min",
        type=float,
        default=0.5,
        help="Minimum p_touch threshold evaluated during the sweep.",
    )
    parser.add_argument(
        "--p-touch-max",
        type=float,
        default=0.8,
        help="Maximum p_touch threshold evaluated during the sweep.",
    )
    parser.add_argument(
        "--p-touch-step",
        type=float,
        default=0.05,
        help="Step size between p_touch thresholds.",
    )
    parser.add_argument(
        "--p-up-min",
        type=float,
        default=0.5,
        help="Minimum p_up threshold evaluated during the sweep.",
    )
    parser.add_argument(
        "--p-up-max",
        type=float,
        default=0.7,
        help="Maximum p_up threshold evaluated during the sweep.",
    )
    parser.add_argument(
        "--p-up-step",
        type=float,
        default=0.05,
        help="Step size between p_up thresholds.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional identifier used when storing reports. Defaults to a UTC timestamp.",
    )
    return parser


def _iter_groups(df: pd.DataFrame, column: str | None) -> Iterable[tuple[str | int | float | None, pd.DataFrame]]:
    if column is None or column not in df.columns:
        yield None, df
        return

    values = df[column].dropna().unique()
    if values.size == 0:
        yield None, df
        return

    for value in sorted(values):
        mask = df[column] == value
        subset = df.loc[mask]
        if subset.empty:
            continue
        yield value, subset


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

    group_col = _find_horizon_column(normalised, args.group_column)
    p_touch_grid = _build_grid(args.p_touch_min, args.p_touch_max, args.p_touch_step)
    p_up_grid = _build_grid(args.p_up_min, args.p_up_max, args.p_up_step)

    params = SweepParams(
        p_touch_values=p_touch_grid,
        p_up_values=p_up_grid,
        fee_bps=args.fee_bps,
        slippage_bps=args.slip_bps,
        latency_steps=latency_steps,
        p_touch_col="p_hat",
        p_up_col=args.p_up_column,
    )

    frames: list[pd.DataFrame] = []
    for horizon, subset in _iter_groups(normalised, group_col):
        frame = sweep_threshold_grid(subset, params=params, horizon_label=horizon)
        frames.append(frame)

    if not frames:
        raise RuntimeError("No data available for threshold sweep")

    result = pd.concat(frames, ignore_index=True)

    run_id = args.run_id or pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = reports_dir / f"threshold_sweep_{run_id}.csv"
    result.to_csv(csv_path, index=False)

    pivot_groups = []
    for horizon, subset in _iter_groups(result, "horizon"):
        pivot = subset.pivot_table(
            index="p_touch_thr",
            columns="p_up_thr",
            values="ev",
            aggfunc="mean",
        )
        pivot_groups.append((horizon, pivot))

    import matplotlib.pyplot as plt  # deferred import for hygiene tests

    if not pivot_groups:
        pivot_groups.append((None, pd.DataFrame()))

    n_groups = max(1, len(pivot_groups))
    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 4), squeeze=False)

    for ax, (horizon, pivot) in zip(axes.flat, pivot_groups):
        if pivot.empty:
            ax.set_visible(False)
            continue
        data = np.ma.masked_invalid(pivot.to_numpy())
        im = ax.imshow(data, origin="lower", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{val:.2f}" for val in pivot.columns], rotation=45)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{val:.2f}" for val in pivot.index])
        title = f"Horizon {horizon}" if horizon is not None else "All horizons"
        ax.set_title(title)
        ax.set_xlabel("p_up_thr")
        ax.set_ylabel("p_touch_thr")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="EV")

    plt.tight_layout()
    png_path = reports_dir / f"threshold_sweep_{run_id}.png"
    plt.savefig(png_path)
    plt.close(fig)

    top = result.sort_values("ev", ascending=False).head(10)
    print("Top threshold combinations by expected value:")
    print(top[["horizon", "p_touch_thr", "p_up_thr", "ev", "pnl", "sharpe", "trades"]])
    print(f"Sweep results saved to {csv_path}")
    print(f"Heatmap saved to {png_path}")

    return csv_path, png_path


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    main()
