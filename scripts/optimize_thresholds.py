"""Sweep touch/up thresholds using a Typer CLI."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from crypto_analyzer.eval.threshold_sweep import SweepParams, sweep_threshold_grid
from crypto_analyzer.utils.cli import run_cli
from crypto_analyzer.utils.config import CONFIG
from crypto_analyzer.utils.errors import DataValidationError
from crypto_analyzer.utils.io import build_path, initialize_run, save_csv, save_json, save_png
from crypto_analyzer.utils.logging import get_logger

app = typer.Typer(add_completion=False, no_args_is_help=True)
logger = get_logger(__name__)


def _read_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataValidationError(f"Predictions file '{path}' does not exist")
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
        raise DataValidationError("Missing required columns: " + ", ".join(sorted(missing)))

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
        raise DataValidationError("Step must be positive")
    values = np.arange(start, stop + step * 0.5, step)
    if values.size == 0:
        raise DataValidationError("Invalid grid range")
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


def _iter_groups(
    df: pd.DataFrame, column: str | None
) -> Iterable[tuple[str | int | float | None, pd.DataFrame]]:
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


def _run_sweep(
    *,
    predictions: Path,
    timestamp_column: str,
    prediction_column: str,
    target_column: str,
    price_column: str,
    p_up_column: str,
    group_column: str,
    fee_bps: float,
    slip_bps: float,
    latency_min: float,
    p_touch_min: float,
    p_touch_max: float,
    p_touch_step: float,
    p_up_min: float,
    p_up_max: float,
    p_up_step: float,
    run_id: str | None,
    dry_run: bool,
) -> tuple[Path, Path]:
    df = _read_predictions(predictions)
    normalised = _normalise_columns(
        df,
        timestamp_col=timestamp_column,
        prediction_col=prediction_column,
        target_col=target_column,
        price_col=price_column,
    )

    latency_steps = _infer_latency_steps(normalised["timestamp"], latency_min)

    group_col = _find_horizon_column(normalised, group_column)
    p_touch_grid = _build_grid(p_touch_min, p_touch_max, p_touch_step)
    p_up_grid = _build_grid(p_up_min, p_up_max, p_up_step)

    params = SweepParams(
        p_touch_values=p_touch_grid,
        p_up_values=p_up_grid,
        fee_bps=fee_bps,
        slippage_bps=slip_bps,
        latency_steps=latency_steps,
        p_touch_col="p_hat",
        p_up_col=p_up_column,
    )

    frames: list[pd.DataFrame] = []
    for horizon, subset in _iter_groups(normalised, group_col):
        frame = sweep_threshold_grid(subset, params=params, horizon_label=horizon)
        frames.append(frame)

    if not frames:
        raise DataValidationError("No data available for threshold sweep")

    result = pd.concat(frames, ignore_index=True)

    run_id_value, run_dir, reports_dir = initialize_run(run_id, deterministic_torch=False)
    logger.info(
        "Prepared threshold sweep run",
        extra={"event": "initialised", "run_id": run_id_value, "rows": int(len(result))},
    )

    csv_path = build_path(
        f"threshold_sweep_{run_id_value}.csv", run_id=run_id_value, location="reports"
    )
    png_path = build_path(
        f"threshold_sweep_{run_id_value}.png", run_id=run_id_value, location="reports"
    )

    if dry_run:
        typer.echo("Dry run requested; skipping report generation.")
        typer.echo(f"Sweep results would be stored at {csv_path} and {png_path}")
        return csv_path, png_path

    csv_report_path = save_csv(
        result,
        f"threshold_sweep_{run_id_value}.csv",
        run_id=run_id_value,
        location="reports",
        index=False,
    )
    save_csv(result, "threshold_sweep.csv", run_id=run_id_value, index=False)

    pivot_groups = []
    for horizon, subset in _iter_groups(result, "horizon"):
        pivot = subset.pivot_table(
            index="p_touch_thr",
            columns="p_up_thr",
            values="ev",
            aggfunc="mean",
        )
        pivot_groups.append((horizon, pivot))

    import matplotlib.pyplot as plt  # deferred import

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
    png_report_path = save_png(
        fig,
        f"threshold_sweep_{run_id_value}.png",
        run_id=run_id_value,
        location="reports",
    )
    save_png(fig, "threshold_sweep.png", run_id=run_id_value)
    plt.close(fig)

    metadata = {
        "run_id": run_id_value,
        "fee_bps": fee_bps,
        "slip_bps": slip_bps,
        "latency_min": latency_min,
        "group_column": group_column,
        "p_touch_grid": {
            "min": p_touch_min,
            "max": p_touch_max,
            "step": p_touch_step,
        },
        "p_up_grid": {
            "min": p_up_min,
            "max": p_up_max,
            "step": p_up_step,
        },
        "probability_column": prediction_column,
        "p_up_column": p_up_column,
    }
    save_json(metadata, "config_dump.json", run_id=run_id_value)

    logger.info(
        "Generated threshold sweep artefacts",
        extra={
            "event": "artefacts",
            "run_id": run_id_value,
            "csv": str(csv_report_path),
            "png": str(png_report_path),
        },
    )

    top = result.sort_values("ev", ascending=False).head(10)
    typer.echo("Top threshold combinations by expected value:")
    typer.echo(top[["horizon", "p_touch_thr", "p_up_thr", "ev", "pnl", "sharpe", "trades"]])
    typer.echo(f"Sweep results saved to {csv_report_path}")
    typer.echo(f"Heatmap saved to {png_report_path}")

    return csv_report_path, png_report_path


@app.command()
def main(
    predictions: Path = typer.Argument(..., exists=True, resolve_path=True),
    timestamp_column: str = typer.Option(
        "timestamp", help="Column containing the prediction timestamp."
    ),
    prediction_column: str = typer.Option(
        "p_hat", help="Column with the touch probability or price forecast."
    ),
    target_column: str = typer.Option(
        "target", help="Column with the realised label used for P&L computation."
    ),
    price_column: str = typer.Option(
        "last_price", help="Reference price column for trade returns."
    ),
    p_up_column: str = typer.Option(
        "p_up", help="Directional probability column gating long/short decisions."
    ),
    group_column: str = typer.Option(
        "horizon", help="Column used to split predictions before sweeping."
    ),
    fee_bps: float = typer.Option(
        getattr(CONFIG.backtest, "fee_bps", 4.0),
        help="Proportional transaction cost per trade in basis points.",
    ),
    slip_bps: float = typer.Option(
        getattr(CONFIG.backtest, "slippage_bps", 0.0),
        help="Slippage assumption in basis points added to the fee.",
    ),
    latency_min: float = typer.Option(
        0.0, help="Execution latency in minutes applied via timestamp shifts."
    ),
    p_touch_min: float = typer.Option(0.5, help="Minimum p_touch threshold evaluated."),
    p_touch_max: float = typer.Option(0.8, help="Maximum p_touch threshold evaluated."),
    p_touch_step: float = typer.Option(0.05, help="Step size between p_touch thresholds."),
    p_up_min: float = typer.Option(0.5, help="Minimum p_up threshold evaluated."),
    p_up_max: float = typer.Option(0.7, help="Maximum p_up threshold evaluated."),
    p_up_step: float = typer.Option(0.05, help="Step size between p_up thresholds."),
    run_id: Optional[str] = typer.Option(
        None, help="Optional identifier used when storing reports."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without writing."),
) -> None:
    if fee_bps < 0 or slip_bps < 0:
        raise DataValidationError("--fee-bps and --slip-bps must be non-negative")
    if p_touch_min > p_touch_max:
        raise DataValidationError("--p-touch-min must be <= --p-touch-max")
    if p_up_min > p_up_max:
        raise DataValidationError("--p-up-min must be <= --p-up-max")

    _run_sweep(
        predictions=predictions,
        timestamp_column=timestamp_column,
        prediction_column=prediction_column,
        target_column=target_column,
        price_column=price_column,
        p_up_column=p_up_column,
        group_column=group_column,
        fee_bps=fee_bps,
        slip_bps=slip_bps,
        latency_min=latency_min,
        p_touch_min=p_touch_min,
        p_touch_max=p_touch_max,
        p_touch_step=p_touch_step,
        p_up_min=p_up_min,
        p_up_max=p_up_max,
        p_up_step=p_up_step,
        run_id=run_id,
        dry_run=dry_run,
    )


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    run_cli(app)
