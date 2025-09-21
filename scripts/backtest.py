"""Backtest runner with Typer CLI integration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from crypto_analyzer.eval.backtest import run_backtest
from crypto_analyzer.eval.threshold_sweep import SweepParams, sweep_threshold_grid
from crypto_analyzer.utils.cli import run_cli
from crypto_analyzer.utils.config import CONFIG
from crypto_analyzer.utils.errors import DataValidationError
from crypto_analyzer.utils.io import build_path, initialize_run, save_csv, save_json
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


def _find_horizon_column(df: pd.DataFrame) -> str | None:
    for column in ("horizon", "horizon_min", "horizon_minutes"):
        if column in df.columns:
            return column
    return None


def _run_backtest(
    *,
    predictions: Path,
    timestamp_column: str,
    prediction_column: str,
    target_column: str,
    price_column: str,
    prob_column: Optional[str],
    fee_bps: float,
    slip_bps: float,
    latency_min: float,
    p_touch_thr: Optional[float],
    p_up_thr: Optional[float],
    run_id: Optional[str],
    optimize_thresholds: bool,
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

    prob_col: Optional[str] = prob_column or None
    if prob_col is not None and prob_col == prediction_column:
        prob_col = "p_hat"
    p_up_col = "p_up" if "p_up" in normalised.columns else None

    if optimize_thresholds:
        horizon_col = _find_horizon_column(normalised)
        touch_grid = np.round(np.arange(0.5, 0.8001, 0.05), 4)
        up_grid = np.round(np.arange(0.5, 0.7001, 0.05), 4)
        params = SweepParams(
            p_touch_values=touch_grid,
            p_up_values=up_grid,
            fee_bps=fee_bps,
            slippage_bps=slip_bps,
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
        typer.echo("Threshold sweep preview (top 5 by EV):")
        typer.echo(top_preview[["horizon", "p_touch_thr", "p_up_thr", "ev", "pnl", "sharpe", "trades"]])

    result = run_backtest(
        normalised,
        fee_bps=fee_bps,
        slippage_bps=slip_bps,
        prob_col=prob_col,
        latency_steps=latency_steps,
        p_touch_col="p_hat" if p_touch_thr is not None else None,
        p_touch_threshold=p_touch_thr,
        p_up_col=p_up_col,
        p_up_threshold=p_up_thr,
    )

    run_id_value, run_dir, reports_dir = initialize_run(run_id, deterministic_torch=False)
    logger.info(
        "Prepared backtest run",
        extra={"event": "initialised", "run_id": run_id_value, "rows": int(len(normalised))},
    )

    equity = result["equity"].assign(run_id=run_id_value)
    metrics = result["metrics"]

    summary_metrics = {
        "ev": metrics.get("ev"),
        "hit_rate": metrics.get("hit_rate"),
        "sharpe": metrics.get("sharpe"),
        "maxDD": metrics.get("maxDD", metrics.get("max_drawdown")),
    }

    for key, value in summary_metrics.items():
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
        typer.echo(f"{key}: {display}")

    equity_output = build_path(
        f"equity_{run_id_value}.csv", run_id=run_id_value, location="reports"
    )
    summary_output = build_path(
        f"summary_{run_id_value}.json", run_id=run_id_value, location="reports"
    )

    if dry_run:
        typer.echo("Dry run requested; skipping file writes.")
        typer.echo(f"Equity would be written to {equity_output}")
        typer.echo(f"Summary would be written to {summary_output}")
        return summary_output, equity_output

    equity_report_path = save_csv(
        equity,
        f"equity_{run_id_value}.csv",
        run_id=run_id_value,
        location="reports",
        index=False,
    )
    save_csv(equity, "equity.csv", run_id=run_id_value, index=False)

    summary = {
        "run_id": run_id_value,
        "parameters": {
            "fee_bps": fee_bps,
            "slip_bps": slip_bps,
            "latency_minutes": latency_min,
            "p_touch_threshold": p_touch_thr,
            "p_up_threshold": p_up_thr,
            "prob_column": prob_col,
            "latency_steps": latency_steps,
        },
        "metrics": summary_metrics,
    }
    summary_report_path = save_json(
        summary,
        f"summary_{run_id_value}.json",
        run_id=run_id_value,
        location="reports",
    )
    save_json(summary, "summary.json", run_id=run_id_value)

    equity_output = equity_report_path
    summary_output = summary_report_path

    config_dump = {
        "args": {
            "timestamp_column": timestamp_column,
            "prediction_column": prediction_column,
            "target_column": target_column,
            "price_column": price_column,
            "prob_column": prob_col,
            "fee_bps": fee_bps,
            "slip_bps": slip_bps,
            "latency_min": latency_min,
            "p_touch_thr": p_touch_thr,
            "p_up_thr": p_up_thr,
            "run_id": run_id_value,
        },
        "config": CONFIG.config_path.as_posix() if CONFIG.config_path else None,
    }
    save_json(config_dump, "config_dump.json", run_id=run_id_value)

    logger.info(
        "Saved backtest artefacts",
        extra={
            "event": "artefacts",
            "run_id": run_id_value,
            "equity_report": str(equity_report_path),
            "summary_report": str(summary_report_path),
        },
    )

    typer.echo(
        "Backtest complete. Final equity: "
        f"{float(equity['equity'].iloc[-1]):.4f}, PnL: {metrics['pnl']:.4f}, "
        f"Sharpe: {metrics['sharpe']:.4f}, EV: {metrics['ev']:.6f}, "
        f"MaxDD: {metrics['max_drawdown']:.4f}"
    )

    return summary_output, equity_output


@app.command()
def main(
    predictions: Path = typer.Argument(..., exists=True, resolve_path=True),
    timestamp_column: str = typer.Option(
        "timestamp", help="Column containing the prediction timestamp."
    ),
    prediction_column: str = typer.Option(
        "p_hat", help="Column with the model's predicted price or probability."
    ),
    target_column: str = typer.Option(
        "target", help="Column with the realised target used for P&L computation."
    ),
    price_column: str = typer.Option(
        "last_price", help="Reference price column used when computing trade returns."
    ),
    prob_column: Optional[str] = typer.Option(
        None, help="Optional probability column for EV-based backtest rules."
    ),
    fee_bps: float = typer.Option(4.0, help="Proportional transaction cost in basis points."),
    slip_bps: float = typer.Option(0.0, help="Slippage assumption in basis points."),
    latency_min: float = typer.Option(0.0, help="Execution latency in minutes."),
    p_touch_thr: Optional[float] = typer.Option(
        None, help="Minimum probability of touching the target required to open a trade."
    ),
    p_up_thr: Optional[float] = typer.Option(
        None,
        help="Directional probability threshold. Values above open longs, below (1-thr) open shorts.",
    ),
    run_id: Optional[str] = typer.Option(None, help="Optional identifier used when storing reports."),
    optimize_thresholds: bool = typer.Option(
        False, help="Run an EV sweep before executing the backtest."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without writing."),
) -> None:
    if fee_bps < 0 or slip_bps < 0:
        raise DataValidationError("--fee-bps and --slip-bps must be non-negative")
    if p_touch_thr is not None and not (0 <= p_touch_thr <= 1):
        raise DataValidationError("--p-touch-thr must lie in [0, 1]")
    if p_up_thr is not None and not (0 <= p_up_thr <= 1):
        raise DataValidationError("--p-up-thr must lie in [0, 1]")

    _run_backtest(
        predictions=predictions,
        timestamp_column=timestamp_column,
        prediction_column=prediction_column,
        target_column=target_column,
        price_column=price_column,
        prob_column=prob_column,
        fee_bps=fee_bps,
        slip_bps=slip_bps,
        latency_min=latency_min,
        p_touch_thr=p_touch_thr,
        p_up_thr=p_up_thr,
        run_id=run_id,
        optimize_thresholds=optimize_thresholds,
        dry_run=dry_run,
    )


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    run_cli(app)

