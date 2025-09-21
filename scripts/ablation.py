"""Feature ablation experiments using a Typer CLI."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import typer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crypto_analyzer.data.db_connector import get_price_data
from crypto_analyzer.eval.cv import purged_walkforward_splits
from crypto_analyzer.features.engineering import (
    FEATURE_COLUMNS,
    create_features,
    get_feature_columns,
)
from crypto_analyzer.features.engineering import make_targets as make_default_targets
from crypto_analyzer.models.calibration import brier_score
from crypto_analyzer.utils.cli import run_cli
from crypto_analyzer.utils.config import CONFIG, FeatureSettings, override_feature_settings
from crypto_analyzer.utils.errors import DataValidationError
from crypto_analyzer.utils.io import build_path, initialize_run, save_csv, save_json, save_png
from crypto_analyzer.utils.logging import get_logger

app = typer.Typer(add_completion=False, no_args_is_help=True)
logger = get_logger(__name__)

GROUPS = ["price", "volatility", "multi_tf", "derivatives", "orderbook"]
PATTERNS = {
    "price": [r"^ret", r"^rel_", r"^mom_", r"taker", r"close", r"volume", r"price"],
    "volatility": [r"^vol", r"atr", r"^rv_", r"^bv_"],
    "multi_tf": [r"_15m", r"_1h", r"_4h", r"_1d", r"roll_", r"ema", r"multi"],
    "derivatives": [r"^deriv", r"funding", r"basis", r"oi_"],
    "orderbook": [r"^lob", r"^ofi", r"^wall", r"order_flow"],
}


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataValidationError(f"Input file '{path}' does not exist")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _prepare_settings(
    *,
    include_onchain: bool | None,
    include_orderbook: bool | None,
    include_derivatives: bool | None,
) -> FeatureSettings:
    settings = CONFIG.features
    overrides: dict[str, Any] = {}
    if include_onchain is not None:
        overrides["include_onchain"] = include_onchain
    if include_orderbook is not None:
        overrides["include_orderbook"] = include_orderbook
    if include_derivatives is not None:
        overrides["include_derivatives"] = include_derivatives
    if overrides:
        settings = override_feature_settings(settings, **overrides)
    return settings


def _load_features(
    *,
    features_path: Path | None,
    symbol: str,
    db_path: Path,
    settings: FeatureSettings,
) -> pd.DataFrame:
    if features_path is not None:
        df = _read_table(features_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return df
    raw = get_price_data(symbol, db_path=db_path)
    return create_features(raw, settings=settings)


def _match_columns(columns: list[str], patterns: list[str]) -> list[str]:
    matched: list[str] = []
    for pattern in patterns:
        regex = re.compile(pattern)
        matched.extend([col for col in columns if regex.search(col)])
    return sorted(set(matched))


def _build_pipeline(feature_names: list[str]) -> Pipeline:
    transformer = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                feature_names,
            )
        ]
    )
    clf = LogisticRegression(max_iter=1000)
    return Pipeline([("transform", transformer), ("clf", clf)])


def _execute_ablation(
    *,
    features: Path | None,
    symbol: str,
    db_path: Path,
    label: Optional[str],
    horizon: int,
    run_id: str | None,
    include_onchain: Optional[bool],
    include_orderbook: Optional[bool],
    include_derivatives: Optional[bool],
    dry_run: bool,
) -> Path:
    if horizon <= 0:
        raise DataValidationError("--horizon must be positive")

    settings = _prepare_settings(
        include_onchain=include_onchain,
        include_orderbook=include_orderbook,
        include_derivatives=include_derivatives,
    )
    df = _load_features(
        features_path=features,
        symbol=symbol,
        db_path=db_path,
        settings=settings,
    )

    label_name = label or f"cls_sign_{horizon}m"
    if label_name not in df.columns:
        df = make_default_targets(df, horizon=horizon)
    if label_name not in df.columns:
        raise DataValidationError(f"Label column '{label_name}' not found")

    df = df.dropna(subset=[label_name]).sort_values("timestamp")

    feature_cols = get_feature_columns(settings) or FEATURE_COLUMNS
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise DataValidationError("Missing features: " + ", ".join(sorted(missing)))

    run_id_value, run_dir, reports_dir = initialize_run(run_id, deterministic_torch=False)
    logger.info(
        "Prepared ablation run",
        extra={"event": "initialised", "run_id": run_id_value, "rows": int(len(df))},
    )

    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    splits = purged_walkforward_splits(
        timestamps,
        CONFIG.cv.n_splits,
        CONFIG.cv.embargo_min,
        run_id=run_id_value,
        reports_dir=reports_dir,
    )
    if not splits:
        raise DataValidationError("No walk-forward splits generated")
    train_idx, test_idx = splits[-1]

    X = df[feature_cols].astype(np.float32)
    y = df[label_name].astype(int)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    results: list[dict[str, Any]] = []

    baseline_pipeline = _build_pipeline(feature_cols)
    baseline_pipeline.fit(X_train, y_train)
    baseline_probs = baseline_pipeline.predict_proba(X_test)[:, 1]
    results.append(
        {
            "group": "baseline",
            "brier": brier_score(y_test, baseline_probs),
            "log_loss": log_loss(y_test, baseline_probs, labels=[0, 1]),
            "auc": roc_auc_score(y_test, baseline_probs),
            "features": len(feature_cols),
        }
    )

    for group in GROUPS:
        drop_cols = _match_columns(feature_cols, PATTERNS.get(group, []))
        active_cols = [col for col in feature_cols if col not in drop_cols]
        if not active_cols:
            continue
        pipeline = _build_pipeline(active_cols)
        pipeline.fit(X_train[active_cols], y_train)
        probs = pipeline.predict_proba(X_test[active_cols])[:, 1]
        metrics_row = {
            "group": group,
            "brier": brier_score(y_test, probs),
            "log_loss": log_loss(y_test, probs, labels=[0, 1]),
            "auc": roc_auc_score(y_test, probs),
            "features": len(active_cols),
        }
        results.append(metrics_row)

    result_df = pd.DataFrame(results).sort_values("brier")

    csv_path = build_path(
        f"ablation_{run_id_value}.csv", run_id=run_id_value, location="reports"
    )
    png_path = build_path(
        f"ablation_{run_id_value}.png", run_id=run_id_value, location="reports"
    )

    if dry_run:
        typer.echo("Dry run requested; skipping report generation.")
        typer.echo(f"Results would be stored at {csv_path} and {png_path}")
        return csv_path

    csv_report_path = save_csv(
        result_df,
        f"ablation_{run_id_value}.csv",
        run_id=run_id_value,
        location="reports",
        index=False,
    )
    save_csv(result_df, "ablation.csv", run_id=run_id_value, index=False)

    import matplotlib.pyplot as plt  # local import for hygiene tests

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.25
    x = np.arange(len(result_df))
    ax.bar(x - width, result_df["brier"], width=width, label="Brier")
    ax.bar(x, result_df["log_loss"], width=width, label="LogLoss")
    ax.bar(x + width, result_df["auc"], width=width, label="AUC")
    ax.set_xticks(x)
    ax.set_xticklabels(result_df["group"], rotation=45)
    ax.legend()
    fig.tight_layout()
    png_report_path = save_png(
        fig,
        f"ablation_{run_id_value}.png",
        run_id=run_id_value,
        location="reports",
    )
    save_png(fig, "ablation.png", run_id=run_id_value)
    plt.close(fig)

    metadata = {
        "run_id": run_id_value,
        "horizon": horizon,
        "include_onchain": include_onchain,
        "include_orderbook": include_orderbook,
        "include_derivatives": include_derivatives,
        "label": label,
        "features_path": str(features) if features else None,
        "symbol": symbol,
        "db_path": str(db_path),
    }
    save_json(metadata, "config_dump.json", run_id=run_id_value)

    logger.info(
        "Generated ablation reports",
        extra={
            "event": "artefacts",
            "run_id": run_id_value,
            "csv": str(csv_report_path),
            "png": str(png_report_path),
        },
    )

    typer.echo(f"Ablation results stored at {csv_report_path} and {png_report_path}")
    return csv_report_path


@app.command()
def main(
    features: Optional[Path] = typer.Option(
        None,
        "--features",
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Optional feature table (CSV/Parquet).",
    ),
    symbol: str = typer.Option(CONFIG.symbol, help="Symbol used when loading price data."),
    db_path: Path = typer.Option(
        CONFIG.db_path,
        "--db-path",
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="SQLite database used when generating features.",
    ),
    label: Optional[str] = typer.Option(None, help="Custom label column name."),
    horizon: int = typer.Option(CONFIG.core.forward_steps * 15, help="Horizon in minutes."),
    run_id: Optional[str] = typer.Option(None, help="Optional run identifier."),
    include_onchain: Optional[bool] = typer.Option(
        None,
        "--include-onchain/--exclude-onchain",
        help="Override on-chain features toggle.",
    ),
    include_orderbook: Optional[bool] = typer.Option(
        None,
        "--include-orderbook/--exclude-orderbook",
        help="Override orderbook features toggle.",
    ),
    include_derivatives: Optional[bool] = typer.Option(
        None,
        "--include-derivatives/--exclude-derivatives",
        help="Override derivative features toggle.",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without writing."),
) -> None:
    _execute_ablation(
        features=features,
        symbol=symbol,
        db_path=db_path,
        label=label,
        horizon=horizon,
        run_id=run_id,
        include_onchain=include_onchain,
        include_orderbook=include_orderbook,
        include_derivatives=include_derivatives,
        dry_run=dry_run,
    )


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    run_cli(app)

