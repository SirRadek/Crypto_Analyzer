#!/usr/bin/env python
"""Report per-feature coverage/variance to spot dead signals."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer

from crypto_analyzer.data.store import PriceDataStore, resolve_data_store
from crypto_analyzer.features.engineering import (
    FEATURE_COLUMNS,
    create_features,
    get_feature_columns,
    make_targets as make_default_targets,
)
from crypto_analyzer.utils.config import CONFIG, FeatureSettings, override_feature_settings
from crypto_analyzer.utils.errors import DataValidationError
from crypto_analyzer.utils.io import build_path, initialize_run, save_csv, save_json
from crypto_analyzer.utils.logging import get_logger

app = typer.Typer(add_completion=False, no_args_is_help=True)
logger = get_logger(__name__)


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
    include_sentiment: bool | None,
    forward_fill_limit: int | None,
    fillna_value: float | None,
) -> FeatureSettings:
    settings = CONFIG.features
    overrides: dict[str, bool] = {}
    if include_onchain is not None:
        overrides["include_onchain"] = include_onchain
    if include_orderbook is not None:
        overrides["include_orderbook"] = include_orderbook
    if include_derivatives is not None:
        overrides["include_derivatives"] = include_derivatives
    if include_sentiment is not None:
        overrides["include_sentiment"] = include_sentiment
    if overrides:
        settings = override_feature_settings(settings, **overrides)

    if forward_fill_limit is not None or fillna_value is not None:
        settings = FeatureSettings(
            include_onchain=settings.include_onchain,
            include_orderbook=settings.include_orderbook,
            include_derivatives=settings.include_derivatives,
            include_sentiment=settings.include_sentiment,
            forward_fill_limit=(
                forward_fill_limit
                if forward_fill_limit is not None
                else settings.forward_fill_limit
            ),
            fillna_value=(fillna_value if fillna_value is not None else settings.fillna_value),
        )
    return settings


def _load_features(
    *,
    features_path: Path | None,
    symbol: str,
    data_store: PriceDataStore | None,
    settings: FeatureSettings,
) -> pd.DataFrame:
    if features_path is not None:
        df = _read_table(features_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return df
    if data_store is None:
        raise DataValidationError("Database store is required when --features is not provided")
    raw = data_store.fetch_prices(symbol)
    return create_features(raw, settings=settings)


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
    store_choice: str = typer.Option(
        "auto",
        "--store",
        help="Database backend (auto, sqlite, timescale).",
    ),
    db_path: Optional[Path] = typer.Option(
        CONFIG.db_path,
        "--db-path",
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="SQLite database path when using the local store.",
    ),
    db_url: Optional[str] = typer.Option(
        CONFIG.db_url,
        "--db-url",
        help="SQLAlchemy URL when using Timescale/PostgreSQL.",
    ),
    horizon: int = typer.Option(CONFIG.horizons[0], help="Prediction horizon in minutes."),
    label: Optional[str] = typer.Option(None, help="Optional label column name."),
    include_onchain: bool = typer.Option(
        None,
        "--include-onchain/--exclude-onchain",
        help="Override on-chain features toggle.",
        flag_value=True,
    ),
    include_orderbook: bool = typer.Option(
        None,
        "--include-orderbook/--exclude-orderbook",
        help="Override orderbook features toggle.",
        flag_value=True,
    ),
    include_derivatives: bool = typer.Option(
        None,
        "--include-derivatives/--exclude-derivatives",
        help="Override derivative features toggle.",
        flag_value=True,
    ),
    include_sentiment: bool = typer.Option(
        None,
        "--include-sentiment/--exclude-sentiment",
        help="Override sentiment features toggle.",
        flag_value=True,
    ),
    forward_fill_limit: Optional[int] = typer.Option(
        None, "--forward-fill-limit", help="Override forward-fill window."
    ),
    fillna_value: Optional[float] = typer.Option(
        None, "--fillna-value", help="Override fillna fallback."
    ),
    nonzero_threshold: float = typer.Option(
        0.01,
        "--nonzero-threshold",
        help="Minimum non-zero ratio to avoid being flagged.",
    ),
    variance_threshold: float = typer.Option(
        1e-8,
        "--variance-threshold",
        help="Minimum variance to avoid being flagged.",
    ),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Optional run identifier."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without writing."),
) -> None:
    settings = _prepare_settings(
        include_onchain=include_onchain,
        include_orderbook=include_orderbook,
        include_derivatives=include_derivatives,
        include_sentiment=include_sentiment,
        forward_fill_limit=forward_fill_limit,
        fillna_value=fillna_value,
    )
    data_store = None
    if features is None:
        data_store = resolve_data_store(
            store_choice,
            sqlite_path=db_path,
            timescale_url=db_url,
        )
    df = _load_features(
        features_path=features,
        symbol=symbol,
        data_store=data_store,
        settings=settings,
    )

    label_name = label or f"cls_sign_{horizon}m"
    if label_name not in df.columns:
        df = make_default_targets(df, horizon=horizon)

    feature_cols = get_feature_columns(settings) or FEATURE_COLUMNS
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise DataValidationError("Missing features: " + ", ".join(sorted(missing)))

    X = df[feature_cols].astype(np.float32)
    stats = []
    total = len(X)
    for col in X.columns:
        series = X[col]
        non_na = series.notna().sum()
        non_zero = (series != 0).sum()
        variance = float(series.var(skipna=True))
        stats.append(
            {
                "feature": col,
                "non_na_ratio": non_na / total if total else 0.0,
                "non_zero_ratio": non_zero / total if total else 0.0,
                "variance": variance,
                "flag_low_signal": non_zero / total <= nonzero_threshold
                or variance <= variance_threshold,
            }
        )

    result = pd.DataFrame(stats).sort_values("non_zero_ratio")
    flagged = result[result["flag_low_signal"]]

    run_id_value, _, _ = initialize_run(run_id, deterministic_torch=False)
    logger.info(
        "Prepared feature coverage report",
        extra={"event": "initialised", "run_id": run_id_value, "rows": int(len(result))},
    )

    if dry_run:
        typer.echo("Dry run requested; skipping report generation.")
        typer.echo(result.head(10).to_string(index=False))
        return

    csv_path = save_csv(
        result,
        f"feature_coverage_{run_id_value}.csv",
        run_id=run_id_value,
        location="reports",
        index=False,
    )
    save_csv(
        flagged,
        "feature_coverage_flagged.csv",
        run_id=run_id_value,
        location="reports",
        index=False,
    )

    meta = {
        "run_id": run_id_value,
        "horizon": horizon,
        "features_path": str(features) if features else None,
        "thresholds": {
            "nonzero_ratio": nonzero_threshold,
            "variance": variance_threshold,
        },
        "flagged_count": int(len(flagged)),
    }
    save_json(
        meta,
        f"feature_coverage_{run_id_value}.json",
        run_id=run_id_value,
        location="reports",
    )
    typer.echo(f"Feature coverage stored at {csv_path}")


if __name__ == "__main__":  # pragma: no cover
    app()
