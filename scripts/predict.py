#!/usr/bin/env python
"""Generate real-time predictions from the latest feature snapshot."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import typer

from crypto_analyzer.data.store import PriceDataStore, resolve_data_store
from crypto_analyzer.features.engineering import create_features
from crypto_analyzer.models.utils import match_model_features
from crypto_analyzer.utils.cli import run_cli
from crypto_analyzer.utils.config import CONFIG, FeatureSettings, override_feature_settings
from crypto_analyzer.utils.errors import DataValidationError
from crypto_analyzer.utils.io import initialize_run, save_json
from crypto_analyzer.utils.logging import get_logger


class StoreChoice(str, Enum):
    auto = "auto"
    sqlite = "sqlite"
    timescale = "timescale"


app = typer.Typer(add_completion=False, no_args_is_help=True)
logger = get_logger(__name__)


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
            fillna_value=fillna_value if fillna_value is not None else settings.fillna_value,
        )
    return settings


def _load_feature_list(path: Path) -> list[str]:
    if not path.exists():
        raise DataValidationError(f"Feature list file '{path}' does not exist")
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise DataValidationError("Feature list JSON must be an array of strings")
        return [str(item) for item in raw]
    frame = pd.read_csv(path)
    if "feature" in frame.columns:
        return frame["feature"].dropna().astype(str).tolist()
    if frame.shape[1] == 1:
        return frame.iloc[:, 0].dropna().astype(str).tolist()
    raise DataValidationError("Feature list CSV must have a 'feature' column or a single column")


def _predict_probability(model: object, X: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        arr = np.asarray(proba)
        if arr.ndim == 2 and arr.shape[1] > 1:
            return float(arr[0, 1])
        if arr.size:
            return float(arr.ravel()[0])
    preds = model.predict(X)
    return float(np.asarray(preds).ravel()[0])


def _store_metadata(data_store: PriceDataStore) -> dict[str, str | None]:
    label = getattr(data_store, "label", None)
    if label == "sqlite":
        location = str(getattr(data_store, "path", None))
    elif label == "timescale":
        location = getattr(data_store, "url", None)
    else:
        location = None
    return {"data_store": label, "store_location": location}


@app.command()
def main(
    symbol: str = typer.Option(CONFIG.symbol, "--symbol", help="Trading symbol to score."),
    store_choice: StoreChoice = typer.Option(
        StoreChoice.auto,
        "--store",
        help="Database backend used to fetch candles (auto follows configuration).",
    ),
    db_path: Optional[Path] = typer.Option(
        CONFIG.db_path,
        "--db-path",
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Override SQLite database path when using the local store.",
    ),
    db_url: Optional[str] = typer.Option(
        CONFIG.db_url,
        "--db-url",
        help="Override SQLAlchemy URL when using Timescale/PostgreSQL.",
    ),
    history_days: int = typer.Option(
        CONFIG.core.history_days,
        "--history-days",
        min=1,
        help="Number of trailing days to load before generating features.",
    ),
    model_path: Path = typer.Option(
        Path("artifacts/meta_model.joblib"),
        "--model-path",
        resolve_path=True,
        help="Model artefact used for prediction.",
    ),
    threshold: float = typer.Option(
        0.5, "--threshold", help="Decision threshold for the positive class."
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        resolve_path=True,
        help="Optional JSON file receiving the prediction result.",
    ),
    include_onchain: bool = typer.Option(
        None,
        "--include-onchain/--exclude-onchain",
        help="Override on-chain feature toggle before scoring.",
        flag_value=True,
    ),
    include_orderbook: bool = typer.Option(
        None,
        "--include-orderbook/--exclude-orderbook",
        help="Override orderbook feature toggle before scoring.",
        flag_value=True,
    ),
    include_derivatives: bool = typer.Option(
        None,
        "--include-derivatives/--exclude-derivatives",
        help="Override derivative feature toggle before scoring.",
        flag_value=True,
    ),
    include_sentiment: bool = typer.Option(
        None,
        "--include-sentiment/--exclude-sentiment",
        help="Override sentiment feature toggle before scoring.",
        flag_value=True,
    ),
    feature_list: Optional[Path] = typer.Option(
        None,
        "--feature-list",
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Optional JSON/CSV list of feature names to keep.",
    ),
    forward_fill_limit: Optional[int] = typer.Option(
        None,
        "--forward-fill-limit",
        min=0,
        help="Override forward-fill limit applied during preprocessing.",
    ),
    fillna_value: Optional[float] = typer.Option(
        None,
        "--fillna-value",
        help="Override fillna fallback applied after forward fills.",
    ),
    run_id: Optional[str] = typer.Option(
        None, "--run-id", help="Optional run identifier for artefact storage."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Compute the prediction without writing artefacts."
    ),
) -> None:
    if threshold <= 0 or threshold >= 1:
        raise DataValidationError("--threshold must lie in (0, 1)")

    data_store = resolve_data_store(
        store_choice.value,
        sqlite_path=db_path,
        timescale_url=db_url,
    )

    history_start = pd.Timestamp.utcnow() - pd.Timedelta(days=int(history_days))
    start_ts = int(history_start.timestamp() * 1000)
    price_frame = data_store.fetch_prices(symbol, start_ts=start_ts)
    if price_frame.empty:
        raise DataValidationError("No price data available for the requested window")

    settings = _prepare_settings(
        include_onchain=include_onchain,
        include_orderbook=include_orderbook,
        include_derivatives=include_derivatives,
        include_sentiment=include_sentiment,
        forward_fill_limit=forward_fill_limit,
        fillna_value=fillna_value,
    )
    feature_frame = create_features(price_frame, settings=settings)
    feature_frame = feature_frame.dropna().sort_values("timestamp")
    if feature_frame.empty:
        raise DataValidationError("Generated feature matrix is empty after preprocessing")

    model = joblib.load(model_path)
    feature_payload = match_model_features(
        feature_frame.drop(columns=["timestamp"], errors="ignore"),
        model,
    )
    if feature_list is not None:
        keep = set(_load_feature_list(feature_list))
        feature_payload = feature_payload.loc[
            :, [col for col in feature_payload.columns if col in keep]
        ]
    latest_features = feature_payload.tail(1)
    probability = _predict_probability(model, latest_features)
    prediction = int(probability >= threshold)
    latest_timestamp = pd.to_datetime(feature_frame["timestamp"].iloc[-1], utc=True)

    result = {
        "symbol": symbol,
        "timestamp": latest_timestamp.isoformat(),
        "probability": probability,
        "prediction": prediction,
        "threshold": threshold,
        **_store_metadata(data_store),
        "model_path": str(model_path),
    }

    typer.echo(
        f"Prediction for {symbol} at {result['timestamp']}: prob_up={probability:.4f} -> {prediction}"
    )

    if dry_run:
        if output is not None:
            typer.echo(f"Dry run requested; prediction would be written to {output}")
        return

    run_id_value, _, _ = initialize_run(run_id, deterministic_torch=False)
    save_json(result, "prediction.json", run_id=run_id_value)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    run_cli(app)
