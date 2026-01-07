#!/usr/bin/env python
"""Compute per-feature importance for the trained classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import typer
import re
from sklearn.inspection import permutation_importance

from crypto_analyzer.features.engineering import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    create_features,
    get_feature_columns,
    make_targets as make_default_targets,
    assign_feature_groups,
)
from crypto_analyzer.models.utils import match_model_features
from crypto_analyzer.data.store import PriceDataStore, resolve_data_store
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


def _match_columns(columns: list[str], patterns: list[str]) -> set[str]:
    matched: set[str] = set()
    for pattern in patterns:
        regex = re.compile(pattern)
        matched.update([col for col in columns if regex.search(col)])
    return matched


def _apply_feature_group_drops(feature_cols: list[str], drop_groups: list[str]) -> list[str]:
    if not drop_groups:
        return feature_cols
    unknown = [group for group in drop_groups if group not in FEATURE_GROUPS]
    if unknown:
        raise DataValidationError("Unknown feature groups: " + ", ".join(sorted(unknown)))
    to_drop: set[str] = set()
    for group in drop_groups:
        to_drop.update(_match_columns(feature_cols, FEATURE_GROUPS.get(group, [])))
    return [col for col in feature_cols if col not in to_drop]


def _booster_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    if not hasattr(model, "get_booster"):
        return pd.DataFrame(columns=["feature", "gain", "weight", "cover"])
    booster = model.get_booster()
    mapping = {f"f{idx}": name for idx, name in enumerate(feature_cols)}
    frames = []
    for imp_type in ("gain", "weight", "cover"):
        scores = booster.get_score(importance_type=imp_type)
        if not scores:
            continue
        rows = []
        for key, value in scores.items():
            rows.append({"feature": mapping.get(key, key), imp_type: float(value)})
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame(columns=["feature", "gain", "weight", "cover"])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="feature", how="outer")
    return merged.fillna(0.0)


def _permutation_importance(model, X: pd.DataFrame, y: pd.Series, n_repeats: int) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=CONFIG.models.random_seed,
        scoring="roc_auc",
    )
    return pd.DataFrame(
        {
            "feature": X.columns,
            "perm_importance": result.importances_mean,
            "perm_std": result.importances_std,
        }
    )


def _shap_importance(model, X: pd.DataFrame, samples: int) -> pd.DataFrame:
    try:
        import shap  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover
        raise DataValidationError("Install shap to compute SHAP importance") from exc

    sample = X.sample(n=min(samples, len(X)), random_state=CONFIG.models.random_seed)
    try:
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(sample)
    except Exception:
        explainer = shap.Explainer(model.predict, sample)
        values = explainer(sample).values
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., 1]
    mean_abs = np.abs(values).mean(axis=0)
    return pd.DataFrame({"feature": sample.columns, "shap_mean_abs": mean_abs})


@app.command()
def main(
    model_path: Path = typer.Option(
        Path("artifacts/meta_model.joblib"),
        "--model-path",
        exists=False,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
        help="Path to the trained model.",
    ),
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
    feature_drop_groups: Optional[str] = typer.Option(
        None,
        "--feature-drop-groups",
        help="Comma-separated feature group names to drop (e.g. lob,derivatives).",
    ),
    shap_samples: int = typer.Option(2000, "--shap-samples", help="SHAP sample size."),
    perm_repeats: int = typer.Option(5, "--perm-repeats", help="Permutation repeats."),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Optional run identifier."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without writing."),
) -> None:
    if horizon <= 0:
        raise DataValidationError("--horizon must be positive")

    settings = _prepare_settings(
        include_onchain=include_onchain,
        include_orderbook=include_orderbook,
        include_derivatives=include_derivatives,
        include_sentiment=include_sentiment,
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
    if label_name not in df.columns:
        raise DataValidationError(f"Label column '{label_name}' not found")

    df = df.dropna(subset=[label_name]).sort_values("timestamp").reset_index(drop=True)
    feature_cols = get_feature_columns(settings) or FEATURE_COLUMNS
    drop_groups = (
        [g.strip() for g in feature_drop_groups.split(",") if g.strip()]
        if feature_drop_groups
        else []
    )
    if drop_groups:
        feature_cols = _apply_feature_group_drops(feature_cols, drop_groups)

    X = df[feature_cols].astype(np.float32)
    y = df[label_name].astype(int)

    model = joblib.load(model_path)
    X = match_model_features(X, model)

    run_id_value, _, _ = initialize_run(run_id, deterministic_torch=False)
    logger.info(
        "Prepared feature importance run",
        extra={"event": "initialised", "run_id": run_id_value, "rows": int(len(X))},
    )

    gain_df = _booster_importance(model, list(X.columns))
    perm_df = _permutation_importance(model, X, y, perm_repeats)
    try:
        shap_df = _shap_importance(model, X, shap_samples)
    except DataValidationError:
        shap_df = pd.DataFrame(columns=["feature", "shap_mean_abs"])

    merged = gain_df.merge(perm_df, on="feature", how="outer").merge(
        shap_df, on="feature", how="outer"
    )
    merged = merged.fillna(0.0)

    groups = assign_feature_groups(list(merged["feature"]))
    merged["group"] = merged["feature"].map(groups).fillna("other")

    group_summary = (
        merged.groupby("group", as_index=False)
        .agg(
            {
                "gain": "sum",
                "weight": "sum",
                "cover": "sum",
                "perm_importance": "sum",
                "shap_mean_abs": "sum",
            }
        )
        .sort_values("gain", ascending=False)
    )

    if dry_run:
        typer.echo("Dry run requested; skipping report generation.")
        return

    csv_path = save_csv(
        merged,
        f"feature_importance_{run_id_value}.csv",
        run_id=run_id_value,
        location="reports",
        index=False,
    )
    save_csv(group_summary, "feature_importance_groups.csv", run_id=run_id_value, index=False)

    meta = {
        "run_id": run_id_value,
        "horizon": horizon,
        "model_path": str(model_path),
        "features_path": str(features) if features else None,
        "feature_drop_groups": drop_groups,
        "shap_samples": shap_samples,
        "perm_repeats": perm_repeats,
    }
    save_json(meta, f"feature_importance_{run_id_value}.json", run_id=run_id_value)

    typer.echo(f"Feature importance stored at {csv_path}")


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    app()
