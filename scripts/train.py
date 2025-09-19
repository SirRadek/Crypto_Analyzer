#!/usr/bin/env python
"""Train entry point for the gradient boosted meta-classifier."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from crypto_analyzer.data.db_connector import get_price_data
from crypto_analyzer.features.engineering import (
    FEATURE_COLUMNS,
    create_features,
    get_feature_columns,
)
from crypto_analyzer.features.engineering import make_targets as make_default_targets
from crypto_analyzer.models.train import train_model
from crypto_analyzer.utils.config import CONFIG, FeatureSettings, override_feature_settings


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _prepare_settings(args: argparse.Namespace) -> FeatureSettings:
    settings = CONFIG.features
    overrides: dict[str, Any] = {}
    if args.include_onchain is not None:
        overrides["include_onchain"] = args.include_onchain
    if args.include_orderbook is not None:
        overrides["include_orderbook"] = args.include_orderbook
    if args.include_derivatives is not None:
        overrides["include_derivatives"] = args.include_derivatives
    if overrides:
        settings = override_feature_settings(settings, **overrides)

    if args.forward_fill_limit is not None or args.fillna_value is not None:
        settings = FeatureSettings(
            include_onchain=settings.include_onchain,
            include_orderbook=settings.include_orderbook,
            include_derivatives=settings.include_derivatives,
            forward_fill_limit=(
                args.forward_fill_limit
                if args.forward_fill_limit is not None
                else settings.forward_fill_limit
            ),
            fillna_value=(
                args.fillna_value if args.fillna_value is not None else settings.fillna_value
            ),
        )
    return settings


def _load_features(args: argparse.Namespace, settings: FeatureSettings) -> pd.DataFrame:
    if args.features is not None:
        df = _read_table(args.features)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return df

    raw = get_price_data(args.symbol, db_path=args.db_path)
    return create_features(raw, settings=settings)


def _ensure_label(df: pd.DataFrame, horizon: int, label: str) -> tuple[pd.DataFrame, str]:
    if label in df.columns:
        return df, label

    labeled = make_default_targets(df, horizon=horizon)
    target_col = f"cls_sign_{horizon}m"
    if target_col not in labeled.columns:
        raise ValueError(
            "Unable to infer training targets. Provide a --label column or ensure the input data "
            "contains OHLC prices so targets can be generated."
        )
    return labeled, target_col


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Crypto Analyzer meta-model")
    parser.add_argument(
        "--features",
        type=Path,
        help="Optional engineered feature table (CSV/Parquet). If omitted data is pulled from the DB.",
    )
    parser.add_argument(
        "--symbol",
        default=CONFIG.symbol,
        help="Trading symbol used when sourcing data from the database.",
    )
    parser.add_argument(
        "--db-path",
        default=CONFIG.db_path,
        help="SQLite database file to read raw price data from.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=120,
        help="Target horizon in minutes for label generation when not provided in the dataset.",
    )
    parser.add_argument(
        "--label",
        help="Existing label column to use. Defaults to cls_sign_<horizon>m.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("artifacts/meta_model.joblib"),
        help="Output path for the trained model.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("artifacts/oob_metrics.json"),
        help="Where to store evaluation metrics gathered during training.",
    )
    parser.add_argument(
        "--split",
        choices=("holdout", "walkforward"),
        default="holdout",
        help="Evaluation split used during training.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Hold-out fraction used for the validation split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for model training.",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration even when available.",
    )
    parser.add_argument(
        "--include-onchain",
        dest="include_onchain",
        action="store_true",
        help="Force-enable on-chain features regardless of config defaults.",
    )
    parser.add_argument(
        "--exclude-onchain",
        dest="include_onchain",
        action="store_false",
        help="Force-disable on-chain features regardless of config defaults.",
    )
    parser.add_argument(
        "--include-orderbook",
        dest="include_orderbook",
        action="store_true",
        help="Force-enable orderbook features regardless of config defaults.",
    )
    parser.add_argument(
        "--exclude-orderbook",
        dest="include_orderbook",
        action="store_false",
        help="Force-disable orderbook features regardless of config defaults.",
    )
    parser.add_argument(
        "--include-derivatives",
        dest="include_derivatives",
        action="store_true",
        help="Force-enable derivative features regardless of config defaults.",
    )
    parser.add_argument(
        "--exclude-derivatives",
        dest="include_derivatives",
        action="store_false",
        help="Force-disable derivative features regardless of config defaults.",
    )
    parser.add_argument(
        "--forward-fill-limit",
        type=int,
        help="Override forward-fill window for NaN handling.",
    )
    parser.add_argument(
        "--fillna-value",
        type=float,
        help="Override fallback value used when forward fill runs out.",
    )
    parser.add_argument(
        "--wfs-train-days",
        type=int,
        help="Training window size in days for walk-forward evaluation.",
    )
    parser.add_argument(
        "--wfs-test-days",
        type=int,
        help="Test window size in days for walk-forward evaluation.",
    )
    parser.add_argument(
        "--wfs-step-days",
        type=int,
        help="Step size in days when rolling the walk-forward window.",
    )
    parser.add_argument(
        "--wfs-min-train-days",
        type=int,
        help="Minimal amount of training data required for walk-forward evaluation.",
    )
    parser.set_defaults(include_onchain=None, include_orderbook=None, include_derivatives=None)
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = _prepare_settings(args)
    df = _load_features(args, settings)

    label = args.label or f"cls_sign_{args.horizon}m"
    df, label_col = _ensure_label(df, args.horizon, label)
    df = df.dropna(subset=[label_col]).sort_values("timestamp")

    feature_cols = get_feature_columns(settings)
    if not feature_cols:
        feature_cols = FEATURE_COLUMNS

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise KeyError(
            "Feature columns missing from dataset: " + ", ".join(sorted(missing))
        )

    X = df[feature_cols].astype("float32")
    if args.split == "walkforward" and "timestamp" in df.columns:
        X = X.assign(timestamp=df["timestamp"].values)

    y = df[label_col].astype("int8")

    if args.model_path:
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
    if args.log_path:
        args.log_path.parent.mkdir(parents=True, exist_ok=True)

    wfs_params = None
    if args.split == "walkforward":
        wfs_params = {
            "train_span_days": args.wfs_train_days or 30,
            "test_span_days": args.wfs_test_days or 7,
            "step_days": args.wfs_step_days or 7,
            "min_train_days": args.wfs_min_train_days or 30,
        }

    model = train_model(
        X,
        y,
        model_path=str(args.model_path),
        test_size=args.test_size,
        random_state=args.random_state,
        use_gpu=not args.no_gpu,
        log_path=str(args.log_path),
        split=args.split,
        wfs_params=wfs_params,
    )
    print(f"Model trained and stored at {args.model_path}")
    return args.model_path


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    main()
