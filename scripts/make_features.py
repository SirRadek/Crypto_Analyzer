#!/usr/bin/env python
"""Command-line entry point for feature engineering."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd

from crypto_analyzer.data.db_connector import get_price_data
from crypto_analyzer.features.engineering import create_features
from crypto_analyzer.utils.config import CONFIG, FeatureSettings, override_feature_settings


def _load_price_data(
    source: Literal["db", "file"],
    *,
    path: Path | None,
    symbol: str,
    db_path: str,
) -> pd.DataFrame:
    if source == "db":
        return get_price_data(symbol, db_path=db_path)
    if path is None:
        raise ValueError("Path must be provided when source='file'")
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return df


def _configure_features(
    *,
    settings: FeatureSettings,
    include_onchain: bool | None,
    include_orderbook: bool | None,
    include_derivatives: bool | None,
) -> FeatureSettings:
    overrides: dict[str, bool] = {}
    if include_onchain is not None:
        overrides["include_onchain"] = include_onchain
    if include_orderbook is not None:
        overrides["include_orderbook"] = include_orderbook
    if include_derivatives is not None:
        overrides["include_derivatives"] = include_derivatives
    if overrides:
        settings = override_feature_settings(settings, **overrides)
    return settings


def _write_output(df: pd.DataFrame, path: Path, fmt: Literal["parquet", "csv"]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate engineered features")
    parser.add_argument(
        "--source",
        choices=("db", "file"),
        default="db",
        help="Where to load raw price data from.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Optional CSV/Parquet file with raw OHLCV data when source=file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("features.parquet"),
        help="Destination file where engineered features will be stored.",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
        help="Output file format.",
    )
    parser.add_argument(
        "--symbol",
        default=CONFIG.symbol,
        help="Trading symbol to load when pulling data from the configured database.",
    )
    parser.add_argument(
        "--db-path",
        default=CONFIG.db_path,
        help="SQLite database file used when source=db.",
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
    parser.set_defaults(include_onchain=None, include_orderbook=None, include_derivatives=None)
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = _build_parser()
    args = parser.parse_args(argv)

    settings = CONFIG.features
    if args.forward_fill_limit is not None or args.fillna_value is not None:
        settings = FeatureSettings(
            include_onchain=settings.include_onchain,
            include_orderbook=settings.include_orderbook,
            include_derivatives=settings.include_derivatives,
            forward_fill_limit=args.forward_fill_limit
            if args.forward_fill_limit is not None
            else settings.forward_fill_limit,
            fillna_value=args.fillna_value if args.fillna_value is not None else settings.fillna_value,
        )

    settings = _configure_features(
        settings=settings,
        include_onchain=args.include_onchain,
        include_orderbook=args.include_orderbook,
        include_derivatives=args.include_derivatives,
    )

    df = _load_price_data(
        args.source,
        path=args.input,
        symbol=args.symbol,
        db_path=args.db_path,
    )
    feature_df = create_features(df, settings=settings)
    _write_output(feature_df, args.output, args.format)
    print(f"Features written to {args.output}")
    return args.output


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    main()
