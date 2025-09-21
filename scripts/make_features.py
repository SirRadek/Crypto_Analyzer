#!/usr/bin/env python
"""Command-line entry point for feature engineering."""
from __future__ import annotations

import argparse
import json
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
        "--use_derivatives",
        action="store_true",
        help="Convenience flag to enable derivative features from auxiliary loaders.",
    )
    parser.add_argument(
        "--use_orderbook",
        action="store_true",
        help="Convenience flag to enable order book feature engineering.",
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
    parser.add_argument("--run-id", type=str, default=None, help="Optional run identifier.")
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

    if args.use_derivatives:
        args.include_derivatives = True
    if args.use_orderbook:
        args.include_orderbook = True

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

    run_id = args.run_id or pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs") / f"run_id={run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    default_output = parser.get_default("output")
    if args.output == default_output:
        target_output = run_dir / args.output.name
    else:
        target_output = args.output

    run_output = run_dir / target_output.name
    _write_output(feature_df, run_output, args.format)
    if target_output != run_output:
        _write_output(feature_df, target_output, args.format)

    config_dump = {
        "config": CONFIG.config_path.as_posix() if CONFIG.config_path else None,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    config_path = run_dir / "config_dump.json"
    config_path.write_text(json.dumps(config_dump, indent=2), encoding="utf-8")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Features written to {run_output}")
    if target_output != run_output:
        print(f"Features copied to {target_output}")
    return run_output


if __name__ == "__main__":  # pragma: no cover - CLI behaviour
    main()
