"""Generate engineered features using a Typer based CLI."""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import typer

from crypto_analyzer.data.store import PriceDataStore, resolve_data_store
from crypto_analyzer.features.engineering import create_features
from crypto_analyzer.utils.cli import run_cli
from crypto_analyzer.utils.config import CONFIG, FeatureSettings, override_feature_settings
from crypto_analyzer.utils.errors import DataValidationError
from crypto_analyzer.utils.io import initialize_run, save_json
from crypto_analyzer.utils.logging import get_logger
from crypto_analyzer.utils.feature_cache import (
    build_cache_key,
    build_cache_payload,
    cache_paths,
    default_cache_dir,
)
from crypto_analyzer.utils.profiling import profile_section


class SourceChoice(str, Enum):
    db = "db"
    file = "file"


class FormatChoice(str, Enum):
    parquet = "parquet"
    csv = "csv"


class StoreChoice(str, Enum):
    auto = "auto"
    sqlite = "sqlite"
    timescale = "timescale"


app = typer.Typer(add_completion=False, no_args_is_help=True)
logger = get_logger(__name__)


def _load_price_data(
    source: SourceChoice,
    *,
    path: Path | None,
    symbol: str,
    data_store: PriceDataStore | None,
) -> pd.DataFrame:
    if source == SourceChoice.db:
        if data_store is None:
            raise DataValidationError("Database source requested but no data store was configured")
        return data_store.fetch_prices(symbol)
    if path is None:
        raise DataValidationError("--input must be provided when --source=file")
    if not path.exists():
        raise DataValidationError(f"Input file '{path}' does not exist")
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
    include_sentiment: bool | None,
) -> FeatureSettings:
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


def _write_output(df: pd.DataFrame, path: Path, fmt: FormatChoice) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == FormatChoice.parquet:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _prepare_settings(
    *,
    forward_fill_limit: int | None,
    fillna_value: float | None,
    include_onchain: bool | None,
    include_orderbook: bool | None,
    include_derivatives: bool | None,
    include_sentiment: bool | None,
) -> FeatureSettings:
    settings = CONFIG.features
    if forward_fill_limit is not None or fillna_value is not None:
        settings = FeatureSettings(
            include_onchain=settings.include_onchain,
            include_orderbook=settings.include_orderbook,
            include_derivatives=settings.include_derivatives,
            include_sentiment=settings.include_sentiment,
            forward_fill_limit=forward_fill_limit
            if forward_fill_limit is not None
            else settings.forward_fill_limit,
            fillna_value=fillna_value if fillna_value is not None else settings.fillna_value,
        )
    return _configure_features(
        settings=settings,
        include_onchain=include_onchain,
        include_orderbook=include_orderbook,
        include_derivatives=include_derivatives,
        include_sentiment=include_sentiment,
    )


def _resolve_output(
    *,
    output: Path,
    run_dir: Path,
) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if output.is_absolute():
        target_output = output
    elif output.parent != Path("."):
        target_output = output
    else:
        target_output = run_dir / output.name
    return target_output, run_dir / output.name


def _persist_metadata(run_id: str, args: dict[str, Any]) -> None:
    config_dump = {
        "config": CONFIG.config_path.as_posix() if CONFIG.config_path else None,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in args.items()},
    }
    save_json(config_dump, "config_dump.json", run_id=run_id)


def _print_outputs(primary: Path, secondary: Path | None) -> None:
    typer.echo(f"Features written to {primary}")
    if secondary and secondary != primary:
        typer.echo(f"Features copied to {secondary}")


def _generate_features(
    *,
    source: SourceChoice,
    input_path: Path | None,
    output: Path,
    fmt: FormatChoice,
    symbol: str,
    data_store: PriceDataStore | None,
    forward_fill_limit: int | None,
    fillna_value: float | None,
    include_onchain: bool | None,
    include_orderbook: bool | None,
    include_derivatives: bool | None,
    include_sentiment: bool | None,
    run_id: str | None,
    dry_run: bool,
    use_cache: bool,
    cache_dir: Path | None,
) -> Path:
    if forward_fill_limit is not None and forward_fill_limit < 0:
        raise DataValidationError("--forward-fill-limit must be non-negative")

    settings = _prepare_settings(
        forward_fill_limit=forward_fill_limit,
        fillna_value=fillna_value,
        include_onchain=include_onchain,
        include_orderbook=include_orderbook,
        include_derivatives=include_derivatives,
        include_sentiment=include_sentiment,
    )

    cache_root = None if not use_cache else (cache_dir if cache_dir is not None else default_cache_dir())
    cache_key: str | None = None
    cache_parquet: Path | None = None
    cache_meta: Path | None = None
    cache_payload: dict[str, Any] | None = None

    feature_df: pd.DataFrame | None = None
    if use_cache and cache_root is not None:
        store_label = getattr(data_store, "label", None)
        store_location = None
        if data_store is not None:
            location = getattr(data_store, "path", None) or getattr(data_store, "url", None)
            if location is not None:
                store_location = str(location)
        latest_open = None
        if data_store is not None:
            try:
                latest_open = data_store.latest_open_time(symbol=symbol, interval=CONFIG.interval)
            except Exception:  # pragma: no cover - cache is best-effort
                latest_open = None

        cache_payload = build_cache_payload(
            source=source.value,
            symbol=symbol,
            input_path=input_path,
            settings=settings,
            store_label=store_label,
            store_location=store_location,
            latest_open_time=latest_open,
        )
        cache_key = build_cache_key(cache_payload)
        cache_parquet, cache_meta = cache_paths(cache_root, cache_key)
        if cache_parquet.exists():
            logger.info(
                "Loaded features from cache",
                extra={"event": "cache_hit", "path": str(cache_parquet)},
            )
            feature_df = pd.read_parquet(cache_parquet)

    if feature_df is None:
        df = _load_price_data(source, path=input_path, symbol=symbol, data_store=data_store)
        feature_df = create_features(df, settings=settings)

    run_id_value, run_dir, _ = initialize_run(run_id, deterministic_torch=False)
    logger.info(
        "Prepared feature generation run",
        extra={"event": "initialised", "run_id": run_id_value, "rows": int(len(feature_df))},
    )
    target_output, run_output = _resolve_output(output=output, run_dir=run_dir)

    if dry_run:
        typer.echo("Dry run requested; skipping file writes.")
        _print_outputs(run_output, None)
        return run_output

    _write_output(feature_df, run_output, fmt)
    if target_output != run_output:
        _write_output(feature_df, target_output, fmt)

    if cache_parquet is not None and cache_key is not None:
        feature_df.to_parquet(cache_parquet, index=False)
        if cache_meta is not None and cache_payload is not None:
            cache_payload["output_path"] = str(cache_parquet)
            cache_payload["rows"] = int(len(feature_df))
            cache_meta.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")

    store_label = getattr(data_store, "label", None)
    if store_label == "sqlite":
        store_location = str(getattr(data_store, "path", None))
    elif store_label == "timescale":
        store_location = getattr(data_store, "url", None)
    else:
        store_location = None

    _persist_metadata(
        run_id_value,
        {
            "source": source,
            "input": input_path,
            "output": output,
            "format": fmt,
            "symbol": symbol,
            "data_store": store_label,
            "store_location": store_location,
            "forward_fill_limit": forward_fill_limit,
            "fillna_value": fillna_value,
            "include_onchain": include_onchain,
            "include_orderbook": include_orderbook,
            "include_derivatives": include_derivatives,
            "include_sentiment": include_sentiment,
            "run_id": run_id_value,
            "dry_run": dry_run,
        },
    )

    logger.info(
        "Persisted engineered features",
        extra={"event": "artefacts", "run_id": run_id_value, "path": str(run_output)},
    )

    _print_outputs(run_output, target_output if target_output != run_output else None)
    return run_output


@app.command()
def main(
    source: SourceChoice = typer.Option(SourceChoice.db, help="Where to load raw price data from."),
    input_path: Optional[Path] = typer.Option(
        None,
        "--input",
        exists=False,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
        help="Optional CSV/Parquet file when --source=file.",
    ),
    output: Path = typer.Option(
        Path("features.parquet"),
        "--output",
        help="Destination path for engineered features.",
    ),
    fmt: FormatChoice = typer.Option(FormatChoice.parquet, "--format", help="Output file format."),
    symbol: str = typer.Option(CONFIG.symbol, "--symbol", help="Trading symbol to load."),
    store_choice: StoreChoice = typer.Option(
        StoreChoice.auto,
        "--store",
        help="Database backend to use when --source=db. 'auto' follows config defaults.",
    ),
    db_path: Optional[Path] = typer.Option(
        CONFIG.db_path,
        "--db-path",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Override SQLite database path when using the local store.",
    ),
    db_url: Optional[str] = typer.Option(
        CONFIG.db_url,
        "--db-url",
        help="Override SQLAlchemy URL for Timescale/PostgreSQL connections.",
    ),
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
        None, help="Override forward-fill window for NaN handling."
    ),
    fillna_value: Optional[float] = typer.Option(
        None, help="Override fallback value when forward fill runs out."
    ),
    run_id: Optional[str] = typer.Option(None, help="Optional run identifier."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without writing."),
    use_cache: bool = typer.Option(
        CONFIG.database.feature_store is not None,
        "--use-cache/--no-cache",
        help="Cache engineered features to speed up reruns.",
    ),
    cache_dir: Optional[Path] = typer.Option(
        None, "--cache-dir", help="Override feature cache directory."
    ),
    profile: bool = typer.Option(False, "--profile", help="Enable cProfile output."),
    profile_path: Optional[Path] = typer.Option(
        None, "--profile-path", help="Optional path for cProfile output."
    ),
) -> None:
    if source == SourceChoice.file and input_path is None:
        raise DataValidationError("--input is required when --source=file")
    if source == SourceChoice.db and input_path is not None:
        typer.secho("Ignoring --input because --source=db", fg=typer.colors.YELLOW)

    data_store: PriceDataStore | None = None
    if source == SourceChoice.db:
        data_store = resolve_data_store(
            store_choice.value,
            sqlite_path=db_path,
            timescale_url=db_url,
        )

    with profile_section(profile, output=profile_path):
        _generate_features(
            source=source,
            input_path=input_path,
            output=output,
            fmt=fmt,
            symbol=symbol,
            data_store=data_store,
            forward_fill_limit=forward_fill_limit,
            fillna_value=fillna_value,
            include_onchain=include_onchain,
            include_orderbook=include_orderbook,
        include_derivatives=include_derivatives,
        include_sentiment=include_sentiment,
        run_id=run_id,
        dry_run=dry_run,
        use_cache=use_cache,
        cache_dir=cache_dir,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    run_cli(app)
