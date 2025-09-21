"""Generate engineered features using a Typer based CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

import pandas as pd
import typer

from crypto_analyzer.data.db_connector import get_price_data
from crypto_analyzer.features.engineering import create_features
from crypto_analyzer.utils.config import CONFIG, FeatureSettings, override_feature_settings
from crypto_analyzer.utils.cli import run_cli
from crypto_analyzer.utils.errors import DataValidationError
from crypto_analyzer.utils.io import initialize_run, save_json
from crypto_analyzer.utils.logging import get_logger


app = typer.Typer(add_completion=False, no_args_is_help=True)
logger = get_logger(__name__)


def _load_price_data(
    source: Literal["db", "file"],
    *,
    path: Path | None,
    symbol: str,
    db_path: Path,
) -> pd.DataFrame:
    if source == "db":
        return get_price_data(symbol, db_path=db_path)
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


def _prepare_settings(
    *,
    forward_fill_limit: Optional[int],
    fillna_value: Optional[float],
    include_onchain: Optional[bool],
    include_orderbook: Optional[bool],
    include_derivatives: Optional[bool],
) -> FeatureSettings:
    settings = CONFIG.features
    if forward_fill_limit is not None or fillna_value is not None:
        settings = FeatureSettings(
            include_onchain=settings.include_onchain,
            include_orderbook=settings.include_orderbook,
            include_derivatives=settings.include_derivatives,
            forward_fill_limit=
            forward_fill_limit if forward_fill_limit is not None else settings.forward_fill_limit,
            fillna_value=fillna_value if fillna_value is not None else settings.fillna_value,
        )
    return _configure_features(
        settings=settings,
        include_onchain=include_onchain,
        include_orderbook=include_orderbook,
        include_derivatives=include_derivatives,
    )


def _resolve_output(
    *,
    output: Path,
    run_dir: Path,
) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    if not output.is_absolute():
        target_output = run_dir / output.name
    else:
        target_output = output
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
    source: Literal["db", "file"],
    input_path: Path | None,
    output: Path,
    fmt: Literal["parquet", "csv"],
    symbol: str,
    db_path: Path,
    forward_fill_limit: Optional[int],
    fillna_value: Optional[float],
    include_onchain: Optional[bool],
    include_orderbook: Optional[bool],
    include_derivatives: Optional[bool],
    use_derivatives: bool,
    use_orderbook: bool,
    run_id: str | None,
    dry_run: bool,
) -> Path:
    if forward_fill_limit is not None and forward_fill_limit < 0:
        raise DataValidationError("--forward-fill-limit must be non-negative")

    include_derivatives = True if use_derivatives else include_derivatives
    include_orderbook = True if use_orderbook else include_orderbook

    settings = _prepare_settings(
        forward_fill_limit=forward_fill_limit,
        fillna_value=fillna_value,
        include_onchain=include_onchain,
        include_orderbook=include_orderbook,
        include_derivatives=include_derivatives,
    )

    df = _load_price_data(source, path=input_path, symbol=symbol, db_path=db_path)
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

    _persist_metadata(run_id_value, {
        "source": source,
        "input": input_path,
        "output": output,
        "format": fmt,
        "symbol": symbol,
        "db_path": db_path,
        "forward_fill_limit": forward_fill_limit,
        "fillna_value": fillna_value,
        "include_onchain": include_onchain,
        "include_orderbook": include_orderbook,
        "include_derivatives": include_derivatives,
        "use_derivatives": use_derivatives,
        "use_orderbook": use_orderbook,
        "run_id": run_id_value,
        "dry_run": dry_run,
    })

    logger.info(
        "Persisted engineered features",
        extra={"event": "artefacts", "run_id": run_id_value, "path": str(run_output)},
    )

    _print_outputs(run_output, target_output if target_output != run_output else None)
    return run_output


@app.command()
def main(
    source: Literal["db", "file"] = typer.Option(
        "db", help="Where to load raw price data from."
    ),
    input_path: Path | None = typer.Option(
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
    fmt: Literal["parquet", "csv"] = typer.Option(
        "parquet", "--format", help="Output file format."
    ),
    symbol: str = typer.Option(CONFIG.symbol, "--symbol", help="Trading symbol to load."),
    db_path: Path = typer.Option(
        CONFIG.db_path,
        "--db-path",
        exists=False,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="SQLite database path when source=db.",
    ),
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
    use_derivatives: bool = typer.Option(
        False, "--use-derivatives", help="Convenience flag to enable derivative features."
    ),
    use_orderbook: bool = typer.Option(
        False, "--use-orderbook", help="Convenience flag to enable orderbook features."
    ),
    forward_fill_limit: Optional[int] = typer.Option(
        None, help="Override forward-fill window for NaN handling."
    ),
    fillna_value: Optional[float] = typer.Option(
        None, help="Override fallback value when forward fill runs out."
    ),
    run_id: str | None = typer.Option(None, help="Optional run identifier."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without writing."),
) -> None:
    if source == "file" and input_path is None:
        raise DataValidationError("--input is required when --source=file")
    if source == "db" and input_path is not None:
        typer.secho("Ignoring --input because --source=db", fg=typer.colors.YELLOW)

    _generate_features(
        source=source,
        input_path=input_path,
        output=output,
        fmt=fmt,
        symbol=symbol,
        db_path=db_path,
        forward_fill_limit=forward_fill_limit,
        fillna_value=fillna_value,
        include_onchain=include_onchain,
        include_orderbook=include_orderbook,
        include_derivatives=include_derivatives,
        use_derivatives=use_derivatives,
        use_orderbook=use_orderbook,
        run_id=run_id,
        dry_run=dry_run,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    run_cli(app)

