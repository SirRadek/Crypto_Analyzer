"""Purge cached feature files."""

from __future__ import annotations

import string
import time
from pathlib import Path
from typing import Iterable

import typer

from crypto_analyzer.utils.cli import run_cli
from crypto_analyzer.utils.errors import DataValidationError
from crypto_analyzer.utils.feature_cache import default_cache_dir
from crypto_analyzer.utils.logging import get_logger

app = typer.Typer(add_completion=False, no_args_is_help=True)
logger = get_logger(__name__)

_HEX = set(string.hexdigits.lower())


def _is_cache_file(path: Path) -> bool:
    if path.suffix not in {".parquet", ".json"}:
        return False
    stem = path.stem.lower()
    if len(stem) != 64:
        return False
    return all(ch in _HEX for ch in stem)


def _collect_cache_entries(cache_dir: Path) -> list[tuple[str, float, list[Path]]]:
    entries: dict[str, list[Path]] = {}
    for path in cache_dir.iterdir():
        if not path.is_file() or not _is_cache_file(path):
            continue
        entries.setdefault(path.stem, []).append(path)

    results: list[tuple[str, float, list[Path]]] = []
    for stem, paths in entries.items():
        newest = max(p.stat().st_mtime for p in paths)
        results.append((stem, newest, paths))
    return results


def _filter_entries(
    entries: list[tuple[str, float, list[Path]]],
    *,
    max_age_days: float | None,
    keep_latest: int,
    purge_all: bool,
) -> list[tuple[str, float, list[Path]]]:
    if not purge_all and max_age_days is None and keep_latest <= 0:
        raise DataValidationError("Specify --all, --max-age-days, or --keep-latest to purge cache")

    selected = entries
    if max_age_days is not None and max_age_days > 0:
        cutoff = time.time() - (max_age_days * 86400.0)
        selected = [entry for entry in selected if entry[1] < cutoff]

    if keep_latest > 0:
        sorted_entries = sorted(entries, key=lambda item: item[1], reverse=True)
        keep = {stem for stem, _, _ in sorted_entries[:keep_latest]}
        selected = [entry for entry in selected if entry[0] not in keep]

    return selected


def _flatten_paths(entries: Iterable[tuple[str, float, list[Path]]]) -> list[Path]:
    paths: list[Path] = []
    for _, _, files in entries:
        paths.extend(files)
    return paths


@app.command()
def main(
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", help="Cache directory to purge."
    ),
    max_age_days: float | None = typer.Option(
        None, "--max-age-days", help="Delete cache files older than this many days."
    ),
    keep_latest: int = typer.Option(
        0, "--keep-latest", help="Keep the newest N cache entries."
    ),
    purge_all: bool = typer.Option(
        False, "--all", help="Remove all cached feature files."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview deletions without removing."),
) -> None:
    target_dir = cache_dir if cache_dir is not None else default_cache_dir()
    if target_dir is None:
        raise DataValidationError("No cache directory configured")
    if not target_dir.exists():
        raise DataValidationError(f"Cache directory '{target_dir}' does not exist")

    entries = _collect_cache_entries(target_dir)
    if not entries:
        typer.echo("No cache entries found.")
        return

    selected = _filter_entries(
        entries,
        max_age_days=max_age_days,
        keep_latest=keep_latest,
        purge_all=purge_all,
    )
    paths = _flatten_paths(selected)

    if not paths:
        typer.echo("No cache entries matched the purge criteria.")
        return

    for path in paths:
        if dry_run:
            typer.echo(f"Would remove {path}")
        else:
            path.unlink(missing_ok=True)

    logger.info(
        "Purged feature cache entries",
        extra={"event": "cache_purge", "entries": len(selected), "files": len(paths)},
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    run_cli(app)
