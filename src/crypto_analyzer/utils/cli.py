"""Utilities shared across Typer-based CLI entrypoints."""

from __future__ import annotations

import typer

from crypto_analyzer.utils.errors import ConfigError, DataValidationError, ModelError


def run_cli(app: typer.Typer) -> None:
    """Execute *app* while mapping domain exceptions to exit codes."""

    try:
        app()
    except DataValidationError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise SystemExit(2) from exc
    except ConfigError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise SystemExit(3) from exc
    except ModelError as exc:
        typer.secho(str(exc), err=True, fg=typer.colors.RED)
        raise SystemExit(4) from exc


__all__ = ["run_cli"]
