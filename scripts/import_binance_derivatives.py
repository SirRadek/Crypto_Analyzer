#!/usr/bin/env python
"""Backfill Binance futures funding/OI (and latest basis) into SQLite."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from crypto_analyzer.data.data_collector import (
    fetch_binance_basis,
    fetch_binance_funding_rates,
    fetch_binance_open_interest,
)
from crypto_analyzer.data.ingestion_store import ensure_schema, store_derivatives
from crypto_analyzer.utils.config import CONFIG
from crypto_analyzer.utils.logging import get_logger

app = typer.Typer(add_completion=False, no_args_is_help=True)
logger = get_logger(__name__)


def _symbol_list(symbol: Optional[str]) -> list[str]:
    if symbol:
        return [symbol]
    universe = list(CONFIG.live_universe) if CONFIG.live_universe else []
    return universe or [CONFIG.symbol]


@app.command()
def main(
    symbol: Optional[str] = typer.Option(
        None, "--symbol", help="Optional symbol to backfill (defaults to live_universe)."
    ),
    days: int = typer.Option(7, "--days", help="Number of trailing days to backfill."),
    period: Optional[str] = typer.Option(
        None, "--period", help="Open interest period (defaults to config interval)."
    ),
    db_path: Optional[Path] = typer.Option(
        CONFIG.db_path,
        "--db-path",
        resolve_path=True,
        help="SQLite database path.",
    ),
) -> None:
    if days <= 0:
        raise typer.BadParameter("--days must be positive")
    period_value = period or CONFIG.interval

    engine = ensure_schema(db_path=str(db_path))
    end = datetime.now(tz=UTC)
    start = end - timedelta(days=days)

    for sym in _symbol_list(symbol):
        logger.info("Backfilling derivatives", extra={"symbol": sym, "start": start, "end": end})
        funding = fetch_binance_funding_rates(sym, start, end)
        open_interest = fetch_binance_open_interest(sym, start, end, period=period_value)

        basis = pd.DataFrame()
        try:
            basis = fetch_binance_basis(sym)
        except Exception:  # pragma: no cover - best effort
            logger.warning("Failed to fetch basis", extra={"symbol": sym})

        result = store_derivatives(
            funding,
            open_interest,
            basis,
            engine=engine,
            symbol=sym,
        )
        logger.info(
            "Stored derivatives rows",
            extra={"symbol": sym, "rows": result.inserted},
        )


if __name__ == "__main__":
    app()
