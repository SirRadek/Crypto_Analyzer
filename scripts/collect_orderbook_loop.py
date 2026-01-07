#!/usr/bin/env python
"""Collect Binance orderbook snapshots on a fixed interval."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

import typer

from crypto_analyzer.data.data_collector import fetch_binance_order_book
from crypto_analyzer.data.ingestion_store import ensure_schema, store_orderbook
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
        None, "--symbol", help="Optional symbol to snapshot (defaults to live_universe)."
    ),
    depth: int = typer.Option(
        CONFIG.orderbook.depth_levels,
        "--depth",
        help="Orderbook depth levels to request.",
    ),
    interval_seconds: int = typer.Option(
        60, "--interval-seconds", help="Interval between snapshots."
    ),
    iterations: Optional[int] = typer.Option(
        None, "--iterations", help="Optional number of loops before exiting."
    ),
    db_path: Optional[Path] = typer.Option(
        CONFIG.db_path,
        "--db-path",
        resolve_path=True,
        help="SQLite database path.",
    ),
) -> None:
    if interval_seconds <= 0:
        raise typer.BadParameter("--interval-seconds must be positive")
    if iterations is not None and iterations <= 0:
        raise typer.BadParameter("--iterations must be positive when provided")

    engine = ensure_schema(db_path=str(db_path))
    symbols = _symbol_list(symbol)

    counter = 0
    while True:
        loop_start = time.time()
        for sym in symbols:
            try:
                snapshot = fetch_binance_order_book(sym, depth=depth)
                result = store_orderbook(snapshot, engine=engine, symbol=sym)
                logger.info(
                    "Stored orderbook snapshot",
                    extra={"symbol": sym, "rows": result.inserted},
                )
            except Exception as exc:  # pragma: no cover - best effort loop
                logger.warning("Orderbook fetch failed", extra={"symbol": sym, "error": str(exc)})

        counter += 1
        if iterations is not None and counter >= iterations:
            break

        elapsed = time.time() - loop_start
        sleep_for = max(0.0, interval_seconds - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)
        logger.info(
            "Orderbook loop tick",
            extra={
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "iteration": counter,
                "sleep_seconds": round(sleep_for, 2),
            },
        )


if __name__ == "__main__":
    app()
