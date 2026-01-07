#!/usr/bin/env python
"""Store a fresh Binance order book snapshot for each symbol."""

from __future__ import annotations

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
    db_path: Optional[Path] = typer.Option(
        CONFIG.db_path,
        "--db-path",
        resolve_path=True,
        help="SQLite database path.",
    ),
) -> None:
    engine = ensure_schema(db_path=str(db_path))
    for sym in _symbol_list(symbol):
        logger.info("Fetching orderbook", extra={"symbol": sym, "depth": depth})
        try:
            snapshot = fetch_binance_order_book(sym, depth=depth)
            result = store_orderbook(snapshot, engine=engine, symbol=sym)
            logger.info("Stored orderbook snapshot", extra={"symbol": sym, "rows": result.inserted})
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning(
                "Orderbook fetch failed",
                extra={"symbol": sym, "error": str(exc)},
            )


if __name__ == "__main__":
    app()
