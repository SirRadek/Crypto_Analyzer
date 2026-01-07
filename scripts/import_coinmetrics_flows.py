#!/usr/bin/env python
"""Backfill CoinMetrics exchange flows into the local database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from crypto_analyzer.data.ingestion_store import ensure_schema, store_coinmetrics_flows
from crypto_analyzer.data.onchain_fetcher import fetch_coinmetrics_exchange_flows
from crypto_analyzer.utils.config import CONFIG
from crypto_analyzer.utils.logging import get_logger

app = typer.Typer(add_completion=False, no_args_is_help=True)
logger = get_logger(__name__)


@app.command()
def main(
    assets: str = typer.Option(
        "btc",
        "--assets",
        help="Comma-separated CoinMetrics asset tickers (e.g. btc,eth).",
    ),
    days: int = typer.Option(180, "--days", help="Lookback window in days."),
    db_path: Optional[Path] = typer.Option(
        CONFIG.db_path,
        "--db-path",
        resolve_path=True,
        help="SQLite database path.",
    ),
) -> None:
    if days <= 0:
        raise typer.BadParameter("--days must be positive")

    end = pd.Timestamp(datetime.now(tz=UTC))
    start = end - timedelta(days=days)

    asset_list = [item.strip().lower() for item in assets.split(",") if item.strip()]
    if not asset_list:
        raise typer.BadParameter("--assets must include at least one asset")

    engine = ensure_schema(db_path=str(db_path))
    for asset in asset_list:
        logger.info(
            "Fetching CoinMetrics exchange flows",
            extra={"asset": asset, "start": start.isoformat(), "end": end.isoformat()},
        )
        flows = fetch_coinmetrics_exchange_flows(
            asset=asset,
            start=start,
            end=end,
        )
        result = store_coinmetrics_flows(flows, engine=engine, asset=asset)
        logger.info(
            "Stored CoinMetrics exchange flows",
            extra={"asset": asset, "rows": result.inserted},
        )


if __name__ == "__main__":
    app()
