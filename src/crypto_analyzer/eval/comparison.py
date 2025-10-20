from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from sqlalchemy import select, text

from crypto_analyzer.utils.config import CONFIG
from crypto_analyzer.utils.helpers import get_logger
from crypto_analyzer.data.db_connector import PRICES_TABLE, get_engine

logger = get_logger(__name__)


def _build_prices_query(symbol: str, target_times: Iterable[int]):
    """Return a parametrised SQLAlchemy statement for the prices lookup."""

    unique_times = sorted({int(ts) for ts in target_times})
    if not unique_times:
        return select(text("NULL AS ts_ms"), text("NULL AS close")).where(text("0=1"))

    return (
        select(PRICES_TABLE.c.open_time.label("ts_ms"), PRICES_TABLE.c.close)
        .where(PRICES_TABLE.c.symbol == symbol)
        .where(PRICES_TABLE.c.open_time.in_(unique_times))
    )


def backfill_actuals_and_errors(
    db_path: str | Path = CONFIG.db_path,
    table_pred: str = CONFIG.table_pred,
    symbol: str = CONFIG.symbol,
) -> None:
    """Fill in ``y_true_hat`` and ``abs_error`` for pending predictions.

    Only price rows matching the outstanding predictions are read which keeps
    memory usage small even for large tables.
    """

    engine = get_engine(db_path)
    with engine.begin() as conn:
        preds = pd.read_sql(
            text(
                f"""
                SELECT id, target_time_ms, p_hat
                FROM {table_pred}
                WHERE y_true_hat IS NULL AND symbol = :symbol
                """
            ),
            conn,
            params={"symbol": symbol},
        ).dropna(subset=["target_time_ms"])
        if preds.empty:
            logger.info("No predictions to backfill")
            return

        query = _build_prices_query(symbol, preds["target_time_ms"].to_list())
        actuals = pd.read_sql(query, conn)

        merged = preds.merge(
            actuals,
            left_on="target_time_ms",
            right_on="ts_ms",
            how="left",
        ).rename(columns={"close": "y_true_hat"})

        updates = [
            {
                "y_true_hat": float(row.y_true_hat),
                "abs_error": float(abs(row.p_hat - row.y_true_hat)),
                "id": int(row.id),
            }
            for row in merged.itertuples(index=False)
            if pd.notna(row.y_true_hat)
        ]
        if not updates:
            logger.info("No matching price data found for pending predictions")
            return

        conn.execute(
            text(
                f"""
                UPDATE {table_pred}
                SET y_true_hat = :y_true_hat, abs_error = :abs_error
                WHERE id = :id
                """
            ),
            updates,
        )
        logger.info("Backfill complete")
