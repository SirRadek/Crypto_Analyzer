from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


def _build_prices_query(symbol: str, target_times: Iterable[int]) -> tuple[str, list[int]]:
    """Return SQL query and parameters for the prices lookup."""

    unique_times = sorted({int(ts) for ts in target_times})
    if not unique_times:
        return "SELECT NULL AS ts_ms, NULL AS close WHERE 0", []

    placeholders = ",".join("?" for _ in unique_times)
    query = (
        "SELECT open_time AS ts_ms, close FROM prices "
        f"WHERE symbol = ? AND open_time IN ({placeholders})"
    )
    params = [symbol, *unique_times]
    return query, params


def backfill_actuals_and_errors(
    db_path: str | Path = "data/crypto_data.sqlite",
    table_pred: str = "predictions",
    symbol: str = "BTCUSDT",
) -> None:
    """Fill in ``y_true_hat`` and ``abs_error`` for pending predictions.

    Only price rows matching the outstanding predictions are read which keeps
    memory usage small even for large tables.
    """

    with sqlite3.connect(str(db_path)) as conn:
        preds = pd.read_sql(
            f"""
            SELECT id, target_time_ms, p_hat
            FROM {table_pred}
            WHERE y_true_hat IS NULL AND symbol = ?
            """,
            conn,
            params=(symbol,),
        ).dropna(subset=["target_time_ms"])
        if preds.empty:
            print("No predictions to backfill.")
            return

        query, params = _build_prices_query(symbol, preds["target_time_ms"].to_list())
        actuals = pd.read_sql(query, conn, params=params)

        merged = preds.merge(
            actuals,
            left_on="target_time_ms",
            right_on="ts_ms",
            how="left",
        ).rename(columns={"close": "y_true_hat"})

        updates = [
            (float(row.y_true_hat), float(abs(row.p_hat - row.y_true_hat)), int(row.id))
            for row in merged.itertuples(index=False)
            if pd.notna(row.y_true_hat)
        ]
        if not updates:
            print("No matching price data found for pending predictions.")
            return

        conn.executemany(
            f"UPDATE {table_pred} SET y_true_hat = ?, abs_error = ? WHERE id = ?",
            updates,
        )
        conn.commit()
        print("Backfill complete.")
