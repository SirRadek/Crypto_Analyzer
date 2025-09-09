import sqlite3

import pandas as pd


def backfill_actuals_and_errors(
    db_path="db/data/crypto_data.sqlite",
    table_pred="prediction",
    symbol="BTCUSDT"
):
    conn = sqlite3.connect(db_path)

    preds = pd.read_sql(
        f"SELECT id, target_time_ms FROM {table_pred} WHERE y_true IS NULL AND symbol = ?",
        conn, params=(symbol,)
    )
    if preds.empty:
        conn.close()
        print("No predictions to backfill."); return

    actuals = pd.read_sql(
        "SELECT open_time AS ts_ms, close FROM prices WHERE symbol = ?",
        conn, params=(symbol,)
    )

    merged = preds.merge(actuals, left_on="target_time_ms", right_on="ts_ms", how="left")
    merged.rename(columns={"close": "y_true"}, inplace=True)

    cur = conn.cursor()
    for _, r in merged.iterrows():
        if pd.notna(r.get("y_true")):
            row = cur.execute(
                f"SELECT y_pred FROM {table_pred} WHERE id = ?", (int(r["id"]),)
            ).fetchone()
            if row is None:
                continue
            y_pred = float(row[0])
            y_true = float(r["y_true"])
            abs_err = abs(y_pred - y_true)
            cur.execute(
                f"UPDATE {table_pred} SET y_true = ?, abs_error = ? WHERE id = ?",
                (y_true, abs_err, int(r["id"]))
            )

    conn.commit()
    conn.close()
    print("Backfill complete.")
