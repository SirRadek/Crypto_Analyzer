import sqlite3

from crypto_analyzer.eval.comparison import backfill_actuals_and_errors
from crypto_analyzer.data.predictions_store import create_predictions_table


def test_backfill_updates_predictions(tmp_path):
    db_path = tmp_path / "preds.sqlite"
    create_predictions_table(db_path=db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO predictions (symbol, interval, target_time_ms, p_hat)"
            " VALUES (?, ?, ?, ?)",
            ("BTCUSDT", "15m", 123, 10.0),
        )
        conn.execute(
            "CREATE TABLE prices (symbol TEXT, open_time INTEGER, close REAL)"
        )
        conn.execute(
            "INSERT INTO prices (symbol, open_time, close) VALUES (?, ?, ?)",
            ("BTCUSDT", 123, 9.5),
        )
        conn.commit()

    backfill_actuals_and_errors(db_path=db_path, table_pred="predictions", symbol="BTCUSDT")

    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT y_true_hat, abs_error FROM predictions WHERE symbol = ?",
            ("BTCUSDT",),
        )
        y_true, abs_err = cur.fetchone()

    assert y_true == 9.5
    assert abs_err == 0.5
