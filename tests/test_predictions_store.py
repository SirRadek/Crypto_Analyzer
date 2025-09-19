import sqlite3

from crypto_analyzer.data.predictions_store import create_predictions_table, save_predictions


def test_predictions_upsert(tmp_path):
    db_path = tmp_path / "preds.sqlite"
    create_predictions_table(db_path=str(db_path))
    row1 = ("BTC", "15m", 1_000, 10.0, 0.25)
    row2 = ("BTC", "15m", 1_000, 10.5, 0.75)
    save_predictions([row1], db_path=str(db_path))
    save_predictions([row2], db_path=str(db_path))
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM predictions")
        assert c.fetchone()[0] == 1
        c.execute("SELECT p_hat, prob_move_ge_05 FROM predictions WHERE symbol='BTC'")
        assert c.fetchone() == (10.5, 0.75)
