import sqlite3

from db.predictions_store import create_predictions_table, save_predictions


def test_predictions_upsert(tmp_path):
    db_path = tmp_path / "preds.sqlite"
    create_predictions_table(db_path=str(db_path))
    row1 = ("BTC", "5m", 1_000, 10.0)
    row2 = ("BTC", "5m", 1_000, 10.5)
    save_predictions([row1], db_path=str(db_path))
    save_predictions([row2], db_path=str(db_path))
    with sqlite3.connect(db_path) as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM predictions")
        assert c.fetchone()[0] == 1
        c.execute("SELECT p_hat FROM predictions WHERE symbol='BTC'")
        assert c.fetchone() == (10.5,)
