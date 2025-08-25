import sqlite3
db = "db/data/crypto_data.sqlite"
with sqlite3.connect(db) as conn:
    cur = conn.cursor()
    # nejdřív zkontroluj kolik řádků to bude:
    cur.execute("""
        SELECT COUNT(*) FROM prediction
        WHERE target_time_ms <= 1755993600000
    """)
    print("k vymazání:", cur.fetchone()[0])

    # smaž
    cur.execute("""
        DELETE FROM prediction
        WHERE target_time_ms <= 1755993600000
    """)
    print("smazáno:", cur.rowcount)

    # lehká údržba
    cur.execute("PRAGMA optimize;")
