import sqlite3
db = "db/data/crypto_data.sqlite"
with sqlite3.connect(db) as conn:
    cur = conn.cursor()
    # nejdřív zkontroluj kolik řádků to bude:
    cur.execute("""
        SELECT COUNT(*) FROM prices
        WHERE open_time <= 1694279700000

    """)
    print("k vymazání:", cur.fetchone()[0])

    # smaž
    cur.execute("""
        DELETE FROM prices
        WHERE open_time <= 1694279700000
    """)
    print("smazáno:", cur.rowcount)

    # lehká údržba
    cur.execute("PRAGMA optimize;")
