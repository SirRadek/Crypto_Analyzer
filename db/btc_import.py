import requests
import sqlite3
import time
from datetime import datetime, timezone
import os

# Database location relative to repository root
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "crypto_data.sqlite")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
TABLE_NAME = "prices"
SYMBOL = "BTCUSDT"
INTERVAL = "5m"

# Download 5-minute candles between the specified timestamps


def get_klines(symbol, interval, start_ts, end_ts, limit=1000):
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ts,
        "endTime": end_ts,
        "limit": limit,
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"API error: {response.status_code}, {response.text}")
        return []
    return response.json()


# Create database and table if not exists


def create_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        open_time INTEGER PRIMARY KEY,
        symbol TEXT,
        interval TEXT,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        close_time INTEGER,
        quote_asset_volume REAL,
        number_of_trades INTEGER,
        taker_buy_base REAL,
        taker_buy_quote REAL
    )"""
    )
    conn.commit()
    conn.close()


# Save rows into the database


def save_to_db(rows, symbol, interval):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for row in rows:
        c.execute(
            f"""
        INSERT OR IGNORE INTO {TABLE_NAME}
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                int(row[0]),
                symbol,
                interval,
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
                int(row[6]),
                float(row[7]),
                int(row[8]),
                float(row[9]),
                float(row[10]),
            ),
        )
    conn.commit()
    conn.close()


# Import new data from the last stored timestamp up to now


def import_latest_data():
    create_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"SELECT MAX(open_time) FROM {TABLE_NAME}")
    row = cur.fetchone()
    conn.close()
    start_ts = row[0] + 1 if row and row[0] is not None else 0
    end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    if start_ts >= end_ts:
        print("Database already up to date.")
        return
    curr_ts = start_ts
    print(
        f"Downloading {SYMBOL}, interval {INTERVAL} from {datetime.fromtimestamp(start_ts/1000, timezone.utc)} to {datetime.fromtimestamp(end_ts/1000, timezone.utc)}"
    )
    while curr_ts < end_ts:
        klines = get_klines(SYMBOL, INTERVAL, curr_ts, end_ts)
        if not klines:
            break
        save_to_db(klines, SYMBOL, INTERVAL)
        curr_ts = klines[-1][0] + 1
        time.sleep(0.4)
    print("Done! Data saved in database:", DB_PATH)


def main():
    import_latest_data()


if __name__ == "__main__":
    main()
