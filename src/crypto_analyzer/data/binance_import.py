import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

from crypto_analyzer.data.db_connector import (
    get_latest_open_time,
    init_timescale,
    save_to_db,
)
from crypto_analyzer.utils.config import CONFIG
from crypto_analyzer.utils.helpers import ensure_dir_exists, get_logger

logger = get_logger(__name__)

DB_PATH = Path(CONFIG.db_path)
ensure_dir_exists(DB_PATH.parent)
DB_TARGET = getattr(CONFIG, "db_url", None) or str(DB_PATH)
TABLE_NAME = "prices"
SYMBOL = CONFIG.symbol
SYMBOLS = list(getattr(CONFIG, "live_universe", []) or [SYMBOL])
INTERVAL = CONFIG.interval
LOOKBACK_DAYS = CONFIG.core.history_days
SPOT_KLINES_ENDPOINT = "https://api.binance.com/api/v3/klines"
FUTURES_KLINES_ENDPOINT = "https://fapi.binance.com/fapi/v1/klines"


def get_klines(symbol, interval, start_ts, end_ts, limit=1000):
    use_futures = symbol.endswith(("USDT", "USDC"))
    url = FUTURES_KLINES_ENDPOINT if use_futures else SPOT_KLINES_ENDPOINT
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ts,
        "endTime": end_ts,
        "limit": limit,
    }
    response = requests.get(url, params=params, timeout=10)
    if response.status_code != 200:
        logger.error(
            "Binance klines API error",
            extra={"status": response.status_code, "body": response.text},
        )
        return []
    return response.json()


def create_db():
    init_timescale()


def import_latest_data():
    create_db()

    # hranice: teď a teď - LOOKBACK_DAYS
    now_utc = datetime.now(UTC)
    end_ts = int(now_utc.timestamp() * 1000)
    lookback_ts = int((now_utc - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)

    for symbol in SYMBOLS:
        # pokračuj od posledního záznamu, ale nikdy ne dřív než lookback
        last_db_ts = get_latest_open_time(symbol=symbol, interval=INTERVAL)
        if last_db_ts is None:
            start_ts = lookback_ts
        else:
            start_ts = max(last_db_ts + 1, lookback_ts)

        if start_ts >= end_ts:
            logger.info(
                "V daném okně není co stáhnout.",
                extra={"symbol": symbol, "interval": INTERVAL},
            )
            continue

        logger.info(
            "Downloading klines",
            extra={
                "symbol": symbol,
                "interval": INTERVAL,
                "start": datetime.fromtimestamp(start_ts / 1000, UTC).isoformat(),
                "end": datetime.fromtimestamp(end_ts / 1000, UTC).isoformat(),
            },
        )

        curr_ts = start_ts
        while curr_ts < end_ts:
            klines = get_klines(symbol, INTERVAL, curr_ts, end_ts)
            if not klines:
                break
            save_to_db(klines, symbol, INTERVAL)
            curr_ts = klines[-1][0] + 1
            time.sleep(0.4)

    logger.info("Import hotov", extra={"db_target": DB_TARGET})


def main():
    import_latest_data()


if __name__ == "__main__":
    main()
