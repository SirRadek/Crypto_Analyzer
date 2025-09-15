import argparse
import sqlite3
import time
from typing import Iterable, Iterator, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry

COLUMNS = [
    "onch_fee_fast_satvb",
    "onch_fee_30m_satvb",
    "onch_fee_60m_satvb",
    "onch_fee_min_satvb",
    "onch_mempool_count",
    "onch_mempool_vsize_vB",
    "onch_mempool_total_fee_sat",
    "onch_fee_wavg_satvb",
    "onch_fee_p50_satvb",
    "onch_fee_p90_satvb",
    "onch_diff_progress_pct",
    "onch_diff_change_pct",
    "onch_blocks_remaining",
    "onch_retarget_ts",
]

USER_AGENT = "CryptoAnalyzer/1.0"


def _session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _get_json(
    session: requests.Session, url: str, *, params: dict | None = None, timeout: int = 30
) -> dict:
    delay = 1.0
    for attempt in range(5):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt == 4:
                return {}
            time.sleep(delay)
            delay *= 2


def _weighted_avg(histogram: Iterable[Iterable[float]]) -> float | None:
    """Compute weighted average feerate from a histogram.

    The histogram is expected to be an iterable of ``(fee, vsize)`` pairs.
    Returns ``None`` when no weights are present.
    """

    total = 0.0
    weighted = 0.0
    for fee, vsize in histogram:
        total += vsize
        weighted += fee * vsize
    if total == 0:
        return None
    return weighted / total


def _percentile(histogram: Iterable[Iterable[float]], pct: float) -> float | None:
    """Compute the percentile of a fee histogram.

    Parameters
    ----------
    histogram:
        Iterable of ``(fee, vsize)`` pairs sorted by ``fee``.
    pct:
        Desired percentile in the range ``0``--``1``.
    """

    hist = list(histogram)
    if not hist:
        return None
    total = sum(v for _, v in hist)
    if total == 0:
        return None
    cutoff = total * pct
    acc = 0.0
    for fee, vsize in hist:
        acc += vsize
        if acc >= cutoff:
            return fee
    return hist[-1][0]


def _reindex_5m(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(index=idx)
    return df.sort_index().reindex(idx, method="nearest", tolerance=pd.Timedelta("5min"))


def fetch_hoenicke_fees(
    start: pd.Timestamp, end: pd.Timestamp, session: requests.Session
) -> pd.DataFrame:
    """Fetch historical fee histogram data from Jochen Hoenicke's dataset.

    The dataset contains snapshots of the mempool fee histogram.  From this we
    derive weighted average fee and percentiles.  The function gracefully
    handles network failures by returning an empty DataFrame.
    """

    url = "https://mempool.jhoenicke.de/api/v1/fees/mempool.json"
    raw = _get_json(session, url, timeout=30).get("data", [])

    records = []
    for entry in raw:
        ts, hist = entry[0], entry[1]
        ts = pd.to_datetime(ts, unit="s", utc=True)
        if ts < start or ts > end:
            continue
        wavg = _weighted_avg(hist)
        p50 = _percentile(hist, 0.5)
        p90 = _percentile(hist, 0.9)
        vsize = sum(v for _, v in hist)
        records.append(
            {
                "ts": ts,
                "onch_fee_wavg_satvb": wavg,
                "onch_fee_p50_satvb": p50,
                "onch_fee_p90_satvb": p90,
                "onch_mempool_vsize_vB": vsize,
            }
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).set_index("ts")


def fetch_blockchain_mempool(
    start: pd.Timestamp, end: pd.Timestamp, session: requests.Session
) -> pd.DataFrame:
    """Fetch mempool statistics from the Blockchain.com Charts API."""

    url = "https://api.blockchain.info/charts/mempool-size"
    params = {
        "format": "json",
        "start": int(start.timestamp()),
        "end": int(end.timestamp()),
    }
    data = _get_json(session, url, params=params, timeout=30).get("values", [])

    # Blockchain.com does not expose count/fees directly; we map size to
    # ``onch_mempool_vsize_vB`` for validation.  Additional fields can be
    # joined from other sources if available.
    records = [
        {
            "ts": pd.to_datetime(item["x"], unit="s", utc=True),
            "onch_mempool_vsize_vB": item["y"],
        }
        for item in data
        if start <= pd.to_datetime(item["x"], unit="s", utc=True) <= end
    ]
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).set_index("ts")


def fetch_mining_difficulty(
    start: pd.Timestamp, end: pd.Timestamp, session: requests.Session
) -> pd.DataFrame:
    """Fetch historical difficulty adjustment data from mempool.space."""

    url = "https://mempool.space/api/v1/mining/difficulty-adjustments/1y"
    data = _get_json(session, url, timeout=30)

    records = []
    for item in data:
        ts = pd.to_datetime(item.get("time"), unit="s", utc=True)
        if ts < start or ts > end:
            continue
        records.append(
            {
                "ts": ts,
                "onch_diff_progress_pct": item.get("progressPercent"),
                "onch_diff_change_pct": item.get("difficultyChange"),
                "onch_blocks_remaining": item.get("remainingBlocks"),
                "onch_retarget_ts": item.get("estimatedRetargetDate"),
            }
        )
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).set_index("ts")


def fetch_current_snapshot(ts: pd.Timestamp, session: requests.Session) -> pd.DataFrame:
    """Fetch current mempool snapshot for nowcasting missing data."""

    fees = _get_json(session, "https://mempool.space/api/v1/fees/recommended", timeout=30)
    mempool = _get_json(session, "https://mempool.space/api/mempool", timeout=30)
    hist = mempool.get("fee_histogram", []) if isinstance(mempool, dict) else []

    record = {
        "ts": ts,
        "onch_fee_fast_satvb": fees.get("fastestFee"),
        "onch_fee_30m_satvb": fees.get("halfHourFee"),
        "onch_fee_60m_satvb": fees.get("hourFee"),
        "onch_fee_min_satvb": fees.get("minimumFee"),
        "onch_mempool_count": mempool.get("count"),
        "onch_mempool_vsize_vB": mempool.get("vsize"),
        "onch_mempool_total_fee_sat": mempool.get("total_fee"),
    }
    if hist:
        record["onch_fee_wavg_satvb"] = _weighted_avg(hist)
        record["onch_fee_p50_satvb"] = _percentile(hist, 0.5)
        record["onch_fee_p90_satvb"] = _percentile(hist, 0.9)
    df = pd.DataFrame([record]).set_index("ts")
    return df


def _date_chunks(
    start: pd.Timestamp, end: pd.Timestamp
) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp]]:
    cur = start
    step = pd.Timedelta(days=1)
    while cur <= end:
        nxt = min(cur + step - pd.Timedelta(minutes=5), end)
        yield cur, nxt
        cur = nxt + pd.Timedelta(minutes=5)


def backfill_onchain_history(start: str, end: str, db_path: str) -> None:
    """Backfill on-chain history into a SQLite database."""

    start_ts = pd.to_datetime(start, utc=True).floor("5min")
    end_ts = pd.to_datetime(end, utc=True).ceil("5min")
    idx = pd.date_range(start_ts, end_ts, freq="5min")
    base = pd.DataFrame(index=idx)

    session = _session()
    fee_parts = []
    bc_parts = []
    diff_parts = []
    for s, e in _date_chunks(start_ts, end_ts):
        fee_parts.append(fetch_hoenicke_fees(s, e, session))
        bc_parts.append(fetch_blockchain_mempool(s, e, session))
        diff_parts.append(fetch_mining_difficulty(s, e, session))
    fee_df = _reindex_5m(pd.concat(fee_parts) if fee_parts else pd.DataFrame(), idx)
    bc_df = _reindex_5m(pd.concat(bc_parts) if bc_parts else pd.DataFrame(), idx)
    diff_df = _reindex_5m(pd.concat(diff_parts) if diff_parts else pd.DataFrame(), idx)
    df = base.join([fee_df, bc_df, diff_df])

    # Fill last point with current snapshot if empty
    if df.iloc[-1].isna().all():
        snap = _reindex_5m(fetch_current_snapshot(idx[-1], session), idx)
        df.update(snap)

    if not df.index.is_monotonic_increasing:
        raise ValueError("non-monotonic index")
    if len(df) != len(idx):
        raise ValueError("unexpected number of rows")

    df.index = df.index.view("int64") // 10**9
    df.index.name = "ts_utc"
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS onchain_5m (
                ts_utc INTEGER PRIMARY KEY,
                onch_fee_fast_satvb REAL,
                onch_fee_30m_satvb REAL,
                onch_fee_60m_satvb REAL,
                onch_fee_min_satvb REAL,
                onch_mempool_count REAL,
                onch_mempool_vsize_vB REAL,
                onch_mempool_total_fee_sat REAL,
                onch_fee_wavg_satvb REAL,
                onch_fee_p50_satvb REAL,
                onch_fee_p90_satvb REAL,
                onch_diff_progress_pct REAL,
                onch_diff_change_pct REAL,
                onch_blocks_remaining REAL,
                onch_retarget_ts INTEGER
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS ix_onchain_ts ON onchain_5m(ts_utc)")
        cols = ["ts_utc"] + COLUMNS
        placeholders = ",".join(["?"] * len(cols))
        rows = []
        for ts, row in df.iterrows():
            values = [ts] + [row.get(c) if pd.notna(row.get(c)) else None for c in COLUMNS]
            rows.append(values)
        for i in range(0, len(rows), 1000):
            conn.executemany(
                f"INSERT OR REPLACE INTO onchain_5m ({','.join(cols)}) VALUES ({placeholders})",
                rows[i : i + 1000],
            )
        conn.commit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill on-chain history")
    parser.add_argument("--start", default="2020-07-22", help="start date (UTC)")
    parser.add_argument("--end", required=True, help="end date (UTC)")
    parser.add_argument("--db", required=True, help="SQLite database path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backfill_onchain_history(args.start, args.end, args.db)


if __name__ == "__main__":  # pragma: no cover
    main()
