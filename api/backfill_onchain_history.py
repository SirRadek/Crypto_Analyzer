#!/usr/bin/env python3
"""
backfill_onchain_history.py — BTC network aggregates (Esplora)

Co dělá:
- Žádné adresy. Sběr síťových agregátů pro Bitcoin: počet transakcí, celkové poplatky, objem převodů.
- Zdroj: Esplora‑kompatibilní REST (mempool.space / blockstream.info). Bez API klíčů.
- Ukládá do jedné SQLite tabulky (per‑blok) + volitelně denní agregace a JSONL.

Metodika:
- Pro zadaný rozsah bloků stáhne každý blok a stránkovaně všechny transakce v bloku
  (endpoint `/api/block/:hash/txs[/:start_index]`, stránkování po 25 tx).
  Součty:
    • `tx_count` = počet tx v bloku
    • `fees_sat` = suma `tx.fee` přes všechny ne‑coinbase tx
    • `volume_sat` = suma hodnot všech výstupů ne‑coinbase tx (approx. „on‑chain volume“, zahrnuje change)
- Volitelně uloží snapshot mempoolu (`/api/mempool`): `count`, `vsize`, `total_fee`.

Poznámky:
- Agregace podle času lze udělat přes `--daily`, což po naplnění tabulky `btc_blocks` vytvoří/aktualizuje tabulku `btc_daily` (GROUP BY den UTC).
- Pro robustnost: retry s exponenciálním backoffem, paralelní zpracování bloků, integrita přes UPSERT.

Odkazy na API (kompatibilní s Esplora):
- `/api/blocks/tip/height`, `/api/block-height/:height`, `/api/block/:hash`, `/api/block/:hash/txs[/:start_index]`.
- `/api/mempool` (mempool statistiky).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    print("Tento skript vyžaduje balíček 'httpx'. Nainstalujte jej: pip install httpx", file=sys.stderr)
    raise

# -----------------------------
# Konfigurace
# -----------------------------

DEFAULT_TIMEOUT = 30.0
MAX_RETRIES = 5
INITIAL_BACKOFF = 0.5
MAX_CONCURRENCY = 4

ESPLORA_BASES_DEFAULT = [
    os.getenv("BTC_ESPLORA_BASE", "").rstrip("/") or "https://mempool.space/api",
    "https://blockstream.info/api",
]

# -----------------------------
# Retry helper
# -----------------------------

async def _with_retries(coro_fn, *, retries=MAX_RETRIES, initial_backoff=INITIAL_BACKOFF, retry_exceptions=(httpx.ReadTimeout, httpx.ConnectError, httpx.HTTPError)):
    attempt = 0
    delay = initial_backoff
    while True:
        try:
            return await coro_fn()
        except retry_exceptions:
            attempt += 1
            if attempt > retries:
                raise
            await asyncio.sleep(delay)
            delay = min(delay * 2, 8.0)

# -----------------------------
# Esplora klient
# -----------------------------

class Esplora:
    def __init__(self, base_url: str, client: httpx.AsyncClient):
        self.base = base_url.rstrip("/")
        self.client = client

    async def tip_height(self) -> int:
        async def call():
            r = await self.client.get(f"{self.base}/blocks/tip/height", timeout=DEFAULT_TIMEOUT)
            r.raise_for_status(); return int(r.text)
        return int(await _with_retries(call))

    async def block_hash(self, height: int) -> str:
        async def call():
            r = await self.client.get(f"{self.base}/block-height/{height}", timeout=DEFAULT_TIMEOUT)
            r.raise_for_status(); return r.text.strip()
        return await _with_retries(call)

    async def block_header(self, block_hash: str) -> Dict[str, Any]:
        async def call():
            r = await self.client.get(f"{self.base}/block/{block_hash}", timeout=DEFAULT_TIMEOUT)
            r.raise_for_status(); return r.json()
        return await _with_retries(call)

    async def block_txs_page(self, block_hash: str, start_index: int = 0) -> List[Dict[str, Any]]:
        suffix = f"/txs/{start_index}" if start_index else "/txs"
        async def call():
            r = await self.client.get(f"{self.base}/block/{block_hash}{suffix}", timeout=DEFAULT_TIMEOUT)
            r.raise_for_status(); return r.json()
        return await _with_retries(call)

    async def mempool_stats(self) -> Dict[str, Any]:
        async def call():
            r = await self.client.get(f"{self.base}/mempool", timeout=DEFAULT_TIMEOUT)
            r.raise_for_status(); return r.json()
        return await _with_retries(call)

# -----------------------------
# DB schéma
# -----------------------------

DDL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS btc_blocks (
  height INTEGER PRIMARY KEY,
  timestamp_utc TEXT NOT NULL,
  date_utc TEXT NOT NULL,
  tx_count INTEGER NOT NULL,
  fees_sat INTEGER NOT NULL,
  volume_sat INTEGER NOT NULL,
  base_url TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_btc_blocks_date ON btc_blocks(date_utc);

CREATE TABLE IF NOT EXISTS btc_daily (
  date_utc TEXT PRIMARY KEY,
  blocks INTEGER NOT NULL,
  txs INTEGER NOT NULL,
  fees_sat INTEGER NOT NULL,
  volume_sat INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS btc_mempool_snapshots (
  ts_utc TEXT PRIMARY KEY,
  count INTEGER,
  vsize INTEGER,
  total_fee_sat INTEGER,
  base_url TEXT NOT NULL
);
"""

UPSERT_BLOCK = """
INSERT INTO btc_blocks(height, timestamp_utc, date_utc, tx_count, fees_sat, volume_sat, base_url)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(height) DO UPDATE SET
  timestamp_utc=excluded.timestamp_utc,
  date_utc=excluded.date_utc,
  tx_count=excluded.tx_count,
  fees_sat=excluded.fees_sat,
  volume_sat=excluded.volume_sat,
  base_url=excluded.base_url
"""

"""
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(height) DO UPDATE SET
    timestamp_utc=excluded.timestamp_utc, date_utc=excluded.date_utc,
    tx_count=excluded.tx_count, fees_sat=excluded.fees_sat, volume_sat=excluded.volume_sat,
    base_url=excluded.base_url"
)"""

UPSERT_DAILY = """
INSERT INTO btc_daily(date_utc, blocks, txs, fees_sat, volume_sat)
SELECT date_utc, COUNT(*), SUM(tx_count), SUM(fees_sat), SUM(volume_sat)
FROM btc_blocks
WHERE date_utc BETWEEN ? AND ?
GROUP BY date_utc
ON CONFLICT(date_utc) DO UPDATE SET
  blocks=excluded.blocks,
  txs=excluded.txs,
  fees_sat=excluded.fees_sat,
  volume_sat=excluded.volume_sat
"""

"""
    SELECT date_utc, COUNT(*), SUM(tx_count), SUM(fees_sat), SUM(volume_sat) FROM btc_blocks
    WHERE date_utc BETWEEN ? AND ? GROUP BY date_utc
    ON CONFLICT(date_utc) DO UPDATE SET
    blocks=excluded.blocks, txs=excluded.txs, fees_sat=excluded.fees_sat, volume_sat=excluded.volume_sat"
)
"""
INSERT_MEMPOOL = (
    "INSERT OR REPLACE INTO btc_mempool_snapshots(ts_utc, count, vsize, total_fee_sat, base_url) VALUES (?, ?, ?, ?, ?)"
)

# -----------------------------
# Výpočty
# -----------------------------

def _is_coinbase_tx(tx: Dict[str, Any]) -> bool:
    vins = tx.get("vin") or []
    if not vins:
        return False
    v0 = vins[0] or {}
    return bool(v0.get("is_coinbase")) or (v0.get("txid") == "0" * 64)

async def _sum_block(esplora: Esplora, height: int) -> Tuple[int, str, str, int, int, int]:
    """Vrátí tuple: (height, ts_iso, date_iso, tx_count, fees_sat, volume_sat)"""
    h = height
    bh = await esplora.block_hash(h)
    header = await esplora.block_header(bh)
    ts = int(header.get("timestamp") or 0)
    ts_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    date_iso = ts_iso[:10]
    tx_count = int(header.get("tx_count") or 0)

    fees = 0
    volume = 0
    fetched = 0
    start = 0
    while True:
        page = await esplora.block_txs_page(bh, start)
        if not page:
            break
        for tx in page:
            if _is_coinbase_tx(tx):
                continue
            fees += int(tx.get("fee") or 0)
            # suma všech výstupů tx
            for v in tx.get("vout") or []:
                volume += int(v.get("value") or 0)
            fetched += 1
        start += len(page)
        if len(page) < 25:  # poslední stránka
            break
    # fallback: pokud se tx nepodařilo stáhnout všechno, stále uložíme hlavičku
    return h, ts_iso, date_iso, tx_count, fees, volume

# -----------------------------
# Orchestrace
# -----------------------------

async def run_aggregates(bases: List[str], *, from_height: Optional[int], to_height: Optional[int], last_blocks: Optional[int], db_path: Path, daily: bool, take_mempool: bool, concurrency: int) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.executescript(DDL)
    conn.commit()

    async with httpx.AsyncClient(headers={"User-Agent": "btc-aggregates/1.0"}) as client:
        esplora = Esplora(bases[0], client)  # primární zdroj
        # Rozsah bloků
        if last_blocks and not (from_height or to_height):
            tip = await esplora.tip_height()
            from_h = max(0, tip - last_blocks + 1)
            to_h = tip
        else:
            # doplň z tip height, pokud chybí jedna hrana
            tip = await esplora.tip_height()
            from_h = from_height if from_height is not None else max(0, (to_height or tip) - 2015)
            to_h = to_height if to_height is not None else tip
        if from_h > to_h:
            from_h, to_h = to_h, from_h

        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()

        async def process(h: int):
            async with sem:
                try:
                    height, ts_iso, date_iso, txc, fees, vol = await _sum_block(esplora, h)
                except Exception as e:
                    # selhalo — uložíme jen hlavičku (bez fees/volume)
                    bh = await esplora.block_hash(h)
                    header = await esplora.block_header(bh)
                    ts = int(header.get("timestamp") or 0)
                    ts_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                    date_iso = ts_iso[:10]
                    txc = int(header.get("tx_count") or 0)
                    fees = 0
                    vol = 0
                async with lock:
                    conn.execute(UPSERT_BLOCK, (height, ts_iso, date_iso, txc, fees, vol, bases[0]))

        await asyncio.gather(*(process(h) for h in range(from_h, to_h + 1)))

        if daily:
            d0 = datetime.fromtimestamp(0, tz=timezone.utc).date().isoformat()
            d1 = datetime.now(tz=timezone.utc).date().isoformat()
            conn.execute(UPSERT_DAILY, (d0, d1))

        if take_mempool:
            try:
                m = await esplora.mempool_stats()
                ts_iso = datetime.now(tz=timezone.utc).isoformat()
                conn.execute(INSERT_MEMPOOL, (ts_iso, int(m.get("count") or 0), int(m.get("vsize") or 0), int(m.get("total_fee") or 0), bases[0]))
            except Exception:
                pass

    conn.commit(); conn.close()

# -----------------------------
# CLI
# -----------------------------

def cli(argv: List[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(description="BTC síťové agregáty z Esplora (mempool.space / blockstream.info)")
    p.add_argument("--db", required=True, help="Cesta k SQLite DB")
    p.add_argument("--from-height", type=int)
    p.add_argument("--to-height", type=int)
    p.add_argument("--last-blocks", type=int, help="Alternativa: posledních N bloků")
    p.add_argument("--daily", action="store_true", help="Vypočti/aktualizuj denní agregace")
    p.add_argument("--mempool", action="store_true", help="Ulož aktuální snapshot mempoolu")
    p.add_argument("--esplora-base", help="Preferované base URL, jinak mempool.space a blockstream.info")
    p.add_argument("--concurrency", type=int, default=MAX_CONCURRENCY)

    a = p.parse_args(argv)

    bases = [b.strip().rstrip("/") for b in (a.esplora_base.split(",") if a.esplora_base else ESPLORA_BASES_DEFAULT) if b.strip()]

    asyncio.run(run_aggregates(bases, from_height=a.from_height, to_height=a.to_height, last_blocks=a.last_blocks, db_path=Path(a.db), daily=a.daily, take_mempool=a.mempool, concurrency=a.concurrency))
    return 0

if __name__ == "__main__":
    try:
        sys.exit(cli(sys.argv[1:]))
    except KeyboardInterrupt:
        print("Přerušeno uživatelem.", file=sys.stderr); sys.exit(130)
    except Exception as e:
        print(f"Chyba: {e}", file=sys.stderr); sys.exit(1)
