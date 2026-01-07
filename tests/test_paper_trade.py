from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from crypto_analyzer.live.paper_broker import execute_order
from crypto_analyzer.live.risk import RiskLimits, allow_order
from crypto_analyzer.live.types import Order, Portfolio, Position
from scripts.paper_trade import _summary_from_ledger, _write_ledger, _write_summary


def test_drawdown_limit_blocks_order() -> None:
    portfolio = Portfolio(equity=10_000.0, cash=10_000.0, positions={})
    position = Position(symbol="BTCUSDC", quantity=1.0, entry_price=100.0, leverage=1.0)
    portfolio.positions["BTCUSDC"] = position

    order = Order(symbol="BTCUSDC", side="buy", quantity=0.1, price=90.0)
    limits = RiskLimits(max_position_drawdown_pct=0.05)
    allowed = allow_order(order, portfolio, limits=limits, position=position, current_price=90.0)
    assert not allowed


def test_paper_execute_updates_position_and_cash(tmp_path: Path) -> None:
    portfolio = Portfolio(equity=10_000.0, cash=10_000.0, positions={})
    order = Order(symbol="BTCUSDC", side="buy", quantity=1.0, price=100.0)
    result = execute_order(order, portfolio, fee_bps=10.0, slippage_bps=0.0)
    assert result.filled_qty == 1.0
    assert portfolio.positions["BTCUSDC"].quantity == 1.0
    assert portfolio.cash < 10_000.0


def test_write_ledger_creates_csv(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.csv"
    ledger = [
        {
            "run_id": "test",
            "timestamp": pd.Timestamp(datetime(2024, 1, 1, tzinfo=timezone.utc)),
            "symbol": "BTCUSDC",
            "signal": 0.7,
            "price": 100.0,
            "equity": 10_000.0,
            "cash": 9_990.0,
            "position_qty": 0.01,
        }
    ]
    _write_ledger(ledger, ledger_path)
    assert ledger_path.exists()
    frame = pd.read_csv(ledger_path)
    assert "symbol" in frame.columns


def test_write_summary_creates_json(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    ledger = [
        {"equity": 10_000.0},
        {"equity": 9_500.0},
        {"equity": 10_200.0},
    ]
    summary = _summary_from_ledger(ledger)
    _write_summary(summary, summary_path)
    assert summary_path.exists()
