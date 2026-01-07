"""Risk checks for paper trading."""

from __future__ import annotations

from dataclasses import dataclass

from .types import Order, Portfolio, Position


@dataclass(slots=True)
class RiskLimits:
    max_wallet_exposure_pct: float = 0.15
    max_position_drawdown_pct: float = 0.40
    max_leverage: float = 100.0


def allow_order(
    order: Order,
    portfolio: Portfolio,
    *,
    limits: RiskLimits,
    position: Position | None = None,
    current_price: float | None = None,
) -> bool:
    if order.quantity <= 0 or order.price <= 0:
        return False
    if portfolio.equity <= 0:
        return False
    if position is not None and current_price is not None and current_price > 0:
        if position.quantity > 0:
            drawdown = (position.entry_price - current_price) / position.entry_price
        else:
            drawdown = (current_price - position.entry_price) / position.entry_price
        if drawdown >= limits.max_position_drawdown_pct:
            return False
    notional = order.quantity * order.price
    exposure_pct = notional / portfolio.equity
    if exposure_pct > limits.max_wallet_exposure_pct:
        return False
    return True
