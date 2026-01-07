"""Paper trading execution simulator."""

from __future__ import annotations

from dataclasses import dataclass

from .types import Order, Portfolio, Position


@dataclass(slots=True)
class PaperFillResult:
    filled_qty: float
    fill_price: float
    fee_paid: float


def execute_order(
    order: Order,
    portfolio: Portfolio,
    *,
    fee_bps: float = 10.0,
    slippage_bps: float = 10.0,
    partial_fill_pct: float = 1.0,
) -> PaperFillResult:
    fill_qty = max(0.0, min(order.quantity, order.quantity * partial_fill_pct))
    fee = (fee_bps / 10_000.0) * (fill_qty * order.price)
    slip = (slippage_bps / 10_000.0) * order.price
    fill_price = order.price + slip if order.side == "buy" else order.price - slip

    pos = portfolio.positions.get(order.symbol)
    delta = fill_qty if order.side == "buy" else -fill_qty
    if pos is None:
        portfolio.positions[order.symbol] = Position(
            symbol=order.symbol,
            quantity=delta,
            entry_price=fill_price,
            leverage=1.0,
        )
    else:
        new_qty = pos.quantity + delta
        # Update average entry for adds in the same direction.
        if pos.quantity == 0 or (pos.quantity > 0 and delta > 0) or (pos.quantity < 0 and delta < 0):
            total_abs = abs(pos.quantity) + abs(delta)
            if total_abs > 0:
                pos.entry_price = (
                    (pos.entry_price * abs(pos.quantity)) + (fill_price * abs(delta))
                ) / total_abs
        # If position flips direction, reset entry price to new fill.
        if pos.quantity != 0 and (pos.quantity > 0 > new_qty or pos.quantity < 0 < new_qty):
            pos.entry_price = fill_price
        pos.quantity = new_qty

    portfolio.cash -= fee
    return PaperFillResult(filled_qty=fill_qty, fill_price=fill_price, fee_paid=fee)
