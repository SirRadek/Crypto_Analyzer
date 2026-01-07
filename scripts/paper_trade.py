"""Paper trading runner (skeleton)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from crypto_analyzer.data.store import PriceDataStore, resolve_data_store
from crypto_analyzer.features.engineering import create_features
from crypto_analyzer.live.paper_broker import execute_order
from crypto_analyzer.live.risk import RiskLimits, allow_order
from crypto_analyzer.live.types import Order, Portfolio, Position
from crypto_analyzer.live.universe import DEFAULT_USDC_FUTURES
from crypto_analyzer.models.utils import match_model_features
from crypto_analyzer.utils.cli import run_cli
from crypto_analyzer.utils.config import CONFIG
from crypto_analyzer.utils.logging import generate_run_id, get_logger

app = typer.Typer(add_completion=False, no_args_is_help=True)
logger = get_logger(__name__)


def _fetch_prices_with_fallback(
    store: PriceDataStore, symbol: str
) -> tuple[str, pd.DataFrame] | tuple[None, None]:
    try:
        frame = store.fetch_prices(symbol)
    except Exception:  # pragma: no cover - best effort fallback
        frame = pd.DataFrame()
    if not frame.empty:
        return symbol, frame

    if symbol.endswith("USDC"):
        fallback = f"{symbol[:-4]}USDT"
        try:
            frame = store.fetch_prices(fallback)
        except Exception:  # pragma: no cover - best effort fallback
            frame = pd.DataFrame()
        if not frame.empty:
            return fallback, frame
    return None, None


def _resolve_asset_limits(symbol: str) -> dict[str, float]:
    limits = CONFIG.live_asset_limits or {}
    direct = limits.get(symbol)
    if isinstance(direct, dict):
        return direct
    base = symbol[:-4] if symbol.endswith(("USDC", "USDT")) else symbol
    for suffix in ("USDC", "USDT"):
        key = f"{base}{suffix}"
        if isinstance(limits.get(key), dict):
            return limits[key]
    return {}


def _position_drawdown_pct(position: Position, current_price: float) -> float:
    if current_price <= 0 or position.entry_price <= 0:
        return 0.0
    if position.quantity > 0:
        return (position.entry_price - current_price) / position.entry_price
    return (current_price - position.entry_price) / position.entry_price


def _mark_to_market(portfolio: Portfolio, prices: dict[str, float]) -> float:
    equity = portfolio.cash
    for symbol, pos in portfolio.positions.items():
        price = prices.get(symbol)
        if price is None:
            continue
        equity += pos.quantity * price
    portfolio.equity = equity
    return equity


def _position_size_from_volatility(realized_vol: float | None) -> float:
    if realized_vol is None or realized_vol <= 0:
        return CONFIG.live_position_size
    raw = CONFIG.live_position_size * (CONFIG.live_target_volatility / realized_vol)
    return max(CONFIG.live_min_position_size, min(CONFIG.live_max_position_size, raw))


def _write_ledger(ledger: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ledger).to_csv(path, index=False)


def _summary_from_ledger(ledger: list[dict[str, object]]) -> dict[str, float]:
    if not ledger:
        return {"pnl": 0.0, "max_drawdown": 0.0}
    equity_series = pd.Series([float(row.get("equity", 0.0)) for row in ledger])
    running_max = equity_series.cummax()
    drawdown = (equity_series / running_max) - 1.0
    pnl = float(equity_series.iloc[-1] - equity_series.iloc[0])
    return {"pnl": pnl, "max_drawdown": float(drawdown.min())}


def _write_summary(summary: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


@app.command()
def main(
    model_path: Optional[Path] = typer.Option(
        None, "--model-path", help="Optional model for prediction-based signals."
    ),
    symbols: Optional[str] = typer.Option(
        None, "--symbols", help="Comma-separated symbols to trade."
    ),
    fee_bps: float = typer.Option(10.0, help="Paper fee model in bps."),
    slip_bps: float = typer.Option(10.0, help="Paper slippage model in bps."),
    ledger_path: Optional[Path] = typer.Option(
        None, "--ledger-path", help="Optional CSV output path for paper ledger."
    ),
    iterations: int = typer.Option(1, "--iterations", help="Number of loop iterations."),
    sleep_seconds: float = typer.Option(
        0.0, "--sleep-seconds", help="Delay between iterations in seconds."
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Preview without persisting.",
    ),
) -> None:
    if symbols:
        symbol_list = tuple(s.strip().upper() for s in symbols.split(",") if s.strip())
    elif CONFIG.live_universe:
        symbol_list = CONFIG.live_universe
    else:
        symbol_list = DEFAULT_USDC_FUTURES
    store: PriceDataStore = resolve_data_store("auto")
    portfolio = Portfolio(
        equity=CONFIG.live_initial_equity,
        cash=CONFIG.live_initial_cash,
        positions={},
    )
    feature_settings = CONFIG.live
    run_id = generate_run_id()
    latest_prices: dict[str, float] = {}
    ledger: list[dict[str, object]] = []

    for iteration in range(max(1, iterations)):
        for symbol in symbol_list:
            resolved, frame = _fetch_prices_with_fallback(store, symbol)
            if resolved is None or frame is None or frame.empty:
                logger.warning(
                    "No price data available for symbol",
                    extra={"event": "paper_skip", "symbol": symbol},
                )
                continue

            feats = create_features(frame, settings=feature_settings)
            latest = feats.iloc[[-1]]
            signal = 0.0
            if model_path is not None:
                import joblib

                model = joblib.load(model_path)
                payload = match_model_features(latest, model)
                proba = float(model.predict_proba(payload)[0, 1])
                signal = proba

            last_ts = frame["timestamp"].iloc[-1]
            last_price = float(frame["close"].iloc[-1])
            latest_prices[resolved] = last_price
            _mark_to_market(portfolio, latest_prices)
            asset_limits = _resolve_asset_limits(resolved)
            per_limits = RiskLimits(
                max_wallet_exposure_pct=float(asset_limits.get("max_exposure_pct", 0.15)),
                max_position_drawdown_pct=float(asset_limits.get("max_drawdown_pct", 0.40)),
                max_leverage=float(asset_limits.get("max_leverage", 100.0)),
            )
            fee_override = asset_limits.get("fee_bps")
            slip_override = asset_limits.get("slippage_bps")
            existing = portfolio.positions.get(resolved)
            if existing is not None:
                drawdown = _position_drawdown_pct(existing, last_price)
                if drawdown >= per_limits.max_position_drawdown_pct:
                    close_side = "sell" if existing.quantity > 0 else "buy"
                    close_qty = abs(existing.quantity)
                    close_order = Order(
                        symbol=resolved,
                        side=close_side,  # type: ignore[arg-type]
                        quantity=close_qty,
                        price=last_price,
                    )
                    execute_order(
                        close_order,
                        portfolio,
                        fee_bps=float(fee_override if fee_override is not None else fee_bps),
                        slippage_bps=float(slip_override if slip_override is not None else slip_bps),
                    )
                    _mark_to_market(portfolio, latest_prices)
                    logger.warning(
                        "Closed position due to drawdown limit",
                        extra={"event": "paper_stop", "symbol": resolved, "drawdown": drawdown},
                    )
                    continue

            if signal <= 0.5:
                continue
            realized_vol = None
            if "vol_realized_7d" in latest.columns:
                realized_vol = float(latest["vol_realized_7d"].iloc[0])
            position_size = _position_size_from_volatility(realized_vol)
            order = Order(symbol=resolved, side="buy", quantity=position_size, price=last_price)
            if not allow_order(
                order,
                portfolio,
                limits=per_limits,
                position=existing,
                current_price=last_price,
            ):
                continue
            execute_order(
                order,
                portfolio,
                fee_bps=float(fee_override if fee_override is not None else fee_bps),
                slippage_bps=float(slip_override if slip_override is not None else slip_bps),
            )
            _mark_to_market(portfolio, latest_prices)

            ledger.append(
                {
                    "run_id": run_id,
                    "iteration": iteration,
                    "timestamp": pd.to_datetime(last_ts, utc=True, errors="coerce"),
                    "symbol": resolved,
                    "signal": signal,
                    "price": last_price,
                    "equity": portfolio.equity,
                    "cash": portfolio.cash,
                    "position_qty": portfolio.positions.get(resolved).quantity
                    if portfolio.positions.get(resolved)
                    else 0.0,
                }
            )
        if sleep_seconds > 0 and iteration < iterations - 1:
            time.sleep(sleep_seconds)

    if dry_run:
        typer.echo("Paper loop complete (dry run).")
        return

    if ledger_path is None:
        ledger_path = Path("reports") / f"paper_ledger_{run_id}.csv"
    _write_ledger(ledger, ledger_path)
    typer.echo(f"Paper ledger written to {ledger_path}")

    summary = _summary_from_ledger(ledger)
    summary_path = Path("reports") / f"paper_summary_{run_id}.json"
    _write_summary(summary, summary_path)
    typer.echo(f"Paper summary written to {summary_path}")


if __name__ == "__main__":  # pragma: no cover
    run_cli(app)
