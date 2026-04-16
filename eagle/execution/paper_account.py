"""
Eagle Execution: Paper Trading Account
========================================
Simulates a BTC trading account with cash + BTC position.
No real orders are placed — this is purely for P&L tracking.

Features:
    - Buy/sell with configurable commission (default 0.1% Binance rate)
    - Tracks full trade history with entry price, P&L, etc.
    - Computes unrealised P&L against current market price
    - Enforces minimum trade size and cooldown to avoid over-trading
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from eagle.strategies.realtime.aggregator import TradeRecommendation
from eagle.strategies.realtime.base import SignalDirection

logger = logging.getLogger(__name__)

COMMISSION_PCT = 0.001   # 0.10% — Binance taker fee
MIN_TRADE_USD = 50.0     # minimum trade value to bother executing
COOLDOWN_CANDLES = 3     # wait at least N new candles between trades


@dataclass
class Trade:
    """A completed (or open) paper trade."""

    trade_id: int
    side: Literal["BUY", "SELL"]
    price: float
    btc_qty: float
    usd_value: float
    commission: float
    timestamp: datetime
    label: str              # recommendation label at time of trade
    realised_pnl: float = 0.0
    close_price: float | None = None

    @property
    def net_usd(self) -> float:
        """Cash impact of this trade (negative for buys, positive for sells)."""
        if self.side == "BUY":
            return -(self.usd_value + self.commission)
        return self.usd_value - self.commission


class PaperAccount:
    """
    Simulates a BTC/USD paper trading account.

    Usage::

        account = PaperAccount(initial_cash=10_000.0)
        account.execute(recommendation, current_price)
        print(account.summary(current_price))
    """

    def __init__(self, initial_cash: float = 10_000.0) -> None:
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._btc: float = 0.0
        self._trades: list[Trade] = []
        self._candles_since_trade = COOLDOWN_CANDLES  # start ready
        self._trade_counter = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(self) -> None:
        """Call once per closed candle to advance the cooldown counter."""
        self._candles_since_trade += 1

    def execute(self, rec: TradeRecommendation, price: float) -> Trade | None:
        """
        Attempt to execute the given recommendation at ``price``.
        Returns the Trade if executed, or None if skipped.
        """
        if self._candles_since_trade < COOLDOWN_CANDLES:
            return None

        if rec.direction == SignalDirection.BUY and self._cash >= MIN_TRADE_USD:
            return self._buy(rec, price)

        if rec.direction == SignalDirection.SELL and self._btc > 0:
            return self._sell(rec, price)

        return None

    def current_equity(self, price: float) -> float:
        """Total account value: cash + BTC mark-to-market."""
        return self._cash + self._btc * price

    def unrealised_pnl(self, price: float) -> float:
        """Unrealised P&L on open BTC position."""
        if self._btc <= 0:
            return 0.0
        # Average entry cost of open position
        avg_entry = self._avg_entry_price()
        return (price - avg_entry) * self._btc if avg_entry > 0 else 0.0

    def total_pnl(self, price: float) -> float:
        return self.current_equity(price) - self._initial_cash

    def total_pnl_pct(self, price: float) -> float:
        return self.total_pnl(price) / self._initial_cash * 100

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def btc(self) -> float:
        return self._btc

    @property
    def trades(self) -> list[Trade]:
        return list(self._trades)

    @property
    def recent_trades(self) -> list[Trade]:
        return self._trades[-10:]

    def summary(self, price: float) -> dict:
        return {
            "cash_usd": self._cash,
            "btc_held": self._btc,
            "btc_value_usd": self._btc * price,
            "equity_usd": self.current_equity(price),
            "unrealised_pnl": self.unrealised_pnl(price),
            "total_pnl_usd": self.total_pnl(price),
            "total_pnl_pct": self.total_pnl_pct(price),
            "total_trades": len(self._trades),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _buy(self, rec: TradeRecommendation, price: float) -> Trade:
        # Use a fraction of cash based on signal strength
        fraction = self._position_fraction(rec.score)
        usd_to_spend = self._cash * fraction
        commission = usd_to_spend * COMMISSION_PCT
        net_spend = usd_to_spend + commission
        btc_received = usd_to_spend / price

        self._cash -= net_spend
        self._btc += btc_received
        self._candles_since_trade = 0
        self._trade_counter += 1

        trade = Trade(
            trade_id=self._trade_counter,
            side="BUY",
            price=price,
            btc_qty=btc_received,
            usd_value=usd_to_spend,
            commission=commission,
            timestamp=datetime.now(tz=timezone.utc),
            label=rec.label,
        )
        self._trades.append(trade)
        logger.info(
            f"[PAPER BUY]  {btc_received:.6f} BTC @ ${price:,.2f} "
            f"| spent ${net_spend:,.2f} | {rec.label}"
        )
        return trade

    def _sell(self, rec: TradeRecommendation, price: float) -> Trade:
        # Sell a fraction of holdings based on signal strength
        fraction = self._position_fraction(abs(rec.score))
        btc_to_sell = self._btc * fraction
        usd_received = btc_to_sell * price
        commission = usd_received * COMMISSION_PCT
        net_received = usd_received - commission

        # Realised P&L
        avg_entry = self._avg_entry_price()
        realised = (price - avg_entry) * btc_to_sell if avg_entry > 0 else 0.0

        self._btc -= btc_to_sell
        self._cash += net_received
        self._candles_since_trade = 0
        self._trade_counter += 1

        trade = Trade(
            trade_id=self._trade_counter,
            side="SELL",
            price=price,
            btc_qty=btc_to_sell,
            usd_value=usd_received,
            commission=commission,
            timestamp=datetime.now(tz=timezone.utc),
            label=rec.label,
            realised_pnl=realised,
            close_price=price,
        )
        self._trades.append(trade)
        logger.info(
            f"[PAPER SELL] {btc_to_sell:.6f} BTC @ ${price:,.2f} "
            f"| received ${net_received:,.2f} | P&L ${realised:+,.2f} | {rec.label}"
        )
        return trade

    @staticmethod
    def _position_fraction(abs_score: float) -> float:
        """Map |score| to position fraction: 0.55→0.40, 1.0→0.90."""
        return min(0.40 + (abs_score - 0.30) * 1.43, 0.90)

    def _avg_entry_price(self) -> float:
        """Compute average entry price for current open BTC position."""
        btc_cost = 0.0
        btc_qty = 0.0
        for t in self._trades:
            if t.side == "BUY":
                btc_cost += t.usd_value
                btc_qty += t.btc_qty
            else:
                # Reduce proportionally
                if btc_qty > 0:
                    frac = t.btc_qty / btc_qty
                    btc_cost *= (1 - frac)
                    btc_qty -= t.btc_qty
        return btc_cost / btc_qty if btc_qty > 0 else 0.0
