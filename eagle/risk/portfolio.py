"""
Eagle Risk: Portfolio Tracking
================================
Real-time portfolio state tracking with position management,
P&L calculation, and performance metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from eagle.core.config import EagleConfig

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open trading position."""

    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    quantity: float
    entry_time: datetime = field(default_factory=datetime.utcnow)
    stop_loss: float | None = None
    take_profit: float | None = None
    trailing_stop: float | None = None
    current_price: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = float("inf")

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.side == "long":
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        cost = self.entry_price * self.quantity
        return self.unrealized_pnl / cost if cost > 0 else 0.0

    @property
    def holding_period(self) -> float:
        """Holding period in days."""
        return (datetime.utcnow() - self.entry_time).total_seconds() / 86400

    def update_price(self, price: float) -> None:
        """Update current market price and track extremes."""
        self.current_price = price
        self.highest_price = max(self.highest_price, price)
        self.lowest_price = min(self.lowest_price, price)

    def should_stop_loss(self) -> bool:
        """Check if stop loss has been triggered."""
        if self.stop_loss is None:
            return False
        if self.side == "long":
            return self.current_price <= self.stop_loss
        return self.current_price >= self.stop_loss

    def should_take_profit(self) -> bool:
        """Check if take profit has been triggered."""
        if self.take_profit is None:
            return False
        if self.side == "long":
            return self.current_price >= self.take_profit
        return self.current_price <= self.take_profit

    def update_trailing_stop(self, trail_pct: float) -> None:
        """Update trailing stop based on highest/lowest price."""
        if self.side == "long":
            new_stop = self.highest_price * (1 - trail_pct)
            if self.trailing_stop is None or new_stop > self.trailing_stop:
                self.trailing_stop = new_stop
                self.stop_loss = new_stop
        else:
            new_stop = self.lowest_price * (1 + trail_pct)
            if self.trailing_stop is None or new_stop < self.trailing_stop:
                self.trailing_stop = new_stop
                self.stop_loss = new_stop


@dataclass
class ClosedTrade:
    """Record of a completed trade."""

    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    commission: float = 0.0
    exit_reason: str = ""

    @property
    def duration_days(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 86400


class Portfolio:
    """
    Real-time portfolio state tracker.

    Maintains:
        - Open positions
        - Closed trade history
        - Cash balance
        - Equity curve
        - Performance metrics (Sharpe, Sortino, Win Rate, etc.)
    """

    def __init__(self, config: EagleConfig) -> None:
        self.config = config
        self._initial_capital = config.execution.paper.initial_capital
        self._cash = self._initial_capital
        self._positions: dict[str, Position] = {}
        self._closed_trades: list[ClosedTrade] = []
        self._equity_curve: list[tuple[datetime, float]] = [
            (datetime.utcnow(), self._initial_capital)
        ]
        self._peak_equity = self._initial_capital
        self._daily_starting_equity = self._initial_capital

    # -------------------------------------------------------
    # Position Management
    # -------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Position:
        """Open a new position."""
        cost = price * quantity
        commission = cost * self.config.execution.paper.commission_pct

        if cost + commission > self._cash:
            raise ValueError(f"Insufficient cash: need {cost + commission:.2f}, have {self._cash:.2f}")

        position = Position(
            symbol=symbol,
            side=side,
            entry_price=price,
            quantity=quantity,
            current_price=price,
            highest_price=price,
            lowest_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        self._cash -= cost + commission
        self._positions[symbol] = position
        logger.info(f"Opened {side} position: {symbol} x{quantity:.2f} @ {price:.2f}")
        return position

    def close_position(
        self,
        symbol: str,
        price: float,
        reason: str = "signal",
    ) -> ClosedTrade | None:
        """Close an existing position."""
        if symbol not in self._positions:
            logger.warning(f"No position to close for {symbol}")
            return None

        pos = self._positions[symbol]
        proceeds = price * pos.quantity
        commission = proceeds * self.config.execution.paper.commission_pct

        if pos.side == "long":
            pnl = (price - pos.entry_price) * pos.quantity - commission
        else:
            pnl = (pos.entry_price - price) * pos.quantity - commission

        pnl_pct = pnl / (pos.entry_price * pos.quantity)

        trade = ClosedTrade(
            symbol=symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=price,
            quantity=pos.quantity,
            entry_time=pos.entry_time,
            exit_time=datetime.utcnow(),
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            exit_reason=reason,
        )

        self._cash += proceeds - commission
        self._closed_trades.append(trade)
        del self._positions[symbol]

        logger.info(f"Closed {pos.side} {symbol}: PnL={pnl:.2f} ({pnl_pct:.2%}), reason={reason}")
        return trade

    def update(self, market_data: pd.DataFrame) -> None:
        """Update all positions with latest market prices."""
        for symbol, position in self._positions.items():
            sym_data = market_data[market_data["symbol"] == symbol] if "symbol" in market_data.columns else market_data
            if not sym_data.empty:
                price = sym_data["close"].iloc[-1]
                position.update_price(price)

                # Update trailing stop
                if self.config.risk.trailing_stop_pct:
                    position.update_trailing_stop(self.config.risk.trailing_stop_pct)

        # Update equity curve
        self._equity_curve.append((datetime.utcnow(), self.equity))
        self._peak_equity = max(self._peak_equity, self.equity)

    # -------------------------------------------------------
    # Properties and Metrics
    # -------------------------------------------------------

    @property
    def equity(self) -> float:
        """Total equity (cash + unrealized value)."""
        positions_value = sum(p.market_value for p in self._positions.values())
        return self._cash + positions_value

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    @property
    def total_exposure_ratio(self) -> float:
        """Total position exposure as fraction of equity."""
        total_pos = sum(p.market_value for p in self._positions.values())
        return total_pos / self.equity if self.equity > 0 else 0

    @property
    def current_drawdown(self) -> float:
        """Current drawdown from peak equity."""
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self.equity) / self._peak_equity

    @property
    def daily_pnl(self) -> float:
        """P&L since start of day."""
        return self.equity - self._daily_starting_equity

    @property
    def total_return(self) -> float:
        """Total return since inception."""
        return (self.equity - self._initial_capital) / self._initial_capital

    @property
    def win_rate(self) -> float:
        """Percentage of winning trades."""
        if not self._closed_trades:
            return 0.0
        wins = sum(1 for t in self._closed_trades if t.pnl > 0)
        return wins / len(self._closed_trades)

    @property
    def profit_factor(self) -> float:
        """Gross profits / Gross losses."""
        gross_profit = sum(t.pnl for t in self._closed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self._closed_trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")

    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe ratio from equity curve."""
        if len(self._equity_curve) < 10:
            return 0.0
        equities = pd.Series([e[1] for e in self._equity_curve])
        returns = equities.pct_change().dropna()
        if returns.std() == 0:
            return 0.0
        return float(returns.mean() / returns.std() * np.sqrt(252))

    @property
    def sortino_ratio(self) -> float:
        """Annualized Sortino ratio (penalizes downside only)."""
        if len(self._equity_curve) < 10:
            return 0.0
        equities = pd.Series([e[1] for e in self._equity_curve])
        returns = equities.pct_change().dropna()
        downside = returns[returns < 0]
        if downside.std() == 0:
            return 0.0
        return float(returns.mean() / downside.std() * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        """Maximum drawdown from equity curve."""
        if len(self._equity_curve) < 2:
            return 0.0
        equities = pd.Series([e[1] for e in self._equity_curve])
        peak = equities.expanding().max()
        dd = (equities - peak) / peak
        return float(abs(dd.min()))

    @property
    def avg_trade_pnl(self) -> float:
        """Average P&L per trade."""
        if not self._closed_trades:
            return 0.0
        return sum(t.pnl for t in self._closed_trades) / len(self._closed_trades)

    @property
    def total_trades(self) -> int:
        return len(self._closed_trades)

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of the current portfolio state."""
        return {
            "equity": self.equity,
            "cash": self.cash,
            "positions": len(self._positions),
            "total_return": self.total_return,
            "daily_pnl": self.daily_pnl,
            "drawdown": self.current_drawdown,
            "win_rate": self.win_rate,
            "sharpe": self.sharpe_ratio,
            "total_trades": self.total_trades,
            "profit_factor": self.profit_factor,
        }

    def reset_daily(self) -> None:
        """Reset daily tracking (call at start of each trading day)."""
        self._daily_starting_equity = self.equity
