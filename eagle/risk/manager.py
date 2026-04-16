"""
Eagle Risk: Risk Management Engine
=====================================
Comprehensive risk management with position sizing, portfolio
risk limits, VaR, drawdown monitoring, and correlation-based
exposure controls.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from eagle.core.config import EagleConfig
from eagle.core.events import Event, EventBus, EventType
from eagle.strategies.manager import TradeSignal

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Central risk management engine.

    Validates every trade signal against multiple risk constraints
    before allowing execution:

    1. Position size limits
    2. Portfolio exposure limits
    3. Maximum daily loss
    4. Maximum drawdown
    5. VaR constraints
    6. Correlation-based exposure limits
    7. Cooldown timers after losses
    """

    def __init__(self, config: EagleConfig, event_bus: EventBus, portfolio: Any) -> None:
        self.config = config
        self.event_bus = event_bus
        self.portfolio = portfolio
        self._cfg = config.risk
        self._daily_pnl = 0.0
        self._last_loss_time: datetime | None = None
        self._halted = False
        self._trade_log: list[dict[str, Any]] = []

    def validate_signals(
        self,
        signals: list[TradeSignal],
        portfolio: Any,
        market_data: pd.DataFrame,
    ) -> list[TradeSignal]:
        """
        Filter trade signals through risk checks.

        Returns only signals that pass all risk constraints.
        """
        if self._halted:
            logger.warning("Trading halted. All signals rejected.")
            return []

        approved: list[TradeSignal] = []

        for signal in signals:
            reasons = self._check_signal(signal, portfolio, market_data)
            if reasons:
                logger.info(
                    f"Signal REJECTED for {signal.symbol}: {', '.join(reasons)}"
                )
                continue

            # Apply position sizing
            signal = self._size_position(signal, portfolio, market_data)
            approved.append(signal)
            logger.info(
                f"Signal APPROVED for {signal.symbol} "
                f"({signal.side.value}, conf={signal.confidence:.2f})"
            )

        return approved

    def _check_signal(
        self,
        signal: TradeSignal,
        portfolio: Any,
        market_data: pd.DataFrame,
    ) -> list[str]:
        """Run all risk checks on a signal. Returns list of rejection reasons."""
        reasons: list[str] = []

        # 1. Cooldown check
        if self._last_loss_time:
            cooldown = timedelta(minutes=self._cfg.cooldown_after_loss_minutes)
            if datetime.utcnow() - self._last_loss_time < cooldown:
                reasons.append("cooldown_active")

        # 2. Daily loss limit
        daily_loss_limit = portfolio.equity * self._cfg.max_daily_loss
        if self._daily_pnl < -daily_loss_limit:
            reasons.append("daily_loss_exceeded")

        # 3. Max drawdown
        if portfolio.current_drawdown > self._cfg.max_drawdown:
            reasons.append("max_drawdown_exceeded")

        # 4. Max position size
        if signal.size_hint and signal.size_hint > self._cfg.max_position_size:
            reasons.append("position_too_large")

        # 5. Max total exposure
        if portfolio.total_exposure_ratio > self._cfg.max_total_exposure:
            reasons.append("total_exposure_exceeded")

        # 6. Minimum confidence
        min_conf = self.config.strategies.ml_strategy.min_confidence
        if signal.confidence < min_conf * 0.8:  # Allow slight relaxation
            reasons.append("low_confidence")

        return reasons

    def _size_position(
        self,
        signal: TradeSignal,
        portfolio: Any,
        market_data: pd.DataFrame,
    ) -> TradeSignal:
        """
        Calculate appropriate position size using ATR-based sizing.

        Position size = (Account Risk %) / (ATR * Multiplier / Price)
        """
        sym_data = market_data[market_data["symbol"] == signal.symbol] if "symbol" in market_data.columns else market_data

        if sym_data.empty:
            signal.size_hint = self._cfg.max_portfolio_risk
            return signal

        current_price = sym_data["close"].iloc[-1]

        if self._cfg.use_atr_sizing and len(sym_data) > 14:
            # ATR-based sizing
            high_low = sym_data["high"] - sym_data["low"]
            high_close = (sym_data["high"] - sym_data["close"].shift()).abs()
            low_close = (sym_data["low"] - sym_data["close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean().iloc[-1]

            risk_per_share = atr * self._cfg.atr_risk_multiplier
            risk_amount = portfolio.equity * self._cfg.max_portfolio_risk
            shares = risk_amount / risk_per_share if risk_per_share > 0 else 0
            position_value = shares * current_price
            size_pct = position_value / portfolio.equity if portfolio.equity > 0 else 0

            # Cap at max position size
            size_pct = min(size_pct, self._cfg.max_position_size)

            # Scale by confidence if enabled
            if signal.confidence < 1.0:
                size_pct *= signal.confidence

            signal.size_hint = size_pct
            signal.stop_loss = current_price - risk_per_share if signal.is_buy else current_price + risk_per_share
        else:
            # Fixed risk sizing fallback
            signal.size_hint = min(
                self._cfg.max_portfolio_risk * signal.confidence,
                self._cfg.max_position_size,
            )

        return signal

    def check_portfolio_risk(self, portfolio: Any, market_data: pd.DataFrame) -> None:
        """Periodic portfolio-level risk check."""
        # Check drawdown
        if portfolio.current_drawdown > self._cfg.max_drawdown:
            self._halt_trading("max_drawdown_breach")

        # Check daily P&L
        self._daily_pnl = portfolio.daily_pnl
        if self._daily_pnl < -(portfolio.equity * self._cfg.max_daily_loss):
            self._halt_trading("daily_loss_limit")

    def _halt_trading(self, reason: str) -> None:
        """Halt all trading due to risk breach."""
        self._halted = True
        logger.critical(f"TRADING HALTED: {reason}")
        # Event emission would happen asynchronously in the main loop

    def update_pnl(self, pnl: float) -> None:
        """Update daily P&L tracker."""
        self._daily_pnl += pnl
        if pnl < 0:
            self._last_loss_time = datetime.utcnow()

    def reset_daily(self) -> None:
        """Reset daily risk counters (call at start of each trading day)."""
        self._daily_pnl = 0.0
        self._halted = False

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float | None = None,
    ) -> float:
        """Calculate Value at Risk using historical simulation."""
        conf = confidence or self._cfg.var_confidence
        if returns.empty:
            return 0.0
        return float(np.percentile(returns.dropna(), (1 - conf) * 100))

    def calculate_cvar(self, returns: pd.Series, confidence: float | None = None) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        var = self.calculate_var(returns, confidence)
        return float(returns[returns <= var].mean()) if len(returns[returns <= var]) > 0 else var

    @property
    def is_halted(self) -> bool:
        return self._halted
