"""
Eagle Strategies: EMA Crossover Trend-Following Strategy
==========================================================
Uses the alignment of EMA(9), EMA(21), and EMA(50) to detect trend
direction, combined with price position relative to EMAs.

Signal logic:
    Full bull stack (EMA9 > EMA21 > EMA50) and price > EMA9   → BUY
    Full bear stack (EMA9 < EMA21 < EMA50) and price < EMA9   → SELL
    Mixed: EMA9 > EMA21 only (short-term momentum)             → mild BUY
    Mixed: EMA9 < EMA21 only (short-term pullback)             → mild SELL
    Price in between EMAs, no clear stack                      → HOLD

Score is amplified when the price has crossed an EMA cleanly.
Volume ratio further confirms or discounts the signal.
"""

from __future__ import annotations

from eagle.indicators.realtime_calculator import Indicators
from eagle.strategies.realtime.base import RealtimeStrategy, SignalDirection, StrategySignal


class EMAStrategy(RealtimeStrategy):

    @property
    def name(self) -> str:
        return "EMA(9/21/50)"

    def compute(self, indicators: Indicators) -> StrategySignal:
        price = indicators.price
        e9 = indicators.ema_9
        e21 = indicators.ema_21
        e50 = indicators.ema_50

        # Volume confirmation boost (capped at 0.15 extra confidence)
        vol_boost = min((indicators.volume_ratio - 1.0) * 0.10, 0.15) if indicators.volume_ratio > 1.0 else 0.0

        full_bull = e9 > e21 > e50
        full_bear = e9 < e21 < e50
        short_bull = e9 > e21  # short-term momentum only
        price_above_e9 = price > e9

        if full_bull and price_above_e9:
            score = 0.80
            confidence = min(0.75 + vol_boost, 0.95)
            direction = SignalDirection.BUY
            reason = f"Full EMA bull stack: EMA9({e9:.0f}) > EMA21({e21:.0f}) > EMA50({e50:.0f})"

        elif full_bull and not price_above_e9:
            # Bull stack but price pulled back below EMA9 — possible dip entry
            score = 0.45
            confidence = 0.55
            direction = SignalDirection.BUY
            reason = f"Bull stack but price dipped below EMA9 — potential dip buy"

        elif full_bear and not price_above_e9:
            score = -0.80
            confidence = min(0.75 + vol_boost, 0.95)
            direction = SignalDirection.SELL
            reason = f"Full EMA bear stack: EMA9({e9:.0f}) < EMA21({e21:.0f}) < EMA50({e50:.0f})"

        elif full_bear and price_above_e9:
            # Bear stack but price bounced above EMA9 — dead cat?
            score = -0.35
            confidence = 0.45
            direction = SignalDirection.SELL
            reason = f"Bear stack, price briefly above EMA9 — possible dead-cat bounce"

        elif short_bull:
            score = 0.25
            confidence = 0.40
            direction = SignalDirection.HOLD
            reason = f"EMA9 > EMA21 (short-term momentum only)"

        else:
            score = -0.25
            confidence = 0.40
            direction = SignalDirection.HOLD
            reason = f"EMA9 < EMA21 (short-term bearish drift)"

        return StrategySignal(
            strategy_name=self.name,
            direction=direction,
            score=score,
            confidence=confidence,
            reason=reason,
        )
