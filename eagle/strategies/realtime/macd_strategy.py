"""
Eagle Strategies: MACD Momentum Strategy
==========================================
Uses MACD(12, 26, 9) line, signal line, and histogram.

Signal logic:
    Histogram cross from negative → positive  → Strong BUY  (fresh momentum)
    MACD > Signal and histogram growing       → BUY
    MACD > Signal but histogram shrinking     → HOLD (losing steam)
    MACD < Signal but histogram growing       → HOLD (recovering)
    Histogram cross from positive → negative  → Strong SELL
    MACD < Signal and histogram falling more  → SELL

Score magnitude scales with histogram size relative to price (normalised).
"""

from __future__ import annotations

from eagle.indicators.realtime_calculator import Indicators
from eagle.strategies.realtime.base import RealtimeStrategy, SignalDirection, StrategySignal


class MACDStrategy(RealtimeStrategy):

    @property
    def name(self) -> str:
        return "MACD(12,26,9)"

    def compute(self, indicators: Indicators) -> StrategySignal:
        hist = indicators.macd_hist
        hist_prev = indicators.macd_hist_prev
        macd = indicators.macd
        signal = indicators.macd_signal

        # Normalise histogram relative to price
        norm_hist = abs(hist / indicators.price * 10_000)  # basis points

        bullish_cross = indicators.macd_cross_up
        bearish_cross = indicators.macd_cross_down
        hist_growing = abs(hist) > abs(hist_prev)

        if bullish_cross:
            score = 0.90
            confidence = 0.88
            direction = SignalDirection.BUY
            reason = "MACD histogram crossed above zero — fresh bullish momentum"

        elif bearish_cross:
            score = -0.90
            confidence = 0.88
            direction = SignalDirection.SELL
            reason = "MACD histogram crossed below zero — fresh bearish momentum"

        elif macd > signal and hist_growing:
            raw = min(norm_hist / 5.0, 1.0)
            score = 0.40 + 0.30 * raw
            confidence = 0.65
            direction = SignalDirection.BUY
            reason = f"MACD above signal & histogram expanding (hist={hist:.2f})"

        elif macd > signal and not hist_growing:
            score = 0.15
            confidence = 0.40
            direction = SignalDirection.HOLD
            reason = "MACD above signal but histogram shrinking — momentum fading"

        elif macd < signal and hist_growing:
            raw = min(norm_hist / 5.0, 1.0)
            score = -(0.40 + 0.30 * raw)
            confidence = 0.55
            direction = SignalDirection.SELL
            reason = f"MACD below signal & histogram expanding downward (hist={hist:.2f})"

        elif macd < signal and not hist_growing:
            # histogram negative but recovering
            score = -0.15
            confidence = 0.35
            direction = SignalDirection.HOLD
            reason = "MACD below signal but histogram recovering"

        else:
            score = 0.0
            confidence = 0.20
            direction = SignalDirection.HOLD
            reason = "MACD neutral"

        return StrategySignal(
            strategy_name=self.name,
            direction=direction,
            score=score,
            confidence=confidence,
            reason=reason,
        )
