"""
Eagle Strategies: RSI Mean-Reversion Strategy
===============================================
Uses the 14-period RSI to identify overbought / oversold conditions.

Signal logic:
    RSI < 25            → Strong BUY  (deeply oversold)
    25 ≤ RSI < 35       → BUY         (oversold)
    35 ≤ RSI ≤ 65       → HOLD        (neutral)
    65 < RSI ≤ 75       → SELL        (overbought)
    RSI > 75            → Strong SELL (deeply overbought)

Score is also boosted when RSI is trending in the right direction
(i.e., RSI was even more extreme on the previous candle).
"""

from __future__ import annotations

from eagle.indicators.realtime_calculator import Indicators
from eagle.strategies.realtime.base import RealtimeStrategy, SignalDirection, StrategySignal


class RSIStrategy(RealtimeStrategy):

    @property
    def name(self) -> str:
        return "RSI(14)"

    def compute(self, indicators: Indicators) -> StrategySignal:
        rsi = indicators.rsi

        if rsi < 25:
            score = 1.0
            confidence = 0.90
            direction = SignalDirection.BUY
            reason = f"RSI={rsi:.1f} — deeply oversold (< 25)"

        elif rsi < 35:
            score = 0.65
            confidence = 0.75
            direction = SignalDirection.BUY
            reason = f"RSI={rsi:.1f} — oversold (< 35)"

        elif rsi < 45:
            score = 0.20
            confidence = 0.40
            direction = SignalDirection.HOLD
            reason = f"RSI={rsi:.1f} — mild buy bias"

        elif rsi <= 55:
            score = 0.0
            confidence = 0.20
            direction = SignalDirection.HOLD
            reason = f"RSI={rsi:.1f} — neutral"

        elif rsi <= 65:
            score = -0.20
            confidence = 0.40
            direction = SignalDirection.HOLD
            reason = f"RSI={rsi:.1f} — mild sell bias"

        elif rsi <= 75:
            score = -0.65
            confidence = 0.75
            direction = SignalDirection.SELL
            reason = f"RSI={rsi:.1f} — overbought (> 65)"

        else:
            score = -1.0
            confidence = 0.90
            direction = SignalDirection.SELL
            reason = f"RSI={rsi:.1f} — deeply overbought (> 75)"

        return StrategySignal(
            strategy_name=self.name,
            direction=direction,
            score=score,
            confidence=confidence,
            reason=reason,
        )
