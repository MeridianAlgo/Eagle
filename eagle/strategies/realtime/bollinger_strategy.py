"""
Eagle Strategies: Bollinger Band Mean-Reversion Strategy
==========================================================
Uses Bollinger Bands(20, 2σ) to detect price extremes.

%B = (price - lower) / (upper - lower)

Signal logic:
    %B < 0.05   → Strong BUY  (price below lower band)
    %B < 0.20   → BUY         (price near lower band)
    %B < 0.40   → HOLD        (lower half, mild bias)
    %B < 0.60   → HOLD        (neutral zone)
    %B < 0.80   → HOLD        (upper half, mild bias)
    %B < 0.95   → SELL        (price near upper band)
    %B ≥ 0.95   → Strong SELL (price above upper band)

Band width modulates confidence — wider bands mean more volatile market
so signals carry less certainty.
"""

from __future__ import annotations

from eagle.indicators.realtime_calculator import Indicators
from eagle.strategies.realtime.base import RealtimeStrategy, SignalDirection, StrategySignal


class BollingerStrategy(RealtimeStrategy):

    @property
    def name(self) -> str:
        return "Bollinger(20,2σ)"

    def compute(self, indicators: Indicators) -> StrategySignal:
        pct_b = indicators.bb_pct
        width = indicators.bb_width

        # Wide bands = high volatility = less reliable mean-reversion signal
        vol_discount = max(0.5, 1.0 - width * 5)

        if pct_b < 0.05:
            score = 1.0
            confidence = 0.88 * vol_discount
            direction = SignalDirection.BUY
            reason = f"%B={pct_b:.2f} — price below lower band (strong oversold)"

        elif pct_b < 0.20:
            score = 0.70
            confidence = 0.72 * vol_discount
            direction = SignalDirection.BUY
            reason = f"%B={pct_b:.2f} — price near lower band"

        elif pct_b < 0.40:
            score = 0.20
            confidence = 0.40 * vol_discount
            direction = SignalDirection.HOLD
            reason = f"%B={pct_b:.2f} — lower half, mild buy bias"

        elif pct_b <= 0.60:
            score = 0.0
            confidence = 0.20
            direction = SignalDirection.HOLD
            reason = f"%B={pct_b:.2f} — neutral zone"

        elif pct_b <= 0.80:
            score = -0.20
            confidence = 0.40 * vol_discount
            direction = SignalDirection.HOLD
            reason = f"%B={pct_b:.2f} — upper half, mild sell bias"

        elif pct_b <= 0.95:
            score = -0.70
            confidence = 0.72 * vol_discount
            direction = SignalDirection.SELL
            reason = f"%B={pct_b:.2f} — price near upper band"

        else:
            score = -1.0
            confidence = 0.88 * vol_discount
            direction = SignalDirection.SELL
            reason = f"%B={pct_b:.2f} — price above upper band (strong overbought)"

        return StrategySignal(
            strategy_name=self.name,
            direction=direction,
            score=score,
            confidence=confidence,
            reason=reason,
        )
