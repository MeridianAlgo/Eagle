"""
Eagle Strategies: Signal Aggregator
=====================================
Combines signals from multiple strategies into a single trade recommendation.

Weighting:
    MACD       30%  — best for momentum / trend confirmation
    RSI        25%  — best for mean-reversion / extremes
    Bollinger  25%  — best for volatility & band extremes
    EMA        20%  — best for trend direction

Final score = Σ(strategy_weight × weighted_score) / Σ(strategy_weight)

Recommendation thresholds:
    score ≥  0.55   STRONG BUY
    score ≥  0.30   BUY
    score ≤ -0.55   STRONG SELL
    score ≤ -0.30   SELL
    otherwise       HOLD
"""

from __future__ import annotations

from dataclasses import dataclass

from eagle.indicators.realtime_calculator import Indicators
from eagle.strategies.realtime.base import SignalDirection, StrategySignal
from eagle.strategies.realtime.bollinger_strategy import BollingerStrategy
from eagle.strategies.realtime.ema_strategy import EMAStrategy
from eagle.strategies.realtime.macd_strategy import MACDStrategy
from eagle.strategies.realtime.rsi_strategy import RSIStrategy


@dataclass
class TradeRecommendation:
    """Final aggregated trade recommendation."""

    direction: SignalDirection
    score: float          # -1.0 … +1.0
    confidence: float     # 0.0 – 1.0
    signals: list[StrategySignal]
    summary: str

    @property
    def is_actionable(self) -> bool:
        return self.direction != SignalDirection.HOLD

    @property
    def label(self) -> str:
        if self.score >= 0.55:
            return "STRONG BUY"
        if self.score >= 0.30:
            return "BUY"
        if self.score <= -0.55:
            return "STRONG SELL"
        if self.score <= -0.30:
            return "SELL"
        return "HOLD"

    @property
    def emoji(self) -> str:
        lbl = self.label
        return {
            "STRONG BUY":  "🚀",
            "BUY":         "📈",
            "HOLD":        "⏸️ ",
            "SELL":        "📉",
            "STRONG SELL": "🔻",
        }.get(lbl, "⏸️ ")


# Strategy weights (must sum to 1.0)
_WEIGHTS: dict[str, float] = {
    "MACD(12,26,9)":   0.30,
    "RSI(14)":         0.25,
    "Bollinger(20,2σ)": 0.25,
    "EMA(9/21/50)":    0.20,
}


class SignalAggregator:
    """
    Runs all real-time strategies and aggregates their signals.

    Usage::

        agg = SignalAggregator()
        recommendation = agg.evaluate(indicators)
    """

    def __init__(self) -> None:
        self._strategies = [
            MACDStrategy(),
            RSIStrategy(),
            BollingerStrategy(),
            EMAStrategy(),
        ]

    def evaluate(self, indicators: Indicators) -> TradeRecommendation:
        signals = [s.compute(indicators) for s in self._strategies]
        score = self._aggregate_score(signals)
        confidence = self._aggregate_confidence(signals)
        direction = self._score_to_direction(score)
        summary = self._build_summary(signals, score)

        return TradeRecommendation(
            direction=direction,
            score=score,
            confidence=confidence,
            signals=signals,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_score(signals: list[StrategySignal]) -> float:
        total_weight = 0.0
        weighted_sum = 0.0
        for sig in signals:
            w = _WEIGHTS.get(sig.strategy_name, 0.25)
            total_weight += w
            weighted_sum += w * sig.weighted_score
        return weighted_sum / total_weight if total_weight else 0.0

    @staticmethod
    def _aggregate_confidence(signals: list[StrategySignal]) -> float:
        """Confidence = weighted average of individual confidences."""
        total_weight = 0.0
        weighted_sum = 0.0
        for sig in signals:
            w = _WEIGHTS.get(sig.strategy_name, 0.25)
            total_weight += w
            weighted_sum += w * sig.confidence
        return weighted_sum / total_weight if total_weight else 0.0

    @staticmethod
    def _score_to_direction(score: float) -> SignalDirection:
        if score >= 0.30:
            return SignalDirection.BUY
        if score <= -0.30:
            return SignalDirection.SELL
        return SignalDirection.HOLD

    @staticmethod
    def _build_summary(signals: list[StrategySignal], final_score: float) -> str:
        parts = [f"{s.strategy_name}: {s.score:+.2f}" for s in signals]
        return f"Score {final_score:+.3f} | " + " | ".join(parts)
