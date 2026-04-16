"""
Eagle Strategies: Real-Time Strategy Base
==========================================
All real-time strategies inherit from ``RealtimeStrategy``.
They receive an ``Indicators`` snapshot and return a ``StrategySignal``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from eagle.indicators.realtime_calculator import Indicators


class SignalDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class StrategySignal:
    """Signal produced by a single strategy."""

    strategy_name: str
    direction: SignalDirection
    score: float        # -1.0 (strong sell) … 0 (neutral) … +1.0 (strong buy)
    confidence: float   # 0.0 – 1.0
    reason: str         # human-readable explanation

    @property
    def weighted_score(self) -> float:
        """Score weighted by confidence."""
        return self.score * self.confidence


class RealtimeStrategy(ABC):
    """Abstract base class for all real-time trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def compute(self, indicators: Indicators) -> StrategySignal:
        """Compute a trade signal from the current indicator snapshot."""
        ...
