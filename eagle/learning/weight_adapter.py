"""
Eagle Learning: Adaptive Strategy Weight System
=================================================
After each completed trade (BUY then SELL pair), grades each strategy
on whether its signal at entry time was directionally correct, then
nudges the strategy's weight up or down using exponential moving average.

Algorithm (online perceptron / EMA accuracy tracker):
  For trade outcome r = +1 (profit) or -1 (loss):
    For each strategy s with score v_s at entry:
      alignment_s = sign(v_s) == sign(r)   →  1 if correct, 0 if wrong
      accuracy_s  = (1-α) * accuracy_s + α * alignment_s
  Weights = softmax(accuracy_s)   (sum to 1, all > 0)

Learning rate α = 0.08 — slow enough to avoid noise-chasing,
fast enough to adapt to changing market regimes in ~12 trades.

Weights persist to ``data/learned_weights.json`` across restarts.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

WEIGHTS_PATH = Path("data/learned_weights.json")
ALPHA = 0.08          # learning rate
MIN_WEIGHT = 0.05     # floor so no strategy is completely ignored
INITIAL_ACCURACY = 0.5  # start neutral (50% assumed accuracy)

# Default strategy list (must match aggregator)
DEFAULT_STRATEGIES = [
    "MACD(12,26,9)",
    "RSI(14)",
    "Bollinger(20,2s)",
    "EMA(9/21/50)",
]


@dataclass
class StrategyStats:
    """Running stats for one strategy."""

    name: str
    accuracy: float = INITIAL_ACCURACY   # EMA of correct-call rate
    weight: float = 0.25                 # current normalised weight
    total_graded: int = 0
    correct_calls: int = 0

    @property
    def accuracy_pct(self) -> float:
        return self.accuracy * 100

    @property
    def win_rate_pct(self) -> float:
        if self.total_graded == 0:
            return 50.0
        return self.correct_calls / self.total_graded * 100


@dataclass
class LearningState:
    """Full snapshot of the learning system."""

    strategies: dict[str, StrategyStats] = field(default_factory=dict)
    total_trades_learned: int = 0
    last_weight_update: str = ""


class WeightAdapter:
    """
    Tracks per-strategy accuracy and outputs updated weights for the aggregator.

    Usage::

        adapter = WeightAdapter()
        adapter.grade_trade(outcome_pct=+2.3, strategy_scores={"RSI(14)": 0.8, ...})
        new_weights = adapter.weights   # dict[str, float]
    """

    def __init__(self, path: Path = WEIGHTS_PATH) -> None:
        self._path = path
        self._stats: dict[str, StrategyStats] = {}
        self._total_learned = 0
        self._load()
        self._ensure_all_strategies()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def grade_trade(
        self,
        outcome_pct: float,
        strategy_scores: dict[str, float],
    ) -> dict[str, float]:
        """
        Grade each strategy based on trade outcome and return updated weights.

        Args:
            outcome_pct:      Return % of the closed trade (+positive = profit).
            strategy_scores:  {strategy_name: score_at_entry}

        Returns:
            Updated weight dict {strategy_name: weight}.
        """
        outcome_sign = 1.0 if outcome_pct >= 0 else -1.0
        self._total_learned += 1

        for name, score in strategy_scores.items():
            if name not in self._stats:
                self._stats[name] = StrategyStats(name=name)

            stat = self._stats[name]
            # Was the strategy directionally correct?
            correct = (score * outcome_sign) > 0
            stat.accuracy = (1 - ALPHA) * stat.accuracy + ALPHA * (1.0 if correct else 0.0)
            stat.total_graded += 1
            if correct:
                stat.correct_calls += 1

            logger.info(
                f"Grade [{name}]: score={score:+.2f} outcome={outcome_pct:+.2f}% "
                f"{'CORRECT' if correct else 'WRONG'} "
                f"-> accuracy={stat.accuracy_pct:.1f}%"
            )

        self._recompute_weights()
        self._save()
        return self.weights

    @property
    def weights(self) -> dict[str, float]:
        return {name: stat.weight for name, stat in self._stats.items()}

    @property
    def stats(self) -> dict[str, StrategyStats]:
        return dict(self._stats)

    @property
    def total_learned(self) -> int:
        return self._total_learned

    def summary_table(self) -> list[tuple[str, float, float, int]]:
        """Returns list of (name, accuracy%, weight%, graded_count)."""
        return [
            (name, s.accuracy_pct, s.weight * 100, s.total_graded)
            for name, s in self._stats.items()
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recompute_weights(self) -> None:
        """Softmax over accuracy values, with a minimum floor per strategy."""
        accs = {n: s.accuracy for n, s in self._stats.items()}
        # Softmax: exp(acc) / sum(exp(acc))
        exps = {n: math.exp(a * 4) for n, a in accs.items()}  # *4 sharpens the distribution
        total = sum(exps.values())
        raw = {n: v / total for n, v in exps.items()}

        # Apply floor and renormalise
        floored = {n: max(v, MIN_WEIGHT) for n, v in raw.items()}
        floor_total = sum(floored.values())
        for name, val in floored.items():
            self._stats[name].weight = val / floor_total

    def _ensure_all_strategies(self) -> None:
        for name in DEFAULT_STRATEGIES:
            # Normalise sigma symbol stored vs. displayed
            if name not in self._stats:
                self._stats[name] = StrategyStats(name=name)
        self._recompute_weights()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_learned": self._total_learned,
            "strategies": {
                name: {
                    "accuracy": s.accuracy,
                    "total_graded": s.total_graded,
                    "correct_calls": s.correct_calls,
                }
                for name, s in self._stats.items()
            },
        }
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._total_learned = data.get("total_learned", 0)
            for name, vals in data.get("strategies", {}).items():
                self._stats[name] = StrategyStats(
                    name=name,
                    accuracy=vals["accuracy"],
                    total_graded=vals["total_graded"],
                    correct_calls=vals["correct_calls"],
                )
            logger.info(
                f"Loaded learning state: {self._total_learned} trades learned, "
                f"{len(self._stats)} strategies tracked"
            )
        except Exception as exc:
            logger.warning(f"Could not load learned weights: {exc}")
