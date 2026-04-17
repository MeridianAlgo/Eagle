"""
Eagle Learning: Trade Journal
================================
Persists every paper trade with its full indicator + signal state
to a JSON file so the learning system can replay and grade past decisions.

Each journal entry stores:
  - Trade metadata (side, price, qty, timestamp)
  - Indicator snapshot at the moment of the trade decision
  - Per-strategy scores that led to the decision
  - Outcome (filled in when position is later closed)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JOURNAL_PATH = Path("data/trade_journal.json")


@dataclass
class JournalEntry:
    """One recorded trade decision with its full context."""

    trade_id: int
    side: str                           # "BUY" | "SELL"
    price: float
    btc_qty: float
    timestamp: str                      # ISO-8601
    label: str                          # e.g. "STRONG BUY"
    agg_score: float                    # aggregated score at decision time
    strategy_scores: dict[str, float]   # {strategy_name: score}
    indicator_snapshot: dict[str, Any]  # key indicator values
    outcome_pct: float | None = None    # % return when closed (filled later)
    closed_at_price: float | None = None
    is_closed: bool = False


class TradeJournal:
    """
    Append-only journal of all trade decisions with their context.
    Persists to ``data/trade_journal.json`` so learning survives restarts.
    """

    def __init__(self, path: Path = JOURNAL_PATH) -> None:
        self._path = path
        self._entries: dict[int, JournalEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_trade(
        self,
        trade_id: int,
        side: str,
        price: float,
        btc_qty: float,
        label: str,
        agg_score: float,
        strategy_scores: dict[str, float],
        indicator_snapshot: dict[str, Any],
    ) -> JournalEntry:
        entry = JournalEntry(
            trade_id=trade_id,
            side=side,
            price=price,
            btc_qty=btc_qty,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            label=label,
            agg_score=agg_score,
            strategy_scores=strategy_scores,
            indicator_snapshot=indicator_snapshot,
        )
        self._entries[trade_id] = entry
        self._save()
        return entry

    def close_trade(
        self,
        open_trade_id: int,
        close_price: float,
    ) -> JournalEntry | None:
        """Mark the matching BUY entry as closed, compute return %."""
        entry = self._entries.get(open_trade_id)
        if entry is None or entry.is_closed:
            return None
        entry.closed_at_price = close_price
        entry.outcome_pct = (close_price - entry.price) / entry.price * 100
        entry.is_closed = True
        self._save()
        logger.info(
            f"Journal: trade #{open_trade_id} closed at ${close_price:,.2f} "
            f"— return {entry.outcome_pct:+.3f}%"
        )
        return entry

    def get_closed_entries(self) -> list[JournalEntry]:
        return [e for e in self._entries.values() if e.is_closed]

    def get_open_buy(self) -> JournalEntry | None:
        """Return the last unclosed BUY entry (our open position)."""
        buys = [e for e in self._entries.values() if e.side == "BUY" and not e.is_closed]
        return buys[-1] if buys else None

    @property
    def total_entries(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {str(k): asdict(v) for k, v in self._entries.items()}
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for k, v in raw.items():
                self._entries[int(k)] = JournalEntry(**v)
            logger.info(f"Loaded {len(self._entries)} journal entries from {self._path}")
        except Exception as exc:
            logger.warning(f"Could not load trade journal: {exc}")
