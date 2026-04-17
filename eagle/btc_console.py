"""
Eagle: Console-mode BTC Bot (no full-screen required)
=======================================================
Runs the full trading + self-learning pipeline and prints
a clear, formatted log to stdout every candle.

Lower trade threshold (0.15) vs the dashboard bot (0.30)
so trades fire more often and learning is observable sooner.

Usage:
    python -m eagle.btc_console
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime, timezone

from eagle.data.kraken_ws import Candle, KrakenWebSocket
from eagle.execution.paper_account import PaperAccount
from eagle.indicators import realtime_calculator as calc
from eagle.learning.trade_journal import TradeJournal
from eagle.learning.weight_adapter import WeightAdapter
from eagle.strategies.realtime.aggregator import SignalAggregator

logger = logging.getLogger(__name__)

TRADE_THRESHOLD = 0.15   # lower than dashboard so trades fire more often
COOLDOWN        = 1      # candles between trades


def _now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%H:%M:%S")


def _bar(value: float, lo: float, hi: float, width: int = 12) -> str:
    """ASCII progress bar for a value in [lo, hi]."""
    pct = max(0.0, min(1.0, (value - lo) / (hi - lo))) if hi > lo else 0.5
    filled = int(pct * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _dir_arrow(score: float) -> str:
    if score >= TRADE_THRESHOLD:
        return "  BUY >>>"
    if score <= -TRADE_THRESHOLD:
        return "<<< SELL "
    return "  HOLD   "


def _print_header() -> None:
    print()
    print("=" * 90)
    print("  EAGLE  |  BTC/USD Real-Time Trading Bot  |  Self-Learning Engine Active")
    print("  Data: Kraken WebSocket  |  Paper trading: $10,000 initial capital")
    print("=" * 90)
    print(f"  {'TIME':8s}  {'PRICE':>10s}  {'RSI':>6s}  {'MACD-H':>8s}  "
          f"{'BB%':>5s}  {'SIGNAL':9s}  {'SCORE':>7s}  {'EQUITY':>11s}  {'P&L':>9s}")
    print("-" * 90)


def _print_candle(
    price: float,
    ind: "calc.Indicators",
    label: str,
    score: float,
    equity: float,
    pnl: float,
    trade_tag: str,
) -> None:
    rsi_flag = " OB" if ind.rsi > 70 else (" OS" if ind.rsi < 30 else "   ")
    line = (
        f"  {_now():8s}  "
        f"${price:>9,.2f}  "
        f"{ind.rsi:>5.1f}{rsi_flag}  "
        f"{ind.macd_hist:>+8.2f}  "
        f"{ind.bb_pct:>5.2f}  "
        f"{_dir_arrow(score):9s}  "
        f"{score:>+7.3f}  "
        f"${equity:>10,.2f}  "
        f"{pnl:>+9.2f}"
        f"{trade_tag}"
    )
    print(line, flush=True)


def _print_trade(side: str, price: float, btc: float, usd: float, label: str) -> None:
    arrow = ">>>" if side == "BUY" else "<<<"
    print(f"\n  {arrow} TRADE: {side} {btc:.6f} BTC @ ${price:,.2f}  "
          f"(${usd:,.2f})  [{label}]")


def _print_learning(
    trade_id: int,
    outcome_pct: float,
    strategy_scores: dict[str, float],
    old_weights: dict[str, float],
    new_weights: dict[str, float],
) -> None:
    direction = "PROFIT" if outcome_pct >= 0 else "LOSS"
    print(f"\n  *** LEARNING from trade #{trade_id}: {direction} {outcome_pct:+.3f}% ***")
    for name, score in strategy_scores.items():
        correct = (score * outcome_pct) > 0
        mark = "CORRECT (+)" if correct else "WRONG   (-)"
        old_w = old_weights.get(name, 0.25) * 100
        new_w = new_weights.get(name, 0.25) * 100
        delta = new_w - old_w
        arrow = "^" if delta > 0 else ("v" if delta < 0 else "=")
        print(f"       {name:20s}  {mark}  "
              f"weight: {old_w:.1f}% -> {new_w:.1f}%  ({arrow}{abs(delta):.1f}%)")
    print()


class BTCConsoleBot:

    def __init__(self) -> None:
        self._ws       = KrakenWebSocket(history_size=210)
        self._agg      = SignalAggregator()
        self._account  = PaperAccount(initial_cash=10_000.0)
        self._journal  = TradeJournal()
        self._adapter  = WeightAdapter()
        self._candles  = 0
        self._cooldown = 0

        # Seed with previously-learned weights
        self._agg.update_weights(self._adapter.weights)
        self._ws.on_candle(self._on_candle)

    async def run(self) -> None:
        print(f"\n  [Eagle] Bootstrapping historical data from Kraken...", flush=True)
        await self._ws.bootstrap()
        print(f"  Loaded {len(self._ws.history)} candles. "
              f"Current BTC/USD: ${self._ws.current_price:,.2f}\n", flush=True)

        if self._adapter.total_learned > 0:
            print(f"  Resuming: {self._adapter.total_learned} trades already learned.")
            print(f"  Current weights:", end="")
            for name, w in self._adapter.weights.items():
                print(f"  {name}: {w*100:.1f}%", end="")
            print("\n")

        _print_header()
        try:
            await self._ws.start()
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass

    async def _on_candle(self, candle: Candle, price: float) -> None:
        if not candle.is_closed:
            return

        self._candles += 1
        if self._cooldown > 0:
            self._cooldown -= 1
        self._account.tick()

        ind = calc.compute(self._ws.history, price)
        if ind is None:
            print(f"  [{_now()}] Warming up... {len(self._ws.history)}/50 candles", flush=True)
            return

        rec = self._agg.evaluate(ind)
        score = rec.score
        eq = self._account.current_equity(price)
        pnl = self._account.total_pnl(price)

        # Determine if we should trade (lower threshold, manual cooldown)
        will_trade = (
            self._cooldown == 0
            and (
                (score >= TRADE_THRESHOLD and self._account.cash >= 50)
                or (score <= -TRADE_THRESHOLD and self._account.btc > 0)
            )
        )
        trade_tag = "  *** TRADE ***" if will_trade else ""

        _print_candle(price, ind, rec.label, score, eq, pnl, trade_tag)

        if not will_trade:
            return

        self._cooldown = COOLDOWN

        # Execute paper trade
        from eagle.strategies.realtime.base import SignalDirection
        from eagle.execution.paper_account import Trade

        if score >= TRADE_THRESHOLD:
            # Override: force BUY signal regardless of cooldown state in account
            from eagle.strategies.realtime.aggregator import TradeRecommendation
            from eagle.strategies.realtime.base import SignalDirection
            rec_buy = TradeRecommendation(
                direction=SignalDirection.BUY,
                score=score, confidence=rec.confidence,
                signals=rec.signals, summary=rec.summary
            )
            # Bypass account cooldown by directly creating the trade
            self._account._candles_since_trade = 999
            trade = self._account.execute(rec_buy, price)
        else:
            self._account._candles_since_trade = 999
            trade = self._account.execute(rec, price)

        if trade is None:
            return

        _print_trade(trade.side, price, trade.btc_qty, trade.usd_value, rec.label)

        # Journal
        scores = {s.strategy_name: s.score for s in rec.signals}
        self._journal.record_trade(
            trade_id=trade.trade_id, side=trade.side, price=price,
            btc_qty=trade.btc_qty, label=rec.label, agg_score=score,
            strategy_scores=scores,
            indicator_snapshot={"rsi": round(ind.rsi, 1), "bb_pct": round(ind.bb_pct, 3)},
        )

        # Learning: grade after SELL
        if trade.side == "SELL":
            open_e = self._journal.get_open_buy()
            if open_e is not None:
                self._journal.close_trade(open_e.trade_id, price)
                outcome_pct = (price - open_e.price) / open_e.price * 100
                old_w = dict(self._adapter.weights)
                new_w = self._adapter.grade_trade(outcome_pct, open_e.strategy_scores)
                self._agg.update_weights(new_w)
                _print_learning(open_e.trade_id, outcome_pct, open_e.strategy_scores, old_w, new_w)
