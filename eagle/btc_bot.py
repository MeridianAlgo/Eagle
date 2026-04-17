"""
Eagle: Real-Time Bitcoin Trading Bot with Self-Learning
=========================================================
Orchestrates the full live trading + learning pipeline:

    1. Bootstrap historical candles from Kraken REST
    2. Stream live BTC/USD klines via Kraken WebSocket
    3. Compute indicators on every closed candle
    4. Run 4 strategies, aggregate signals → trade recommendation
    5. Paper-trade the recommendation (no real money)
    6. Record every trade to the journal with its full indicator context
    7. After each SELL, grade each strategy (correct / wrong) and
       update their weights via exponential-moving-average learning
    8. Hot-swap updated weights into the aggregator — it learns live

Usage:
    python run_btc_bot.py
    python run_btc_bot.py --capital 5000
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from rich.live import Live

from eagle.data.kraken_ws import Candle, KrakenWebSocket
from eagle.display.live_dashboard import DashboardSnapshot, LiveDashboard
from eagle.execution.paper_account import PaperAccount
from eagle.indicators import realtime_calculator as calc
from eagle.learning.trade_journal import TradeJournal
from eagle.learning.weight_adapter import WeightAdapter
from eagle.strategies.realtime.aggregator import SignalAggregator

logger = logging.getLogger(__name__)


class BTCBot:
    """
    Real-time BTC/USD paper trading bot with a self-learning weight system.

    After every completed trade cycle (BUY → SELL), each strategy is graded
    on whether its signal at entry time was directionally correct.  Weights
    are updated via EMA and hot-swapped into the aggregator immediately.
    """

    def __init__(self, initial_cash: float = 10_000.0, refresh_rate: float = 2.0) -> None:
        self._ws          = KrakenWebSocket(history_size=210)
        self._aggregator  = SignalAggregator()
        self._account     = PaperAccount(initial_cash=initial_cash)
        self._journal     = TradeJournal()
        self._adapter     = WeightAdapter()
        self._dashboard   = LiveDashboard()
        self._refresh     = refresh_rate
        self._candles_done = 0

        # Seed the aggregator with any previously-learned weights
        self._aggregator.update_weights(self._adapter.weights)

        snap = DashboardSnapshot(
            account=self._account,
            weight_adapter=self._adapter,
            status_msg="Starting...",
        )
        self._snap = snap
        self._ws.on_candle(self._on_candle)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        print("\n  [Eagle] BTC Bot starting -- fetching historical data from Kraken...\n")
        await self._ws.bootstrap()

        self._snap.connected = True
        self._snap.status_msg = "Live"

        with Live(
            self._dashboard.render(),
            refresh_per_second=self._refresh,
            screen=True,
        ) as live:
            ws_task = asyncio.create_task(self._ws.start())
            try:
                while True:
                    self._dashboard.update(self._snap)
                    live.update(self._dashboard.render())
                    await asyncio.sleep(1.0 / self._refresh)
            except asyncio.CancelledError:
                pass
            finally:
                await self._ws.stop()
                ws_task.cancel()
                try:
                    await ws_task
                except asyncio.CancelledError:
                    pass

    # ------------------------------------------------------------------
    # WebSocket callback
    # ------------------------------------------------------------------

    async def _on_candle(self, candle: Candle, price: float) -> None:
        self._snap.price       = price
        self._snap.last_update = datetime.now(tz=timezone.utc)
        self._snap.connected   = True

        if not candle.is_closed:
            return  # live tick — price already updated above, skip heavy work

        self._candles_done        += 1
        self._snap.candles_received = self._candles_done
        self._account.tick()

        history = self._ws.history
        indicators = calc.compute(history, price)
        if indicators is None:
            self._snap.status_msg = f"Warming up ({len(history)}/50 candles)..."
            return

        self._snap.indicators  = indicators
        self._snap.status_msg  = "Live"

        # ── Evaluate strategies ──────────────────────────────────────
        rec = self._aggregator.evaluate(indicators)
        self._snap.recommendation = rec

        # ── Paper trade ──────────────────────────────────────────────
        trade = self._account.execute(rec, price)
        if trade is None:
            return

        self._snap.last_trade  = trade
        self._snap.account     = self._account

        # ── Journal: record every trade ──────────────────────────────
        strategy_scores = {s.strategy_name: s.score for s in rec.signals}
        indicator_snap  = {
            "rsi":       round(indicators.rsi, 2),
            "macd_hist": round(indicators.macd_hist, 4),
            "bb_pct":    round(indicators.bb_pct, 3),
            "ema_9":     round(indicators.ema_9, 2),
            "ema_21":    round(indicators.ema_21, 2),
            "ema_50":    round(indicators.ema_50, 2),
            "vol_ratio": round(indicators.volume_ratio, 3),
            "atr_pct":   round(indicators.atr_pct * 100, 4),
        }
        self._journal.record_trade(
            trade_id=trade.trade_id,
            side=trade.side,
            price=price,
            btc_qty=trade.btc_qty,
            label=rec.label,
            agg_score=rec.score,
            strategy_scores=strategy_scores,
            indicator_snapshot=indicator_snap,
        )

        # ── Learning: grade strategies after a SELL closes a position ─
        if trade.side == "SELL":
            open_entry = self._journal.get_open_buy()
            if open_entry is not None:
                self._journal.close_trade(open_entry.trade_id, price)
                outcome_pct = (price - open_entry.price) / open_entry.price * 100
                updated_weights = self._adapter.grade_trade(
                    outcome_pct=outcome_pct,
                    strategy_scores=open_entry.strategy_scores,
                )
                self._aggregator.update_weights(updated_weights)
                self._snap.weight_adapter = self._adapter

                direction = "profit" if outcome_pct >= 0 else "loss"
                self._snap.last_learn_event = (
                    f"Learned from trade #{open_entry.trade_id}: "
                    f"{direction} {outcome_pct:+.2f}% -- weights updated"
                )
                logger.info(self._snap.last_learn_event)
