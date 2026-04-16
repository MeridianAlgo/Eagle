"""
Eagle: Real-Time Bitcoin Trading Bot
========================================
Orchestrates the entire live trading pipeline:

    1. Bootstrap 200 historical 1-minute candles from Binance REST
    2. Connect to Binance WebSocket for live BTC/USDT kline stream
    3. Compute indicators on every new closed candle
    4. Run all strategies and aggregate into a single recommendation
    5. Paper-trade the recommendation (no real money)
    6. Render a live Rich dashboard that updates every 0.5 s

Run via:
    python run_btc_bot.py
or:
    eagle btc       (if eagle CLI is installed)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from rich.live import Live

from eagle.data.binance_ws import BinanceWebSocket, Candle
from eagle.display.live_dashboard import DashboardSnapshot, LiveDashboard
from eagle.execution.paper_account import PaperAccount
from eagle.indicators import realtime_calculator as calc
from eagle.strategies.realtime.aggregator import SignalAggregator

logger = logging.getLogger(__name__)


class BTCBot:
    """
    Real-time Bitcoin trading bot.

    Args:
        initial_cash: Starting capital for paper trading (USD).
        refresh_rate: Dashboard refresh rate (renders per second).
    """

    def __init__(self, initial_cash: float = 10_000.0, refresh_rate: float = 2.0) -> None:
        self._ws = BinanceWebSocket(symbol="BTCUSDT", interval="1m", history_size=210)
        self._aggregator = SignalAggregator()
        self._account = PaperAccount(initial_cash=initial_cash)
        self._dashboard = LiveDashboard()
        self._snap = DashboardSnapshot(account=self._account)
        self._refresh_rate = refresh_rate
        self._candles_processed = 0

        # Register WebSocket callback
        self._ws.on_candle(self._on_candle)

    async def run(self) -> None:
        """Start the bot. Blocks until KeyboardInterrupt."""
        print("\n  🦅  Eagle BTC Bot starting — fetching historical data…\n")

        # Pre-fill history so indicators are ready immediately
        await self._ws.bootstrap()

        self._snap.connected = True
        self._snap.status_msg = "Live"

        with Live(
            self._dashboard.render(),
            refresh_per_second=self._refresh_rate,
            screen=True,
        ) as live:
            # Launch WebSocket in background
            ws_task = asyncio.create_task(self._ws.start())

            try:
                while True:
                    self._dashboard.update(self._snap)
                    live.update(self._dashboard.render())
                    await asyncio.sleep(1 / self._refresh_rate)
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
        """Called on every kline update (live candle or closed candle)."""
        self._snap.price = price
        self._snap.last_update = datetime.now(tz=timezone.utc)
        self._snap.connected = True

        if not candle.is_closed:
            # Live tick — update price only, skip heavy computation
            self._dashboard.update(self._snap)
            return

        # A candle just closed — run full analysis
        self._candles_processed += 1
        self._snap.candles_received = self._candles_processed

        # Advance cooldown counter
        self._account.tick()

        history = self._ws.history
        indicators = calc.compute(history, price)

        if indicators is None:
            self._snap.status_msg = f"Warming up… ({len(history)}/50 candles)"
            return

        self._snap.indicators = indicators

        # Run strategies
        recommendation = self._aggregator.evaluate(indicators)
        self._snap.recommendation = recommendation

        # Paper trade
        trade = self._account.execute(recommendation, price)
        if trade:
            self._snap.last_trade = trade
            logger.info(
                f"Trade executed: {trade.side} {trade.btc_qty:.6f} BTC "
                f"@ ${trade.price:,.2f} [{recommendation.label}]"
            )

        self._snap.account = self._account
        self._snap.status_msg = "Live"
