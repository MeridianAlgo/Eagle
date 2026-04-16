"""
Eagle Data: Binance WebSocket Real-Time Feed
=============================================
Public WebSocket connection to Binance for BTC/USDT real-time data.
No API key required — uses Binance's public market data streams.

On startup, bootstraps 200 historical 1-minute candles via REST so that
indicators have enough data immediately, then switches to live WebSocket
updates for subsequent candles.

Streams used:
    wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Deque

import aiohttp
import websockets
import websockets.exceptions

logger = logging.getLogger(__name__)

BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"
BINANCE_REST_BASE = "https://api.binance.com/api/v3"

# Callback type: (candle, current_price) -> None (async)
CandleCallback = Callable[["Candle", float], Awaitable[None]]


@dataclass
class Candle:
    """Single OHLCV candle."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool  # True once the interval has ended

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def body_pct(self) -> float:
        """Candle body as a percentage of the high-low range."""
        rng = self.high - self.low
        return abs(self.close - self.open) / rng if rng > 0 else 0.0


class BinanceWebSocket:
    """
    Streams 1-minute BTC/USDT klines from Binance in real time.

    Usage::

        ws = BinanceWebSocket()
        ws.on_candle(my_async_callback)
        await ws.bootstrap()   # pre-fill history from REST
        await ws.start()       # blocks — runs the live stream

    The ``history`` property always contains the last ``history_size``
    *closed* candles (oldest first), ready for indicator computation.
    ``current_price`` reflects the latest trade price seen.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        history_size: int = 210,
    ) -> None:
        self.symbol = symbol.upper()
        self._symbol_lower = symbol.lower()
        self.interval = interval
        self._history: Deque[Candle] = deque(maxlen=history_size)
        self._current_candle: Candle | None = None
        self._current_price: float = 0.0
        self._callbacks: list[CandleCallback] = []
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_candle(self, callback: CandleCallback) -> None:
        """Register an async callback invoked on every incoming candle update."""
        self._callbacks.append(callback)

    @property
    def history(self) -> list[Candle]:
        """Snapshot of closed candles, oldest-first."""
        return list(self._history)

    @property
    def current_price(self) -> float:
        return self._current_price

    @property
    def current_candle(self) -> Candle | None:
        return self._current_candle

    @property
    def ready(self) -> bool:
        """True once we have at least 50 closed candles (enough for indicators)."""
        return len(self._history) >= 50

    async def bootstrap(self) -> None:
        """
        Pre-fill candle history by fetching the last 200 closed 1-minute
        candles from Binance REST API. Call this once before start().
        """
        url = (
            f"{BINANCE_REST_BASE}/klines"
            f"?symbol={self.symbol}&interval={self.interval}&limit=200"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            for row in data:
                candle = Candle(
                    timestamp=datetime.fromtimestamp(row[0] / 1000, tz=timezone.utc),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                    is_closed=True,
                )
                self._history.append(candle)

            if self._history:
                self._current_price = self._history[-1].close

            logger.info(
                f"Bootstrapped {len(self._history)} historical candles "
                f"for {self.symbol} — current price: ${self._current_price:,.2f}"
            )
        except Exception as exc:
            logger.warning(f"REST bootstrap failed ({exc}); starting cold from WebSocket")

    async def start(self) -> None:
        """
        Connect to the Binance kline WebSocket and stream candles forever.
        Auto-reconnects with exponential back-off on failure.
        """
        self._running = True
        url = f"{BINANCE_WS_BASE}/{self._symbol_lower}@kline_{self.interval}"
        delay = 1.0

        while self._running:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    open_timeout=15,
                ) as ws:
                    delay = 1.0  # reset back-off on successful connect
                    logger.info(f"Live stream connected: {url}")

                    async for raw in ws:
                        if not self._running:
                            return
                        await self._handle_message(raw)

            except websockets.exceptions.ConnectionClosed as exc:
                logger.warning(f"Stream closed ({exc}), reconnecting in {delay:.0f}s")
            except OSError as exc:
                logger.error(f"Network error ({exc}), reconnecting in {delay:.0f}s")
            except Exception as exc:
                logger.error(f"Unexpected stream error: {exc}")

            if self._running:
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)

    async def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _handle_message(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        if msg.get("e") != "kline":
            return

        k = msg["k"]
        candle = Candle(
            timestamp=datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc),
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            is_closed=bool(k["x"]),
        )

        self._current_price = candle.close
        self._current_candle = candle

        if candle.is_closed:
            self._history.append(candle)

        for cb in self._callbacks:
            try:
                await cb(candle, self._current_price)
            except Exception as exc:
                logger.error(f"Candle callback error: {exc}")
