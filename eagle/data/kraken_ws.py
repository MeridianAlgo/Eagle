"""
Eagle Data: Kraken WebSocket Real-Time Feed
=============================================
Public WebSocket connection to Kraken for BTC/USD real-time data.
No API key required — uses Kraken's public market data streams.

Bootstraps up to 720 historical 1-minute candles via REST, then
switches to live WebSocket kline updates.

REST:      https://api.kraken.com/0/public/OHLC
WebSocket: wss://ws.kraken.com  (v1 API)
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

KRAKEN_REST = "https://api.kraken.com/0/public"
KRAKEN_WS   = "wss://ws.kraken.com"

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
    is_closed: bool

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def body_pct(self) -> float:
        rng = self.high - self.low
        return abs(self.close - self.open) / rng if rng > 0 else 0.0


class KrakenWebSocket:
    """
    Streams 1-minute BTC/USD klines from Kraken in real time.

    Usage::

        ws = KrakenWebSocket()
        ws.on_candle(my_async_callback)
        await ws.bootstrap()   # pre-fill history via REST
        await ws.start()       # blocks — runs the live stream
    """

    def __init__(self, history_size: int = 210) -> None:
        self._history: Deque[Candle] = deque(maxlen=history_size)
        self._current_price: float = 0.0
        self._current_candle: Candle | None = None
        self._callbacks: list[CandleCallback] = []
        self._running = False
        # Track the interval END time (etime / k[1]) — fixed for all trades in the same minute
        self._current_interval_end: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def on_candle(self, callback: CandleCallback) -> None:
        self._callbacks.append(callback)

    @property
    def history(self) -> list[Candle]:
        return list(self._history)

    @property
    def current_price(self) -> float:
        return self._current_price

    @property
    def current_candle(self) -> Candle | None:
        return self._current_candle

    @property
    def ready(self) -> bool:
        return len(self._history) >= 50

    async def bootstrap(self) -> None:
        """
        Fetch the last 720 closed 1-minute candles from Kraken REST API.
        Kraken returns up to 720 rows, oldest first.
        """
        url = f"{KRAKEN_REST}/OHLC?pair=XBTUSD&interval=1"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            if data.get("error"):
                raise ValueError(f"Kraken error: {data['error']}")

            rows = data["result"].get("XXBTZUSD", [])
            # Last row is the current (open) candle — exclude it
            for row in rows[:-1]:
                ts, o, h, l, c, _vwap, vol, _count = row
                candle = Candle(
                    timestamp=datetime.fromtimestamp(float(ts), tz=timezone.utc),
                    open=float(o),
                    high=float(h),
                    low=float(l),
                    close=float(c),
                    volume=float(vol),
                    is_closed=True,
                )
                self._history.append(candle)

            if self._history:
                self._current_price = self._history[-1].close

            logger.info(
                f"Bootstrapped {len(self._history)} Kraken candles — "
                f"current price: ${self._current_price:,.2f}"
            )
        except Exception as exc:
            logger.warning(f"REST bootstrap failed ({exc}); starting cold from WebSocket")

    async def start(self) -> None:
        """Stream live klines via Kraken WebSocket, with auto-reconnect."""
        self._running = True
        subscribe_msg = json.dumps({
            "event": "subscribe",
            "pair": ["XBT/USD"],
            "subscription": {"name": "ohlc", "interval": 1},
        })
        delay = 1.0

        while self._running:
            try:
                async with websockets.connect(
                    KRAKEN_WS,
                    ping_interval=20,
                    ping_timeout=10,
                    open_timeout=15,
                ) as ws:
                    delay = 1.0
                    await ws.send(subscribe_msg)
                    logger.info("Connected to Kraken WebSocket — subscribed to ohlc-1 XBT/USD")

                    async for raw in ws:
                        if not self._running:
                            return
                        await self._handle_message(raw)

            except websockets.exceptions.ConnectionClosed as exc:
                logger.warning(f"Kraken WS closed ({exc}), reconnecting in {delay:.0f}s")
            except OSError as exc:
                logger.error(f"Network error ({exc}), reconnecting in {delay:.0f}s")
            except Exception as exc:
                logger.error(f"Unexpected WS error: {exc}")

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

        # Control messages are dicts
        if isinstance(msg, dict):
            if msg.get("event") not in ("heartbeat", "systemStatus", "subscriptionStatus"):
                logger.debug(f"Kraken control: {msg}")
            return

        # Data messages are lists: [channelID, ohlc_dict, "ohlc-1", "XBT/USD"]
        if not isinstance(msg, list) or len(msg) < 4:
            return
        if not str(msg[2]).startswith("ohlc"):
            return

        k = msg[1]
        # Kraken ohlc fields: time(last-trade-ts), etime(interval-end), open, high, low, close, vwap, volume, count
        # IMPORTANT: k[0] = last-trade timestamp (changes every message — do NOT use for interval ID)
        #            k[1] = etime = fixed end-of-interval timestamp (same for all trades in the minute)
        interval_end = int(float(k[1]))          # use as stable interval identifier
        interval_open_ts = interval_end - 60     # 1-minute candle: open = end - 60 s

        o   = float(k[2])
        h   = float(k[3])
        l   = float(k[4])
        c   = float(k[5])
        vol = float(k[7])

        # Detect a new 1-minute interval when etime advances
        new_interval = interval_end != self._current_interval_end
        if new_interval and self._current_interval_end != 0:
            # The previous interval just ended — emit its final candle as closed
            if self._current_candle is not None:
                closed = Candle(
                    timestamp=self._current_candle.timestamp,
                    open=self._current_candle.open,
                    high=self._current_candle.high,
                    low=self._current_candle.low,
                    close=self._current_candle.close,
                    volume=self._current_candle.volume,
                    is_closed=True,
                )
                self._history.append(closed)
                logger.debug(
                    f"Candle closed: {closed.timestamp} "
                    f"O={closed.open:.2f} H={closed.high:.2f} "
                    f"L={closed.low:.2f} C={closed.close:.2f}"
                )
                await self._dispatch(closed, closed.close)

        self._current_interval_end = interval_end
        self._current_price = c
        live_candle = Candle(
            timestamp=datetime.fromtimestamp(interval_open_ts, tz=timezone.utc),
            open=o, high=h, low=l, close=c, volume=vol,
            is_closed=False,
        )
        self._current_candle = live_candle
        await self._dispatch(live_candle, c)

    async def _dispatch(self, candle: Candle, price: float) -> None:
        for cb in self._callbacks:
            try:
                await cb(candle, price)
            except Exception as exc:
                logger.error(f"Candle callback error: {exc}")
