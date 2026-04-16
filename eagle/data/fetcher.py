"""
Eagle Data: Async Market Data Fetcher
======================================
Multi-provider async data fetching with caching, rate limiting,
and automatic failover.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from eagle.core.config import EagleConfig
from eagle.core.events import Event, EventBus, EventType

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    async def fetch_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data."""
        ...

    @abstractmethod
    async def fetch_latest(self, symbol: str, timeframe: str = "1d") -> pd.DataFrame:
        """Fetch latest available data."""
        ...

    @abstractmethod
    async def connect(self) -> None:
        """Initialize provider connection."""
        ...

    @abstractmethod
    async def disconnect(self) -> None:
        """Clean up provider connection."""
        ...


class YahooProvider(DataProvider):
    """Yahoo Finance data provider using yfinance."""

    def __init__(self, config: EagleConfig) -> None:
        self.config = config
        self._rate_limit = config.data.providers.yahoo.rate_limit
        self._semaphore = asyncio.Semaphore(self._rate_limit)

    async def connect(self) -> None:
        logger.info("Yahoo Finance provider ready")

    async def disconnect(self) -> None:
        pass

    async def fetch_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance."""
        import yfinance as yf

        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1h", "4h": "4h", "1d": "1d", "1w": "1wk", "1M": "1mo",
        }
        interval = interval_map.get(timeframe, "1d")

        async with self._semaphore:
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)
            df = await loop.run_in_executor(
                None,
                lambda: ticker.history(start=start, end=end, interval=interval),
            )

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Standardize column names
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df.index.name = "datetime"

        # Add metadata
        df["symbol"] = symbol
        df["timeframe"] = timeframe

        return df

    async def fetch_latest(self, symbol: str, timeframe: str = "1d") -> pd.DataFrame:
        """Fetch the most recent data."""
        end = datetime.utcnow()
        start = end - timedelta(days=5)
        df = await self.fetch_historical(symbol, start, end, timeframe)
        if not df.empty:
            return df.tail(1)
        return pd.DataFrame()


class BinanceProvider(DataProvider):
    """Binance data provider using ccxt."""

    def __init__(self, config: EagleConfig) -> None:
        self.config = config
        self._exchange: Any = None

    async def connect(self) -> None:
        try:
            import ccxt.async_support as ccxt

            exchange_config: dict[str, Any] = {
                "enableRateLimit": True,
            }

            if self.config.data.providers.binance.api_key:
                exchange_config["apiKey"] = self.config.data.providers.binance.api_key
                exchange_config["secret"] = self.config.data.providers.binance.api_secret

            if self.config.data.providers.binance.testnet:
                exchange_config["sandbox"] = True

            self._exchange = ccxt.binance(exchange_config)
            logger.info("Binance provider connected")
        except ImportError:
            logger.warning("ccxt not installed, Binance provider unavailable")

    async def disconnect(self) -> None:
        if self._exchange:
            await self._exchange.close()

    async def fetch_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        if not self._exchange:
            return pd.DataFrame()

        since = int(start.timestamp() * 1000)
        ohlcv = await self._exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)

        if not ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)
        df.drop(columns=["timestamp"], inplace=True)
        df["symbol"] = symbol
        df["timeframe"] = timeframe

        # Filter to end date
        df = df[df.index <= pd.Timestamp(end)]
        return df

    async def fetch_latest(self, symbol: str, timeframe: str = "1d") -> pd.DataFrame:
        if not self._exchange:
            return pd.DataFrame()

        ohlcv = await self._exchange.fetch_ohlcv(symbol, timeframe, limit=1)
        if not ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("datetime", inplace=True)
        df.drop(columns=["timestamp"], inplace=True)
        df["symbol"] = symbol
        df["timeframe"] = timeframe
        return df


class DataCache:
    """In-memory LRU cache for market data with TTL support."""

    def __init__(self, ttl_minutes: int = 15, max_size: int = 1000) -> None:
        self._cache: dict[str, tuple[datetime, pd.DataFrame]] = {}
        self._ttl = timedelta(minutes=ttl_minutes)
        self._max_size = max_size

    def get(self, key: str) -> pd.DataFrame | None:
        """Get cached data if not expired."""
        if key in self._cache:
            timestamp, data = self._cache[key]
            if datetime.utcnow() - timestamp < self._ttl:
                return data.copy()
            else:
                del self._cache[key]
        return None

    def set(self, key: str, data: pd.DataFrame) -> None:
        """Cache data with current timestamp."""
        if len(self._cache) >= self._max_size:
            # Remove oldest entry
            oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
            del self._cache[oldest_key]
        self._cache[key] = (datetime.utcnow(), data.copy())

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate a specific key or entire cache."""
        if key:
            self._cache.pop(key, None)
        else:
            self._cache.clear()


class DataManager:
    """
    Central data management layer.

    Coordinates between multiple data providers with caching,
    automatic failover, and event emission.
    """

    def __init__(self, config: EagleConfig, event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._providers: dict[str, DataProvider] = {}
        self._cache = DataCache(
            ttl_minutes=config.data.cache_ttl_minutes if config.data.cache_enabled else 0,
        )
        self._default_provider = config.data.default_provider

    async def initialize(self) -> None:
        """Initialize enabled data providers."""
        provider_map: dict[str, tuple[bool, type[DataProvider]]] = {
            "yahoo": (self.config.data.providers.yahoo.enabled, YahooProvider),
            "binance": (self.config.data.providers.binance.enabled, BinanceProvider),
        }

        for name, (enabled, provider_class) in provider_map.items():
            if enabled:
                provider = provider_class(self.config)
                await provider.connect()
                self._providers[name] = provider
                logger.info(f"Data provider '{name}' initialized")

        if not self._providers:
            logger.warning("No data providers enabled, using Yahoo as fallback")
            provider = YahooProvider(self.config)
            await provider.connect()
            self._providers["yahoo"] = provider

    async def fetch_historical(
        self,
        symbols: list[str],
        start: datetime | None = None,
        end: datetime | None = None,
        timeframe: str = "1d",
        provider: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical data for multiple symbols.

        Returns a MultiIndex DataFrame with symbol and datetime.
        """
        if end is None:
            end = datetime.utcnow()
        if start is None:
            start = end - timedelta(days=self.config.data.history_days)

        provider_name = provider or self._default_provider
        data_provider = self._providers.get(provider_name)

        if not data_provider:
            raise ValueError(f"Data provider '{provider_name}' not available")

        all_data: list[pd.DataFrame] = []
        tasks = []

        for symbol in symbols:
            cache_key = f"{symbol}_{timeframe}_{start.date()}_{end.date()}"
            cached = self._cache.get(cache_key)

            if cached is not None:
                all_data.append(cached)
                continue

            tasks.append((symbol, cache_key, data_provider.fetch_historical(symbol, start, end, timeframe)))

        # Fetch missing data concurrently
        if tasks:
            results = await asyncio.gather(
                *[t[2] for t in tasks],
                return_exceptions=True,
            )

            for (symbol, cache_key, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to fetch {symbol}: {result}")
                    continue
                if not result.empty:
                    self._cache.set(cache_key, result)
                    all_data.append(result)

                    # Emit market data event
                    await self.event_bus.emit(Event(
                        event_type=EventType.MARKET_DATA,
                        source="data_manager",
                        data={"symbol": symbol, "rows": len(result)},
                    ))

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, axis=0)

    async def fetch_latest(
        self,
        symbols: list[str],
        timeframe: str | None = None,
    ) -> pd.DataFrame:
        """Fetch latest data for all symbols in the universe."""
        if timeframe is None:
            timeframe = self.config.universe.timeframes[0] if self.config.universe.timeframes else "1d"

        provider = self._providers.get(self._default_provider)
        if not provider:
            provider = next(iter(self._providers.values()), None)

        if not provider:
            return pd.DataFrame()

        results = await asyncio.gather(
            *[provider.fetch_latest(s, timeframe) for s in symbols],
            return_exceptions=True,
        )

        dfs = []
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch latest {symbol}: {result}")
            elif not result.empty:
                dfs.append(result)

        return pd.concat(dfs, axis=0) if dfs else pd.DataFrame()

    async def shutdown(self) -> None:
        """Disconnect all providers."""
        for name, provider in self._providers.items():
            await provider.disconnect()
            logger.info(f"Data provider '{name}' disconnected")
