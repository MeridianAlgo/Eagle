"""
Eagle Core: Main Engine Orchestrator
======================================
The central nervous system of Eagle. Initializes all subsystems,
manages the lifecycle, and coordinates the trading loop.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from datetime import datetime
from typing import Any

from eagle.core.config import EagleConfig, load_config
from eagle.core.events import Event, EventBus, EventType
from eagle.utils.logger import setup_logging

logger = logging.getLogger(__name__)


class EagleEngine:
    """
    Main trading engine that orchestrates all Eagle subsystems.

    Lifecycle:
        1. Initialize configuration and logging
        2. Start event bus
        3. Initialize data, strategy, risk, and execution subsystems
        4. Enter main trading loop
        5. Graceful shutdown on signal

    Modes:
        - paper: Simulated trading with paper money
        - live: Real trading with actual broker
        - backtest: Historical backtesting
    """

    def __init__(self, config: EagleConfig | None = None, config_path: str | None = None) -> None:
        self.config = config or load_config(config_path)
        self.event_bus = EventBus()
        self._running = False
        self._tasks: list[asyncio.Task[Any]] = []

        # Subsystem registries (populated during initialization)
        self._data_manager: Any = None
        self._feature_engine: Any = None
        self._model_manager: Any = None
        self._strategy_manager: Any = None
        self._risk_manager: Any = None
        self._execution_engine: Any = None
        self._portfolio: Any = None
        self._dashboard: Any = None

        # Performance tracking
        self._start_time: datetime | None = None
        self._cycle_count = 0

    async def initialize(self) -> None:
        """Initialize all subsystems."""
        setup_logging(self.config)
        logger.info("=" * 60)
        logger.info(f"  {self.config.app.name} v{self.config.app.version}")
        logger.info(f"  Mode: {self.config.app.mode}")
        logger.info("=" * 60)

        # Initialize data subsystem
        await self._init_data()

        # Initialize intelligence subsystem
        await self._init_intelligence()

        # Initialize risk subsystem
        await self._init_risk()

        # Initialize execution subsystem
        await self._init_execution()

        # Subscribe to system events
        self.event_bus.subscribe(EventType.ERROR, self._handle_error)
        self.event_bus.subscribe(EventType.TRADING_HALTED, self._handle_halt)

        logger.info("All subsystems initialized successfully")

    async def _init_data(self) -> None:
        """Initialize data fetching and storage."""
        from eagle.data.fetcher import DataManager

        self._data_manager = DataManager(self.config, self.event_bus)
        await self._data_manager.initialize()
        logger.info("Data subsystem initialized")

    async def _init_intelligence(self) -> None:
        """Initialize feature engineering, ML models, and strategies."""
        from eagle.features.engineer import FeatureEngine
        from eagle.models.manager import ModelManager
        from eagle.strategies.manager import StrategyManager

        self._feature_engine = FeatureEngine(self.config)
        logger.info("Feature engine initialized")

        self._model_manager = ModelManager(self.config, self.event_bus)
        await self._model_manager.initialize()
        logger.info("Model manager initialized")

        self._strategy_manager = StrategyManager(self.config, self.event_bus)
        self._strategy_manager.initialize()
        logger.info("Strategy manager initialized")

    async def _init_risk(self) -> None:
        """Initialize risk management and portfolio tracking."""
        from eagle.risk.manager import RiskManager
        from eagle.risk.portfolio import Portfolio

        self._portfolio = Portfolio(self.config)
        self._risk_manager = RiskManager(self.config, self.event_bus, self._portfolio)
        logger.info("Risk subsystem initialized")

    async def _init_execution(self) -> None:
        """Initialize execution engine and broker connections."""
        from eagle.execution.engine import ExecutionEngine

        self._execution_engine = ExecutionEngine(self.config, self.event_bus)
        await self._execution_engine.initialize()
        logger.info("Execution subsystem initialized")

    async def run(self) -> None:
        """Main entry point. Starts the trading engine."""
        await self.initialize()
        self._running = True
        self._start_time = datetime.utcnow()

        # Emit engine start event
        await self.event_bus.emit(Event(
            event_type=EventType.ENGINE_START,
            source="engine",
            data={"mode": self.config.app.mode, "start_time": str(self._start_time)},
        ))

        # Start background tasks
        self._tasks.append(asyncio.create_task(self.event_bus.process_queue()))
        self._tasks.append(asyncio.create_task(self._heartbeat_loop()))

        if self.config.app.mode == "backtest":
            await self._run_backtest()
        else:
            await self._run_trading_loop()

    async def _run_trading_loop(self) -> None:
        """Main trading loop for paper/live modes."""
        logger.info("Starting trading loop...")

        while self._running:
            try:
                self._cycle_count += 1

                # 1. Fetch latest market data
                market_data = await self._data_manager.fetch_latest(
                    symbols=[a.symbol for a in self.config.universe.assets],
                )

                if market_data is None or market_data.empty:
                    logger.debug("No new market data, waiting...")
                    await asyncio.sleep(1)
                    continue

                # 2. Generate features
                features = self._feature_engine.compute(market_data)

                # 3. Get model predictions
                predictions = await self._model_manager.predict(features)

                # 4. Generate signals from strategies
                signals = self._strategy_manager.generate_signals(
                    market_data=market_data,
                    features=features,
                    predictions=predictions,
                )

                # 5. Validate signals through risk management
                approved_signals = self._risk_manager.validate_signals(
                    signals=signals,
                    portfolio=self._portfolio,
                    market_data=market_data,
                )

                # 6. Execute approved signals
                for sig in approved_signals:
                    await self._execution_engine.execute(sig)

                # 7. Update portfolio state
                self._portfolio.update(market_data)

                # 8. Check risk limits
                self._risk_manager.check_portfolio_risk(self._portfolio, market_data)

                # Emit portfolio update event
                await self.event_bus.emit(Event(
                    event_type=EventType.PORTFOLIO_UPDATED,
                    source="engine",
                    data=self._portfolio.snapshot(),
                ))

                # Adaptive sleep based on market conditions
                await asyncio.sleep(self._get_loop_interval())

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trading loop error: {e}", exc_info=True)
                await self.event_bus.emit(Event(
                    event_type=EventType.ERROR,
                    source="engine",
                    data={"error": str(e), "cycle": self._cycle_count},
                ))
                await asyncio.sleep(5)

    async def _run_backtest(self) -> None:
        """Run backtesting mode."""
        from eagle.backtest.engine import BacktestEngine

        backtester = BacktestEngine(
            config=self.config,
            event_bus=self.event_bus,
            data_manager=self._data_manager,
            feature_engine=self._feature_engine,
            model_manager=self._model_manager,
            strategy_manager=self._strategy_manager,
            risk_manager=self._risk_manager,
        )
        await backtester.run()

    async def _heartbeat_loop(self) -> None:
        """Emit periodic heartbeat events."""
        while self._running:
            await self.event_bus.emit(Event(
                event_type=EventType.HEARTBEAT,
                source="engine",
                data={
                    "cycles": self._cycle_count,
                    "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds()
                    if self._start_time
                    else 0,
                    "pending_events": self.event_bus.pending_count,
                },
            ))
            await asyncio.sleep(30)

    async def _handle_error(self, event: Event) -> None:
        """Handle error events."""
        logger.error(f"System error: {event.data}")

    async def _handle_halt(self, event: Event) -> None:
        """Handle trading halt events."""
        logger.critical(f"TRADING HALTED: {event.data}")
        self._running = False

    def _get_loop_interval(self) -> float:
        """Get adaptive loop interval based on timeframe."""
        timeframe = self.config.universe.timeframes[0] if self.config.universe.timeframes else "1d"
        intervals = {
            "1m": 1.0,
            "5m": 5.0,
            "15m": 15.0,
            "1h": 60.0,
            "4h": 240.0,
            "1d": 300.0,
        }
        return intervals.get(timeframe, 60.0)

    async def shutdown(self) -> None:
        """Graceful shutdown of all subsystems."""
        logger.info("Shutting down Eagle engine...")
        self._running = False

        # Emit stop event
        await self.event_bus.emit(Event(
            event_type=EventType.ENGINE_STOP,
            source="engine",
            data={"total_cycles": self._cycle_count},
        ))

        # Stop event bus
        self.event_bus.stop()

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

        # Close subsystems
        if self._execution_engine:
            await self._execution_engine.shutdown()
        if self._data_manager:
            await self._data_manager.shutdown()

        logger.info("Eagle engine shut down successfully")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def uptime(self) -> float:
        """Return uptime in seconds."""
        if self._start_time:
            return (datetime.utcnow() - self._start_time).total_seconds()
        return 0.0


def run_eagle(config_path: str | None = None) -> None:
    """Convenience function to run Eagle with signal handling."""
    engine = EagleEngine(config_path=config_path)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Register signal handlers for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.create_task(engine.shutdown()))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        loop.run_until_complete(engine.run())
    except KeyboardInterrupt:
        loop.run_until_complete(engine.shutdown())
    finally:
        loop.close()
