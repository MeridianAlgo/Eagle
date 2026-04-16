"""
Eagle Backtest: Backtesting Engine
====================================
Historical backtesting with walk-forward analysis, performance
metrics, and detailed trade reporting.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from eagle.core.config import EagleConfig
from eagle.core.events import EventBus
from eagle.risk.portfolio import Portfolio

logger = logging.getLogger(__name__)


class BacktestMetrics:
    """Calculate comprehensive backtest performance metrics."""

    @staticmethod
    def calculate(portfolio: Portfolio, benchmark_returns: pd.Series | None = None) -> dict[str, Any]:
        """Calculate all metrics from portfolio state."""
        metrics: dict[str, Any] = {
            # Returns
            "total_return": portfolio.total_return,
            "total_return_pct": f"{portfolio.total_return * 100:.2f}%",

            # Risk metrics
            "sharpe_ratio": portfolio.sharpe_ratio,
            "sortino_ratio": portfolio.sortino_ratio,
            "max_drawdown": portfolio.max_drawdown,
            "max_drawdown_pct": f"{portfolio.max_drawdown * 100:.2f}%",

            # Trade statistics
            "total_trades": portfolio.total_trades,
            "win_rate": portfolio.win_rate,
            "win_rate_pct": f"{portfolio.win_rate * 100:.1f}%",
            "profit_factor": portfolio.profit_factor,
            "avg_trade_pnl": portfolio.avg_trade_pnl,

            # Portfolio
            "final_equity": portfolio.equity,
            "peak_equity": portfolio._peak_equity,
        }

        # Calmar ratio
        if portfolio.max_drawdown > 0:
            # Approximate annualized return
            n_days = len(portfolio._equity_curve)
            ann_return = portfolio.total_return * (252 / max(n_days, 1))
            metrics["calmar_ratio"] = ann_return / portfolio.max_drawdown
        else:
            metrics["calmar_ratio"] = 0.0

        # Benchmark comparison
        if benchmark_returns is not None and not benchmark_returns.empty:
            benchmark_total = (1 + benchmark_returns).prod() - 1
            metrics["benchmark_return"] = benchmark_total
            metrics["alpha"] = portfolio.total_return - benchmark_total

        return metrics

    @staticmethod
    def print_report(metrics: dict[str, Any]) -> str:
        """Generate a formatted backtest report."""
        lines = [
            "",
            "=" * 60,
            "  EAGLE BACKTEST REPORT",
            "=" * 60,
            "",
            "  RETURNS",
            f"    Total Return:        {metrics.get('total_return_pct', 'N/A')}",
            f"    Final Equity:        ${metrics.get('final_equity', 0):,.2f}",
            "",
            "  RISK METRICS",
            f"    Sharpe Ratio:        {metrics.get('sharpe_ratio', 0):.3f}",
            f"    Sortino Ratio:       {metrics.get('sortino_ratio', 0):.3f}",
            f"    Calmar Ratio:        {metrics.get('calmar_ratio', 0):.3f}",
            f"    Max Drawdown:        {metrics.get('max_drawdown_pct', 'N/A')}",
            "",
            "  TRADE STATISTICS",
            f"    Total Trades:        {metrics.get('total_trades', 0)}",
            f"    Win Rate:            {metrics.get('win_rate_pct', 'N/A')}",
            f"    Profit Factor:       {metrics.get('profit_factor', 0):.2f}",
            f"    Avg Trade P&L:       ${metrics.get('avg_trade_pnl', 0):,.2f}",
            "",
        ]

        if "alpha" in metrics:
            lines.extend([
                "  BENCHMARK COMPARISON",
                f"    Benchmark Return:    {metrics['benchmark_return'] * 100:.2f}%",
                f"    Alpha:               {metrics['alpha'] * 100:.2f}%",
                "",
            ])

        lines.append("=" * 60)
        report = "\n".join(lines)
        logger.info(report)
        return report


class BacktestEngine:
    """
    Event-driven backtesting engine with walk-forward analysis.

    Features:
        - Vectorized data processing for speed
        - Walk-forward optimization
        - Transaction cost modeling
        - Slippage simulation
        - Detailed metric computation
        - Equity curve generation
    """

    def __init__(
        self,
        config: EagleConfig,
        event_bus: EventBus,
        data_manager: Any,
        feature_engine: Any,
        model_manager: Any,
        strategy_manager: Any,
        risk_manager: Any,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.data_manager = data_manager
        self.feature_engine = feature_engine
        self.model_manager = model_manager
        self.strategy_manager = strategy_manager
        self.risk_manager = risk_manager
        self._portfolio: Portfolio | None = None

    async def run(self) -> dict[str, Any]:
        """Run the full backtest."""
        cfg = self.config.backtest
        logger.info(f"Starting backtest: {cfg.start_date} to {cfg.end_date}")

        # Fetch historical data
        symbols = [a.symbol for a in self.config.universe.assets]
        start = datetime.strptime(cfg.start_date, "%Y-%m-%d")
        end = datetime.strptime(cfg.end_date, "%Y-%m-%d")

        data = await self.data_manager.fetch_historical(
            symbols=symbols,
            start=start,
            end=end,
        )

        if data.empty:
            logger.error("No data available for backtesting")
            return {"error": "no_data"}

        # Compute features
        features = self.feature_engine.compute(data)

        if self.config.backtest.walk_forward.enabled:
            return await self._walk_forward_backtest(data, features, symbols)
        else:
            return await self._simple_backtest(data, features, symbols)

    async def _simple_backtest(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        symbols: list[str],
    ) -> dict[str, Any]:
        """Run a simple backtest over the full period."""
        cfg = self.config.backtest
        self._portfolio = Portfolio(self.config)

        # Train models on warmup period
        warmup = cfg.warmup_period
        train_data = features.iloc[:warmup]
        X_train, y_train = self.feature_engine.get_training_data(train_data)

        if not X_train.empty:
            await self.model_manager.train_all(X_train, y_train)

        # Iterate through each bar
        for i in range(warmup, len(data)):
            bar_data = data.iloc[[i]]
            bar_features = features.iloc[[i]]

            # Check stop losses and take profits
            self._check_exits(bar_data)

            # Get predictions
            window_features = features.iloc[max(0, i - 60) : i + 1]
            predictions = await self.model_manager.predict(window_features)

            # Generate signals
            window_data = data.iloc[max(0, i - 60) : i + 1]
            signals = self.strategy_manager.generate_signals(
                window_data, window_features, predictions
            )

            # Validate through risk
            approved = self.risk_manager.validate_signals(
                signals, self._portfolio, bar_data
            )

            # Execute signals
            for signal in approved:
                symbol = signal.symbol
                price = bar_data["close"].iloc[0]

                if signal.is_buy and symbol not in self._portfolio.positions:
                    quantity = (self._portfolio.equity * (signal.size_hint or 0.05)) / price
                    if quantity > 0:
                        try:
                            self._portfolio.open_position(
                                symbol=symbol,
                                side="long",
                                price=price * (1 + cfg.slippage_pct),
                                quantity=quantity,
                                stop_loss=signal.stop_loss,
                                take_profit=signal.take_profit,
                            )
                        except ValueError:
                            pass  # Insufficient funds

                elif signal.is_sell and symbol in self._portfolio.positions:
                    self._portfolio.close_position(
                        symbol=symbol,
                        price=price * (1 - cfg.slippage_pct),
                        reason="signal",
                    )

            # Update portfolio
            self._portfolio.update(bar_data)

        # Close all remaining positions at end
        for symbol in list(self._portfolio.positions.keys()):
            sym_data = data[data["symbol"] == symbol] if "symbol" in data.columns else data
            if not sym_data.empty:
                self._portfolio.close_position(symbol, sym_data["close"].iloc[-1], "backtest_end")

        # Calculate metrics
        metrics = BacktestMetrics.calculate(self._portfolio)
        BacktestMetrics.print_report(metrics)
        return metrics

    async def _walk_forward_backtest(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        symbols: list[str],
    ) -> dict[str, Any]:
        """
        Walk-forward optimization: train on rolling window,
        test on out-of-sample period, step forward.
        """
        wf = self.config.backtest.walk_forward
        self._portfolio = Portfolio(self.config)

        total_bars = len(data)
        train_size = wf.train_window_days
        test_size = wf.test_window_days
        step_size = wf.step_days

        window_results: list[dict[str, Any]] = []
        i = 0

        while i + train_size + test_size <= total_bars:
            train_end = i + train_size
            test_end = train_end + test_size

            # Train on training window
            train_features = features.iloc[i:train_end]
            X_train, y_train = self.feature_engine.get_training_data(train_features)

            if not X_train.empty:
                await self.model_manager.train_all(X_train, y_train)
                logger.info(f"Walk-forward window {i}-{train_end}: trained on {len(X_train)} samples")

            # Test on out-of-sample window
            for j in range(train_end, test_end):
                if j >= total_bars:
                    break

                bar_data = data.iloc[[j]]
                self._check_exits(bar_data)

                window_features = features.iloc[max(0, j - 60) : j + 1]
                predictions = await self.model_manager.predict(window_features)

                window_data = data.iloc[max(0, j - 60) : j + 1]
                signals = self.strategy_manager.generate_signals(
                    window_data, window_features, predictions
                )

                approved = self.risk_manager.validate_signals(
                    signals, self._portfolio, bar_data
                )

                for signal in approved:
                    price = bar_data["close"].iloc[0]
                    symbol = signal.symbol

                    if signal.is_buy and symbol not in self._portfolio.positions:
                        quantity = (self._portfolio.equity * (signal.size_hint or 0.05)) / price
                        if quantity > 0:
                            try:
                                self._portfolio.open_position(
                                    symbol=symbol, side="long",
                                    price=price, quantity=quantity,
                                    stop_loss=signal.stop_loss,
                                    take_profit=signal.take_profit,
                                )
                            except ValueError:
                                pass
                    elif signal.is_sell and symbol in self._portfolio.positions:
                        self._portfolio.close_position(symbol, price, "signal")

                self._portfolio.update(bar_data)

            i += step_size

        # Close remaining positions
        for symbol in list(self._portfolio.positions.keys()):
            sym_data = data[data["symbol"] == symbol] if "symbol" in data.columns else data
            if not sym_data.empty:
                self._portfolio.close_position(symbol, sym_data["close"].iloc[-1], "backtest_end")

        metrics = BacktestMetrics.calculate(self._portfolio)
        metrics["walk_forward_windows"] = len(window_results)
        BacktestMetrics.print_report(metrics)
        return metrics

    def _check_exits(self, bar_data: pd.DataFrame) -> None:
        """Check stop-loss and take-profit for all positions."""
        if self._portfolio is None:
            return

        for symbol in list(self._portfolio.positions.keys()):
            pos = self._portfolio.positions[symbol]
            sym_data = bar_data[bar_data["symbol"] == symbol] if "symbol" in bar_data.columns else bar_data

            if sym_data.empty:
                continue

            price = sym_data["close"].iloc[0]
            pos.update_price(price)

            if pos.should_stop_loss():
                self._portfolio.close_position(symbol, price, "stop_loss")
            elif pos.should_take_profit():
                self._portfolio.close_position(symbol, price, "take_profit")
