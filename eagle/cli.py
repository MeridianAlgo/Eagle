"""
Eagle CLI: Command Line Interface
===================================
Entry point for running Eagle in various modes.
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    """Eagle Trading Bot CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="eagle",
        description="Eagle: AI-Powered Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  eagle run                    Run Eagle in paper trading mode (default)
  eagle run --mode live        Run Eagle in live trading mode
  eagle backtest               Run historical backtesting
  eagle train                  Train ML models on historical data
  eagle dashboard              Launch the monitoring dashboard
  eagle status                 Show current portfolio status
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Start the trading engine")
    run_parser.add_argument(
        "--config", "-c", type=str, default=None, help="Path to config YAML file"
    )
    run_parser.add_argument(
        "--mode", "-m", choices=["paper", "live"], default="paper",
        help="Trading mode (default: paper)"
    )

    # Backtest command
    bt_parser = subparsers.add_parser("backtest", help="Run backtesting")
    bt_parser.add_argument("--config", "-c", type=str, default=None)
    bt_parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    bt_parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    bt_parser.add_argument("--symbols", nargs="+", help="Symbols to backtest")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train ML models")
    train_parser.add_argument("--config", "-c", type=str, default=None)
    train_parser.add_argument(
        "--model", choices=["lstm", "transformer", "xgboost", "ensemble", "all"],
        default="all", help="Model to train"
    )

    # Dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Launch monitoring dashboard")
    dash_parser.add_argument("--port", type=int, default=5050)

    # Status command
    subparsers.add_parser("status", help="Show current status")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "backtest":
        _cmd_backtest(args)
    elif args.command == "train":
        _cmd_train(args)
    elif args.command == "dashboard":
        _cmd_dashboard(args)
    elif args.command == "status":
        _cmd_status()


def _cmd_run(args: argparse.Namespace) -> None:
    """Start the trading engine."""
    from eagle.core.config import load_config
    from eagle.core.engine import EagleEngine

    overrides = {"app": {"mode": args.mode}}
    config = load_config(args.config, overrides=overrides)
    engine = EagleEngine(config=config)

    print(f"\n  Starting Eagle in {args.mode.upper()} mode...\n")

    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        asyncio.run(engine.shutdown())


def _cmd_backtest(args: argparse.Namespace) -> None:
    """Run backtesting."""
    from eagle.core.config import load_config
    from eagle.core.engine import EagleEngine

    overrides: dict = {"app": {"mode": "backtest"}}
    if args.start:
        overrides.setdefault("backtest", {})["start_date"] = args.start
    if args.end:
        overrides.setdefault("backtest", {})["end_date"] = args.end

    config = load_config(args.config, overrides=overrides)
    engine = EagleEngine(config=config)

    print("\n  Starting Eagle Backtest...\n")

    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        print("\n  Backtest interrupted")


def _cmd_train(args: argparse.Namespace) -> None:
    """Train ML models."""
    from eagle.core.config import load_config
    from eagle.core.events import EventBus
    from eagle.data.fetcher import DataManager
    from eagle.features.engineer import FeatureEngine
    from eagle.models.manager import ModelManager
    from eagle.utils.logger import setup_logging

    config = load_config(args.config)
    setup_logging(config)

    print(f"\n  Training model(s): {args.model}\n")

    async def _train() -> None:
        event_bus = EventBus()
        data_mgr = DataManager(config, event_bus)
        await data_mgr.initialize()

        from datetime import datetime, timedelta

        symbols = [a.symbol for a in config.universe.assets]
        end = datetime.utcnow()
        start = end - timedelta(days=config.data.history_days)

        print("  Fetching historical data...")
        data = await data_mgr.fetch_historical(symbols, start, end)

        if data.empty:
            print("  ERROR: No data available for training")
            return

        print(f"  Data loaded: {len(data)} rows")

        feature_engine = FeatureEngine(config)
        features = feature_engine.compute(data)
        X, y = feature_engine.get_training_data(features)

        print(f"  Features: {len(X.columns)} columns, {len(X)} samples")

        # Split into train/val
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        model_mgr = ModelManager(config, event_bus)
        await model_mgr.initialize()

        print("  Training in progress...\n")
        results = await model_mgr.train_all(X_train, y_train, X_val, y_val)

        print("\n  Training Results:")
        for model_name, metrics in results.items():
            print(f"    {model_name}: {metrics}")

        await data_mgr.shutdown()

    asyncio.run(_train())


def _cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the monitoring dashboard."""
    print(f"\n  Launching Eagle Dashboard on port {args.port}...")
    print(f"  Open http://127.0.0.1:{args.port} in your browser\n")

    # Dashboard implementation would go here
    print("  Dashboard module coming soon!")


def _cmd_status() -> None:
    """Show current status."""
    print("\n  Eagle Trading Bot Status")
    print("  " + "=" * 40)
    print("  Mode: Not running")
    print("  Use 'eagle run' to start\n")


if __name__ == "__main__":
    main()
