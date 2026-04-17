"""
Eagle BTC Bot — Entry Point
============================
Run this script to start the real-time Bitcoin trading bot.

Usage:
    python run_btc_bot.py
    python run_btc_bot.py --capital 5000    # start with $5,000 paper capital

The bot will:
  • Fetch 200 historical 1-minute candles from Binance (no API key needed)
  • Connect to Binance's live WebSocket stream for BTC/USDT
  • Compute RSI, MACD, Bollinger Bands, and EMA indicators in real time
  • Run 4 strategies and aggregate their signals into a single recommendation
  • Paper-trade the best signal automatically
  • Display a live dashboard in your terminal

Press Ctrl+C to stop cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

# Force UTF-8 on Windows so Rich emoji render correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_btc_bot",
        description="Eagle real-time BTC/USDT paper trading bot",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10_000.0,
        metavar="USD",
        help="Starting paper capital in USD (default: 10000)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: WARNING — keeps terminal clean)",
    )
    parser.add_argument(
        "--log-file",
        default="logs/eagle_btc.log",
        metavar="PATH",
        help="Write logs to this file (default: logs/eagle_btc.log)",
    )
    return parser.parse_args()


def _setup_logging(level: str, log_file: str) -> None:
    import os
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    handlers: list[logging.Handler] = [logging.FileHandler(log_file, encoding="utf-8")]
    # Only add console handler for DEBUG — dashboard takes over the screen
    if level == "DEBUG":
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(level=level, format=fmt, handlers=handlers)


async def _main(capital: float) -> None:
    from eagle.btc_bot import BTCBot

    bot = BTCBot(initial_cash=capital, refresh_rate=2.0)
    try:
        await bot.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    args = _parse_args()
    _setup_logging(args.log_level, args.log_file)

    print(f"\n  Starting Eagle BTC Bot | capital: ${args.capital:,.0f} USD")
    print("  Press Ctrl+C to stop\n")

    try:
        asyncio.run(_main(args.capital))
    except KeyboardInterrupt:
        print("\n\n  Bot stopped. Goodbye.\n")
