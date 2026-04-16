# Project Eagle

A highly versatile, adaptable, trainable, efficient, and intelligent AI-powered trading bot built in Python.

## Architecture

```
eagle/
├── core/           Event bus, configuration, engine orchestration
├── data/           Async multi-provider data fetching and caching
├── indicators/     150+ vectorized technical indicators
├── features/       Feature engineering pipeline (100+ features)
├── models/         ML models (LSTM, Transformer, XGBoost, Ensemble)
├── strategies/     Plugin-based trading strategies
├── risk/           Risk management, position sizing, VaR
├── execution/      Order management and broker adapters
├── backtest/       Walk-forward backtesting engine
├── dashboard/      Real-time web monitoring (coming soon)
├── utils/          Logging, alerts, and helpers
└── cli.py          Command line interface
```

## Key Features

**Intelligence**
- Bidirectional LSTM with attention mechanism
- Transformer encoder with positional encoding
- XGBoost gradient boosted trees
- Weighted voting, stacking, and majority vote ensembles
- Automatic retraining with walk-forward optimization

**Strategies**
- Momentum / trend-following
- Mean reversion with z-score detection
- ML-driven signal generation
- Plugin architecture for custom strategies

**Risk Management**
- ATR-based dynamic position sizing
- Value at Risk (VaR) and Conditional VaR
- Maximum drawdown limits with auto-halt
- Daily loss limits with cooldown
- Correlation-based exposure controls
- Trailing stop losses

**Execution**
- Paper trading with slippage/commission simulation
- Alpaca Markets integration (paper and live)
- Binance crypto exchange support
- Order type support (market, limit, stop)

**Data**
- Async multi-provider data fetching (Yahoo, Binance, Alpaca, Alpha Vantage)
- In-memory LRU cache with TTL
- Automatic failover between providers

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/eagle.git
cd eagle

# Install in development mode
pip install -e ".[dev]"
```

### Configuration

```bash
# Copy default config
cp config/default.yaml config/config.yaml

# Edit with your API keys and preferences
# You can also set environment variables:
# EAGLE__DATA__PROVIDERS__ALPACA__API_KEY=your_key
```

### Usage

```bash
# Run in paper trading mode
eagle run

# Run backtesting
eagle backtest --start 2023-01-01 --end 2024-12-31

# Train ML models
eagle train --model all

# Train specific model
eagle train --model lstm

# Run with custom config
eagle run --config path/to/config.yaml --mode paper
```

### Python API

```python
import asyncio
from eagle.core.engine import EagleEngine
from eagle.core.config import load_config

async def main():
    config = load_config("config/config.yaml")
    engine = EagleEngine(config=config)
    await engine.run()

asyncio.run(main())
```

## Configuration

All parameters are configurable via `config/default.yaml`. Key sections:

| Section | Description |
|---------|-------------|
| `app` | Mode (paper/live/backtest), logging level |
| `data` | Data providers, caching, history length |
| `universe` | Asset watchlist and timeframes |
| `features` | Technical indicator parameters |
| `models` | ML model hyperparameters |
| `strategies` | Active strategies and their parameters |
| `risk` | Risk limits, position sizing, VaR |
| `execution` | Broker config, order types, slippage |
| `backtest` | Date range, walk-forward settings |

Environment variables override YAML config using the format:
```
EAGLE__SECTION__KEY=value
```

## Models

| Model | Type | Best For |
|-------|------|----------|
| LSTM | Deep Learning | Sequential patterns, regime detection |
| Transformer | Deep Learning | Long-range dependencies, attention patterns |
| XGBoost | Gradient Boosting | Tabular features, fast inference |
| Ensemble | Meta-learner | Combining all models for robust signals |

## Project Structure

```
Eagle/
├── config/
│   └── default.yaml            Default configuration
├── eagle/
│   ├── __init__.py
│   ├── cli.py                  Command line interface
│   ├── core/
│   │   ├── config.py           Pydantic configuration
│   │   ├── engine.py           Main engine orchestrator
│   │   └── events.py           Async event bus
│   ├── data/
│   │   └── fetcher.py          Multi-provider data fetching
│   ├── indicators/
│   │   └── technical.py        Technical indicator library
│   ├── features/
│   │   └── engineer.py         Feature engineering pipeline
│   ├── models/
│   │   ├── base_model.py       Base model interface
│   │   ├── lstm.py             BiLSTM with attention
│   │   ├── transformer.py      Transformer encoder
│   │   ├── ensemble.py         XGBoost and ensemble
│   │   └── manager.py          Model lifecycle manager
│   ├── strategies/
│   │   └── manager.py          Strategy framework
│   ├── risk/
│   │   ├── manager.py          Risk management engine
│   │   └── portfolio.py        Portfolio tracker
│   ├── execution/
│   │   └── engine.py           Order execution engine
│   ├── backtest/
│   │   └── engine.py           Backtesting engine
│   └── utils/
│       └── logger.py           Structured logging
├── pyproject.toml
└── README.md
```

## License

MIT
