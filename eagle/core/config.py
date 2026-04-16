"""
Eagle Core: Configuration Management
======================================
Pydantic-based configuration with YAML loading, environment variable
overrides, and runtime validation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


# -----------------------------------------------------------
# Data Provider Configs
# -----------------------------------------------------------

class YahooConfig(BaseModel):
    enabled: bool = True
    rate_limit: int = 5


class BinanceConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = True


class AlpacaProviderConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""
    api_secret: str = ""
    paper: bool = True
    base_url: str = "https://paper-api.alpaca.markets"


class AlphaVantageConfig(BaseModel):
    enabled: bool = False
    api_key: str = ""


class DataProviders(BaseModel):
    yahoo: YahooConfig = Field(default_factory=YahooConfig)
    binance: BinanceConfig = Field(default_factory=BinanceConfig)
    alpaca: AlpacaProviderConfig = Field(default_factory=AlpacaProviderConfig)
    alpha_vantage: AlphaVantageConfig = Field(default_factory=AlphaVantageConfig)


class StoreConfig(BaseModel):
    backend: str = "duckdb"
    path: str = "data/eagle.duckdb"


class DataConfig(BaseModel):
    default_provider: str = "yahoo"
    cache_enabled: bool = True
    cache_ttl_minutes: int = 15
    history_days: int = 365
    store: StoreConfig = Field(default_factory=StoreConfig)
    providers: DataProviders = Field(default_factory=DataProviders)


# -----------------------------------------------------------
# Universe
# -----------------------------------------------------------

class AssetConfig(BaseModel):
    symbol: str
    asset_type: str = "equity"


class UniverseConfig(BaseModel):
    assets: list[AssetConfig] = Field(default_factory=list)
    timeframes: list[str] = Field(default_factory=lambda: ["1d"])


# -----------------------------------------------------------
# Features
# -----------------------------------------------------------

class MACDConfig(BaseModel):
    fast: int = 12
    slow: int = 26
    signal: int = 9


class BollingerConfig(BaseModel):
    period: int = 20
    std_dev: float = 2.0


class StochasticConfig(BaseModel):
    k_period: int = 14
    d_period: int = 3


class TechnicalIndicatorConfig(BaseModel):
    sma_periods: list[int] = Field(default_factory=lambda: [10, 20, 50, 200])
    ema_periods: list[int] = Field(default_factory=lambda: [12, 26, 50])
    rsi_period: int = 14
    macd: MACDConfig = Field(default_factory=MACDConfig)
    bollinger: BollingerConfig = Field(default_factory=BollingerConfig)
    atr_period: int = 14
    adx_period: int = 14
    stochastic: StochasticConfig = Field(default_factory=StochasticConfig)
    volume_indicators: bool = True
    volatility_indicators: bool = True
    momentum_indicators: bool = True


class CustomFeatureConfig(BaseModel):
    price_momentum: bool = True
    volume_profile: bool = True
    market_regime: bool = True
    correlation_features: bool = True


class FeatureConfig(BaseModel):
    technical_indicators: TechnicalIndicatorConfig = Field(default_factory=TechnicalIndicatorConfig)
    custom_features: CustomFeatureConfig = Field(default_factory=CustomFeatureConfig)


# -----------------------------------------------------------
# ML Models
# -----------------------------------------------------------

class LSTMConfig(BaseModel):
    enabled: bool = True
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    sequence_length: int = 60
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 10


class TransformerConfig(BaseModel):
    enabled: bool = True
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 4
    dropout: float = 0.1
    sequence_length: int = 60
    learning_rate: float = 0.0001
    epochs: int = 100
    batch_size: int = 32


class XGBoostConfig(BaseModel):
    enabled: bool = True
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 20


class EnsembleConfig(BaseModel):
    enabled: bool = True
    method: str = "weighted_vote"
    weights: dict[str, float] = Field(
        default_factory=lambda: {"lstm": 0.35, "transformer": 0.35, "xgboost": 0.30}
    )
    confidence_threshold: float = 0.6


class ModelsConfig(BaseModel):
    default_model: str = "ensemble"
    lstm: LSTMConfig = Field(default_factory=LSTMConfig)
    transformer: TransformerConfig = Field(default_factory=TransformerConfig)
    xgboost: XGBoostConfig = Field(default_factory=XGBoostConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)


# -----------------------------------------------------------
# Strategies
# -----------------------------------------------------------

class MomentumStrategyConfig(BaseModel):
    lookback_period: int = 20
    entry_threshold: float = 0.02
    exit_threshold: float = -0.01
    max_holding_days: int = 10


class MeanReversionStrategyConfig(BaseModel):
    lookback_period: int = 20
    entry_z_score: float = 2.0
    exit_z_score: float = 0.5
    max_holding_days: int = 5


class MLStrategyConfig(BaseModel):
    model: str = "ensemble"
    retrain_interval_days: int = 7
    min_confidence: float = 0.6
    position_scale_by_confidence: bool = True


class PairsTradingConfig(BaseModel):
    lookback_period: int = 60
    entry_z_score: float = 2.0
    exit_z_score: float = 0.5
    cointegration_pvalue: float = 0.05


class StrategiesConfig(BaseModel):
    active_strategies: list[str] = Field(default_factory=lambda: ["ml_strategy"])
    momentum: MomentumStrategyConfig = Field(default_factory=MomentumStrategyConfig)
    mean_reversion: MeanReversionStrategyConfig = Field(default_factory=MeanReversionStrategyConfig)
    ml_strategy: MLStrategyConfig = Field(default_factory=MLStrategyConfig)
    pairs_trading: PairsTradingConfig = Field(default_factory=PairsTradingConfig)


# -----------------------------------------------------------
# Risk Management
# -----------------------------------------------------------

class RiskConfig(BaseModel):
    max_portfolio_risk: float = 0.02
    max_position_size: float = 0.10
    max_total_exposure: float = 0.95
    max_correlated_exposure: float = 0.30
    max_daily_loss: float = 0.03
    max_drawdown: float = 0.10
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.06
    trailing_stop_pct: float = 0.015
    use_atr_sizing: bool = True
    atr_risk_multiplier: float = 2.0
    var_confidence: float = 0.95
    var_lookback_days: int = 252
    correlation_threshold: float = 0.7
    cooldown_after_loss_minutes: int = 30


# -----------------------------------------------------------
# Execution
# -----------------------------------------------------------

class PaperTradingConfig(BaseModel):
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001
    slippage_model: str = "percentage"
    slippage_pct: float = 0.001


class ExecutionConfig(BaseModel):
    broker: str = "paper"
    order_type: str = "market"
    limit_offset_pct: float = 0.001
    max_slippage_pct: float = 0.005
    retry_attempts: int = 3
    retry_delay_seconds: int = 2
    paper: PaperTradingConfig = Field(default_factory=PaperTradingConfig)


# -----------------------------------------------------------
# Backtesting
# -----------------------------------------------------------

class WalkForwardConfig(BaseModel):
    enabled: bool = True
    train_window_days: int = 252
    test_window_days: int = 63
    step_days: int = 21


class BacktestConfig(BaseModel):
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 100_000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.001
    benchmark: str = "SPY"
    warmup_period: int = 60
    walk_forward: WalkForwardConfig = Field(default_factory=WalkForwardConfig)


# -----------------------------------------------------------
# Dashboard
# -----------------------------------------------------------

class DashboardConfig(BaseModel):
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 5050
    refresh_interval_seconds: int = 5
    theme: str = "dark"


# -----------------------------------------------------------
# Alerts
# -----------------------------------------------------------

class EmailAlertConfig(BaseModel):
    smtp_host: str = ""
    smtp_port: int = 587
    username: str = ""
    password: str = ""
    recipient: str = ""


class DiscordAlertConfig(BaseModel):
    webhook_url: str = ""


class AlertChannelsConfig(BaseModel):
    console: bool = True
    file: bool = True
    email: bool = False
    discord: bool = False


class AlertsConfig(BaseModel):
    enabled: bool = True
    channels: AlertChannelsConfig = Field(default_factory=AlertChannelsConfig)
    email: EmailAlertConfig = Field(default_factory=EmailAlertConfig)
    discord: DiscordAlertConfig = Field(default_factory=DiscordAlertConfig)


# -----------------------------------------------------------
# Logging
# -----------------------------------------------------------

class LoggingConfig(BaseModel):
    console: bool = True
    file: bool = True
    file_path: str = "logs/eagle.log"
    max_file_size_mb: int = 50
    backup_count: int = 5
    structured: bool = True


# -----------------------------------------------------------
# App Config
# -----------------------------------------------------------

class AppConfig(BaseModel):
    name: str = "Eagle Trading Bot"
    version: str = "1.0.0"
    mode: str = "paper"
    log_level: str = "INFO"
    timezone: str = "America/New_York"


# -----------------------------------------------------------
# Root Configuration
# -----------------------------------------------------------

class EagleConfig(BaseModel):
    """Root configuration model for the entire Eagle system."""

    app: AppConfig = Field(default_factory=AppConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(config_dict: dict[str, Any], prefix: str = "EAGLE") -> dict[str, Any]:
    """
    Apply environment variable overrides.
    Format: EAGLE__SECTION__KEY=value (double underscore as separator).
    """
    for key, value in os.environ.items():
        if not key.startswith(f"{prefix}__"):
            continue
        parts = key[len(prefix) + 2:].lower().split("__")
        current = config_dict
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        # Attempt type coercion
        final_key = parts[-1]
        if value.lower() in ("true", "false"):
            current[final_key] = value.lower() == "true"
        elif value.isdigit():
            current[final_key] = int(value)
        else:
            try:
                current[final_key] = float(value)
            except ValueError:
                current[final_key] = value

    return config_dict


def load_config(
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> EagleConfig:
    """
    Load Eagle configuration from YAML file with environment variable overrides.

    Priority: defaults < YAML file < environment variables < explicit overrides

    Args:
        config_path: Path to YAML config file. If None, uses default config.
        overrides: Dictionary of explicit overrides (highest priority).

    Returns:
        Validated EagleConfig instance.
    """
    config_dict: dict[str, Any] = {}

    # Load from YAML file
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    config_dict = yaml_config
    else:
        # Try default locations
        default_paths = [
            Path("config/config.yaml"),
            Path("config/default.yaml"),
            Path("config.yaml"),
        ]
        for path in default_paths:
            if path.exists():
                with open(path) as f:
                    yaml_config = yaml.safe_load(f)
                    if yaml_config:
                        config_dict = yaml_config
                break

    # Apply environment variable overrides
    config_dict = _apply_env_overrides(config_dict)

    # Apply explicit overrides
    if overrides:
        config_dict = _deep_merge(config_dict, overrides)

    return EagleConfig(**config_dict)
