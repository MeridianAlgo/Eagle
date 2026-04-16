"""
Eagle Features: Feature Engineering Pipeline
==============================================
Transforms raw OHLCV data into rich feature matrices for ML models.
Computes technical indicators, custom features, market regime,
and cross-asset correlation features.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from eagle.core.config import EagleConfig
from eagle.indicators.technical import TechnicalIndicators as TI

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Central feature engineering pipeline.

    Takes raw OHLCV data and produces a feature matrix suitable
    for ML model consumption. Features are organized into categories:

    1. Technical Indicators (configurable from YAML)
    2. Price Action Features
    3. Volume Profile Features
    4. Market Regime Features
    5. Cross-Asset Correlation Features
    6. Statistical Features
    """

    def __init__(self, config: EagleConfig) -> None:
        self.config = config
        self._feature_names: list[str] = []

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for the given market data.

        Args:
            df: DataFrame with columns [open, high, low, close, volume, symbol].

        Returns:
            DataFrame with original data plus all computed features.
        """
        if df.empty:
            return df

        # Process each symbol separately if multi-symbol
        if "symbol" in df.columns and df["symbol"].nunique() > 1:
            results = []
            for symbol in df["symbol"].unique():
                symbol_df = df[df["symbol"] == symbol].copy()
                featured = self._compute_single(symbol_df)
                results.append(featured)
            result = pd.concat(results, axis=0)
        else:
            result = self._compute_single(df.copy())

        self._feature_names = [
            c for c in result.columns
            if c not in ["open", "high", "low", "close", "volume", "symbol", "timeframe", "dividends", "stock_splits"]
        ]

        logger.debug(f"Computed {len(self._feature_names)} features")
        return result

    def _compute_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for a single symbol."""
        cfg = self.config.features

        # Technical Indicators
        df = self._add_trend_indicators(df, cfg)
        df = self._add_momentum_indicators(df, cfg)
        df = self._add_volatility_indicators(df, cfg)
        df = self._add_volume_indicators(df, cfg)

        # Custom Features
        if cfg.custom_features.price_momentum:
            df = self._add_price_action_features(df)
        if cfg.custom_features.volume_profile:
            df = self._add_volume_profile_features(df)
        if cfg.custom_features.market_regime:
            df = self._add_market_regime_features(df)

        # Statistical Features
        df = self._add_statistical_features(df)

        # Target variable for training
        df = self._add_target(df)

        return df

    def _add_trend_indicators(self, df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        """Add trend-following indicators."""
        ti = cfg.technical_indicators

        # Simple Moving Averages
        for period in ti.sma_periods:
            df[f"sma_{period}"] = TI.sma(df["close"], period)
            df[f"close_sma_{period}_ratio"] = df["close"] / df[f"sma_{period}"]

        # Exponential Moving Averages
        for period in ti.ema_periods:
            df[f"ema_{period}"] = TI.ema(df["close"], period)
            df[f"close_ema_{period}_ratio"] = df["close"] / df[f"ema_{period}"]

        # MACD
        macd_df = TI.macd(df["close"], ti.macd.fast, ti.macd.slow, ti.macd.signal)
        df = pd.concat([df, macd_df], axis=1)

        # Moving Average Crossovers
        if len(ti.sma_periods) >= 2:
            short_sma = f"sma_{ti.sma_periods[0]}"
            long_sma = f"sma_{ti.sma_periods[-1]}"
            df["ma_crossover"] = (df[short_sma] > df[long_sma]).astype(int)
            df["ma_spread"] = (df[short_sma] - df[long_sma]) / df[long_sma]

        # ADX
        if ti.adx_period:
            adx_df = TI.adx(df, ti.adx_period)
            df = pd.concat([df, adx_df], axis=1)

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        """Add momentum indicators."""
        ti = cfg.technical_indicators

        if ti.momentum_indicators:
            # RSI
            df["rsi"] = TI.rsi(df["close"], ti.rsi_period)
            df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
            df["rsi_overbought"] = (df["rsi"] > 70).astype(int)

            # Stochastic
            stoch_df = TI.stochastic(df, ti.stochastic.k_period, ti.stochastic.d_period)
            df = pd.concat([df, stoch_df], axis=1)

            # Stochastic RSI
            stoch_rsi = TI.stochastic_rsi(df["close"], ti.rsi_period)
            df = pd.concat([df, stoch_rsi], axis=1)

            # Williams %R
            df["williams_r"] = TI.williams_r(df, 14)

            # CCI
            df["cci"] = TI.cci(df, 20)

            # Rate of Change
            df["roc_12"] = TI.roc(df["close"], 12)
            df["roc_6"] = TI.roc(df["close"], 6)

            # TSI
            df["tsi"] = TI.tsi(df["close"])

            # Momentum
            df["momentum_10"] = TI.momentum(df["close"], 10)

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        """Add volatility indicators."""
        ti = cfg.technical_indicators

        if ti.volatility_indicators:
            # ATR
            df["atr"] = TI.atr(df, ti.atr_period)
            df["natr"] = TI.natr(df, ti.atr_period)

            # Bollinger Bands
            bb = TI.bollinger_bands(df["close"], ti.bollinger.period, ti.bollinger.std_dev)
            df = pd.concat([df, bb], axis=1)

            # Keltner Channels
            kc = TI.keltner_channels(df)
            df = pd.concat([df, kc], axis=1)

            # Historical and Parkinson Volatility
            df["hist_vol_20"] = TI.historical_volatility(df["close"], 20)
            df["parkinson_vol"] = TI.parkinson_volatility(df, 20)

            # Squeeze Detection (BB inside KC)
            df["squeeze"] = (
                (df["bb_lower"] > df["kc_lower"]) & (df["bb_upper"] < df["kc_upper"])
            ).astype(int)

        return df

    def _add_volume_indicators(self, df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
        """Add volume indicators."""
        ti = cfg.technical_indicators

        if ti.volume_indicators and "volume" in df.columns:
            # OBV
            df["obv"] = TI.obv(df)
            df["obv_sma_20"] = TI.sma(df["obv"], 20)

            # VWAP
            df["vwap"] = TI.vwap(df)

            # MFI
            df["mfi"] = TI.mfi(df, 14)

            # Chaikin Money Flow
            df["cmf"] = TI.cmf(df, 20)

            # Force Index
            df["force_index"] = TI.force_index(df, 13)

            # Volume SMA ratio
            vol_sma = TI.sma(df["volume"], 20)
            df["volume_ratio"] = df["volume"] / vol_sma.replace(0, np.nan)

            # Volume trend
            df["volume_trend"] = TI.sma(df["volume"], 5) / TI.sma(df["volume"], 20)

        return df

    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action and candlestick features."""
        # Returns at various horizons
        for period in [1, 3, 5, 10, 20]:
            df[f"return_{period}d"] = TI.pct_returns(df["close"], period)

        # Log returns
        df["log_return"] = TI.log_returns(df["close"])

        # Candle body and wick analysis
        df["body"] = df["close"] - df["open"]
        df["body_pct"] = df["body"] / df["open"]
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["range"] = df["high"] - df["low"]
        df["body_range_ratio"] = df["body"].abs() / df["range"].replace(0, np.nan)

        # Gap features
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        df["gap_filled"] = (
            ((df["gap"] > 0) & (df["low"] <= df["close"].shift(1)))
            | ((df["gap"] < 0) & (df["high"] >= df["close"].shift(1)))
        ).astype(int)

        # Distance from high/low
        df["dist_from_high_20"] = (df["close"] - df["high"].rolling(20).max()) / df["high"].rolling(20).max()
        df["dist_from_low_20"] = (df["close"] - df["low"].rolling(20).min()) / df["low"].rolling(20).min()

        # Higher highs / Lower lows
        df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
        df["lower_low"] = (df["low"] < df["low"].shift(1)).astype(int)

        return df

    def _add_volume_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume profile features."""
        if "volume" not in df.columns:
            return df

        # Volume-weighted close
        df["vwc_20"] = (
            (df["close"] * df["volume"]).rolling(20).sum()
            / df["volume"].rolling(20).sum()
        )

        # Relative volume
        df["rvol"] = df["volume"] / df["volume"].rolling(20).mean()

        # Volume momentum
        df["vol_momentum"] = df["volume"].pct_change(5)

        # Accumulation/Distribution
        df["ad_line"] = TI.ad_line(df)

        return df

    def _add_market_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection features."""
        close = df["close"]

        # Trend regime via moving average alignment
        if "sma_20" in df.columns and "sma_50" in df.columns:
            df["trend_regime"] = np.where(
                (close > df["sma_20"]) & (df["sma_20"] > df["sma_50"]),
                1,  # Uptrend
                np.where(
                    (close < df["sma_20"]) & (df["sma_20"] < df["sma_50"]),
                    -1,  # Downtrend
                    0,  # Sideways
                ),
            )

        # Volatility regime
        vol = TI.historical_volatility(close, 20)
        vol_median = vol.rolling(60).median()
        df["vol_regime"] = np.where(vol > vol_median, 1, 0)  # 1 = high vol

        # Mean reversion score
        df["mr_score"] = TI.z_score(close, 20)

        # Fractal dimension
        df["fractal_dim"] = TI.fractal_dimension(close, 30)

        return df

    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        returns = df["close"].pct_change()

        # Rolling statistics
        for period in [10, 20]:
            df[f"return_mean_{period}"] = returns.rolling(period).mean()
            df[f"return_std_{period}"] = returns.rolling(period).std()
            df[f"return_skew_{period}"] = returns.rolling(period).skew()
            df[f"return_kurt_{period}"] = returns.rolling(period).kurt()

        # Autocorrelation
        df["autocorr_1"] = returns.rolling(20).apply(lambda x: x.autocorr(lag=1), raw=False)
        df["autocorr_5"] = returns.rolling(20).apply(lambda x: x.autocorr(lag=5), raw=False)

        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target variable for training (future return direction)."""
        # Binary classification: 1 = up, 0 = down
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

        # Regression target: forward return
        df["target_return"] = df["close"].pct_change().shift(-1)

        return df

    @property
    def feature_names(self) -> list[str]:
        """Return list of computed feature names."""
        return list(self._feature_names)

    def get_training_data(
        self,
        df: pd.DataFrame,
        target_col: str = "target",
        dropna: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Extract feature matrix X and target vector y for training.

        Returns:
            Tuple of (features_df, target_series).
        """
        exclude = {
            "open", "high", "low", "close", "volume", "symbol", "timeframe",
            "target", "target_return", "dividends", "stock_splits",
        }

        feature_cols = [c for c in df.columns if c not in exclude]
        X = df[feature_cols]
        y = df[target_col]

        if dropna:
            mask = X.notna().all(axis=1) & y.notna()
            X = X[mask]
            y = y[mask]

        return X, y
