"""
Eagle Indicators: Technical Analysis Engine
=============================================
Comprehensive technical indicator calculations with vectorized
operations for maximum performance. Supports 150+ indicators.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Vectorized technical indicator calculator.

    All methods are static and operate on pandas DataFrames/Series
    for maximum composability and performance.
    """

    # -------------------------------------------------------
    # Trend Indicators
    # -------------------------------------------------------

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def wma(series: pd.Series, period: int) -> pd.Series:
        """Weighted Moving Average."""
        weights = np.arange(1, period + 1, dtype=float)
        return series.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )

    @staticmethod
    def dema(series: pd.Series, period: int) -> pd.Series:
        """Double Exponential Moving Average."""
        ema1 = series.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        return 2 * ema1 - ema2

    @staticmethod
    def tema(series: pd.Series, period: int) -> pd.Series:
        """Triple Exponential Moving Average."""
        ema1 = series.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3

    @staticmethod
    def kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
        """Kaufman Adaptive Moving Average."""
        direction = abs(series - series.shift(period))
        volatility = series.diff().abs().rolling(window=period).sum()

        er = direction / volatility.replace(0, np.nan)
        fast_sc = 2.0 / (fast + 1)
        slow_sc = 2.0 / (slow + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        kama_values = pd.Series(np.nan, index=series.index)
        kama_values.iloc[period - 1] = series.iloc[period - 1]

        for i in range(period, len(series)):
            if pd.notna(kama_values.iloc[i - 1]) and pd.notna(sc.iloc[i]):
                kama_values.iloc[i] = kama_values.iloc[i - 1] + sc.iloc[i] * (
                    series.iloc[i] - kama_values.iloc[i - 1]
                )

        return kama_values

    @staticmethod
    def supertrend(
        df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
    ) -> pd.DataFrame:
        """
        SuperTrend indicator.

        Returns DataFrame with 'supertrend' and 'supertrend_direction' columns.
        """
        hl2 = (df["high"] + df["low"]) / 2
        atr = TechnicalIndicators.atr(df, period)

        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr

        supertrend = pd.Series(np.nan, index=df.index)
        direction = pd.Series(1, index=df.index)

        for i in range(period, len(df)):
            if df["close"].iloc[i] > upper_band.iloc[i - 1]:
                direction.iloc[i] = 1
            elif df["close"].iloc[i] < lower_band.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]

            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]

        result = pd.DataFrame(index=df.index)
        result["supertrend"] = supertrend
        result["supertrend_direction"] = direction
        return result

    # -------------------------------------------------------
    # Momentum Indicators
    # -------------------------------------------------------

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def stochastic_rsi(series: pd.Series, period: int = 14, smooth: int = 3) -> pd.DataFrame:
        """Stochastic RSI."""
        rsi = TechnicalIndicators.rsi(series, period)
        stoch_rsi = (rsi - rsi.rolling(period).min()) / (
            rsi.rolling(period).max() - rsi.rolling(period).min()
        )
        result = pd.DataFrame(index=series.index)
        result["stoch_rsi_k"] = stoch_rsi.rolling(smooth).mean() * 100
        result["stoch_rsi_d"] = result["stoch_rsi_k"].rolling(smooth).mean()
        return result

    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """Moving Average Convergence Divergence."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        result = pd.DataFrame(index=series.index)
        result["macd"] = macd_line
        result["macd_signal"] = signal_line
        result["macd_histogram"] = histogram
        return result

    @staticmethod
    def stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
    ) -> pd.DataFrame:
        """Stochastic Oscillator."""
        lowest_low = df["low"].rolling(window=k_period).min()
        highest_high = df["high"].rolling(window=k_period).max()

        k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()

        result = pd.DataFrame(index=df.index)
        result["stoch_k"] = k
        result["stoch_d"] = d
        return result

    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Williams %R."""
        highest_high = df["high"].rolling(window=period).max()
        lowest_low = df["low"].rolling(window=period).min()
        return -100 * (highest_high - df["close"]) / (highest_high - lowest_low)

    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        tp = (df["high"] + df["low"] + df["close"]) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        return (tp - sma) / (0.015 * mad)

    @staticmethod
    def roc(series: pd.Series, period: int = 12) -> pd.Series:
        """Rate of Change."""
        return ((series - series.shift(period)) / series.shift(period)) * 100

    @staticmethod
    def momentum(series: pd.Series, period: int = 10) -> pd.Series:
        """Price Momentum."""
        return series - series.shift(period)

    @staticmethod
    def tsi(series: pd.Series, long_period: int = 25, short_period: int = 13) -> pd.Series:
        """True Strength Index."""
        diff = series.diff()
        double_smoothed = diff.ewm(span=long_period).mean().ewm(span=short_period).mean()
        double_smoothed_abs = diff.abs().ewm(span=long_period).mean().ewm(span=short_period).mean()
        return 100 * double_smoothed / double_smoothed_abs.replace(0, np.nan)

    # -------------------------------------------------------
    # Volatility Indicators
    # -------------------------------------------------------

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> pd.DataFrame:
        """Bollinger Bands."""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()

        result = pd.DataFrame(index=series.index)
        result["bb_upper"] = sma + std_dev * std
        result["bb_middle"] = sma
        result["bb_lower"] = sma - std_dev * std
        result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result["bb_middle"]
        result["bb_pct"] = (series - result["bb_lower"]) / (
            result["bb_upper"] - result["bb_lower"]
        )
        return result

    @staticmethod
    def keltner_channels(
        df: pd.DataFrame,
        ema_period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0,
    ) -> pd.DataFrame:
        """Keltner Channels."""
        ema = df["close"].ewm(span=ema_period, adjust=False).mean()
        atr = TechnicalIndicators.atr(df, atr_period)

        result = pd.DataFrame(index=df.index)
        result["kc_upper"] = ema + multiplier * atr
        result["kc_middle"] = ema
        result["kc_lower"] = ema - multiplier * atr
        return result

    @staticmethod
    def donchian_channels(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Donchian Channels."""
        result = pd.DataFrame(index=df.index)
        result["dc_upper"] = df["high"].rolling(window=period).max()
        result["dc_lower"] = df["low"].rolling(window=period).min()
        result["dc_middle"] = (result["dc_upper"] + result["dc_lower"]) / 2
        return result

    @staticmethod
    def historical_volatility(series: pd.Series, period: int = 20) -> pd.Series:
        """Annualized historical volatility."""
        log_returns = np.log(series / series.shift(1))
        return log_returns.rolling(window=period).std() * np.sqrt(252)

    @staticmethod
    def natr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Normalized Average True Range."""
        return (TechnicalIndicators.atr(df, period) / df["close"]) * 100

    # -------------------------------------------------------
    # Volume Indicators
    # -------------------------------------------------------

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """On-Balance Volume."""
        direction = np.where(df["close"] > df["close"].shift(1), 1, -1)
        direction = np.where(df["close"] == df["close"].shift(1), 0, direction)
        return (df["volume"] * direction).cumsum()

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """Volume-Weighted Average Price."""
        tp = (df["high"] + df["low"] + df["close"]) / 3
        return (tp * df["volume"]).cumsum() / df["volume"].cumsum()

    @staticmethod
    def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Money Flow Index."""
        tp = (df["high"] + df["low"] + df["close"]) / 3
        raw_mf = tp * df["volume"]

        positive_mf = raw_mf.where(tp > tp.shift(1), 0).rolling(window=period).sum()
        negative_mf = raw_mf.where(tp < tp.shift(1), 0).rolling(window=period).sum()

        mfr = positive_mf / negative_mf.replace(0, np.nan)
        return 100 - (100 / (1 + mfr))

    @staticmethod
    def ad_line(df: pd.DataFrame) -> pd.Series:
        """Accumulation/Distribution Line."""
        clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
            df["high"] - df["low"]
        ).replace(0, np.nan)
        return (clv * df["volume"]).cumsum()

    @staticmethod
    def cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Chaikin Money Flow."""
        clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
            df["high"] - df["low"]
        ).replace(0, np.nan)
        return (clv * df["volume"]).rolling(window=period).sum() / df["volume"].rolling(
            window=period
        ).sum()

    @staticmethod
    def force_index(df: pd.DataFrame, period: int = 13) -> pd.Series:
        """Force Index."""
        fi = df["close"].diff() * df["volume"]
        return fi.ewm(span=period, adjust=False).mean()

    @staticmethod
    def eom(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Ease of Movement."""
        distance = ((df["high"] + df["low"]) / 2) - ((df["high"].shift(1) + df["low"].shift(1)) / 2)
        box_ratio = (df["volume"] / 1e8) / (df["high"] - df["low"]).replace(0, np.nan)
        return (distance / box_ratio).rolling(window=period).mean()

    # -------------------------------------------------------
    # Trend Strength
    # -------------------------------------------------------

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average Directional Index with +DI and -DI."""
        high_diff = df["high"] - df["high"].shift(1)
        low_diff = df["low"].shift(1) - df["low"]

        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

        atr = TechnicalIndicators.atr(df, period)

        plus_di = 100 * plus_dm.ewm(alpha=1 / period).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1 / period).mean() / atr.replace(0, np.nan)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx_val = dx.ewm(alpha=1 / period).mean()

        result = pd.DataFrame(index=df.index)
        result["adx"] = adx_val
        result["plus_di"] = plus_di
        result["minus_di"] = minus_di
        return result

    @staticmethod
    def ichimoku(
        df: pd.DataFrame,
        tenkan: int = 9,
        kijun: int = 26,
        senkou_b: int = 52,
    ) -> pd.DataFrame:
        """Ichimoku Cloud."""
        tenkan_sen = (
            df["high"].rolling(window=tenkan).max() + df["low"].rolling(window=tenkan).min()
        ) / 2
        kijun_sen = (
            df["high"].rolling(window=kijun).max() + df["low"].rolling(window=kijun).min()
        ) / 2
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        senkou_b_val = (
            (
                df["high"].rolling(window=senkou_b).max()
                + df["low"].rolling(window=senkou_b).min()
            )
            / 2
        ).shift(kijun)
        chikou = df["close"].shift(-kijun)

        result = pd.DataFrame(index=df.index)
        result["tenkan_sen"] = tenkan_sen
        result["kijun_sen"] = kijun_sen
        result["senkou_a"] = senkou_a
        result["senkou_b"] = senkou_b_val
        result["chikou"] = chikou
        return result

    # -------------------------------------------------------
    # Statistical / Custom
    # -------------------------------------------------------

    @staticmethod
    def z_score(series: pd.Series, period: int = 20) -> pd.Series:
        """Rolling Z-Score."""
        mean = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        return (series - mean) / std.replace(0, np.nan)

    @staticmethod
    def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
        """
        Hurst exponent for detecting trend vs mean reversion.
        H > 0.5: trending, H < 0.5: mean reverting, H = 0.5: random walk
        """
        lags = range(2, max_lag)
        tau = [np.std(np.subtract(series[lag:].values, series[:-lag].values)) for lag in lags]
        poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
        return poly[0]

    @staticmethod
    def fractal_dimension(series: pd.Series, period: int = 30) -> pd.Series:
        """Fractal Dimension for market regime detection."""
        n = period // 2
        hl_n = (
            series.rolling(window=n).max() - series.rolling(window=n).min()
        )
        hl_2n = (
            series.rolling(window=period).max() - series.rolling(window=period).min()
        )
        n1 = (hl_n + hl_n.shift(n)) / 2
        fd = np.where(hl_2n > 0, np.log(n1 / hl_2n.replace(0, np.nan)) / np.log(2), np.nan)
        return pd.Series(fd, index=series.index)

    @staticmethod
    def log_returns(series: pd.Series) -> pd.Series:
        """Log returns."""
        return np.log(series / series.shift(1))

    @staticmethod
    def pct_returns(series: pd.Series, period: int = 1) -> pd.Series:
        """Percentage returns."""
        return series.pct_change(periods=period)

    @staticmethod
    def rolling_correlation(
        series1: pd.Series, series2: pd.Series, period: int = 20
    ) -> pd.Series:
        """Rolling correlation between two series."""
        return series1.rolling(window=period).corr(series2)

    @staticmethod
    def rolling_beta(
        asset: pd.Series, benchmark: pd.Series, period: int = 60
    ) -> pd.Series:
        """Rolling beta relative to a benchmark."""
        asset_ret = asset.pct_change()
        bench_ret = benchmark.pct_change()
        covariance = asset_ret.rolling(window=period).cov(bench_ret)
        variance = bench_ret.rolling(window=period).var()
        return covariance / variance.replace(0, np.nan)

    @staticmethod
    def parkinson_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Parkinson volatility estimator (uses high-low range)."""
        log_hl = np.log(df["high"] / df["low"]) ** 2
        return np.sqrt(log_hl.rolling(window=period).mean() / (4 * np.log(2))) * np.sqrt(252)

    @staticmethod
    def garman_klass_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Garman-Klass volatility estimator."""
        log_hl = np.log(df["high"] / df["low"]) ** 2
        log_co = np.log(df["close"] / df["open"]) ** 2
        gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return np.sqrt(gk.rolling(window=period).mean() * 252)
