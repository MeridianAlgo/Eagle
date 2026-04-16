"""
Eagle Indicators: Real-Time Calculator
========================================
Computes technical indicators from a rolling list of closed candles.
Uses the ``ta`` library for all indicator math.

Returns an ``Indicators`` dataclass with all values needed by strategies.
Returns ``None`` if there is insufficient history (< 50 candles).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from eagle.data.binance_ws import Candle

logger = logging.getLogger(__name__)

MIN_CANDLES = 50  # minimum history required before any calculation


@dataclass
class Indicators:
    """Snapshot of all indicator values for the current moment."""

    # Price
    price: float
    price_change_1m: float   # % change vs 1 candle ago
    price_change_5m: float   # % change vs 5 candles ago
    price_change_15m: float  # % change vs 15 candles ago

    # RSI
    rsi: float               # 0–100; <30 oversold, >70 overbought

    # MACD
    macd: float              # MACD line
    macd_signal: float       # Signal line
    macd_hist: float         # Histogram (macd - signal)
    macd_hist_prev: float    # Previous histogram (for trend detection)

    # Bollinger Bands
    bb_upper: float
    bb_mid: float
    bb_lower: float
    bb_pct: float            # %B: 0=lower band, 1=upper band
    bb_width: float          # Band width as % of mid

    # EMAs
    ema_9: float
    ema_21: float
    ema_50: float

    # Volume
    volume_current: float
    volume_avg_20: float
    volume_ratio: float      # current / avg; >1.5 = spike

    # ATR (volatility)
    atr: float
    atr_pct: float           # ATR as % of price

    # Trend helpers
    @property
    def is_ema_bullish(self) -> bool:
        """EMA 9 > EMA 21 > EMA 50 — bullish alignment."""
        return self.ema_9 > self.ema_21 > self.ema_50

    @property
    def is_ema_bearish(self) -> bool:
        return self.ema_9 < self.ema_21 < self.ema_50

    @property
    def macd_cross_up(self) -> bool:
        """MACD histogram just crossed above zero."""
        return self.macd_hist > 0 > self.macd_hist_prev

    @property
    def macd_cross_down(self) -> bool:
        return self.macd_hist < 0 < self.macd_hist_prev

    @property
    def rsi_oversold(self) -> bool:
        return self.rsi < 30

    @property
    def rsi_overbought(self) -> bool:
        return self.rsi > 70


def compute(candles: list[Candle], current_price: float) -> Indicators | None:
    """
    Compute all indicators from the given closed-candle history.

    Args:
        candles:       List of closed Candle objects (oldest first).
        current_price: Latest live price (may differ from last close).

    Returns:
        An ``Indicators`` snapshot, or ``None`` if there is not enough data.
    """
    if len(candles) < MIN_CANDLES:
        return None

    # Build DataFrame
    df = pd.DataFrame(
        {
            "open":   [c.open for c in candles],
            "high":   [c.high for c in candles],
            "low":    [c.low for c in candles],
            "close":  [c.close for c in candles],
            "volume": [c.volume for c in candles],
        }
    )

    closes = df["close"]
    highs = df["high"]
    lows = df["low"]
    volumes = df["volume"]

    # ---- RSI(14) ----------------------------------------------------------
    rsi_val = _rsi(closes, period=14)

    # ---- MACD(12, 26, 9) --------------------------------------------------
    ema12 = closes.ewm(span=12, adjust=False).mean()
    ema26 = closes.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line

    macd_val = float(macd_line.iloc[-1])
    signal_val = float(signal_line.iloc[-1])
    hist_val = float(hist.iloc[-1])
    hist_prev = float(hist.iloc[-2]) if len(hist) >= 2 else 0.0

    # ---- Bollinger Bands(20, 2) -------------------------------------------
    sma20 = closes.rolling(20).mean()
    std20 = closes.rolling(20).std()
    bb_mid_val = float(sma20.iloc[-1])
    bb_upper_val = float(sma20.iloc[-1] + 2 * std20.iloc[-1])
    bb_lower_val = float(sma20.iloc[-1] - 2 * std20.iloc[-1])
    bb_range = bb_upper_val - bb_lower_val
    bb_pct_val = (current_price - bb_lower_val) / bb_range if bb_range > 0 else 0.5
    bb_width_val = bb_range / bb_mid_val if bb_mid_val > 0 else 0.0

    # ---- EMAs -------------------------------------------------------------
    ema9_val = float(closes.ewm(span=9, adjust=False).mean().iloc[-1])
    ema21_val = float(closes.ewm(span=21, adjust=False).mean().iloc[-1])
    ema50_val = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])

    # ---- Volume -----------------------------------------------------------
    vol_cur = float(volumes.iloc[-1])
    vol_avg20 = float(volumes.rolling(20).mean().iloc[-1])
    vol_ratio = vol_cur / vol_avg20 if vol_avg20 > 0 else 1.0

    # ---- ATR(14) ----------------------------------------------------------
    atr_val = _atr(highs, lows, closes, period=14)
    atr_pct = atr_val / current_price if current_price > 0 else 0.0

    # ---- Price change -----------------------------------------------------
    def pct_change(n: int) -> float:
        if len(closes) > n:
            prev = float(closes.iloc[-(n + 1)])
            return (current_price - prev) / prev * 100 if prev > 0 else 0.0
        return 0.0

    return Indicators(
        price=current_price,
        price_change_1m=pct_change(1),
        price_change_5m=pct_change(5),
        price_change_15m=pct_change(15),
        rsi=rsi_val,
        macd=macd_val,
        macd_signal=signal_val,
        macd_hist=hist_val,
        macd_hist_prev=hist_prev,
        bb_upper=bb_upper_val,
        bb_mid=bb_mid_val,
        bb_lower=bb_lower_val,
        bb_pct=bb_pct_val,
        bb_width=bb_width_val,
        ema_9=ema9_val,
        ema_21=ema21_val,
        ema_50=ema50_val,
        volume_current=vol_cur,
        volume_avg_20=vol_avg20,
        volume_ratio=vol_ratio,
        atr=atr_val,
        atr_pct=atr_pct,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _atr(
    highs: pd.Series,
    lows: pd.Series,
    closes: pd.Series,
    period: int = 14,
) -> float:
    prev_close = closes.shift(1)
    tr = pd.concat(
        [
            highs - lows,
            (highs - prev_close).abs(),
            (lows - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return float(tr.ewm(com=period - 1, adjust=False).mean().iloc[-1])
