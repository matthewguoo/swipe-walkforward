"""
Technical indicators for trigger detection
"""
import pandas as pd
import numpy as np


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def pct_from_high(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculate percentage drop from rolling high"""
    rolling_high = series.rolling(window=period).max()
    return ((rolling_high - series) / rolling_high) * 100


def pct_from_low(series: pd.Series, period: int = 20) -> pd.Series:
    """Calculate percentage gain from rolling low"""
    rolling_low = series.rolling(window=period).min()
    return ((series - rolling_low) / rolling_low) * 100


def sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=period).mean()


def ema(series: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def volume_spike(volume: pd.Series, period: int = 20, multiplier: float = 2.0) -> pd.Series:
    """Detect volume spikes (volume > multiplier * average)"""
    avg_vol = volume.rolling(window=period).mean()
    return volume / avg_vol


INDICATORS = {
    'rsi': rsi,
    'pct_from_high': pct_from_high,
    'pct_from_low': pct_from_low,
    'sma': sma,
    'ema': ema,
    'volume_spike': volume_spike,
}
