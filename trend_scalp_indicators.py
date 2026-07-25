"""
Indicators for the trend-following scalping strategy:
  - EMA (200-period trend filter)
  - VWAP (Volume Weighted Average Price, session-based — resets each day)
  - CVD (Cumulative Volume Delta) — approximated from candle direction,
    since true tick-by-tick aggressor-side data isn't available from
    Delta's basic candle API. This is a standard, widely-used
    approximation: up-candles add their volume, down-candles subtract it.
"""
import pandas as pd
import numpy as np


def compute_ema(df, period=200, price_col="close"):
    df = df.copy()
    df[f"ema_{period}"] = df[price_col].ewm(span=period, adjust=False).mean()
    return df


def compute_vwap(df):
    """
    Session VWAP — resets at the start of each calendar day (UTC).
    VWAP = cumulative(price * volume) / cumulative(volume), within each day.
    """
    df = df.copy()
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    day = df["timestamp"].dt.date

    pv = typical_price * df["volume"]
    df["vwap"] = pv.groupby(day).cumsum() / df["volume"].groupby(day).cumsum()
    return df


def compute_cvd(df):
    """
    Approximate Cumulative Volume Delta:
    up-candle (close > open) -> +volume treated as buy pressure
    down-candle (close < open) -> -volume treated as sell pressure
    doji (close == open) -> 0
    Cumulative sum gives a running "buy vs sell pressure" line.

    NOTE: this is an approximation (candle-direction based), not true
    aggressor-side tick data. Good enough as a directional-momentum proxy,
    but treat it as a rule-of-thumb signal, not exact order-flow data.
    """
    df = df.copy()
    direction = np.sign(df["close"] - df["open"])
    delta = direction * df["volume"]
    df["cvd"] = delta.cumsum()
    return df


def compute_rsi(df, period=14):
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def compute_atr(df, period=14):
    df = df.copy()
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/period, adjust=False).mean()
    return df


def cvd_rising(df, idx, lookback=10):
    """True if CVD has a positive slope over the last `lookback` candles."""
    if idx < lookback:
        return False
    return df["cvd"].iloc[idx] > df["cvd"].iloc[idx - lookback]


def cvd_falling(df, idx, lookback=10):
    if idx < lookback:
        return False
    return df["cvd"].iloc[idx] < df["cvd"].iloc[idx - lookback]
