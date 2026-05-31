"""Juge longtou indicator translated from the Pine Script version."""

from __future__ import annotations

import numpy as np
import pandas as pd


WINNER_LEN = 60


def safeDiv(numerator, denominator, fallback=np.nan):
    """Pine safeDiv equivalent, vectorized for pandas/numpy inputs."""
    num = pd.Series(numerator) if not isinstance(numerator, pd.Series) else numerator
    den = pd.Series(denominator, index=num.index) if not isinstance(denominator, pd.Series) else denominator
    result = num / den
    return result.where((den != 0) & den.notna(), fallback)


def intPart(value):
    """Pine int(value) equivalent: truncate toward zero while preserving NaN."""
    series = value if isinstance(value, pd.Series) else pd.Series(value)
    return pd.Series(np.trunc(series.to_numpy(dtype=float)), index=series.index)


def winner(price, low, high, length=WINNER_LEN):
    """Approximate Tongdaxin WINNER using Pine script's rolling high/low model."""
    price_series = price if isinstance(price, pd.Series) else pd.Series(price, index=low.index)
    low_bound = low.rolling(length, min_periods=1).min()
    high_bound = high.rolling(length, min_periods=1).max()
    raw = safeDiv(price_series - low_bound, high_bound - low_bound, 0.5)
    return raw.clip(lower=0.0, upper=1.0)


def countSince(condition, length):
    """Pine countSince equivalent: sum true values over a dynamic lookback."""
    cond = condition.fillna(False).astype(int) if isinstance(condition, pd.Series) else pd.Series(condition).fillna(False).astype(int)

    if isinstance(length, pd.Series):
        cond_values = cond.to_numpy()
        length_values = length.reindex(cond.index).fillna(0).to_numpy()
        out = np.zeros(len(cond), dtype=float)
        for i, raw_len in enumerate(length_values):
            lookback = max(int(raw_len), 1)
            start = max(0, i - lookback + 1)
            out[i] = cond_values[start : i + 1].sum()
        return pd.Series(out, index=cond.index)

    lookback = max(int(0 if pd.isna(length) else length), 1)
    return cond.rolling(lookback, min_periods=1).sum()


def dma(source, alpha):
    """Pine dma equivalent with alpha bounded to [0, 1]."""
    source = source.astype(float)
    bounded_alpha = alpha.reindex(source.index).fillna(0.0).clip(lower=0.0, upper=1.0)
    out = np.full(len(source), np.nan, dtype=float)
    src_values = source.to_numpy()
    alpha_values = bounded_alpha.to_numpy()

    for i, src in enumerate(src_values):
        if i == 0 or np.isnan(out[i - 1]):
            out[i] = src
        else:
            out[i] = alpha_values[i] * src + (1.0 - alpha_values[i]) * out[i - 1]

    return pd.Series(out, index=source.index)


def _ema(source, length):
    return source.ewm(span=length, adjust=False, min_periods=1).mean()


def _barssince(condition):
    cond = condition.fillna(False).astype(bool)
    out = np.full(len(cond), np.nan, dtype=float)
    last_true = None
    for i, is_true in enumerate(cond.to_numpy()):
        if is_true:
            last_true = i
            out[i] = 0.0
        elif last_true is not None:
            out[i] = float(i - last_true)
    return pd.Series(out, index=cond.index)


def _crossover(source, level):
    return (source > level) & (source.shift(1) <= level)


def add_juge_longtou_signals(df):
    """Add Juge longtou indicator columns to an OHLCV DataFrame."""
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    out = df.copy()
    open_ = out["open"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    close = out["close"].astype(float)
    volume = out["volume"].astype(float)

    ZLCM = _ema(winner(close, low, high) * 70, 3)
    SHCM = _ema((winner(close * 1.1, low, high) - winner(close * 0.9, low, high)) * 80, 3)
    sumCM = ZLCM + SHCM
    ZSHTL = safeDiv(SHCM, sumCM, 0.0) * 100
    ZZLKP = safeDiv(ZLCM, sumCM, 0.0) * 100
    ZCMZL = sumCM.rolling(13, min_periods=1).mean()

    _ = _barssince((ZSHTL < 90) & (ZSHTL.shift(1) > 90))
    ZSHJJ = _ema(ZSHTL, 89)
    ZZLJJ = _ema(ZZLKP, 89)
    ZJLRQD = intPart(ZZLKP - ZZLJJ)
    _ = _ema(ZSHTL, 8)
    _ = np.where((ZZLKP - ZZLKP.shift(1)) > (ZSHTL - ZSHTL.shift(1)), 1, 0)

    out["juRef"] = intPart(ZSHTL)
    out["juCoef"] = intPart(ZZLKP)
    out["juStr"] = ZJLRQD
    out["juCycle"] = intPart(ZCMZL) - 50

    K1 = low.rolling(5, min_periods=1).min()
    K2 = high.rolling(5, min_periods=1).max()
    K3 = _ema(safeDiv(close - K1, K2 - K1, 0.0) * 100, 4)
    K5 = (high + low) / 2

    volSum5 = volume.rolling(5, min_periods=1).sum()
    volSum13 = volume.rolling(13, min_periods=1).sum()
    volSum34 = volume.rolling(34, min_periods=1).sum()
    volSum75 = volume.rolling(75, min_periods=1).sum()

    K6 = dma(K5, safeDiv(volume, volSum5, 0.0))
    K7 = dma(K5, safeDiv(volume, volSum13, 0.0))
    K8 = dma(K5, safeDiv(volume, volSum34, 0.0))
    K9 = dma(K5, safeDiv(volume, volSum75, 0.0))

    KA = _ema(winner(0.9 * close, low, high), 5)
    _ = 1 - _ema(winner(1.2 * close, low, high), 5)
    _ = _ema(winner(close, low, high), 5)

    KDY = safeDiv(100 * (close - K6), K6, 0.0)
    KE = safeDiv(100 * (close - K7), K7, 0.0)
    KF = safeDiv(100 * (np.minimum(close, open_) - K8), K8, 0.0)

    KG = _barssince((K8 > K9) & (K8.shift(1) <= K9.shift(1)))
    KGCountLen = KG.fillna(0).clip(lower=1)

    KH = (
        (
            countSince((winner(close, low, high) < 0.11) & (_ema(winner(close, low, high), 5) < 0.15), 2) > 0
        )
        | (((1 - winner(1.2 * close, low, high)) >= 0.8) & (winner(close, low, high) < 0.05))
    ) & (countSince(KE < -16, 2) > 0) & (countSince(KF < -20, 2) > 0)

    K12 = (
        (countSince(KDY < -10, 2) > 0)
        & (countSince(KE < -15, 2) > 0)
        & (countSince(KF < -15, 2) > 0)
        & (countSince(KA > 0.8, KGCountLen) == 0)
    )

    K13 = (
        (countSince(KH | K12, 2) > 0)
        & (K3 > K3.shift(1))
        & ((1 - winner(1.15 * close, low, high)) * 100 > 80)
    )

    K14 = (K13 & (countSince(K13, 3) <= 1)).astype(int)
    ZVF = safeDiv(100 * (close - close.shift(1)), close.shift(1), 0.0)

    juCtrl = (out["juCoef"] > 10) & (out["juStr"] > -15)
    juVsig = _crossover(out["juCoef"], 50)
    buyLookback = 13
    buyRiseMin = 2.5
    buySetup = (K14 == 1) | (K14.shift(1) == 1)
    XGBase = countSince(buySetup, buyLookback) > 0
    XG = XGBase & (ZVF > buyRiseMin) & juCtrl
    juAccel = ZSHTL >= 90

    out["juCtrl"] = juCtrl
    out["juVsig"] = juVsig
    out["XG"] = XG
    out["juAccel"] = juAccel

    return out
