"""Batch yfinance backtest for crypto assets (BTC, ETH, BNB, LTC) using Juge longtou indicator."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("YFINANCE_CACHE_DIR", str(ROOT / ".yfinance_cache"))

try:
    import yfinance.cache as yf_cache

    yf_cache.set_cache_location(os.environ["YFINANCE_CACHE_DIR"])
except Exception:
    pass

from indicators.juge_longtou import add_juge_longtou_signals


PERIOD = "1y"  # 2025年到现在约1年
MODE_1 = "juAccel_only"
MODE_2 = "XG_only"
MODE_3 = "XG_or_juAccel"
MODE_4 = "strong_signal"
MODES = [MODE_1, MODE_2, MODE_3, MODE_4]
HOLD_DAYS = 20  # 固定20天持仓
TICKERS = [
    "BTC-USD",
    "ETH-USD",
    "BNB-USD",
    "LTC-USD",
]
RESULT_PATH = ROOT / "results" / "crypto_backtest_20days.csv"


def normalize_yfinance_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    return df[["open", "high", "low", "close", "volume"]].dropna()


def download_ohlcv(ticker, period=PERIOD):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)
    df = normalize_yfinance_columns(df)
    if df.empty:
        raise RuntimeError(f"No yfinance data downloaded for {ticker} period={period}.")
    return df


def build_signal(df, mode):
    if mode == MODE_1:
        return df["juAccel"].fillna(False)
    if mode == MODE_2:
        return df["XG"].fillna(False)
    if mode == MODE_3:
        return df["XG"].fillna(False) | df["juAccel"].fillna(False)
    if mode == MODE_4:
        return (
            (df["XG"].fillna(False) | df["juAccel"].fillna(False))
            & (df["juCoef"] > 50)
            & (df["juCtrl"] == True)
        )
    raise ValueError(f"Unknown mode: {mode}")


def run_backtest(df, signal, hold_days):
    trades = []
    i = 0

    while i < len(df):
        if bool(signal.iloc[i]):
            sell_i = i + hold_days
            if sell_i >= len(df):
                break

            buy_price = float(df["close"].iloc[i])
            sell_price = float(df["close"].iloc[sell_i])
            return_pct = (sell_price / buy_price - 1.0) * 100
            trades.append(return_pct)

            i = sell_i + 1
            continue

        i += 1

    return pd.Series(trades, dtype=float)


def summarize_returns(returns):
    if returns.empty:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "total_return": 0.0,
        }

    return {
        "trades": int(len(returns)),
        "win_rate": float((returns > 0).mean() * 100),
        "avg_return": float(returns.mean()),
        "best_trade": float(returns.max()),
        "worst_trade": float(returns.min()),
        "total_return": float(((returns / 100 + 1).prod() - 1) * 100),
    }


def backtest_ticker(ticker):
    raw_df = download_ohlcv(ticker)
    df = add_juge_longtou_signals(raw_df)
    rows = []

    for mode in MODES:
        signal = build_signal(df, mode)
        signal_count = int(signal.sum())

        returns = run_backtest(df, signal, HOLD_DAYS)
        stats = summarize_returns(returns)
        rows.append(
            {
                "ticker": ticker,
                "mode": mode,
                "hold_days": HOLD_DAYS,
                "signal_count": signal_count,
                **stats,
            }
        )

    return rows


def format_summary_for_print(summary):
    display = summary.copy()
    for col in ["win_rate", "avg_return", "best_trade", "worst_trade", "total_return"]:
        display[col] = display[col].map("{:.2f}%".format)
    return display


def main():
    tickers = sys.argv[1:] if len(sys.argv) > 1 else TICKERS
    rows = []

    print("=" * 80)
    print("加密货币龙头指标回测 (Crypto Backtest with Juge Longtou Indicator)")
    print("=" * 80)
    print("参数配置:")
    print(f"  周期: {PERIOD}")
    print(f"  持仓天数: {HOLD_DAYS}")
    print(f"  信号模式: {MODES}")
    print(f"  测试资产: {tickers}")
    print()

    for ticker in tickers:
        try:
            print(f"正在测试 {ticker}...")
            rows.extend(backtest_ticker(ticker))
        except Exception as exc:
            print(f"{ticker}: 跳过, 原因: {exc}")

    if not rows:
        raise RuntimeError("没有生成任何回测结果")

    summary = pd.DataFrame(rows)
    summary = summary[
        [
            "ticker",
            "mode",
            "hold_days",
            "signal_count",
            "trades",
            "win_rate",
            "avg_return",
            "best_trade",
            "worst_trade",
            "total_return",
        ]
    ].sort_values(
        by=["win_rate", "avg_return", "trades"],
        ascending=[False, False, False],
    )

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")

    print()
    print("=" * 80)
    print("回测结果汇总:")
    print("=" * 80)
    print(format_summary_for_print(summary).to_string(index=False))
    print()
    print(f"✓ 结果已保存: {RESULT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
