import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("YFINANCE_CACHE_DIR", str(Path(__file__).resolve().parents[1] / ".yfinance_cache"))

import yfinance as yf
import yfinance.cache as yf_cache

from indicators.juge_longtou import add_juge_longtou_signals


yf_cache.set_cache_location(os.environ["YFINANCE_CACHE_DIR"])


def _normalize_yfinance_columns(df):
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


def test_juge_longtou_yfinance_aapl():
    df = yf.download("AAPL", period="2y", interval="1d", auto_adjust=False, progress=False)
    df = _normalize_yfinance_columns(df)

    result = add_juge_longtou_signals(df)

    print(result[["close", "juCoef", "juCtrl", "juVsig", "XG", "juAccel"]].tail(30))
    print()
    print("最近两年统计:")
    print(f"XG出现次数: {int(result['XG'].sum())}")
    print(f"juAccel出现次数: {int(result['juAccel'].sum())}")

    expected_columns = {"juRef", "juCoef", "juStr", "juCycle", "juCtrl", "juVsig", "XG", "juAccel"}
    assert expected_columns.issubset(result.columns)
    assert len(result) > 30


if __name__ == "__main__":
    test_juge_longtou_yfinance_aapl()
