"""Download DBS daily prices for the AI6124 project.

Install: python -m pip install yfinance pandas
Run:     python download_dbs.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:
    raise SystemExit("请先安装依赖：python -m pip install yfinance pandas") from exc


TICKER = "D05.SI"
START = "2020-01-01"
# Yahoo Finance's end date is exclusive, so this includes December 2025.
END = "2026-01-01"
OUTPUT_DIR = Path(__file__).resolve().parent / "data"


def download_dbs():
    """Return raw Yahoo OHLCV, adjusted close, and corporate-action columns."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(OUTPUT_DIR / ".yfinance_cache"))
    df = yf.Ticker(TICKER).history(
        start=START,
        end=END,
        interval="1d",
        auto_adjust=False,
        actions=True,
        raise_errors=True,
        timeout=30,
    )
    if df.empty:
        raise RuntimeError("没有下载到 DBS 数据，请检查网络或稍后重试。")

    required = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"行情缺少必要字段：{missing_columns}")

    # Remove the timezone without changing the local trading date.
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError("发现重复交易日期，请检查数据来源。")
    if not ((df.index >= pd.Timestamp(START)) & (df.index < pd.Timestamp(END))).all():
        raise ValueError("返回数据超出指定日期范围。")
    if not np.isfinite(df[required].to_numpy(dtype=float)).all():
        raise ValueError("价格或成交量存在缺失/非有限值；未自动填充或删除。")
    if (df[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any().any():
        raise ValueError("发现非正价格。")
    if (df["Volume"] < 0).any():
        raise ValueError("发现负成交量。")
    if ((df["High"] < df[["Open", "Close", "Low"]].max(axis=1)) |
        (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))).any():
        raise ValueError("发现不一致的 OHLC 价格范围。")

    # Preserve all returned columns, including dividends and stock splits.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "DBS_D05_SI_daily_2020_2025.csv"
    df.to_csv(csv_path, date_format="%Y-%m-%d", encoding="utf-8")

    metadata = {
        "ticker": TICKER,
        "source": "Yahoo Finance via yfinance",
        "yfinance_version": yf.__version__,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_inclusive": START,
        "end_exclusive": END,
        "interval": "1d",
        "auto_adjust": False,
        "rows": len(df),
        "first_date": str(df.index.min().date()),
        "last_date": str(df.index.max().date()),
        "columns": list(df.columns),
        "missing_values": df.isna().sum().astype(int).to_dict(),
    }
    csv_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"已保存：{csv_path}")
    print(f"交易日数：{len(df)} | {metadata['first_date']} 至 {metadata['last_date']}")
    print("\n各年交易日数：")
    print(df.groupby(df.index.year).size().to_string())
    print("\n前五行：")
    print(df.head().to_string())
    print("\nClose 是未自动复权的收盘价；Adj Close 是 Yahoo 提供的复权价。")
    print("保留休市日缺口，不填充为虚构的交易日，也不在下载阶段归一化。")
    return df


if __name__ == "__main__":
    try:
        download_dbs()
    except Exception as exc:
        raise SystemExit(f"下载或数据检查失败：{exc}") from exc
