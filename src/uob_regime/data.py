"""Download immutable research snapshots and validate daily OHLCV data."""

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def validate_prices(frame):
    required = ["Open", "High", "Low", "Close", "Volume"]
    if frame.empty or not set(required).issubset(frame.columns):
        raise ValueError("A nonempty OHLCV series is required")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Dates must be unique and increasing")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("OHLCV contains missing or non-finite values; no filling is applied")
    if (frame[["Open", "High", "Low", "Close"]] <= 0).any().any():
        raise ValueError("Prices must be positive")
    if (frame["Volume"] < 0).any():
        raise ValueError("Volume cannot be negative")
    if ((frame.High < frame[["Open", "Low", "Close"]].max(axis=1)) |
            (frame.Low > frame[["Open", "High", "Close"]].min(axis=1))).any():
        raise ValueError("Inconsistent OHLC ranges")
    if "Adj Close" in frame:
        adjusted = frame["Adj Close"].to_numpy(dtype=float)
        if not np.isfinite(adjusted).all() or (adjusted <= 0).any():
            raise ValueError("Adjusted close must be finite and positive")
    return {
        "rows": len(frame), "first_date": str(frame.index.min().date()),
        "last_date": str(frame.index.max().date()),
        "zero_volume_dates": frame.index[frame.Volume.eq(0)].strftime("%Y-%m-%d").tolist(),
        "missing_values": frame.isna().sum().astype(int).to_dict(),
        "note": "Calendar coverage and unusual records still require independent review."
    }


def read_prices(path):
    frame = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    validate_prices(frame)
    return frame


def align_daily_series(asset, market):
    """Keep usable asset days with an observed STI close and audit exclusions.

    Zero asset volume is treated as an unverified quote rather than a tradable
    observation. Index volume is not used because it is not an asset volume.
    Original downloaded snapshots are never edited by this function.
    """
    zero_asset = asset.index[asset["Volume"].eq(0)]
    usable = asset.drop(index=zero_asset)
    missing_sti = usable.index.difference(market.index)
    common = usable.index.intersection(market.index)
    report = {
        "raw_asset_rows": len(asset),
        "raw_market_rows": len(market),
        "excluded_zero_asset_volume_dates": zero_asset.strftime("%Y-%m-%d").tolist(),
        "excluded_missing_sti_dates": missing_sti.strftime("%Y-%m-%d").tolist(),
        "aligned_rows": len(common),
        "policy": "Use observed common dates with positive asset volume; do not fill STI prices or edit raw CSVs.",
        "review_required": "Independently verify excluded dates, especially unusual zero-volume asset records, before final reporting."
    }
    if len(common) < 300:
        raise ValueError("Too few aligned trading dates to calculate the planned indicators")
    return asset.loc[common].copy(), market.loc[common].copy(), report


def download(config, raw_dir):
    """Download both series; never silently overwrite an existing snapshot."""
    import yfinance as yf

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    destinations = [raw_dir / f"{role}.csv" for role in ("asset", "market")]
    if any(p.exists() for p in destinations):
        raise FileExistsError("Raw snapshots already exist; archive them before a fresh download")
    yf.set_tz_cache_location(str(raw_dir / ".yfinance_cache"))
    pending = []
    for role in ("asset", "market"):
        ticker = config[role]
        frame = yf.Ticker(ticker).history(
            start=config["download_start"], end=config["download_end_exclusive"],
            interval="1d", auto_adjust=False, actions=True, raise_errors=True,
            timeout=30,
        )
        frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
        frame.index.name = "Date"
        report = validate_prices(frame)
        if frame.index.min() > pd.Timestamp(config["study_start"]):
            raise ValueError(f"{ticker}: missing study-start coverage")
        if frame.index.max() < pd.Timestamp(config["study_end"]) - pd.Timedelta(days=7):
            raise ValueError(f"{ticker}: study-end coverage appears incomplete")
        report.update(ticker=ticker, source="Yahoo Finance via yfinance",
                      downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
                      yfinance_version=importlib.metadata.version("yfinance"),
                      start=config["download_start"],
                      end_exclusive=config["download_end_exclusive"], auto_adjust=False)
        pending.append((role, frame, report))
    for role, frame, report in pending:
        path = raw_dir / f"{role}.csv"
        frame.to_csv(path, date_format="%Y-%m-%d")
        report["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        path.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return {role: report for role, _, report in pending}
