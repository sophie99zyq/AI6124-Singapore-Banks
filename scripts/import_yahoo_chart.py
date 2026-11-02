"""Import a Yahoo Finance chart response as an immutable asset snapshot."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from uob_regime.config import load_config, workspace_root
from uob_regime.data import validate_prices


def event_date(timestamp, timezone_name):
    """Convert a Yahoo event timestamp to its exchange-local calendar date."""
    return pd.to_datetime(timestamp, unit="s", utc=True).tz_convert(timezone_name).tz_localize(None).normalize()


def import_chart(input_path, config):
    """Validate and persist one browser-downloaded Yahoo chart response."""
    input_path = Path(input_path).resolve()
    payload = json.loads(input_path.read_text())
    chart = payload.get("chart", {})
    if chart.get("error") is not None or not chart.get("result"):
        raise ValueError(f"Yahoo chart response contains an error: {chart.get('error')}")

    result = chart["result"][0]
    metadata = result["meta"]
    if metadata.get("symbol") != config["asset"]:
        raise ValueError(f"Expected {config['asset']}, received {metadata.get('symbol')}")

    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    adjusted = result["indicators"]["adjclose"][0]["adjclose"]
    lengths = {len(timestamps), len(adjusted), *(len(quote[name]) for name in
                ("open", "high", "low", "close", "volume"))}
    if len(lengths) != 1:
        raise ValueError("Yahoo chart arrays have inconsistent lengths")

    timezone_name = metadata.get("exchangeTimezoneName", "Asia/Singapore")
    dates = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(timezone_name).tz_localize(None).normalize()
    frame = pd.DataFrame({
        "Open": quote["open"],
        "High": quote["high"],
        "Low": quote["low"],
        "Close": quote["close"],
        "Adj Close": adjusted,
        "Volume": quote["volume"],
        "Dividends": 0.0,
        "Stock Splits": 0.0,
    }, index=dates)
    frame.index.name = "Date"

    # Yahoo occasionally publishes an OHLC row whose close falls just outside
    # the reported high-low range. Apply the smallest range-only correction and
    # retain a complete audit trail; close, adjusted close and volume are never
    # changed by this normalization.
    inconsistent = ((frame["High"] < frame[["Open", "Low", "Close"]].max(axis=1)) |
                    (frame["Low"] > frame[["Open", "High", "Close"]].min(axis=1)))
    ohlc_normalizations = []
    for date, row in frame.loc[inconsistent].iterrows():
        original_high = float(row["High"])
        original_low = float(row["Low"])
        repaired_high = float(max(row["Open"], row["High"], row["Low"], row["Close"]))
        repaired_low = float(min(row["Open"], row["High"], row["Low"], row["Close"]))
        frame.loc[date, "High"] = repaired_high
        frame.loc[date, "Low"] = repaired_low
        ohlc_normalizations.append({
            "date": str(date.date()),
            "reason": "Source close was outside the reported high-low range",
            "original_high": original_high,
            "original_low": original_low,
            "normalized_high": repaired_high,
            "normalized_low": repaired_low,
        })

    events = result.get("events", {})
    for event in events.get("dividends", {}).values():
        date = event_date(event["date"], timezone_name)
        if date not in frame.index:
            raise ValueError(f"Dividend event has no matching price row: {date.date()}")
        frame.loc[date, "Dividends"] = float(event["amount"])
    for event in events.get("splits", {}).values():
        date = event_date(event["date"], timezone_name)
        if date not in frame.index:
            raise ValueError(f"Split event has no matching price row: {date.date()}")
        frame.loc[date, "Stock Splits"] = float(event["numerator"]) / float(event["denominator"])

    report = validate_prices(frame)
    if frame.index.min() > pd.Timestamp(config["study_start"]):
        raise ValueError("Imported snapshot does not cover the configured study start")
    if frame.index.max() < pd.Timestamp(config["study_end"]) - pd.Timedelta(days=7):
        raise ValueError("Imported snapshot does not cover the configured study end")

    raw_dir = workspace_root(config) / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output = raw_dir / "asset.csv"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable snapshot: {output}")
    frame.to_csv(output, date_format="%Y-%m-%d")
    report.update({
        "ticker": config["asset"],
        "source": "Yahoo Finance chart API saved through Google Chrome",
        "source_url_pattern": "query1.finance.yahoo.com/v8/finance/chart/{ticker}",
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": input_path.name,
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "start": config["download_start"],
        "end_exclusive": config["download_end_exclusive"],
        "auto_adjust": False,
        "ohlc_normalizations": ohlc_normalizations,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    })
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    arguments = parser.parse_args()
    output, report = import_chart(arguments.input, load_config(arguments.config))
    print(output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
