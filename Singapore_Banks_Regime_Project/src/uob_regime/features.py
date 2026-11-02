"""Build causal feature windows and explicitly dated forward-return labels."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = ["log_return", "open_gap", "intraday_return", "range_ratio",
            "log_volume_change", "volatility", "trend", "sti_drawdown"]
REGIMES = ["volatility", "trend", "sti_drawdown"]


@dataclass
class Samples:
    X: np.ndarray
    regime: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame
    feature_names: tuple = tuple(FEATURES)


def _adjust(frame, basis):
    prices = frame[["Open", "High", "Low", "Close"]].copy()
    if basis == "adjusted":
        if "Adj Close" not in frame:
            raise ValueError("Adjusted price basis requires Adj Close in both series")
        prices = prices.mul(frame["Adj Close"] / frame.Close, axis=0)
    return prices


def build_features(asset, market, config):
    """Return causal features plus future execution metadata, never model inputs."""
    missing = asset.index.difference(market.index)
    if len(missing):
        raise ValueError(f"STI missing {len(missing)} asset dates; resolve calendar alignment first")
    market = market.reindex(asset.index)
    a, m = _adjust(asset, config["price_basis"]), _adjust(market, config["price_basis"])
    log_close = np.log(a.Close)
    ret = log_close.diff()
    width = config["trend_window"]
    time_axis = np.arange(width, dtype=float)
    time_axis -= time_axis.mean()
    frame = pd.DataFrame(index=asset.index)
    frame["log_return"] = ret
    frame["open_gap"] = np.log(a.Open / a.Close.shift(1))
    frame["intraday_return"] = np.log(a.Close / a.Open)
    frame["range_ratio"] = (a.High - a.Low) / a.Close
    frame["log_volume_change"] = np.log1p(asset.Volume).diff()
    frame["volatility"] = ret.rolling(config["volatility_window"]).std()
    frame["trend"] = log_close.rolling(width).apply(
        lambda values: np.dot(values, time_axis) / np.dot(time_axis, time_axis), raw=True)
    frame["sti_drawdown"] = m.Close / m.Close.rolling(config["drawdown_window"]).max() - 1
    frame["y"] = np.log(a.Close.shift(-1) / a.Close)
    dates = pd.Series(asset.index, index=asset.index)
    frame["target_date"] = dates.shift(-1)
    frame["entry_open"] = a.Open.shift(-1)
    frame["target_close"] = a.Close.shift(-1)
    frame["exit_date"] = dates.shift(-2)
    frame["exit_open"] = a.Open.shift(-2)
    return frame


def build_samples(frame, config):
    """Use t-29..t to predict t+1; assign folds using the target date."""
    lookback = config["lookback"]
    values = frame[FEATURES].to_numpy(dtype=np.float32)
    X, regimes, targets, rows = [], [], [], []
    for i in range(lookback - 1, len(frame)):
        row = frame.iloc[i]
        window = values[i - lookback + 1:i + 1]
        if not np.isfinite(window).all() or not np.isfinite(row.y):
            continue
        if not pd.Timestamp(config["study_start"]) <= row.target_date <= pd.Timestamp(config["study_end"]):
            continue
        X.append(window)
        regimes.append(row[REGIMES].to_numpy(dtype=np.float32))
        targets.append(row.y)
        rows.append({"prediction_date": frame.index[i], "target_date": row.target_date,
                     "observed_return": row.log_return,
                     "entry_open": row.entry_open, "target_close": row.target_close,
                     "exit_date": row.exit_date, "exit_open": row.exit_open,
                     **{key: row[key] for key in REGIMES}})
    if not X:
        raise ValueError("No valid samples; check coverage, warm-up windows and study dates")
    return Samples(np.stack(X), np.stack(regimes), np.array(targets, dtype=np.float32),
                   pd.DataFrame(rows))
