"""Long-or-cash research simulation with next-open execution and explicit costs.

Adjusted OHLC is a total-return price proxy, not a broker execution ledger.
Signals observed at close t enter at open t+1 and earn open t+1 to open t+2.
The last position exits at the final target close. No close-t entry is assumed.
"""

import numpy as np
import pandas as pd


def financial_metrics(net_returns, turnover, config):
    returns = np.asarray(net_returns, dtype=float)
    if not len(returns) or not np.isfinite(returns).all() or (returns <= -1).any():
        raise ValueError("Invalid portfolio returns")
    equity = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(np.r_[1.0, equity])[1:]
    excess = returns - ((1 + config["risk_free_rate"]) ** (1 / config["annualization"]) - 1)
    deviation = np.std(excess, ddof=1) if len(excess) > 1 else 0
    sharpe = float(np.sqrt(config["annualization"]) * np.mean(excess) / deviation) if deviation > 0 else None
    return {"net_cumulative_return": float(equity[-1] - 1), "annualized_sharpe": sharpe,
            "maximum_drawdown": float(np.min(equity / peaks - 1)),
            "annual_turnover": float(np.sum(turnover) * config["annualization"] / len(returns))}


def backtest(predictions, config, test_end, buy_and_hold=False):
    if predictions.empty:
        raise ValueError("At least one aligned prediction is required")
    frame = predictions.sort_values("target_date").copy().reset_index(drop=True)
    for col in ("prediction_date", "target_date", "exit_date"):
        frame[col] = pd.to_datetime(frame[col])
    if frame.target_date.duplicated().any() or not (frame.prediction_date < frame.target_date).all():
        raise ValueError("Invalid signal or target dates")
    if len(frame) > 1:
        exits = frame.exit_date.iloc[:-1].to_numpy()
        entries = frame.target_date.iloc[1:].to_numpy()
        if not np.array_equal(exits, entries):
            raise ValueError("Prediction dates contain gaps; do not treat them as continuous holdings")
    if (frame.target_date > pd.Timestamp(test_end)).any():
        raise ValueError("Entry dates exceed the evaluation boundary")
    exit_price = frame.exit_open.copy()
    # Always liquidate the final row inside the evaluated period.
    frame.loc[frame.index[-1], "exit_date"] = frame.target_date.iloc[-1]
    exit_price.iloc[-1] = frame.target_close.iloc[-1]
    if (frame.exit_date > pd.Timestamp(test_end)).any():
        raise ValueError("Execution would cross the evaluation boundary")
    if not np.isfinite(frame.entry_open).all() or not np.isfinite(exit_price).all():
        raise ValueError("Missing execution prices")
    position = np.ones(len(frame)) if buy_and_hold else (frame.predicted_return > config["signal_threshold"]).to_numpy(dtype=float)
    turnover = np.abs(position - np.r_[0.0, position[:-1]])
    cost = config["cost_bps"] / 10000
    gross = position * (exit_price.to_numpy() / frame.entry_open.to_numpy() - 1)
    net = (1 - cost * turnover) * (1 + gross)
    # Final liquidation is a second transaction if the final interval was long.
    net[-1] *= 1 - cost * position[-1]
    turnover[-1] += position[-1]
    frame["position"] = position
    frame["turnover"] = turnover
    frame["execution_exit_price"] = exit_price
    frame["gross_return"] = gross
    frame["net_return"] = net - 1
    frame["equity"] = np.cumprod(net)
    frame["drawdown"] = frame.equity / np.maximum.accumulate(np.r_[1.0, frame.equity])[1:] - 1
    return frame, financial_metrics(net - 1, turnover, config)
