# Research Protocol

## Task and information timing

At the close of trading day t, observe the last 30 daily feature vectors and the current three regime indicators. Predict `log(C[t+1] / C[t])`. All prices follow the configured common basis. Targets are assigned to partitions by their target date, not their input start date. Past observations may appear in test input windows; future labels may not enter training.

The default adjusted basis multiplies each OHLC field by `Adj Close / Close`. This provides a consistent vendor-adjusted research series. The close basis retains vendor Close and OHLC instead. Neither basis should be described as independently verified historical tape prices. Retrospective vendor adjustments are not a point-in-time corporate-action database. Do not add cash dividends again to adjusted-price returns. Corporate-action treatment and vendor limitations must be discussed in the final report.

## Features

- Close-to-close log return.
- Open gap relative to the previous close.
- Same-day open-to-close log return.
- High-low range divided by close.
- Change in log(1 + volume).
- Rolling 20-day sample standard deviation of daily log returns.
- OLS slope of log close against time over 50 days, measured per trading day.
- STI close relative to its rolling 252-trading-day maximum minus one.

Windows include t because forecasts are produced after the close of t. No backward filling or centered windows are used. Preparation intersects observed UOB and STI dates and excludes UOB rows with zero reported volume. Every excluded date is recorded in `data/processed/alignment.json`; raw snapshots are retained. This is a documented research-data policy, not independent confirmation of the underlying market event. Missing STI prices are never filled. Index volume is not used as a validity filter. History begins in October 2013 so that the one-year indicator can be computed for every day of the first 30-day input window before the 2015 study start. The model receives numeric features; future execution metadata is stored separately and is never fed to the model.

## Models

The ordinary LSTM uses one recurrent layer and one linear output. The fuzzy-feature baseline appends three trainable Gaussian memberships per input feature before an ordinary LSTM. This is a documented adaptation to the common return task, not an exact reproduction of the reference repository's price-defuzzification model or its published numbers.

The main model uses three independently initialized LSTM experts. Each returns a standardized return estimate. The TSK gate has three indicators, three Gaussian sets per indicator, 27 Cartesian-product rules, and three zero-order constant consequents per rule. Gaussian centers start at training-only 20th/50th/80th percentiles. Widths use softplus plus a positive floor. Log firing strengths are normalized with softmax to prevent underflow. Normalized rule strengths aggregate the rule consequents into expert logits; another softmax produces expert weights. The final prediction is the weighted sum of expert forecasts. Expert semantic roles are hypotheses to examine after training.

The equal-mixture ablation averages three expert predictions without a learned gate. The capacity control searches integer hidden sizes for the closest single-LSTM parameter count to the main model, including gate parameters. The actual count and reference count are recorded.

All neural models optimize a train-standardized target with MSE or Huber. Reported prediction metrics are inverse-transformed into original log-return units. Default entropy regularization is zero; mild positive values encourage sample-wise entropy if collapse is observed on training/validation analysis. Overly large values can erase useful specialization.

## Chronological validation

| Test year | Training target years | Validation target year |
|---|---|---|
| 2020 | 2015–2018 | 2019 |
| 2021 | 2015–2019 | 2020 |
| 2022 | 2015–2020 | 2021 |
| 2023 | 2015–2021 | 2022 |
| 2024 | 2015–2022 | 2023 |
| 2025 | 2015–2023 | 2024 |

Each fold fits its own feature, regime and target scalers. Overlapping training windows weight frequent interior training dates multiple times when estimating feature scaling; all observations still come exclusively from training windows. Early stopping monitors validation loss and restores the best validation weights. Training does not shuffle time-series samples. Seeds default to 11, 29 and 47. A single deterministic baseline run has no estimated seed standard deviation; blank standard deviations are expected, not zero uncertainty.

Final holdout: use 2020-2025 as development folds, then refit on every target sample from 2015 through 2025 and evaluate January through September 2026. The refit has no holdout-driven early stopping. Each neural model trains for the rounded-up median of its best development-fold epochs, stored explicitly in `refit_epochs`. The final command remains separate and requires a hashed frozen configuration.

## Trading execution

Use the return forecast from close t to choose a 0/1 position at open t+1. The position earns the price return from open t+1 to open t+2. It may be changed at that next open using the newly available signal. At the final evaluated target date, liquidate at that day's close to avoid crossing the test boundary. Predictions must cover consecutive asset trading dates; gaps raise an error.

This executable holding period differs from the close-to-close forecast label. Predictive accuracy and trading performance answer different questions and must be reported separately. Adjusted OHLC simulates a total-return proxy; it does not model board lots, fees by market venue, bid-ask spread separately, or a cash corporate-action ledger. The configured basis-point charge is a research friction assumption, not an asserted broker tariff.

Costs are proportional to absolute position changes, including entry and final liquidation. Cash earns zero. The benchmark enters at the same initial open and exits at the same final close, with identical cost and price conventions. Net wealth compounds multiplicatively. Sharpe uses daily excess returns and 252-period annualization. Turnover means one-way absolute exposure changes, including liquidation, annualized by 252 / observed intervals. The last intraday interval is treated as one observation, an approximation to disclose. Maximum drawdown is reported as a nonpositive number and includes initial wealth of one.

## Remaining work

Verify literature attribution, finish the presentation and jointly review the fixed model set. Freeze the dated refit protocol only after that review, then conduct the single 2026 holdout evaluation. Do not tune models, epoch counts, thresholds or ensemble rules after observing 2026 results.
