# Multi-Bank Data Inventory

## Study boundary

UOB remains the primary AI6124 study defined by the approved proposal. DBS and OCBC are additional independent-bank studies used to examine whether the same feature, model and backtest pipeline transfers beyond one stock. They do not replace the UOB study and are not pooled into a single training set.

Every bank uses the Straits Times Index (`^STI`) as the market-regime series, the same eight causal features, 30-trading-day inputs, next-day log-return target, annual walk-forward folds and train-only scaling.

## Available snapshots

| Bank | Ticker | Raw rows | First date | Last date | Valid model samples | Asset SHA-256 |
|---|---|---:|---|---|---:|---|
| UOB | `U11.SI` | 3,265 | 2013-10-01 | 2026-09-30 | 2,945 | `79626acecd223fe8d99fccdf66525c7c72ea6b547b93bcb0ac5d07c1ce4a4e6e` |
| DBS | `D05.SI` | 3,265 | 2013-10-01 | 2026-09-30 | 2,945 | `8281c0d278a2f530264b40edbdf9f11d71c17cd222c5abd49ba28db18eed610d` |
| OCBC | `O39.SI` | 3,265 | 2013-10-01 | 2026-09-30 | 2,945 | `9c5d7e36237c7f5a9f3ca90ca45b394bb2f62d74031c8ee7971a9c8af0f63b96` |

The common STI snapshot has 3,258 raw rows and SHA-256 `9c4fe4be59efe4537def8a9223ba32c4a24302683b64b4d69be1e09e75e4712b`.

## Workspace routing

- UOB uses the project-root `data/` and `results/` directories through `configs/default.json`.
- DBS uses `banks/dbs/data/` and `banks/dbs/results/` through `configs/dbs.json`.
- OCBC uses `banks/ocbc/data/` and `banks/ocbc/results/` through `configs/ocbc.json`.

The shared `workspace` resolver prevents absolute paths and parent-directory traversal. Therefore, selecting a bank configuration changes only that bank's data and result workspace while retaining the common implementation.

## Current completion status

- UOB: prepared data, official 2020-2024 core and supplementary runs, 2025 prospective development run, cost and fuzzy-rule analysis.
- DBS: all nine methods completed for 2020-2025. The original five-method run is supplemented by `banks/dbs/results/historical_20261102T061318Z_bc7015` for persistence, capacity-matched LSTM, DLinear and ANFIS.
- OCBC: all nine methods completed for 2020-2025 in `banks/ocbc/results/historical_20261102T061325Z_debf7f`.

The three-bank comparison selects official runs explicitly in `configs/bank_comparison_runs.json`; it audits 138 unique fold/model/seed rows per bank (414 total). See `reports/multi_bank/REPORT.md` for results and `provenance.json` for data and source hashes. Cost scenarios, seed ensembles, annual results and a comparison plot are included. The full TensorFlow-enabled 11-test suite passed after the extension; no packages were installed.

The 2026 holdout remains unopened for the primary UOB study. Additional-bank experiments must be reported as robustness or transfer checks and must not be used to rewrite the predeclared UOB holdout protocol after seeing 2026 results.

## Data caveats

Raw files retain Yahoo-provided OHLCV, adjusted close and corporate-action fields. Preparation uses observed common asset/STI dates, excludes zero-volume asset records, never fills a missing STI price and records every exclusion in the bank's `data/processed/alignment.json`. These snapshots are research data, not independently verified exchange tape or broker execution records.
