# Singapore Banks Regime-Aware Fuzzy-Gated LSTM Project

This independent workspace implements the primary contributor's research infrastructure for AI6124 Group 4. UOB remains the primary study required by the final proposal in `../Proposal/AI6124_Group4_Proposal_Presentation.pptx`; DBS and OCBC are additional bank studies that reuse the same leakage-aware pipeline. Original reference files remain unchanged.

## Included banks

| Bank | Ticker | Configuration | Workspace | Status |
|---|---|---|---|---|
| UOB | `U11.SI` | `configs/default.json` | project root | Primary study; development experiments through 2025 complete |
| DBS | `D05.SI` | `configs/dbs.json` | `banks/dbs` | All nine methods complete on 2020-2025 development folds |
| OCBC | `O39.SI` | `configs/ocbc.json` | `banks/ocbc` | All nine methods complete on 2020-2025 development folds |

All three asset snapshots cover 2013-10-01 through 2026-09-30 and use the same STI (`^STI`) market series, feature definitions, chronological folds and train-only preprocessing. See `docs/MULTI_BANK_DATA.md` for hashes, row counts and the boundary between the primary study and the two extensions.

The completed [three-bank comparison](reports/multi_bank/REPORT.md) contains 414 fold/model/seed results (138 per bank). Its directory includes summary and annual CSVs, cost sensitivity, seed ensembles, provenance and a comparison figure. Rebuild it from the explicit run selection with:

```bash
python scripts/summarize_banks.py configs/bank_comparison_runs.json
```

The notebook loads the generated report directly. UOB final refit epoch counts are predeclared; the additional-bank configurations retain their separate validation-year final procedure. Review and freeze any final evaluation protocol before using 2026 data.

## Current implementation

- UOB (`U11.SI`), DBS (`D05.SI`), OCBC (`O39.SI`) and STI (`^STI`) daily snapshots with metadata and hashes.
- Audited common-date alignment, causal features, 30-trading-day windows and next-day log-return targets.
- Development folds for 2020–2025 and an explicitly separate January–September 2026 holdout.
- Train-only feature, regime and target scaling per fold.
- Persistence, zero-return, ordinary LSTM, adapted fuzzy-feature LSTM, three-expert TSK mixture, parameter-matched single LSTM, equal-weight expert mixture, DLinear and ANFIS.
- Early stopping on validation only; seeded training; saved predictions, histories, weights and preprocessing.
- Next-open long-or-cash research backtest, transaction costs and buy-and-hold comparison.
- CSV summaries, plots, fuzzy-rule exports, temporal checks and numerical integration tests.

Development results through 2025 are complete. The final 2026 holdout remains unopened until the protocol is jointly reviewed and frozen. The final runner refits on all 2015–2025 samples for fixed, preselected epoch counts rather than using holdout labels for early stopping.

## Environment

Use Python 3.11 or 3.12. TensorFlow 2.16.2 and legacy `tf-keras` 2.16.0 match the earlier CPU workflow. Do not mix `keras` 3 objects with `tf_keras` objects.

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[notebook]'
```

For a direct entry point, `python run.py --help` works without installing the local package, as long as dependencies are available. Use `python run.py download`, `python run.py prepare`, and `python run.py run --folds 2020 --models persistence lstm --seeds 11 --epochs 5` to get started.

On this machine, the existing environment below passed the full test suite without installing new packages:

```bash
/Users/sophia/anaconda3/envs/pytorch/bin/python run.py --help
/Users/sophia/anaconda3/envs/pytorch/bin/python run.py download
/Users/sophia/anaconda3/envs/pytorch/bin/python run.py prepare
/Users/sophia/anaconda3/envs/pytorch/bin/python run.py run --folds 2020 --models persistence lstm --seeds 11 --epochs 5
```

In VS Code, select this same Python interpreter for the notebook. Local UOB, DBS, OCBC and STI snapshots are already present in their respective workspaces. Raw-data directories are ignored by version control, so another machine must download its own snapshots or receive an explicitly documented copy.

Alternatively, use an existing environment with the required dependencies and run commands without installation:

```bash
PYTHONPATH=src python -m uob_regime.cli --help
```

The commands below use this second form. Installed users can replace `PYTHONPATH=src python -m uob_regime.cli` with `uob-project`.

## First real-data run

```bash
PYTHONPATH=src python -m uob_regime.cli download
PYTHONPATH=src python -m uob_regime.cli prepare
PYTHONPATH=src python -m uob_regime.cli run --folds 2020 --models persistence zero lstm --seeds 11 --epochs 5
```

Download requires access to Yahoo Finance. The downloader never generates artificial market data or overwrites an existing snapshot. Inspect `data/raw/*.json`, investigate warnings, and review `data/processed/alignment.json` and `data/processed/folds.csv` before interpreting results. Preparation retains raw CSVs and excludes UOB zero-volume rows or dates without an observed STI close; excluded dates require independent review before final reporting. Download starts in October 2013 to warm up both the 252-day STI indicator and the 30-day sequence; the study starts in 2015.

Prepare or run one of the additional banks by selecting its configuration. The `workspace` value in each configuration automatically routes data and results to the corresponding bank directory:

```bash
python run.py --config configs/dbs.json prepare
python run.py --config configs/ocbc.json prepare

# Example small historical checks; omit the overrides only when a full run is intended.
python run.py --config configs/dbs.json run --folds 2020 --models zero lstm --seeds 11 --epochs 5
python run.py --config configs/ocbc.json run --folds 2020 --models zero lstm --seeds 11 --epochs 5
```

Do not pool the three banks into one training set unless a separate cross-sectional protocol is specified. The current design treats each bank as an independent time-series study so that targets, scalers, folds and backtests remain bank-specific.

## Core and full historical experiments

```bash
PYTHONPATH=src python -m uob_regime.cli run --folds 2020 --models fuzzy_feature fuzzy_moe capacity_lstm equal_moe --seeds 11 --epochs 5
PYTHONPATH=src python -m uob_regime.cli run
```

The second command uses all historical folds, configured models and three seeds. It can take substantial time. Five-epoch runs are pipeline checks, not final performance evidence. Review and adjust configuration on historical training/validation data before the full experiment.

Analyze a completed run without retraining:

```bash
python run.py analyze --run-dir results/YOUR_COMPLETED_RUN --cost-bps 0 5 10 20
```

This exports a cost-sensitivity table and learned membership/rule artifacts when a TSK gate or ANFIS model is present. Existing analysis directories are retained; choose another run or archive an old analysis before repeating. Membership set numbers identify their initial ordering; training does not constrain centers to remain ordered.

To use a custom configuration:

```bash
PYTHONPATH=src python -m uob_regime.cli --config configs/my_experiment.json run
```

## Final holdout

Freeze the reviewed settings before the first final evaluation:

```bash
PYTHONPATH=src python -m uob_regime.cli freeze --output configs/frozen_final.json
PYTHONPATH=src python -m uob_regime.cli run-holdout --frozen-config configs/frozen_final.json
```

The holdout command accepts no model, seed or epoch overrides. A hash detects changes inside the saved protocol. This procedural guard cannot prove that a person has never inspected the holdout. Keep a dated experiment log and avoid adapting settings after seeing final results.

The revised final procedure trains on every target observation through 2025 and tests January–September 2026. Because no validation year remains outside the refit set, each model uses the fixed epoch count stored in `refit_epochs`; these counts are rounded-up medians of best epochs selected on the 2020–2025 development folds. Seed forecasts are also exported as a predeclared arithmetic ensemble during analysis.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Tests generate explicitly artificial prices in memory and temporary directories. They never enter `data/raw/` or the formal `results/` directory. The neural integration test trains every implemented model for one epoch on a small synthetic configuration. TensorFlow tests are skipped if `tf_keras` is unavailable.

## Folder map

```text
configs/default.json          UOB primary-study settings
configs/dbs.json              DBS extension settings
configs/ocbc.json             OCBC extension settings
src/uob_regime/data.py        Download and quality checks
src/uob_regime/features.py    Causal features, labels and execution metadata
src/uob_regime/splits.py      Chronological folds and train-only scalers
src/uob_regime/models/        LSTM, DLinear, ANFIS and TSK fuzzy models
src/uob_regime/experiment.py  Training, evaluation and artifact runner
src/uob_regime/backtest.py    Execution and financial metrics
src/uob_regime/cli.py         Command-line entry points
notebooks/                   English notebook entry point
tests/                       Temporal and integration checks
docs/                        Protocol, responsibilities and teammate handoff
data/raw/                    Downloaded asset and market snapshots
data/processed/              Feature/sample/fold audits
results/                     Unique experiment directories
banks/dbs/                   DBS raw/processed data and experiment results
banks/ocbc/                  OCBC raw/processed data and future results
```

See [the data audit](docs/DATA_AUDIT.md), [historical experiment decisions](docs/EXPERIMENT_DECISIONS.md), [historical results](docs/HISTORICAL_RESULTS.md), [the research protocol](docs/RESEARCH_PROTOCOL.md), [the ownership checklist](docs/WORK_PLAN.md), and [the teammate handoff](docs/TEAMMATE_HANDOFF.md) before extending the project.

API references checked during implementation: [yfinance PriceHistory](https://ranaroussi.github.io/yfinance/reference/yfinance.price_history.html), [TensorFlow custom layers](https://www.tensorflow.org/api_docs/python/tf/keras/Layer), and [Keras layer API](https://keras.io/api/layers/base_layer/). The code uses the pinned legacy `tf_keras` implementation.
