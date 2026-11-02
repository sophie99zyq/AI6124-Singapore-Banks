# AI6124 Group 4: Singapore Bank Return Forecasting

Shared coursework workspace for ZHOU YIQING and MANIAR KUSH HEMAL.

## Start here

The implementation lives in [Singapore_Banks_Regime_Project](Singapore_Banks_Regime_Project/README.md). It compares nine forecasting methods on UOB, DBS and OCBC daily returns, with STI market indicators, causal preprocessing, annual chronological validation and next-open trading simulation.

- [Three-bank development results](Singapore_Banks_Regime_Project/reports/multi_bank/REPORT.md)
- [Result comparison figure](Singapore_Banks_Regime_Project/reports/multi_bank/comparison.png)
- [Data inventory](Singapore_Banks_Regime_Project/docs/MULTI_BANK_DATA.md)
- [Research protocol](Singapore_Banks_Regime_Project/docs/RESEARCH_PROTOCOL.md)
- [Notebook](Singapore_Banks_Regime_Project/notebooks/Final_Project.ipynb)
- [Teammate handoff](Singapore_Banks_Regime_Project/docs/TEAMMATE_HANDOFF.md) (original UOB handoff; see the three-bank report for the extension)

All three banks have nine methods evaluated on the six 2020-2025 development folds, totaling 414 fold/model/seed results. The 2026 final holdout has not been run. Literature review, final notebook prose and the final presentation remain to be completed.

## Reproduce the saved comparison

Use Python 3.11 or 3.12 and the pinned dependencies described in the project README:

```bash
cd Singapore_Banks_Regime_Project
python -m pip install -e '.[notebook]'
python scripts/summarize_banks.py configs/bank_comparison_runs.json
```

Data snapshots, historical predictions, model weights, training logs and analysis artifacts are included for group review. Download-tool cookies, local caches and environment directories are excluded. Use existing snapshots rather than downloading over them.

## Other materials

`Proposal/` contains proposal versions and the course-requirements summary. `26S1_AI6124_Project.ipynb` is the supplied course starter. `Paper_Fuzzy_LSTM-main/` and its ZIP are third-party reference materials from https://github.com/MakhovaAnastasia/Paper_Fuzzy_LSTM; `Fuzzy_LSTM_CPU/` is a local compatibility adaptation. These materials retain their original authorship and are not original contributions by this group. Market data comes from Yahoo Finance as recorded in the snapshot metadata. No blanket license is asserted over third-party materials or market data.

`PROJECT_ROADMAP_CN.md` is an earlier planning document; use the current project README and result report for implementation status.
