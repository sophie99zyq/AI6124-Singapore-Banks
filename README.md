# Remaining Work

## Literature Review and Method Description

- [ ] Verify the original references and bibliographic details for LSTM, Mixture of Experts, TSK, ANFIS, DLinear and the reference Fuzzy LSTM implementation.
- [ ] Present the development of related methods chronologically and explain their relationship to this project.
- [ ] Document the fuzzy feature transformation, the 27-rule TSK gate and the three-expert fusion equations, including differences from the reference code.

## Final Notebook

Target file: [Final_Project.ipynb](notebooks/Final_Project.ipynb).

- [ ] Complete the research motivation, data sources, price adjustments, anomalous-date handling and prediction task definition.
- [ ] Explain the eight input features, 30-day windows, chronological splits and training-only standardization.
- [ ] Present the three-bank prediction metrics, backtests, seed variability, transaction-cost sensitivity and ablation results.
- [ ] Discuss differences across banks, the zero-return baseline and the finding that fuzzy gating remains close to equal expert weighting.
- [ ] Complete the limitations and future work, including data-source limitations, sample size, correlations between banks and the difference between forecast and trading intervals.
- [ ] Remove TODOs and placeholder text, run the notebook from start to finish and save its outputs.

Results source: [Three-Bank Development Report](reports/multi_bank/REPORT.md).

## Final Holdout Evaluation

- [ ] Confirm the models, training periods, refit epochs, seed-averaging rules and trading settings for each bank.
- [ ] Resolve and document the difference between UOB's fixed-epoch refit procedure and the retained validation-year procedure currently configured for DBS and OCBC.
- [ ] Save hashed, frozen configurations before inspecting the 2026 evaluation results.
- [ ] Evaluate January–September 2026 and save predictions, metrics, model checkpoints and backtest artifacts.
- [ ] Report final holdout results separately from development results, without tuning settings after inspecting the holdout.

Protocol: [RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md).

## Submission Checks

- [ ] Update documentation to consistently describe the three-bank experiments, holdout boundaries and current paths.
- [ ] Check every numerical claim against saved results; do not describe mean annual-fold returns as a continuous multi-year cumulative return.
- [ ] Verify attribution of reference code and literature.
- [ ] Validate dependency installation, data loading, result aggregation and notebook execution in a clean environment.
- [ ] Confirm that required data, configurations, result files and relative links are included and accessible.
