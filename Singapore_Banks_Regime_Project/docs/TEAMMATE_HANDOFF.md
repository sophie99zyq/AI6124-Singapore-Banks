# Teammate Handoff: Literature and Presentation

## Completed technical work

All historical implementation and model execution are complete. Do not retrain, tune or replace models for the presentation.

- Core run: `results/historical_20261023T080942Z_852f7f`.
- DLinear/ANFIS run: `results/historical_20261023T083105Z_ee4221`.
- 2025 prospective development run: `results/historical_20261023T084348Z_9d8a7e`.
- Combined interpretation: `docs/HISTORICAL_RESULTS.md`.
- Data decisions: `docs/DATA_AUDIT.md`.
- Model settings: `docs/EXPERIMENT_DECISIONS.md`.

The official runs use the same UOB/STI files, train-only preprocessing, seeds, optimization settings, metrics, transaction cost and backtest implementation. The final 2026 holdout has not been run.

## Remaining teammate deliverables

1. Verify primary-source bibliographic details for LSTM, mixture-of-experts, TSK fuzzy systems, ANFIS and DLinear.
2. Prepare a short chronological literature narrative and connect each cited method to the implemented comparison.
3. Convert the completed results into notebook prose, tables and a final presentation.
4. Include limitations: a single stock, small overlapping daily windows, only five historical test years, Yahoo data caveats, transaction-cost sensitivity and no clear fuzzy-gate specialization.
5. Check every numerical claim against `docs/HISTORICAL_RESULTS.md` and the official `metrics.csv` files.
6. Rehearse the final explanation with both team members and jointly check submission files.

## Results to emphasize

- The zero-return forecast retains the lowest mean RMSE, showing how difficult daily return prediction is.
- Across 2020-2025, the fuzzy-feature LSTM has the highest mean directional accuracy at 51.52% and the strongest learned-model mean return at 8.41%.
- ANFIS averages 7.33% with low turnover but loses 11.96% in 2025, demonstrating regime dependence.
- Fuzzy-MoE improves on the ordinary LSTM in most annual fold comparisons, but it is nearly indistinguishable from equal expert weighting.
- Learned fuzzy-gate weights remain close to one third, so clear expert specialization is not supported.
- DLinear underperforms here and loses 7.44% on the mean annual fold after costs.
- Buy and hold averages approximately 11.28% across the six separate annual folds and remains an important benchmark.

Do not describe mean annual-fold returns as one continuous five-year portfolio. Do not claim statistical significance or guaranteed profitability.

## Suggested slide order

1. Motivation and research question.
2. UOB/STI data, eight features and 30-day input windows.
3. Walk-forward timeline and untouched final holdout.
4. LSTM, fuzzy-feature and three-expert Fuzzy-MoE architecture.
5. Baselines and ablations, including ANFIS and DLinear.
6. Predictive comparison table.
7. Trading and cost-sensitivity comparison.
8. Gate weights and the negative specialization finding.
9. Limitations, final-holdout plan and conclusion.

## Forbidden next step

Do not run the 2026 holdout for slide improvement. The team must first review the development results, agree on the final model set and freeze the protocol. Final evaluation is a separate joint step.
