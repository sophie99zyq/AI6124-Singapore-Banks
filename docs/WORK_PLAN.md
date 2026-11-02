# Ownership and Completion Checklist

## Primary contributor: approximately 85 percent

- [x] Create a separate project and common configuration.
- [x] Implement snapshot downloading, quality checks and raw-data provenance.
- [x] Implement causal features, dated samples, folds and train-only scaling.
- [x] Implement ordinary and adapted fuzzy-feature LSTM baselines.
- [x] Implement the three-expert TSK model, equal-mixture ablation and capacity control.
- [x] Implement seeded training, saved artifacts and explicit final-holdout routing.
- [x] Implement next-open research backtesting and primary metrics.
- [x] Add tests for causality, scaling, execution, gate gradients and full integration.
- [x] Obtain and independently review real UOB/STI data coverage and unusual observations.
- [x] Select and document defensible historical experiment settings.
- [x] Run historical folds with repeated seeds and investigate failures.
- [x] Review expert collapse and learned rules across historical folds.
- [x] Review cost scenarios and historical prediction/trading conclusions.
- [x] Implement DLinear and ANFIS through the shared fold interface.
- [x] Run both supplementary models across all historical folds and seeds.
- [x] Aggregate the nine-method predictive, trading and cost comparison.
- [x] Perform a clean synthetic integration run and full automated test suite.
- [ ] Freeze the protocol, perform final evaluation and technically review the submission.

Checked implementation items mean code is present, not that the scientific experiments are complete. Consult `VALIDATION.md` for the actual execution checks performed on this workspace.

## Teammate: approximately 15 percent

- [ ] Prepare chronological literature notes and verify bibliographic details.
- [ ] Turn the completed technical results into concise notebook explanations.
- [ ] Prepare the final PPT, tables and selected figures.
- [ ] Check slide/report claims against the saved metrics and limitations.
- [ ] Rehearse the presentation and jointly check submission files.

The teammate should begin from `docs/TEAMMATE_HANDOFF.md` and `docs/HISTORICAL_RESULTS.md`. No model implementation or experiment execution remains assigned to the teammate.

## Joint decisions

Confirm price/corporate-action treatment, feature definitions, folds, training budget, signal threshold and cost scenarios before final experiments. Keep technical decisions and run IDs in an experiment log. Both members should understand the TSK gate and the evaluation timeline before presenting.
