# Validation Record

Date: 2026-10-23.

## Environment used

The existing local Python 3.11.5 environment at `/Users/sophia/anaconda3/envs/pytorch/bin/python` was used for TensorFlow integration checks. No packages were installed or changed in that environment. The project pins TensorFlow 2.16.2 and tf-keras 2.16.0 to match the reference CPU setup. The system Python 3.12 environment was also used for non-neural checks.

## Checks performed

The final TensorFlow-enabled run passed all 11 tests in approximately 29 seconds after the DLinear and ANFIS integration.

- Temporal causality: changing future prices does not change earlier feature vectors.
- Sequence/label alignment: the final feature date is t and the target is the next trading day's log return.
- Preprocessing isolation: changing test distributions does not change fitted training scaler statistics.
- Calendar/model guards: missing STI dates and unknown model identifiers raise explicit errors.
- Execution accounting: signals enter at the next open, final exit remains inside the test period, and entry/liquidation costs compound correctly.
- Neural gate checks: weights are nonnegative, sum to one, reproduce mixture predictions, and gradients reach the expert networks and gate.
- DLinear/ANFIS gradients: both supplementary models have finite trainable gradients and scalar outputs.
- End-to-end synthetic integration: all nine implemented model variants run; prediction, checkpoint, backtest, plot and seed-summary artifacts are produced.
- Postprocessing: saved forecasts can be re-evaluated under alternative costs without retraining; higher costs reduce wealth for an otherwise identical all-long path.
- Final-holdout guard: the experiment API requires an explicitly frozen protocol for final mode.
- CLI help, Python syntax compilation and the English-only source/documentation scan were checked.

Artificial prices exist only inside tests and temporary directories. They are not UOB data, are not saved as formal experimental results, and provide no evidence of financial forecasting quality.

## Real-data run on 2026-10-23

The user downloaded local UOB and STI snapshots through `python run.py download`. Their recorded SHA-256 values are `79626acecd223fe8d99fccdf66525c7c72ea6b547b93bcb0ac5d07c1ce4a4e6e` and `9c4fe4be59efe4537def8a9223ba32c4a24302683b64b4d69be1e09e75e4712b`. The initial `prepare` attempt detected seven UOB dates absent from STI and stopped. The preparation policy now audits and excludes six zero-volume UOB observations and two further UOB dates without an STI close. Original CSVs remain unchanged. The exact dates and policy are saved in `data/processed/alignment.json`. Independent market-data verification of those excluded dates remains outstanding.

After the alignment fix, `prepare` succeeded. The 2020 fold contains 1,001 train, 250 validation and 251 test samples. All nine automated tests passed, including full one-epoch synthetic model integration. The following real-data smoke command completed successfully:

```bash
python run.py run --folds 2020 --models persistence lstm fuzzy_moe --seeds 11 --epochs 5
```

The completed smoke artifacts are in `results/historical_20261023T075620Z_49f44a`. The 2020 test RMSEs on log returns were 0.023655 for persistence, 0.016075 for ordinary LSTM, and 0.016427 for the fuzzy mixture. The fuzzy gate's mean weights were approximately 0.330, 0.341 and 0.329, with mean entropy 1.0983, close to the three-way maximum `log(3)`. This short run shows the pipeline works; it does not establish a benefit from the fuzzy gate or expert specialization.

The official full historical run is `results/historical_20261023T080942Z_852f7f`. It completed 85 fold-model-seed rows across 2020-2024 and generated 799 files, including predictions, model weights, histories, trading ledgers, figures, fuzzy parameters, rules and cost sensitivity. The run passed a structural audit for model names, fold names, seed counts, manifest completion and analysis output. Its zero-return directional accuracy is 0.5069 under the corrected positive-versus-nonpositive definition. Detailed conclusions are in `docs/HISTORICAL_RESULTS.md`.

The official supplementary run is `results/historical_20261023T083105Z_ee4221`. It completed 30 rows: DLinear and ANFIS across five historical folds and three seeds. Its manifest is complete, it contains 319 files after ANFIS analysis export, and its data hashes and fold dates match the core run. Both smoke runs and the full run completed without model failures.

The prospective 2025 development run is `results/historical_20261023T084348Z_9d8a7e`. It completed all nine methods on 252 aligned 2025 targets. The revised preparation report assigns 2,758 samples dated 2015-2025 to final refit training and 187 samples dated 2026-01-02 through 2026-09-30 to the untouched final test. A synthetic integration test confirms that final refit accepts an empty validation partition, uses fixed epochs and records `fixed_from_pre_holdout_history`.

Analysis now exports aligned arithmetic seed ensembles and their predictive, trading and cost metrics. A Huber candidate run is retained at `results/historical_20261023T083857Z_acc4f1`; it was rejected because it did not consistently improve the development comparison.

The TensorFlow runs emitted Apple Silicon optimizer performance warnings and harmless retracing warnings. They did not prevent training. All development model work is complete; the 2026 holdout remains closed.
