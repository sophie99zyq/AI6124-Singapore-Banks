# Three-bank development comparison

All three banks: nine methods, six annual test folds (2020-2025), three seeds for learned models; 138 rows per bank. No 2026 test is included.

Annual results are first averaged over seeds, then equally over years. Returns are means of separate annual portfolios, not a continuous six-year return. Banks share a market and calendar, so they are not three independent statistical replications.

| Bank | Model | RMSE | MAE | Direction accuracy | Mean net return | Sharpe |
|---|---|---:|---:|---:|---:|---:|
| DBS | anfis | 0.012143 | 0.008507 | 51.75% | 18.75% | 0.955 |
| DBS | capacity_lstm | 0.012169 | 0.008576 | 49.14% | 5.98% | 0.469 |
| DBS | dlinear | 0.012287 | 0.008674 | 49.65% | 3.32% | 0.244 |
| DBS | equal_moe | 0.012150 | 0.008549 | 49.42% | 10.28% | 0.602 |
| DBS | fuzzy_feature | 0.012191 | 0.008582 | 51.23% | 13.69% | 0.862 |
| DBS | fuzzy_moe | 0.012149 | 0.008549 | 49.64% | 10.73% | 0.634 |
| DBS | lstm | 0.012219 | 0.008595 | 50.21% | 10.53% | 0.679 |
| DBS | persistence | 0.016653 | 0.011920 | 51.09% | 3.90% | 0.200 |
| DBS | zero | 0.012127 | 0.008506 | 48.19% | 0.00% | N/A |
| OCBC | anfis | 0.010474 | 0.007417 | 50.30% | 9.83% | 0.827 |
| OCBC | capacity_lstm | 0.010441 | 0.007389 | 49.57% | 6.11% | 0.583 |
| OCBC | dlinear | 0.010575 | 0.007491 | 50.32% | 0.48% | 0.101 |
| OCBC | equal_moe | 0.010436 | 0.007380 | 51.29% | 9.00% | 0.816 |
| OCBC | fuzzy_feature | 0.010495 | 0.007459 | 50.03% | 6.93% | 0.432 |
| OCBC | fuzzy_moe | 0.010436 | 0.007380 | 51.51% | 9.16% | 0.825 |
| OCBC | lstm | 0.010490 | 0.007449 | 49.52% | 4.82% | 0.424 |
| OCBC | persistence | 0.014270 | 0.010231 | 50.04% | 4.56% | 0.444 |
| OCBC | zero | 0.010425 | 0.007339 | 50.10% | 0.00% | N/A |
| UOB | anfis | 0.011159 | 0.007610 | 49.25% | 7.33% | 0.562 |
| UOB | capacity_lstm | 0.011037 | 0.007554 | 48.16% | 2.61% | 0.262 |
| UOB | dlinear | 0.011184 | 0.007701 | 48.95% | -7.44% | -0.540 |
| UOB | equal_moe | 0.011014 | 0.007538 | 50.16% | 4.04% | 0.366 |
| UOB | fuzzy_feature | 0.011048 | 0.007573 | 51.52% | 8.41% | 0.575 |
| UOB | fuzzy_moe | 0.011015 | 0.007542 | 49.96% | 4.50% | 0.392 |
| UOB | lstm | 0.011061 | 0.007601 | 48.24% | -0.13% | 0.022 |
| UOB | persistence | 0.015088 | 0.010372 | 50.57% | -1.12% | -0.091 |
| UOB | zero | 0.010989 | 0.007467 | 50.51% | 0.00% | N/A |

## Buy-and-hold benchmark

- DBS: mean annual-fold net return 23.06%.
- OCBC: mean annual-fold net return 16.84%.
- UOB: mean annual-fold net return 11.28%.

## Per-bank findings

- DBS: lowest RMSE is zero (0.012127); strongest learned-model mean annual-fold return is anfis (18.75%). Fuzzy-MoE minus equal-MoE RMSE is -0.00000017 (negative favors fuzzy gating). These are descriptive comparisons, not significance tests.
- OCBC: lowest RMSE is zero (0.010425); strongest learned-model mean annual-fold return is anfis (9.83%). Fuzzy-MoE minus equal-MoE RMSE is +0.00000009 (negative favors fuzzy gating). These are descriptive comparisons, not significance tests.
- UOB: lowest RMSE is zero (0.010989); strongest learned-model mean annual-fold return is fuzzy_feature (8.41%). Fuzzy-MoE minus equal-MoE RMSE is +0.00000119 (negative favors fuzzy gating). These are descriptive comparisons, not significance tests.

## Gate diagnostics

- DBS: mean gate entropy 1.098463; uniform three-expert entropy is 1.098612. Compare against equal_moe before attributing benefits to learned gating.
- OCBC: mean gate entropy 1.098460; uniform three-expert entropy is 1.098612. Compare against equal_moe before attributing benefits to learned gating.
- UOB: mean gate entropy 1.098365; uniform three-expert entropy is 1.098612. Compare against equal_moe before attributing benefits to learned gating.

## Reproducibility

Exact run IDs, data hashes and source hashes are retained in provenance.json. Training settings and raw hashes were checked; historical runs may have different source revisions, which are retained rather than rewritten. Sample counts agree across models within each bank/fold. Original UOB results remain UOB-specific.

CSV files include annual metrics, between-year variability, cost scenarios and aligned arithmetic seed ensembles. The figure comparison.png compares RMSE and net returns. Final 2026 evaluation requires a separately frozen protocol.
