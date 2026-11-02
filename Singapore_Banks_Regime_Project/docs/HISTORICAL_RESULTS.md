# Development Results: 2020-2025

## Official runs

- Core 2020-2024 run: `results/historical_20261023T080942Z_852f7f`.
- DLinear/ANFIS 2020-2024 run: `results/historical_20261023T083105Z_ee4221`.
- Prospective 2025 development run: `results/historical_20261023T084348Z_9d8a7e`.

All three runs use identical data hashes, chronological fold construction, train-only scaling, seeds, MSE objective, transaction costs and metric code. The 2026 holdout has not been run.

## Six-year per-seed summary

The table averages equally across the six annual folds and then across seeds. Returns are arithmetic means of separate annual-fold results at 10-basis-point costs, not one continuous portfolio.

| Model | RMSE | MAE | Directional accuracy | Mean annual-fold net return | Mean Sharpe | Mean max drawdown | Mean turnover |
|---|---:|---:|---:|---:|---:|---:|---:|
| Zero return | **0.010989** | **0.007467** | 50.51% | 0.00% | N/A | 0.00% | 0.00 |
| Equal expert mixture | 0.011014 | 0.007538 | 50.16% | 4.04% | 0.366 | -14.85% | 38.35 |
| TSK fuzzy mixture | 0.011015 | 0.007542 | 49.96% | 4.50% | 0.392 | -14.53% | 38.58 |
| Capacity-matched LSTM | 0.011037 | 0.007554 | 48.16% | 2.61% | 0.262 | -12.92% | 43.81 |
| Fuzzy-feature LSTM | 0.011048 | 0.007573 | **51.52%** | **8.41%** | **0.575** | **-10.71%** | 41.52 |
| Ordinary LSTM | 0.011061 | 0.007601 | 48.24% | -0.13% | 0.022 | -15.25% | 47.39 |
| ANFIS | 0.011159 | 0.007610 | 49.25% | 7.33% | 0.562 | -15.70% | 9.03 |
| DLinear | 0.011184 | 0.007701 | 48.95% | -7.44% | -0.540 | -17.49% | 100.24 |
| Previous-return persistence | 0.015088 | 0.010372 | 50.57% | -1.12% | -0.091 | -13.56% | 125.06 |

Buy and hold averaged approximately 11.28% net return, 0.709 Sharpe and -17.37% maximum drawdown across the six separate annual periods.

## Seed-ensemble improvement

For every stochastic model, the predeclared ensemble is the arithmetic mean of the three aligned seed forecasts. It introduces no fitted parameter and uses no test label. It mainly reduces prediction variance; it does not guarantee better trading returns in every year.

| Model | Ensemble RMSE | Ensemble directional accuracy | Mean net return | Mean Sharpe |
|---|---:|---:|---:|---:|
| Equal expert mixture | 0.010998 | 49.63% | 1.72% | 0.275 |
| TSK fuzzy mixture | 0.011000 | 49.09% | 1.40% | 0.248 |
| Fuzzy-feature LSTM | 0.011010 | 50.44% | **8.29%** | **0.645** |
| Ordinary LSTM | 0.011015 | 49.04% | 4.52% | 0.339 |
| Capacity-matched LSTM | 0.011017 | 48.64% | 5.24% | 0.476 |
| DLinear | 0.011073 | 49.03% | -3.93% | -0.340 |
| ANFIS | 0.011153 | 49.23% | 7.93% | 0.572 |

The ensemble improves RMSE for every stochastic model. The zero-return forecast still has the best six-year RMSE. Fuzzy-feature LSTM is the most balanced learned method because it has the best directional accuracy, strongest six-year per-seed trading result, strongest ensemble Sharpe and shallower drawdown.

## 2025 prospective check

The 2025 fold trains through 2023 and validates on 2024. It was run only after the 2020-2024 model designs were complete.

- Zero return again had the best RMSE at 0.010127.
- Fuzzy-feature LSTM was the only learned model with a positive mean strategy return, approximately 10.42%; its seed ensemble returned approximately 7.82%.
- ANFIS changed from strong 2020-2024 average performance to -11.96% in 2025.
- Most recurrent and mixture models had negative 2025 trading returns.

This result shows that performance is regime-dependent. It weakens any claim that the earlier ANFIS or mixture returns are stable and supports retaining the simple baseline and fuzzy-feature model in the final comparison.

## Why the models remain weak

- Daily UOB returns have low serial predictability. Across the original results, prediction/target correlations were only about 0.01 to 0.06.
- Actual daily return standard deviation was about 1.11%, while trained forecast standard deviations were roughly 0.06% to 0.21%. The networks correctly shrink toward zero but find little stable incremental signal.
- There are only 2,945 overlapping 30-day samples from one stock. Adjacent windows share 29 observations, so the effective independent sample size is smaller.
- Market regimes change across years; the 2020 pandemic validation distribution and the 2025 result illustrate nonstationarity.
- RMSE rewards predictions near zero. A complex model must identify stable conditional means to beat the zero-return forecast.
- The close-to-close prediction target and executable next-open holding return are different intervals, so predictive and trading rankings need not agree.

A separate Huber-loss candidate run (`results/historical_20261023T083857Z_acc4f1`) did not consistently improve RMSE, MAE or directional accuracy and was rejected. MSE remains the fixed objective. Increasing epochs alone is not supported: validation losses commonly bottom out early and then deteriorate.

## Gate specialization

The fuzzy gate's mean entropy across the original folds was 1.098399, close to the three-expert maximum `log(3) = 1.098612`. Mean expert weights were approximately 0.3369, 0.3310 and 0.3321. Fuzzy-MoE and equal weighting remain nearly indistinguishable, so clear expert specialization is not supported.

## Six-year transaction-cost sensitivity

| Model | 0 bps | 5 bps | 10 bps | 20 bps |
|---|---:|---:|---:|---:|
| Fuzzy-feature LSTM | 12.90% | 10.63% | 8.41% | 4.10% |
| TSK fuzzy mixture | 8.62% | 6.54% | 4.50% | 0.54% |
| Equal expert mixture | 8.13% | 6.06% | 4.04% | 0.11% |
| ANFIS | 8.28% | 7.80% | 7.33% | 6.38% |
| Capacity-matched LSTM | 7.14% | 4.85% | 2.61% | -1.72% |
| Ordinary LSTM | 4.68% | 2.24% | -0.13% | -4.71% |
| DLinear | 2.21% | -2.74% | -7.44% | -16.11% |
| Persistence | 12.04% | 5.26% | -1.12% | -12.73% |

## Final protocol boundary

The development years are now 2020-2025. The final refit will use all 2,758 target samples from 2015-2025, fixed per-model epoch counts derived from the median best development epochs, and three seeds. The only final test period is 2026-01-02 through 2026-09-30 with 187 samples. No 2026 result has been generated.
