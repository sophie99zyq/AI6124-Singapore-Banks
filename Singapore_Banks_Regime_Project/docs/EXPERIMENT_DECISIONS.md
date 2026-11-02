# Historical Experiment Decisions

Decision date: 2026-10-23. These settings apply before the full 2020-2024 historical experiment.

## Fixed settings

- Input length: 30 trading days, as specified in the proposal.
- Base expert size: 16 LSTM units, consistent across the ordinary, fuzzy-feature and expert models.
- Optimizer: Adam with learning rate 0.001.
- Loss: mean squared error on train-standardized next-day log returns.
- Batch size: 32.
- Maximum epochs: 50.
- Early stopping: validation loss, patience 7, restore the best weights.
- Seeds: 11, 29 and 47 for every neural model.
- Transaction cost: 10 basis points per one-way exposure change for the primary trading table.
- Signal threshold: zero predicted return.

The project does not perform a broad hyperparameter search before the historical comparison. With a small daily dataset, searching many choices across the same annual periods would make the reported historical tests increasingly optimistic. The settings are simple, shared and declared in advance. Early stopping selects the training duration separately within each fold using only that fold's validation year.

## Historical models

1. Previous-return persistence.
2. Zero-return forecast.
3. Ordinary LSTM.
4. Adapted fuzzy-feature LSTM.
5. Three-expert TSK fuzzy-gated mixture.
6. Parameter-matched single LSTM.
7. Three-expert equal-weight mixture.

The equal-weight mixture isolates the value of the learned gate. The parameter-matched LSTM tests whether improvement comes mainly from additional capacity. Every model uses the same dates and target.

## Predeclared interpretation checks

- Compare mean and standard deviation across seeds for RMSE, MAE and directional accuracy.
- Compare the fuzzy gate with both the ordinary LSTM and equal-weight experts.
- Treat expert specialization as unsupported when weights stay near one third, entropy stays near `log(3)`, or experts do not show different regime-wise errors.
- Report trading results separately from predictive metrics.
- Recalculate costs at 0, 5, 10 and 20 basis points without retraining.
- Do not change model settings after observing the 2026 final holdout.

## Holdout status

The final 2026 holdout remains closed during development. A frozen protocol file will be created only after all 2020-2025 results have been reviewed.

## Supplementary baseline extension

Before observing their historical test results, two compact supplementary baselines were fixed and added to the same fold runner:

- DLinear uses a centered five-day moving average to decompose every standardized 30-day feature sequence into trend and residual components. Shared linear temporal heads forecast each channel and a final linear layer maps the eight channel forecasts to the standardized return target. It has 71 trainable parameters.
- ANFIS uses the three causal regime descriptors already supplied to the fuzzy gate: 20-day volatility, 50-day trend and 252-day STI drawdown. Three train-only quantile-initialized Gaussian sets per input form 27 rules. Each rule has a first-order Sugeno consequent containing an intercept and three input coefficients. It has 126 trainable parameters.

Both models use the existing Adam learning rate, MSE loss, batch size, 50-epoch maximum, patience-seven early stopping, seeds and train-only scaling. No broad hyperparameter search was performed. Their official supplementary run is `results/historical_20261023T083105Z_ee4221`.

## Revised final refit decision

The user authorized moving 2025 from the final holdout into development and final training. The 2025 fold was evaluated prospectively after the earlier model designs were complete. The only final test period is now January-September 2026.

The final model receives all 2,758 target samples from 2015-2025. With no separate validation year after refitting, epoch counts are fixed to the rounded-up median best epoch across the six development folds: LSTM 4, fuzzy-feature LSTM 4, fuzzy/equal MoE 3, capacity LSTM 2, DLinear 37 and ANFIS 2. The three seed predictions will be averaged as a predeclared ensemble. MSE remains the objective because a separately saved Huber candidate did not consistently improve predictive metrics.
