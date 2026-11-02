"""Analyze saved forecasts without retraining or changing model settings."""

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import backtest
from .metrics import prediction_metrics


def analyze_run(run_directory, costs=(0, 5, 10, 20)):
    """Export cost sensitivity and learned fuzzy sets from a completed run."""
    run_directory = Path(run_directory).resolve()
    manifest = json.loads((run_directory / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only completed experiment runs can be analyzed")
    if not costs or any(not 0 <= cost < 10000 for cost in costs):
        raise ValueError("Costs must be nonempty basis-point values in [0, 10000)")
    config = manifest["config"]
    destination = run_directory / "analysis"
    destination.mkdir(exist_ok=False)
    rows = []
    ensemble_inputs = {}
    for path in sorted(run_directory.glob("*/*/seed_*/predictions.csv")):
        fold, model, seed = path.relative_to(run_directory).parts[:3]
        predictions = pd.read_csv(path, parse_dates=["prediction_date", "target_date", "exit_date"])
        ensemble_inputs.setdefault((fold, model), []).append(predictions)
        end = config["holdout_end"] if fold == "final_holdout" else f"{fold}-12-31"
        for cost in costs:
            settings = {**config, "cost_bps": cost}
            for strategy in ("model", "buy_and_hold"):
                _, metrics = backtest(predictions, settings, end, buy_and_hold=strategy == "buy_and_hold")
                rows.append({"fold": fold, "model": model, "seed": seed,
                             "strategy": strategy, "cost_bps": cost, **metrics})
        gate_path = path.parent / "tsk_gate.npz"
        anfis_path = path.parent / "anfis.npz"
        fuzzy_path = gate_path if gate_path.exists() else anfis_path
        if fuzzy_path.exists():
            fuzzy = np.load(fuzzy_path)
            centers, widths = fuzzy["centers"], fuzzy["widths"]
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
            for i, name in enumerate(("Volatility", "Trend", "STI drawdown")):
                lo, hi = np.min(centers[i] - 3 * widths[i]), np.max(centers[i] + 3 * widths[i])
                x = np.linspace(lo, hi, 250)
                for j in range(3):
                    axes[i].plot(x, np.exp(-0.5 * ((x - centers[i, j]) / widths[i, j]) ** 2),
                                 label=f"Initial set {j + 1}")
                axes[i].set_title(name)
                axes[i].set_xlabel("Training-standardized value")
                axes[i].legend()
            axes[0].set_ylabel("Membership")
            fig.tight_layout()
            prefix = f"{fold}_{model}_{seed}"
            fig.savefig(destination / f"{prefix}_memberships.png", dpi=150)
            plt.close(fig)
            rules = pd.DataFrame(list(itertools.product(range(1, 4), repeat=3)),
                                 columns=["volatility_set", "trend_set", "drawdown_set"])
            if gate_path.exists():
                for i in range(3):
                    rules[f"expert_{i + 1}_logit_consequent"] = fuzzy["consequents"][:, i]
            else:
                for i, name in enumerate(("intercept", "volatility", "trend", "sti_drawdown")):
                    rules[f"{name}_consequent"] = fuzzy["consequents"][:, i]
            rules.to_csv(destination / f"{prefix}_rules.csv", index=False)
    ensemble_rows = []
    ensemble_directory = destination / "seed_ensembles"
    for (fold, model), frames in sorted(ensemble_inputs.items()):
        if len(frames) < 2:
            continue
        reference = frames[0].copy()
        for frame in frames[1:]:
            if not reference.target_date.equals(frame.target_date):
                raise ValueError(f"Seed predictions are not date-aligned for {fold}/{model}")
            if not np.allclose(reference.actual_return, frame.actual_return):
                raise ValueError(f"Seed predictions have inconsistent labels for {fold}/{model}")
        reference["predicted_return"] = np.mean(
            [frame.predicted_return.to_numpy() for frame in frames], axis=0)
        end = config["holdout_end"] if fold == "final_holdout" else f"{fold}-12-31"
        strategy, financial = backtest(reference, config, end)
        ensemble_rows.append({"fold": fold, "model": model, "seeds": len(frames),
                              **prediction_metrics(reference.actual_return,
                                                   reference.predicted_return),
                              **financial})
        ensemble_directory.mkdir(exist_ok=True)
        reference.to_csv(ensemble_directory / f"{fold}_{model}_predictions.csv", index=False)
        strategy.to_csv(ensemble_directory / f"{fold}_{model}_strategy.csv", index=False)
        for cost in costs:
            settings = {**config, "cost_bps": cost}
            for strategy_name in ("model", "buy_and_hold"):
                _, metrics = backtest(reference, settings, end,
                                      buy_and_hold=strategy_name == "buy_and_hold")
                rows.append({"fold": fold, "model": model, "seed": "ensemble",
                             "strategy": strategy_name, "cost_bps": cost, **metrics})
    if not rows:
        raise ValueError("No saved prediction files were found")
    pd.DataFrame(rows).to_csv(destination / "cost_sensitivity.csv", index=False)
    if ensemble_rows:
        pd.DataFrame(ensemble_rows).to_csv(destination / "seed_ensemble_metrics.csv", index=False)
    return destination
