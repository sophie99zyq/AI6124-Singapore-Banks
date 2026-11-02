"""Export result figures without requiring an interactive display."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(predictions, strategy, benchmark, path):
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(strategy.target_date, strategy.equity, label="Long or cash")
    axes[0].plot(benchmark.target_date, benchmark.equity, label="Buy and hold")
    axes[0].set_ylabel("Net wealth")
    axes[0].legend()
    axes[1].plot(strategy.target_date, strategy.drawdown)
    axes[1].set_ylabel("Drawdown")
    weights = [f"weight_{i}" for i in range(1, 4)]
    if set(weights).issubset(predictions):
        for name in weights:
            axes[2].plot(strategy.target_date, predictions[name], label=name)
        axes[2].legend()
        axes[2].set_ylabel("Expert weight")
    else:
        axes[2].plot(strategy.target_date, predictions.actual_return, alpha=0.5, label="Actual")
        axes[2].plot(strategy.target_date, predictions.predicted_return, alpha=0.7, label="Predicted")
        axes[2].set_ylabel("Log return")
        axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
