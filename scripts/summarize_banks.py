"""Audit and summarize the explicitly selected historical bank experiments."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uob_regime.analysis import analyze_run

MODELS = {"zero", "persistence", "lstm", "fuzzy_feature", "fuzzy_moe",
          "equal_moe", "capacity_lstm", "dlinear", "anfis"}


def summarize(selection):
    """Keep source selection explicit; reject duplicates and incomplete coverage."""
    output = ROOT / "reports/multi_bank"
    output.mkdir(parents=True, exist_ok=True)
    tables, costs, ensembles, provenance = [], [], [], {}
    expected = {(str(year), model, seed) for year in range(2020, 2026)
                for model in MODELS
                for seed in ([11] if model in {"zero", "persistence"} else [11, 29, 47])}
    reference = None
    keys = ["lookback", "price_basis", "epochs", "batch_size", "patience",
            "learning_rate", "loss", "hidden_units", "entropy_strength",
            "signal_threshold", "cost_bps", "annualization", "risk_free_rate"]
    for bank, paths in selection.items():
        frames, sources = [], []
        for relative in paths:
            directory = ROOT / relative
            manifest = json.loads((directory / "manifest.json").read_text())
            if manifest["status"] != "complete" or manifest["mode"] != "historical":
                raise ValueError(f"Incomplete or non-historical run: {directory}")
            settings = {key: manifest["config"][key] for key in keys}
            if reference is None:
                reference = settings
            if settings != reference:
                raise ValueError(f"Training/comparison settings differ: {directory}")
            raw = ROOT / ("data/raw" if bank == "UOB" else f"banks/{bank.lower()}/data/raw")
            import hashlib
            for role in ("asset", "market"):
                digest = hashlib.sha256((raw / f"{role}.csv").read_bytes()).hexdigest()
                if digest != manifest[f"{role}_sha256"]:
                    raise ValueError(f"Data hash mismatch: {directory}/{role}")
            frame = pd.read_csv(directory / "metrics.csv", dtype={"fold": str})
            frames.append(frame)
            analysis = directory / "analysis"
            if not analysis.exists():
                analyze_run(directory)
            costs.append(pd.read_csv(analysis / "cost_sensitivity.csv").assign(bank=bank))
            if (analysis / "seed_ensemble_metrics.csv").exists():
                ensembles.append(pd.read_csv(analysis / "seed_ensemble_metrics.csv").assign(bank=bank))
            sources.append({"run": relative, "asset_sha256": manifest["asset_sha256"],
                            "market_sha256": manifest["market_sha256"],
                            "source_sha256": manifest["source_sha256"]})
        frame = pd.concat(frames, ignore_index=True)
        actual = set(zip(frame.fold, frame.model, frame.seed))
        if len(frame) != len(expected) or actual != expected:
            raise ValueError(f"{bank}: missing/duplicate/unexpected fold-model-seed rows")
        if frame.groupby("fold").n_predictions.nunique().max() != 1:
            raise ValueError(f"{bank}: models use different sample counts")
        tables.append(frame.assign(bank=bank))
        provenance[bank] = sources
    full = pd.concat(tables, ignore_index=True)
    full.to_csv(output / "all_metrics.csv", index=False)
    metrics = ["rmse", "mae", "directional_accuracy", "net_cumulative_return",
               "annualized_sharpe", "maximum_drawdown", "annual_turnover"]
    annual = full.groupby(["bank", "fold", "model"])[metrics].mean()
    annual.to_csv(output / "annual_metrics.csv")
    summary = annual.groupby(["bank", "model"]).mean().reset_index()
    summary.to_csv(output / "summary.csv", index=False)
    # This std measures variation between annual fold means, not statistical significance.
    annual.groupby(["bank", "model"]).std().to_csv(output / "annual_variability.csv")
    pd.concat(costs, ignore_index=True).to_csv(output / "cost_sensitivity.csv", index=False)
    pd.concat(ensembles, ignore_index=True).to_csv(output / "seed_ensemble_metrics.csv", index=False)
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, metric, title in zip(axes, ["rmse", "net_cumulative_return"],
                                ["Mean annual RMSE", "Mean annual-fold net return (10 bps)"]):
        pivot = summary.pivot(index="model", columns="bank", values=metric)
        if metric == "net_cumulative_return":
            pivot = pivot * 100
        pivot.plot.bar(ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=65)
        if metric == "net_cumulative_return":
            ax.set_ylabel("Percent")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    lines = ["# Three-bank development comparison", "",
             "All three banks: nine methods, six annual test folds (2020-2025), three seeds for learned models; 138 rows per bank. No 2026 test is included.", "",
             "Annual results are first averaged over seeds, then equally over years. Returns are means of separate annual portfolios, not a continuous six-year return. Banks share a market and calendar, so they are not three independent statistical replications.", "",
             "| Bank | Model | RMSE | MAE | Direction accuracy | Mean net return | Sharpe |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        sharpe = f"{row.annualized_sharpe:.3f}" if pd.notna(row.annualized_sharpe) else "N/A"
        lines.append(f"| {row.bank} | {row.model} | {row.rmse:.6f} | {row.mae:.6f} | {row.directional_accuracy:.2%} | {row.net_cumulative_return:.2%} | {sharpe} |")
    lines += ["", "## Buy-and-hold benchmark", ""]
    for bank, frame in full.groupby("bank"):
        benchmark = frame.groupby("fold").buy_hold_net_cumulative_return.first().mean()
        lines.append(f"- {bank}: mean annual-fold net return {benchmark:.2%}.")
    lines += ["", "## Per-bank findings", ""]
    for bank, frame in summary.groupby("bank"):
        best_error = frame.loc[frame.rmse.idxmin()]
        learned = frame[~frame.model.isin(["zero", "persistence"])]
        best_return = learned.loc[learned.net_cumulative_return.idxmax()]
        indexed = frame.set_index("model")
        delta = indexed.loc["fuzzy_moe", "rmse"] - indexed.loc["equal_moe", "rmse"]
        lines.append(f"- {bank}: lowest RMSE is {best_error.model} ({best_error.rmse:.6f}); strongest learned-model mean annual-fold return is {best_return.model} ({best_return.net_cumulative_return:.2%}). Fuzzy-MoE minus equal-MoE RMSE is {delta:+.8f} (negative favors fuzzy gating). These are descriptive comparisons, not significance tests.")
    lines += ["", "## Gate diagnostics", ""]
    for bank, frame in full[full.model == "fuzzy_moe"].groupby("bank"):
        entropy = frame.mean_gating_entropy.mean()
        lines.append(f"- {bank}: mean gate entropy {entropy:.6f}; uniform three-expert entropy is 1.098612. Compare against equal_moe before attributing benefits to learned gating.")
    lines += ["", "## Reproducibility", "",
              "Exact run IDs, data hashes and source hashes are retained in provenance.json. Training settings and raw hashes were checked; historical runs may have different source revisions, which are retained rather than rewritten. Sample counts agree across models within each bank/fold. Original UOB results remain UOB-specific.", "",
              "CSV files include annual metrics, between-year variability, cost scenarios and aligned arithmetic seed ensembles. The figure comparison.png compares RMSE and net returns. Final 2026 evaluation requires a separately frozen protocol."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(output)


if __name__ == "__main__":
    summarize(json.loads(Path(sys.argv[1]).read_text()))
