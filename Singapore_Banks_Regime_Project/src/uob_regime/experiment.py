"""Reproducible fold runners with prediction, model and trading artifacts."""

import hashlib
import importlib.metadata
import json
import pickle
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import backtest
from .config import workspace_root
from .data import align_daily_series, read_prices
from .features import FEATURES, build_features, build_samples
from .metrics import prediction_metrics
from .models import validate_model_names
from .plots import plot_results
from .splits import FoldScaler, final_fold, historical_folds


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_samples(config, root=None):
    root = workspace_root(config) if root is None else Path(root)
    raw = Path(root) / "data/raw"
    for role in ("asset", "market"):
        if not (raw / f"{role}.csv").exists():
            raise FileNotFoundError(f"Missing {raw / (role + '.csv')}; run the download command first")
    asset, market, alignment = align_daily_series(
        read_prices(raw / "asset.csv"), read_prices(raw / "market.csv"))
    frame = build_features(asset, market, config)
    return frame, build_samples(frame, config), alignment


def prepare(config, root=None):
    root = workspace_root(config) if root is None else Path(root)
    frame, samples, alignment = load_samples(config, root)
    destination = Path(root) / "data/processed"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "alignment.json").write_text(json.dumps(alignment, indent=2))
    frame.to_csv(destination / "features.csv", index_label="Date")
    samples.meta.assign(actual_return=samples.y).to_csv(destination / "samples.csv", index=False)
    report = []
    for fold in historical_folds(samples.meta, config) + [final_fold(samples.meta, config)]:
        for name in ("train", "validation", "test"):
            indices = getattr(fold, name)
            first = samples.meta.target_date.iloc[indices].min() if len(indices) else None
            last = samples.meta.target_date.iloc[indices].max() if len(indices) else None
            report.append({"fold": fold.name, "partition": name, "samples": len(indices),
                           "first_target": str(first.date()) if first is not None else None,
                           "last_target": str(last.date()) if last is not None else None})
    pd.DataFrame(report).to_csv(destination / "folds.csv", index=False)
    metadata = {"feature_names": FEATURES, "n_samples": len(samples.y),
                "config": config, "asset_sha256": file_hash(Path(root) / "data/raw/asset.csv"),
                "market_sha256": file_hash(Path(root) / "data/raw/market.csv")}
    (destination / "preparation.json").write_text(json.dumps(metadata, indent=2))
    return pd.DataFrame(report)


def _fit_neural(name, samples, fold, config, seed, destination, fixed_epochs=None):
    import tensorflow as tf
    import tf_keras as keras

    from .models.neural import build_model

    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    scaler = FoldScaler().fit(samples, fold.train)
    X, z, y = scaler.transform(samples, fold.train)
    Xt, zt, _ = scaler.transform(samples, fold.test)
    model, diagnostics, details = build_model(name, X, z, config)
    started = time.perf_counter()
    if fixed_epochs is None:
        Xv, zv, yv = scaler.transform(samples, fold.validation)
        history = model.fit([X, z], y, validation_data=([Xv, zv], yv),
                            epochs=config["epochs"], batch_size=config["batch_size"],
                            shuffle=False, verbose=2,
                            callbacks=[keras.callbacks.EarlyStopping(
                                monitor="val_loss", patience=config["patience"],
                                restore_best_weights=True), keras.callbacks.TerminateOnNaN()])
        losses = history.history["val_loss"]
        details["best_epoch"] = int(np.argmin(losses)) + 1
        details["epoch_selection"] = "validation_early_stopping"
    else:
        history = model.fit([X, z], y, epochs=fixed_epochs,
                            batch_size=config["batch_size"], shuffle=False, verbose=2,
                            callbacks=[keras.callbacks.TerminateOnNaN()])
        losses = history.history["loss"]
        details["best_epoch"] = fixed_epochs
        details["epoch_selection"] = "fixed_from_pre_holdout_history"
    details["training_seconds"] = time.perf_counter() - started
    if not np.isfinite(losses).all():
        raise ValueError("Non-finite training history; this run is not a valid result")
    predictions = scaler.inverse_target(model.predict([Xt, zt], verbose=0))
    pd.DataFrame(history.history).to_csv(destination / "history.csv", index=False)
    model.save_weights(str(destination / "model.weights.h5"))
    with (destination / "scaler.pkl").open("wb") as stream:
        pickle.dump(scaler, stream)
    extra = {}
    if diagnostics is not None:
        expert_predictions, weights = diagnostics.predict([Xt, zt], verbose=0)
        for i in range(3):
            extra[f"expert_{i + 1}_return"] = scaler.inverse_target(expert_predictions[:, i])
            extra[f"weight_{i + 1}"] = weights[:, i]
        details["mean_gating_entropy"] = float(np.mean(-np.sum(weights * np.log(weights + 1e-8), axis=1)))
        details["mean_expert_weights"] = weights.mean(axis=0).astype(float).tolist()
    for layer in model.layers:
        if layer.name in {"tsk_gate", "fuzzy_features", "anfis"}:
            widths = tf.nn.softplus(layer.raw_widths).numpy() + 1e-3
            np.savez(destination / f"{layer.name}.npz", centers=layer.centers.numpy(), widths=widths,
                     **({"consequents": layer.consequents.numpy()} if hasattr(layer, "consequents") else {}))
    return predictions, extra, details


def run_experiments(config, mode="historical", years=None, root=None, frozen_protocol=None):
    """Evaluate each model on identical samples and retain failures explicitly."""
    root = workspace_root(config) if root is None else Path(root)
    if mode not in {"historical", "holdout"}:
        raise ValueError("mode must be historical or holdout")
    if mode == "holdout" and frozen_protocol is None:
        raise ValueError("Final evaluation requires an explicitly frozen protocol")
    validate_model_names(config["models"])
    _, samples, alignment = load_samples(config, root)
    folds = historical_folds(samples.meta, config) if mode == "historical" else [final_fold(samples.meta, config)]
    if years:
        folds = [fold for fold in folds if fold.name in {str(year) for year in years}]
    if not folds:
        raise ValueError("No folds selected")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = Path(root) / "results" / f"{mode}_{stamp}_{uuid.uuid4().hex[:6]}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "mode": mode, "config": config,
                "started_at_utc": stamp, "python": platform.python_version(),
                "feature_names": FEATURES, "frozen_protocol": frozen_protocol,
                "asset_sha256": file_hash(Path(root) / "data/raw/asset.csv"),
                "market_sha256": file_hash(Path(root) / "data/raw/market.csv"),
                "alignment": alignment,
                "source_sha256": {str(path.relative_to(Path(__file__).parent)): file_hash(path)
                                  for path in sorted(Path(__file__).parent.rglob("*.py"))},
                "price_warning": "Adjusted OHLC is a research total-return proxy, not a broker ledger.",
                "versions": {name: importlib.metadata.version(name) for name in
                             ("numpy", "pandas", "scikit-learn", "matplotlib")}}
    for package in ("tensorflow", "tf-keras", "yfinance"):
        try:
            manifest["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    results = []
    try:
        for fold in folds:
            for name in config["models"]:
                # Deterministic forecasts do not need duplicate seed runs.
                seeds = config["seeds"][:1] if name in {"zero", "persistence"} else config["seeds"]
                for seed in seeds:
                    destination = output / fold.name / name / f"seed_{seed}"
                    destination.mkdir(parents=True)
                    print(f"Running fold={fold.name} model={name} seed={seed}", flush=True)
                    audit = {}
                    for partition in ("train", "validation", "test"):
                        idx = getattr(fold, partition)
                        first = samples.meta.target_date.iloc[idx].min() if len(idx) else None
                        last = samples.meta.target_date.iloc[idx].max() if len(idx) else None
                        audit[partition] = {"samples": len(idx),
                            "first_target": str(first.date()) if first is not None else None,
                            "last_target": str(last.date()) if last is not None else None}
                    (destination / "split_audit.json").write_text(json.dumps(audit, indent=2))
                    if name in {"zero", "persistence"}:
                        predicted = (np.zeros(len(fold.test)) if name == "zero" else
                                     samples.meta.observed_return.iloc[fold.test].to_numpy())
                        extra, details = {}, {"parameters": 0, "training_seconds": 0}
                    else:
                        fixed = (config["refit_epochs"][name]
                                 if mode == "holdout" and config.get("final_refit", False)
                                 else None)
                        predicted, extra, details = _fit_neural(
                            name, samples, fold, config, seed, destination, fixed_epochs=fixed)
                    predictions = samples.meta.iloc[fold.test].reset_index(drop=True).copy()
                    predictions["actual_return"] = samples.y[fold.test]
                    predictions["predicted_return"] = predicted
                    for key, value in extra.items():
                        predictions[key] = value
                    predictions.to_csv(destination / "predictions.csv", index=False)
                    strategy, financial = backtest(predictions, config, fold.test_end)
                    benchmark, benchmark_metrics = backtest(predictions, config, fold.test_end, buy_and_hold=True)
                    strategy.to_csv(destination / "strategy.csv", index=False)
                    benchmark.to_csv(destination / "buy_and_hold.csv", index=False)
                    plot_results(predictions, strategy, benchmark, destination / "overview.png")
                    row = {"fold": fold.name, "model": name, "seed": seed,
                           **prediction_metrics(samples.y[fold.test], predicted), **financial, **details,
                           **{f"buy_hold_{key}": value for key, value in benchmark_metrics.items()}}
                    (destination / "metrics.json").write_text(json.dumps(row, indent=2, allow_nan=False))
                    results.append(row)
                    pd.DataFrame(results).to_csv(output / "metrics.csv", index=False)
        table = pd.DataFrame(results)
        numeric = [col for col in table.select_dtypes(include="number") if col != "seed"]
        summary = table.groupby(["fold", "model"])[numeric].agg(["mean", "std"])
        summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
        summary.to_csv(output / "seed_summary.csv")
        manifest["status"] = "complete"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
        print(f"Artifacts: {output}", flush=True)
    return output
