"""Check temporal causality, independent scaling and execution accounting."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from uob_regime.backtest import backtest
from uob_regime.config import load_config
from uob_regime.data import align_daily_series
from uob_regime.features import FEATURES, build_features, build_samples
from uob_regime.metrics import prediction_metrics
from uob_regime.models import validate_model_names
from uob_regime.splits import FoldScaler, historical_folds


def synthetic_prices():
    """Create artificial business-day prices exclusively for software tests."""
    dates = pd.bdate_range("2017-01-02", "2022-12-30")
    rng = np.random.default_rng(123)
    close = 25 * np.exp(np.cumsum(rng.normal(0.0002, 0.008, len(dates))))
    opening = close * np.exp(rng.normal(0, 0.002, len(dates)))
    frame = pd.DataFrame({"Open": opening, "High": np.maximum(opening, close) * 1.005,
                          "Low": np.minimum(opening, close) * 0.995, "Close": close,
                          "Adj Close": close, "Volume": rng.integers(10000, 50000, len(dates))}, index=dates)
    frame.index.name = "Date"
    return frame


def test_config():
    config = load_config()
    config.update(study_start="2018-01-01", study_end="2022-12-30",
                  test_years=[2020, 2021], holdout_start="2022-01-01",
                  holdout_end="2022-12-30", final_validation_year=2021,
                  epochs=1, patience=1, hidden_units=4, seeds=[11], batch_size=128)
    return config


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.config = test_config()
        self.asset = synthetic_prices()
        self.market = self.asset.copy()
        self.frame = build_features(self.asset, self.market, self.config)
        self.samples = build_samples(self.frame, self.config)

    def test_future_prices_do_not_change_earlier_inputs(self):
        cutoff = pd.Timestamp("2020-06-01")
        changed = self.asset.copy()
        changed.loc[changed.index > cutoff, ["Open", "High", "Low", "Close", "Adj Close"]] *= 2
        rebuilt = build_features(changed, self.market, self.config)
        np.testing.assert_allclose(self.frame.loc[:cutoff, FEATURES], rebuilt.loc[:cutoff, FEATURES], equal_nan=True)

    def test_window_and_label_alignment(self):
        meta = self.samples.meta.iloc[30]
        day = self.asset.index.get_loc(meta.prediction_date)
        expected = np.log(self.asset.Close.iloc[day + 1] / self.asset.Close.iloc[day])
        self.assertAlmostEqual(self.samples.y[30], expected, places=7)
        np.testing.assert_allclose(self.samples.X[30, -1], self.frame.loc[meta.prediction_date, FEATURES].to_numpy(dtype=float))
        self.assertEqual(meta.target_date, self.asset.index[day + 1])

    def test_scaling_ignores_test_distribution(self):
        fold = historical_folds(self.samples.meta, self.config)[0]
        first = FoldScaler().fit(self.samples, fold.train)
        changed = copy.deepcopy(self.samples)
        changed.X[fold.test] += 10000
        changed.regime[fold.test] += 10000
        changed.y[fold.test] += 10000
        second = FoldScaler().fit(changed, fold.train)
        np.testing.assert_equal(first.features.mean_, second.features.mean_)
        np.testing.assert_equal(first.targets.mean_, second.targets.mean_)
        self.assertLess(self.samples.meta.target_date.iloc[fold.train].max(),
                        self.samples.meta.target_date.iloc[fold.validation].min())

    def test_next_open_execution_and_round_trip_cost(self):
        frame = pd.DataFrame({"prediction_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                              "target_date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
                              "exit_date": pd.to_datetime(["2020-01-03", "2020-01-06"]),
                              "entry_open": [100.0, 110.0], "exit_open": [110.0, 9000.0],
                              "target_close": [105.0, 121.0], "predicted_return": [1.0, 1.0]})
        config = {**self.config, "cost_bps": 100}
        ledger, metrics = backtest(frame, config, "2020-01-03")
        expected_wealth = 0.99 * 1.1 * 1.1 * 0.99
        self.assertAlmostEqual(ledger.equity.iloc[-1], expected_wealth)
        self.assertAlmostEqual(metrics["net_cumulative_return"], expected_wealth - 1)
        self.assertEqual(ledger.turnover.sum(), 2)
        self.assertEqual(ledger.exit_date.iloc[-1], pd.Timestamp("2020-01-03"))
        self.assertEqual(ledger.execution_exit_price.iloc[-1], 121)

    def test_zero_forecast_counts_nonpositive_days_as_correct_direction(self):
        metrics = prediction_metrics(np.array([-0.1, 0.0, 0.1]), np.zeros(3))
        self.assertAlmostEqual(metrics["directional_accuracy"], 2 / 3)

    def test_missing_market_dates_and_unknown_models_fail(self):
        with self.assertRaises(ValueError):
            build_features(self.asset, self.market.iloc[1:], self.config)
        with self.assertRaises(ValueError):
            validate_model_names(["invented_model"])

    def test_alignment_audits_zero_volume_and_missing_index_dates(self):
        asset = self.asset.copy()
        market = self.market.drop(index=asset.index[100])
        asset.loc[asset.index[101], "Volume"] = 0
        aligned_asset, aligned_market, report = align_daily_series(asset, market)
        self.assertEqual(len(aligned_asset), len(asset) - 2)
        self.assertTrue(aligned_asset.index.equals(aligned_market.index))
        self.assertEqual(report["excluded_missing_sti_dates"], [str(asset.index[100].date())])
        self.assertEqual(report["excluded_zero_asset_volume_dates"], [str(asset.index[101].date())])
        self.assertEqual(len(asset), len(self.asset))

    def test_saved_forecast_analysis_and_holdout_guard(self):
        from uob_regime.analysis import analyze_run
        from uob_regime.experiment import run_experiments

        with self.assertRaises(ValueError):
            run_experiments(self.config, mode="holdout")
        fold = historical_folds(self.samples.meta, self.config)[0]
        predictions = self.samples.meta.iloc[fold.test].copy()
        predictions["actual_return"] = self.samples.y[fold.test]
        predictions["predicted_return"] = 1.0
        with tempfile.TemporaryDirectory(prefix="ai6124_analysis_test_") as directory:
            root = Path(directory)
            for seed in (11, 29):
                leaf = root / f"2020/lstm/seed_{seed}"
                leaf.mkdir(parents=True)
                predictions.to_csv(leaf / "predictions.csv", index=False)
            (root / "manifest.json").write_text(json.dumps({"status": "complete", "config": self.config}))
            destination = analyze_run(root, costs=[0, 20])
            results = pd.read_csv(destination / "cost_sensitivity.csv")
            values = results[(results.strategy.eq("model")) &
                             (results.seed.astype(str) != "ensemble")].groupby(
                                 "cost_bps").net_cumulative_return.mean().sort_index()
            self.assertGreater(values.iloc[0], values.iloc[1])
            ensemble = pd.read_csv(destination / "seed_ensemble_metrics.csv")
            self.assertEqual(ensemble.seeds.iloc[0], 2)


@unittest.skipUnless(importlib.util.find_spec("tf_keras"), "Optional TensorFlow environment is unavailable")
class NeuralTests(unittest.TestCase):
    def test_dlinear_and_anfis_are_trainable(self):
        import tensorflow as tf
        import tf_keras as keras
        from uob_regime.models.neural import build_model

        keras.utils.set_random_seed(11)
        rng = np.random.default_rng(11)
        X = rng.normal(size=(16, 30, len(FEATURES))).astype("float32")
        z = rng.normal(size=(16, 3)).astype("float32")
        for name in ("dlinear", "anfis"):
            model, _, details = build_model(name, X, z, test_config())
            with tf.GradientTape() as tape:
                output = model([X, z])
                loss = tf.reduce_mean((output - 0.3) ** 2)
            gradients = tape.gradient(loss, model.trainable_variables)
            self.assertEqual(output.shape, (16, 1))
            self.assertGreater(details["parameters"], 0)
            self.assertTrue(all(g is not None and np.isfinite(g.numpy()).all()
                                for g in gradients))

    def test_gate_constraints_and_joint_gradients(self):
        import tensorflow as tf
        import tf_keras as keras
        from uob_regime.models.neural import build_model

        keras.utils.set_random_seed(11)
        rng = np.random.default_rng(11)
        X = rng.normal(size=(16, 30, len(FEATURES))).astype("float32")
        z = rng.normal(size=(16, 3)).astype("float32")
        model, diagnostic, _ = build_model("fuzzy_moe", X, z, test_config())
        with tf.GradientTape() as tape:
            output = model([X, z])
            loss = tf.reduce_mean((output - 0.3) ** 2)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.assertTrue(all(g is not None and np.isfinite(g.numpy()).all() for g in gradients))
        self.assertTrue(any(np.linalg.norm(g.numpy()) > 0 for g, variable in
                            zip(gradients, model.trainable_variables) if "tsk_gate" in variable.name))
        experts, weights = diagnostic([X, z])
        np.testing.assert_allclose(weights.numpy().sum(axis=1), 1, atol=1e-6)
        self.assertTrue((weights.numpy() >= 0).all())
        np.testing.assert_allclose(output.numpy().ravel(), (experts.numpy() * weights.numpy()).sum(axis=1), atol=1e-6)

    def test_all_models_train_and_save_in_an_isolated_synthetic_run(self):
        from uob_regime.analysis import analyze_run
        from uob_regime.experiment import run_experiments

        with tempfile.TemporaryDirectory(prefix="ai6124_synthetic_smoke_") as directory:
            root = Path(directory)
            (root / "data/raw").mkdir(parents=True)
            prices = synthetic_prices()
            for role in ("asset", "market"):
                prices.to_csv(root / "data/raw" / f"{role}.csv")
            config = test_config()
            output = run_experiments(config, years=[2020], root=root)
            metrics = pd.read_csv(output / "metrics.csv")
            self.assertEqual(set(metrics.model), set(config["models"]))
            self.assertTrue(np.isfinite(metrics.rmse).all())
            self.assertTrue((output / "seed_summary.csv").exists())
            analysis = analyze_run(output)
            self.assertTrue(list(analysis.glob("*_memberships.png")))
            self.assertTrue(list(analysis.glob("*_rules.csv")))

            refit = test_config()
            refit.update(models=["lstm"], seeds=[11], refit_epochs={"lstm": 1})
            heldout = run_experiments(
                refit, mode="holdout", root=root,
                frozen_protocol={"test": "synthetic fixed-epoch refit"})
            row = pd.read_csv(heldout / "metrics.csv").iloc[0]
            audit = json.loads((heldout / "final_holdout/lstm/seed_11/split_audit.json").read_text())
            self.assertEqual(row.epoch_selection, "fixed_from_pre_holdout_history")
            self.assertEqual(row.best_epoch, 1)
            self.assertEqual(audit["validation"]["samples"], 0)


if __name__ == "__main__":
    unittest.main()
