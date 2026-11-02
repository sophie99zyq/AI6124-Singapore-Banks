"""Chronological folds and train-only preprocessing."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class Fold:
    name: str
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    test_end: pd.Timestamp


def make_fold(meta, name, validation_start, test_start, test_end):
    target = pd.to_datetime(meta.target_date)
    prediction = pd.to_datetime(meta.prediction_date)
    val, start, end = map(pd.Timestamp, (validation_start, test_start, test_end))
    train = np.flatnonzero(target.lt(val))
    validation = np.flatnonzero(target.ge(val) & target.lt(start))
    test = np.flatnonzero(target.ge(start) & target.le(end))
    if not all(len(indices) for indices in (train, validation, test)):
        raise ValueError(f"{name}: empty training, validation or test partition")
    # Previous labels must already be observable when the next partition is predicted.
    if target.iloc[train].max() > prediction.iloc[validation].min():
        raise ValueError("Training labels overlap the validation prediction time")
    if target.iloc[validation].max() > prediction.iloc[test].min():
        raise ValueError("Validation labels overlap the test prediction time")
    return Fold(name, train, validation, test, end)


def historical_folds(meta, config):
    return [make_fold(meta, str(year), f"{year - 1}-01-01", f"{year}-01-01",
                      f"{year}-12-31") for year in config["test_years"]]


def final_fold(meta, config):
    if config.get("final_refit", False):
        target = pd.to_datetime(meta.target_date)
        prediction = pd.to_datetime(meta.prediction_date)
        start, end = map(pd.Timestamp, (config["holdout_start"], config["holdout_end"]))
        train = np.flatnonzero(target.lt(start))
        test = np.flatnonzero(target.ge(start) & target.le(end))
        if not len(train) or not len(test):
            raise ValueError("Final refit requires nonempty training and test partitions")
        if target.iloc[train].max() > prediction.iloc[test].min():
            raise ValueError("Final refit labels overlap the first holdout prediction time")
        return Fold("final_holdout", train, np.array([], dtype=int), test, end)
    return make_fold(meta, "final_holdout", f"{config['final_validation_year']}-01-01",
                     config["holdout_start"], config["holdout_end"])


class FoldScaler:
    """Fit all statistics on training windows, regimes and labels only."""

    def fit(self, samples, indices):
        X = samples.X[indices]
        self.features = StandardScaler().fit(X.reshape(-1, X.shape[-1]))
        self.regimes = StandardScaler().fit(samples.regime[indices])
        self.targets = StandardScaler().fit(samples.y[indices, None])
        return self

    def transform(self, samples, indices):
        X = samples.X[indices]
        return (self.features.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype("float32"),
                self.regimes.transform(samples.regime[indices]).astype("float32"),
                self.targets.transform(samples.y[indices, None]).astype("float32"))

    def inverse_target(self, predictions):
        return self.targets.inverse_transform(np.asarray(predictions).reshape(-1, 1)).ravel()
