"""Predictive metrics are measured on unscaled daily log returns."""

import numpy as np


def prediction_metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or not len(actual):
        raise ValueError("Predictions and labels must be nonempty and aligned")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Non-finite labels or predictions")
    error = predicted - actual
    # Treat zero as non-positive, matching the long-or-cash signal rule.
    return {"rmse": float(np.sqrt(np.mean(error ** 2))),
            "mae": float(np.mean(np.abs(error))),
            "directional_accuracy": float(np.mean((actual > 0) == (predicted > 0))),
            "n_predictions": len(actual)}
