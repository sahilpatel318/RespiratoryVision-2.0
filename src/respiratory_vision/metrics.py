from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    mean_bias: float


def regression_metrics(y_true, y_pred) -> RegressionMetrics:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have matching shapes.")
    if (
        truth.ndim != 1
        or not truth.size
        or not np.all(np.isfinite(truth))
        or not np.all(np.isfinite(pred))
    ):
        raise ValueError("Metrics require nonempty finite one-dimensional inputs.")
    error = pred - truth
    return RegressionMetrics(
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        mean_bias=float(np.mean(error)),
    )
