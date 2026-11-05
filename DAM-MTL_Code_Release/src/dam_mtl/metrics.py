from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.stats import pearsonr


def regression_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("Prediction and target must both have shape (n, 2)")

    values: Dict[str, float] = {}
    for index, axis in enumerate(("x", "y")):
        predicted = prediction[:, index]
        observed = target[:, index]
        if np.std(predicted) == 0 or np.std(observed) == 0:
            correlation = 0.0
        else:
            correlation = float(pearsonr(predicted, observed)[0])
        residual = predicted - observed
        denominator = float(np.sum((observed - np.mean(observed)) ** 2))
        values["cc_%s" % axis] = correlation
        values["rmse_%s" % axis] = float(np.sqrt(np.mean(residual ** 2)))
        values["r2_%s" % axis] = float(1.0 - np.sum(residual ** 2) / denominator) if denominator > 1e-12 else 0.0
        values["mae_%s" % axis] = float(np.mean(np.abs(residual)))

    for metric in ("cc", "rmse", "r2", "mae"):
        values["%s_avg" % metric] = float((values["%s_x" % metric] + values["%s_y" % metric]) / 2.0)
    return values
