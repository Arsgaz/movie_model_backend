"""Paired user-level percentile bootstrap for a fixed offline evaluation cohort."""

import numpy as np


def paired_bootstrap(values: np.ndarray, *, resamples: int = 2000, seed: int = 2026) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values):
        raise ValueError("Expected nonempty paired baseline/selected user observations")
    if not np.isfinite(values).all() or resamples < 100:
        raise ValueError("Use finite observations and at least 100 bootstrap resamples")
    rng = np.random.default_rng(seed)
    means = np.empty((resamples, 2))
    for start in range(0, resamples, 64):
        stop = min(start + 64, resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    result = {"users": len(values), "confidence_level": 0.95, "resamples": resamples}
    for index, label in enumerate(("popularity", "selected")):
        low, high = np.quantile(means[:, index], [0.025, 0.975])
        result[label] = {"mean": float(values[:, index].mean()), "low": float(low), "high": float(high)}
    low, high = np.quantile(means[:, 1] - means[:, 0], [0.025, 0.975])
    result["paired_difference"] = {
        "mean": float((values[:, 1] - values[:, 0]).mean()),
        "low": float(low),
        "high": float(high),
        "interval_excludes_zero": bool(low > 0 or high < 0),
    }
    return result
