"""Shared transformations for preference estimation and simulation."""

from __future__ import annotations

import numpy as np


def log_count(values):
    """Return ``log(max(count, 1))`` for scalar or array-like counts.

    Counts equal to zero are mapped to zero. This implements the logged wish-count
    interactions reported in the paper while remaining defined for candidates with
    no wishes in the relevant program category.
    """
    array = np.asarray(values, dtype=float)
    transformed = np.log(np.maximum(array, 1.0))
    return float(transformed) if transformed.ndim == 0 else transformed
