"""Robust intensity normalization."""

import numpy as np
import numpy.typing as npt


def percentile_normalize(
    array: npt.ArrayLike, lower: float = 0.5, upper: float = 99.5
) -> npt.NDArray[np.float32]:
    """Clip percentile outliers and z-score nonconstant intensities."""
    values = np.asarray(array, np.float32)
    low, high = np.percentile(values, (lower, upper))
    values = np.clip(values, low, high)
    return (
        np.zeros_like(values)
        if values.std() == 0
        else (values - values.mean()) / values.std()
    )
