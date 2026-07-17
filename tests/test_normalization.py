"""Intensity normalization tests."""

import numpy as np

from medical_image_segmentation.preprocessing.normalization import percentile_normalize


def test_constant_normalization_is_finite() -> None:
    """Constant images normalize to finite zero arrays."""
    assert np.array_equal(percentile_normalize(np.ones((3, 3))), np.zeros((3, 3)))
