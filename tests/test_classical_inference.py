"""Tests for deterministic CT demonstration segmentation."""

import numpy as np
import pytest

from medical_image_segmentation.inference.classical import segment_lungs_ct


def test_segment_lungs_excludes_border_air() -> None:
    """Only enclosed low-density components are retained."""
    image = np.full((64, 64), -1000.0, dtype=np.float32)
    image[8:56, 8:56] = 40.0
    image[18:46, 14:29] = -850.0
    image[18:46, 35:50] = -850.0
    mask = segment_lungs_ct(image)
    assert mask[30, 20]
    assert mask[30, 42]
    assert not mask[0, 0]
    assert not mask[12, 12]


def test_segment_lungs_requires_axial_slice() -> None:
    """Three-dimensional input is rejected explicitly."""
    with pytest.raises(ValueError, match="two-dimensional"):
        segment_lungs_ct(np.zeros((2, 8, 8)))
