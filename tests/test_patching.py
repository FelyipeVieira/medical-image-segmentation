"""Patch extraction tests."""

import numpy as np

from medical_image_segmentation.preprocessing.patching import extract_patches


def test_patch_extraction() -> None:
    """Overlapping extraction returns coordinates and expected shapes."""
    patches = list(extract_patches(np.zeros((4, 4)), (2, 2), (2, 2)))
    assert len(patches) == 4
    assert all(patch.shape == (2, 2) for _, patch in patches)
