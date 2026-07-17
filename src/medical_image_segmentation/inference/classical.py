"""Deterministic classical segmentation for CT demonstration workflows."""

import numpy as np
import numpy.typing as npt
from scipy import ndimage


def segment_lungs_ct(
    image_hu: npt.ArrayLike,
    threshold_hu: float = -400.0,
    components: int = 2,
) -> npt.NDArray[np.bool_]:
    """Segment lung air spaces in one axial CT slice.

    The algorithm thresholds calibrated Hounsfield units, removes air connected
    to the image border, retains the largest enclosed air components, and uses
    morphology to produce a display-ready whole-lung mask. It is intended for
    deterministic examples and quality-control previews, not clinical use.

    Args:
        image_hu: Two-dimensional CT slice calibrated in Hounsfield units.
        threshold_hu: Upper HU threshold used to identify aerated lung.
        components: Maximum number of enclosed components to retain.

    Returns:
        Boolean mask with the same height and width as ``image_hu``.
    """
    image = np.asarray(image_hu)
    if image.ndim != 2:
        raise ValueError("segment_lungs_ct expects one two-dimensional CT slice")
    if components < 1:
        raise ValueError("components must be at least one")

    candidate = image < threshold_hu
    labels, count = ndimage.label(candidate)
    if count == 0:
        return np.zeros_like(candidate, dtype=bool)

    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    enclosed = candidate & ~np.isin(labels, border_labels)
    enclosed_labels, enclosed_count = ndimage.label(enclosed)
    if enclosed_count == 0:
        return np.zeros_like(candidate, dtype=bool)

    sizes = np.bincount(enclosed_labels.ravel())
    sizes[0] = 0
    retained = np.argsort(sizes)[-components:]
    mask = np.isin(enclosed_labels, retained)
    mask = ndimage.binary_closing(mask, iterations=3)
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=1)
    return np.asarray(mask, dtype=bool)
