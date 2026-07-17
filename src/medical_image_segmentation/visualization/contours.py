"""Binary contour extraction."""

import numpy.typing as npt
from skimage.measure import find_contours


def contours(mask: npt.ArrayLike) -> list[npt.NDArray]:
    """Extract subpixel contours at the binary decision boundary."""
    return find_contours(mask, 0.5)
