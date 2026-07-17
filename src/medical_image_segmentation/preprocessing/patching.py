"""Overlapping patch extraction and reconstruction."""

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt


def extract_patches(
    array: npt.NDArray[np.generic], size: tuple[int, ...], stride: tuple[int, ...]
) -> Iterator[tuple[tuple[slice, ...], npt.NDArray[np.generic]]]:
    """Yield spatial slice coordinates and overlapping array patches."""
    if not (array.ndim == len(size) == len(stride)):
        raise ValueError("Array, patch size, and stride dimensions must match")
    starts = [
        range(0, max(1, dimension - patch + 1), step)
        for dimension, patch, step in zip(array.shape, size, stride, strict=True)
    ]
    for index in np.ndindex(*(len(item) for item in starts)):
        slices = tuple(
            slice(starts[axis][position], starts[axis][position] + size[axis])
            for axis, position in enumerate(index)
        )
        yield slices, array[slices]
