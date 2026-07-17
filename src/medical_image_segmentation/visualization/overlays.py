"""Segmentation overlay and difference-map rendering."""

import numpy as np
import numpy.typing as npt


def overlay(
    image: npt.ArrayLike, mask: npt.ArrayLike, alpha: float = 0.4
) -> npt.NDArray[np.float32]:
    """Blend a red binary mask over a normalized grayscale image."""
    values = np.asarray(image, np.float32)
    values = (values - values.min()) / max(float(values.max() - values.min()), 1e-8)
    rgb = np.repeat(values[..., None], 3, axis=-1)
    foreground = np.asarray(mask) > 0
    rgb[foreground] = (1 - alpha) * rgb[foreground] + alpha * np.array([1, 0, 0])
    return rgb


def difference_map(
    prediction: npt.ArrayLike, target: npt.ArrayLike
) -> npt.NDArray[np.int8]:
    """Return -1 for false negatives, +1 for false positives, and 0 otherwise."""
    return np.asarray(prediction, bool).astype(np.int8) - np.asarray(
        target, bool
    ).astype(np.int8)
