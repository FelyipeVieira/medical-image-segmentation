"""PNG and TIFF mask export."""

from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt


def save_mask(array: npt.NDArray[np.generic], path: str | Path) -> Path:
    """Save a 2D binary or multiclass mask as PNG/TIFF."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(array)
    if not cv2.imwrite(
        str(destination), values.astype(np.uint16 if values.max() > 255 else np.uint8)
    ):
        raise OSError(f"Could not export mask: {destination}")
    return destination
