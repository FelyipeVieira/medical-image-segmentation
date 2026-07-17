"""NIfTI mask and probability-map export."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk


def save_nifti(
    array: npt.NDArray[np.generic], reference: sitk.Image, path: str | Path
) -> Path:
    """Save an array as NIfTI using reference physical geometry."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = sitk.GetImageFromArray(np.asarray(array))
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(destination), True)
    return destination
