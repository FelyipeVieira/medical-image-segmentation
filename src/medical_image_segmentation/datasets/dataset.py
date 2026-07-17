"""Manifest-driven multi-format segmentation dataset."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset[dict[str, Any]]):
    """Load paired images and masks from a CSV manifest.

    Example:
        >>> dataset = SegmentationDataset("data/manifest.csv")  # doctest: +SKIP
        >>> sample = dataset[0]  # doctest: +SKIP
    """

    def __init__(
        self, manifest: str | Path | pd.DataFrame, transform: Any = None
    ) -> None:
        """Initialize from columns ``image``, ``mask``, and optional ``id``."""
        self.frame = (
            pd.read_csv(manifest)
            if not isinstance(manifest, pd.DataFrame)
            else manifest.copy()
        )
        if missing := {"image", "mask"}.difference(self.frame.columns):
            raise ValueError(f"Missing manifest columns: {sorted(missing)}")
        self.transform = transform

    def __len__(self) -> int:
        """Return number of image/mask pairs."""
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Load and return one channel-first image/mask sample."""
        row = self.frame.iloc[index]
        image = sitk.GetArrayFromImage(sitk.ReadImage(str(row["image"]))).astype(
            np.float32
        )
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(row["mask"]))).astype(np.int64)
        sample: dict[str, Any] = {
            "image": image[None],
            "mask": mask,
            "id": str(row.get("id", index)),
        }
        if self.transform:
            sample = self.transform(sample)
        sample["image"] = torch.as_tensor(sample["image"], dtype=torch.float32)
        sample["mask"] = torch.as_tensor(sample["mask"], dtype=torch.long)
        return sample
