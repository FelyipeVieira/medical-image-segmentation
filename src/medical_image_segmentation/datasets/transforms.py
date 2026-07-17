"""MONAI training and validation transform pipelines."""

from monai.transforms import (
    Compose,
    EnsureTyped,
    RandAdjustContrastd,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandSpatialCropd,
)


def training_transforms(spatial_size: tuple[int, ...]) -> Compose:
    """Build flips, rotation/scaling/affine, crop, gamma, and noise augmentation."""
    return Compose(
        [
            EnsureTyped(keys=("image", "mask")),
            RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=0),
            RandAffined(
                keys=("image", "mask"),
                prob=0.3,
                rotate_range=0.2,
                scale_range=0.1,
                mode=("bilinear", "nearest"),
            ),
            RandSpatialCropd(
                keys=("image", "mask"), roi_size=spatial_size, random_size=False
            ),
            RandAdjustContrastd(keys="image", prob=0.2, gamma=(0.7, 1.5)),
            RandGaussianNoised(keys="image", prob=0.2, std=0.05),
        ]
    )
