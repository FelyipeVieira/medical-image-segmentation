"""Model-agnostic MONAI and foundation-model factory."""

from typing import Any

import torch.nn as nn
from monai.networks.nets import (
    AttentionUnet,
    BasicUNetPlusPlus,
    DynUNet,
    SegResNet,
    SwinUNETR,
    UNet,
    VNet,
)


class ModelFactory:
    """Construct supported 2D and 3D segmentation networks.

    Example:
        >>> model = ModelFactory.create("unet", 2, 1, 2)
    """

    @staticmethod
    def create(
        name: str,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        image_size: tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> nn.Module:
        """Create a supported convolutional or transformer segmentation model."""
        key = name.lower().replace("_", "").replace("+", "plus")
        common = {
            "spatial_dims": spatial_dims,
            "in_channels": in_channels,
            "out_channels": out_channels,
        }
        if key == "unet":
            return UNet(
                **common,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                **kwargs,
            )
        if key in {"unetplusplus"}:
            return BasicUNetPlusPlus(
                **common,
                features=(32, 32, 64, 128, 256, 32),
                **kwargs,
            )
        if key == "attentionunet":
            return AttentionUnet(
                **common, channels=(16, 32, 64, 128), strides=(2, 2, 2), **kwargs
            )
        if key == "vnet":
            return VNet(**common, **kwargs)
        if key == "segresnet":
            return SegResNet(**common, **kwargs)
        if key == "dynunet":
            kernels = [[3] * spatial_dims] * 4
            strides = [[1] * spatial_dims, *([[2] * spatial_dims] * 3)]
            return DynUNet(
                **common,
                kernel_size=kernels,
                strides=strides,
                upsample_kernel_size=strides[1:],
                **kwargs,
            )
        if key == "swinunetr":
            if image_size is None:
                raise ValueError("Swin UNETR requires image_size")
            return SwinUNETR(
                in_channels=in_channels,
                out_channels=out_channels,
                feature_size=24,
                **kwargs,
            )
        if key in {"sam", "medsam"}:
            raise NotImplementedError(
                "SAM/MedSAM require a checkpoint-specific adapter; install the "
                "sam extra and register an adapter"
            )
        raise ValueError(f"Unsupported model: {name}")
