"""Typed training and inference settings."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """Reproducible experiment configuration.

    Example:
        >>> config = ExperimentConfig(model="unet", spatial_dims=3)
        >>> config.epochs
        100
    """

    model: str = "unet"
    spatial_dims: Literal[2, 3] = 3
    in_channels: int = Field(default=1, ge=1)
    out_channels: int = Field(default=2, ge=1)
    image_size: tuple[int, ...] = (96, 96, 96)
    batch_size: int = Field(default=2, ge=1)
    epochs: int = Field(default=100, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    num_workers: int = Field(default=4, ge=0)
    precision: str = "16-mixed"
    seed: int = 42
    manifest: Path = Path("data/manifest.csv")
    output_directory: Path = Path("outputs")
