"""Deterministic PyTorch data-loader construction."""

from typing import Any

from torch.utils.data import DataLoader, Dataset


def create_loader(
    dataset: Dataset[Any], batch_size: int, shuffle: bool, workers: int = 0
) -> DataLoader[Any]:
    """Create a pinned-memory loader with persistent workers when available."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
