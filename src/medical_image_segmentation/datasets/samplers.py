"""Sampling extension points for imbalanced datasets."""

from collections.abc import Sequence

from torch.utils.data import WeightedRandomSampler


def weighted_sampler(
    weights: Sequence[float], samples: int | None = None
) -> WeightedRandomSampler:
    """Create a replacement sampler from per-case weights."""
    return WeightedRandomSampler(weights, samples or len(weights), replacement=True)
