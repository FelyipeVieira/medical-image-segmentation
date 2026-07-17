"""Weighted composition of segmentation losses."""

from collections.abc import Sequence

from torch import Tensor, nn


class CombinedLoss(nn.Module):
    """Combine arbitrary objectives with explicit weights.

    Example:
        >>> loss = CombinedLoss([dice, focal], [0.7, 0.3])  # doctest: +SKIP
    """

    def __init__(self, losses: Sequence[nn.Module], weights: Sequence[float]) -> None:
        """Validate and register loss modules and weights."""
        super().__init__()
        if len(losses) != len(weights) or not losses:
            raise ValueError("Losses and weights must have equal nonzero length")
        self.losses = nn.ModuleList(losses)
        self.weights = tuple(weights)

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Return the weighted sum of configured losses."""
        return sum(
            weight * loss(prediction, target)
            for loss, weight in zip(self.losses, self.weights, strict=True)
        )
