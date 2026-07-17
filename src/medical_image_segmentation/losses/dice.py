"""Soft Dice loss for binary and multiclass segmentation."""

import torch
from torch import Tensor, nn


class DiceLoss(nn.Module):
    """Compute one minus soft Dice overlap.

    Example:
        >>> loss = DiceLoss()(prediction, target)  # doctest: +SKIP
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        """Initialize numeric stabilization constant."""
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """Return mean channel-wise Dice loss from logits and one-hot targets."""
        probabilities = (
            torch.softmax(logits, dim=1)
            if logits.shape[1] > 1
            else torch.sigmoid(logits)
        )
        dimensions = tuple(range(2, logits.ndim))
        intersection = torch.sum(probabilities * target, dim=dimensions)
        denominator = torch.sum(probabilities + target, dim=dimensions)
        return (
            1 - ((2 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        )
