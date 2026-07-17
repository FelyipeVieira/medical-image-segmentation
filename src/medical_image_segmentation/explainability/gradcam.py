"""Gradient-based activation-map contract."""

from typing import Protocol

from torch import Tensor


class GradCAMProvider(Protocol):
    """Model-specific Grad-CAM adapter contract.

    Example:
        >>> heatmap = provider.compute(image, class_index=1)  # doctest: +SKIP
    """

    def compute(self, image: Tensor, class_index: int) -> Tensor:
        """Return an activation heatmap aligned with the input."""
        ...
