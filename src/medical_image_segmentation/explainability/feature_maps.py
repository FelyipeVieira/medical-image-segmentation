"""Intermediate feature-map collection."""

from torch import Tensor, nn


def capture_feature_map(module: nn.Module) -> tuple[list[Tensor], object]:
    """Register a forward hook and return its live output list and handle."""
    outputs: list[Tensor] = []
    handle = module.register_forward_hook(
        lambda _module, _inputs, output: outputs.append(output.detach())
    )
    return outputs, handle
