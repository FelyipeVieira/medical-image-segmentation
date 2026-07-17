"""Memory-efficient segmentation inference engine."""

from collections.abc import Sequence

import torch
from monai.inferers import sliding_window_inference
from torch import Tensor, nn


class InferenceEngine:
    """Run full-image or sliding-window inference with optional TTA.

    Example:
        >>> prediction = InferenceEngine(model).predict(image)  # doctest: +SKIP
    """

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        """Move a trained model to the selected inference device."""
        self.model = model.to(device).eval()
        self.device = torch.device(device)

    @torch.inference_mode()
    def predict(
        self,
        image: Tensor,
        roi_size: Sequence[int] | None = None,
        overlap: float = 0.25,
        tta: bool = False,
    ) -> Tensor:
        """Return probability maps, optionally averaging flipped predictions."""
        image = image.to(self.device)

        def infer(value: Tensor) -> Tensor:
            """Apply the selected full-image or sliding-window strategy."""
            if roi_size:
                return sliding_window_inference(
                    value, roi_size, 1, self.model, overlap=overlap
                )
            return self.model(value)

        logits = infer(image)
        if tta:
            for axis in range(2, image.ndim):
                logits += torch.flip(infer(torch.flip(image, [axis])), [axis])
            logits /= image.ndim - 1
        return (
            torch.softmax(logits, dim=1)
            if logits.shape[1] > 1
            else torch.sigmoid(logits)
        )
