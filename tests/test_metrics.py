"""Segmentation metric tests."""

import numpy as np

from medical_image_segmentation.metrics.functional import (
    confusion,
    segmentation_metrics,
)


def test_perfect_segmentation() -> None:
    """Perfect binary masks score one for overlap metrics."""
    mask = np.array([[0, 1], [1, 0]])
    metrics = segmentation_metrics(mask, mask)
    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["hausdorff"] == 0.0


def test_confusion_counts() -> None:
    """Confusion counts follow positive-mask semantics."""
    assert confusion([1, 1, 0, 0], [1, 0, 1, 0]) == (1, 1, 1, 1)
