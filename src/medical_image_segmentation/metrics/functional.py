"""NumPy segmentation metric implementations."""

import numpy as np
import numpy.typing as npt
from scipy.ndimage import binary_erosion, distance_transform_edt


def confusion(
    prediction: npt.ArrayLike, target: npt.ArrayLike
) -> tuple[int, int, int, int]:
    """Return true-positive, false-positive, false-negative, true-negative counts."""
    pred, truth = np.asarray(prediction, bool), np.asarray(target, bool)
    return (
        int(np.sum(pred & truth)),
        int(np.sum(pred & ~truth)),
        int(np.sum(~pred & truth)),
        int(np.sum(~pred & ~truth)),
    )


def surface_distances(
    prediction: npt.ArrayLike,
    target: npt.ArrayLike,
    spacing: tuple[float, ...] | None = None,
) -> npt.NDArray[np.float64]:
    """Return symmetric closest-point distances between binary surfaces."""
    pred, truth = np.asarray(prediction, bool), np.asarray(target, bool)
    if not pred.any() or not truth.any():
        return np.array([np.inf])
    pred_surface = pred ^ binary_erosion(pred)
    truth_surface = truth ^ binary_erosion(truth)
    return np.concatenate(
        [
            distance_transform_edt(~truth_surface, sampling=spacing)[pred_surface],
            distance_transform_edt(~pred_surface, sampling=spacing)[truth_surface],
        ]
    )


def segmentation_metrics(
    prediction: npt.ArrayLike,
    target: npt.ArrayLike,
    spacing: tuple[float, ...] | None = None,
) -> dict[str, float]:
    """Compute overlap, classification, surface, and volumetric metrics."""
    tp, fp, fn, tn = confusion(prediction, target)

    def safe(numerator: int, denominator: int) -> float:
        """Divide counts with perfect score for an empty, correct class."""
        return float(numerator / denominator) if denominator else 1.0

    distances = surface_distances(prediction, target, spacing)
    pred_count, target_count = tp + fp, tp + fn
    return {
        "dice": safe(2 * tp, 2 * tp + fp + fn),
        "iou": safe(tp, tp + fp + fn),
        "precision": safe(tp, tp + fp),
        "recall": safe(tp, tp + fn),
        "sensitivity": safe(tp, tp + fn),
        "specificity": safe(tn, tn + fp),
        "hausdorff": float(np.max(distances)),
        "assd": float(np.mean(distances)),
        "volumetric_similarity": 1.0
        - safe(abs(pred_count - target_count), pred_count + target_count),
    }
