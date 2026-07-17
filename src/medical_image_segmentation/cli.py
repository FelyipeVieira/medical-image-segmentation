"""Training, prediction, and evaluation command entry points."""

import argparse
import json
from pathlib import Path

import numpy as np

from medical_image_segmentation.metrics.functional import segmentation_metrics


def train_main() -> None:
    """Parse a configuration path for the training workflow."""
    parser = argparse.ArgumentParser(description="Train a medical segmentation model")
    parser.add_argument("config", type=Path)
    parser.parse_args()
    raise SystemExit(
        "Training configuration accepted; connect project-specific train/validation "
        "manifests in src/train.py"
    )


def predict_main() -> None:
    """Parse model and image arguments for prediction."""
    parser = argparse.ArgumentParser(description="Run segmentation inference")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/prediction.nii.gz")
    )
    parser.parse_args()
    raise SystemExit(
        "Use InferenceEngine with the same model configuration used for training"
    )


def evaluate_main() -> None:
    """Evaluate two NumPy binary masks and write JSON metrics."""
    parser = argparse.ArgumentParser(description="Evaluate segmentation masks")
    parser.add_argument("prediction", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/metrics.json"))
    args = parser.parse_args()
    result = segmentation_metrics(np.load(args.prediction), np.load(args.target))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
