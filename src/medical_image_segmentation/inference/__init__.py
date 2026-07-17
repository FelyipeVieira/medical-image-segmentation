"""Sliding-window and test-time-augmentation inference."""

from medical_image_segmentation.inference.classical import segment_lungs_ct
from medical_image_segmentation.inference.engine import InferenceEngine

__all__ = ["InferenceEngine", "segment_lungs_ct"]
