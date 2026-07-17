"""Transformer attention visualization contracts."""

from medical_image_segmentation.explainability.gradcam import GradCAMProvider

AttentionProvider = GradCAMProvider
__all__ = ["AttentionProvider"]
