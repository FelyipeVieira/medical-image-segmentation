"""Boundary-aware loss extension point."""

from monai.losses import HausdorffDTLoss

BoundaryLoss = HausdorffDTLoss
__all__ = ["BoundaryLoss"]
