"""DICOM SEG export extension contract."""

from pathlib import Path
from typing import Any


def save_dicom_seg(mask: Any, source_instances: list[Path], path: str | Path) -> Path:
    """Declare DICOM SEG export while requiring a standards-aware backend."""
    raise NotImplementedError(
        "DICOM SEG requires coded segment metadata; register a "
        "highdicom-backed exporter"
    )
