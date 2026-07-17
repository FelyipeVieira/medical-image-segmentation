"""STL surface mesh export."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
from skimage.measure import marching_cubes


def save_stl(
    mask: npt.NDArray[np.generic],
    path: str | Path,
    spacing: tuple[float, float, float] = (1, 1, 1),
) -> Path:
    """Extract a marching-cubes surface and export it to STL."""
    try:
        import trimesh
    except ImportError as error:
        raise RuntimeError("Install the mesh extra for STL export") from error
    vertices, faces, _, _ = marching_cubes(np.asarray(mask), 0.5, spacing=spacing)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=vertices, faces=faces).export(destination)
    return destination
