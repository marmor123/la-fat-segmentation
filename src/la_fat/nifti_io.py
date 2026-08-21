"""Unified NIfTI I/O for LA Fat Segmentation.

Provides a single pair of functions (``save_nifti`` / ``load_nifti``) that
replace the two divergent private ``_save_nifti`` implementations previously
scattered across ``pipeline.py`` and ``mesh_extractor.py``.

The pipeline version correctly set spacing, origin, **and** direction.
The mesh_extractor version only set spacing and also reordered axes
(``(sx, sy, sz)`` → ``(sz, sy, sx)``), producing NIfTIs with inconsistent
spatial metadata.  This module fixes that bug.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import SimpleITK as sitk

from la_fat.image_ops import GridGeometry

__all__ = [
    "load_nifti",
    "load_nifti_with_geometry",
    "save_nifti",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_nifti(
    mask: np.ndarray,
    path: str | pathlib.Path,
    spacing: tuple[float, float, float] | None = None,
    origin: tuple[float, float, float] | None = None,
    direction: np.ndarray | None = None,
    geometry: GridGeometry | None = None,
) -> None:
    """Save a numpy array as a NIfTI file with full spatial metadata.

    Parameters
    ----------
    mask:
        3D array to save.
    path:
        Output path (``.nii.gz`` or ``.nii``).  Parent directories are
        created automatically.
    spacing:
        Voxel spacing ``(sx, sy, sz)`` in mm.
    origin:
        World-coordinate origin ``(ox, oy, oz)``.  Defaults to
        ``(0.0, 0.0, 0.0)``.
    direction:
        3×3 direction cosine matrix.  Defaults to identity.
    geometry:
        Optional GridGeometry instance. If provided, spacing, origin,
        and direction are extracted from it.
    """
    if geometry is not None:
        spacing = geometry.spacing
        origin = geometry.origin
        direction = geometry.direction

    if spacing is None:
        spacing = (1.0, 1.0, 1.0)

    img = sitk.GetImageFromArray(mask)
    img.SetSpacing(spacing)

    if origin is not None:
        img.SetOrigin(origin)

    if direction is not None:
        img.SetDirection(direction.ravel().tolist())

    path_str = str(path)
    os.makedirs(os.path.dirname(path_str), exist_ok=True)
    sitk.WriteImage(img, path_str)


def load_nifti(path: str | pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file, returning the array and its 4×4 affine matrix.

    Parameters
    ----------
    path:
        Path to the NIfTI file to load.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(array, affine)`` where *array* is the 3D voxel data and
        *affine* is the 4×4 world-coordinate affine matrix constructed
        from the SimpleITK spacing / origin / direction metadata.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    array, geometry = load_nifti_with_geometry(path)
    return (array, geometry.to_affine())


def load_nifti_with_geometry(path: str | pathlib.Path) -> tuple[np.ndarray, GridGeometry]:
    """Load a NIfTI file, returning the array and its typed GridGeometry.

    Parameters
    ----------
    path:
        Path to the NIfTI file to load.

    Returns
    -------
    tuple[np.ndarray, GridGeometry]
        ``(array, geometry)``
    """
    path_str = str(path)
    if not os.path.isfile(path_str):
        raise FileNotFoundError(f"NIfTI file not found: {path_str}")

    img = sitk.ReadImage(path_str)
    array: np.ndarray = sitk.GetArrayFromImage(img)
    geometry = GridGeometry.from_sitk_image(img)
    return (array, geometry)
