"""Cleanup module for LA Fat Segmentation.

Post-processes the LA Fat mask by removing small islands (connected
components below a configurable volume threshold) and optionally
applying morphological opening and/or vessel filling (hole filling).
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_opening
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as connected_components

from la_fat.config import PipelineConfig


@dataclasses.dataclass(frozen=True)
class CleanupResult:
    """Result of cleaning the LA Fat mask.

    Attributes
    ----------
    cleaned_mask:
        The cleaned LA Fat binary mask (same shape as input).
    islands_removed:
        Number of small connected components that were removed.
    island_volumes_mm3:
        Volume in mm³ for each removed island (for reporting).
    total_removed_volume_mm3:
        Sum of all removed island volumes in mm³.
    morphological_opening_applied:
        Whether binary morphological opening was performed.
    vessel_filling_applied:
        Whether vessel filling (binary hole filling) was performed.
    """

    cleaned_mask: np.ndarray
    islands_removed: int
    island_volumes_mm3: list[float]
    total_removed_volume_mm3: float
    morphological_opening_applied: bool
    vessel_filling_applied: bool


def cleanup_la_fat_mask(
    la_fat_mask: np.ndarray,
    config: PipelineConfig,
    spacing: tuple[float, float, float] = (1.5, 1.5, 1.5),
    apply_opening: bool = False,
    apply_vessel_filling: bool = False,
) -> CleanupResult:
    """Clean the LA Fat mask by removing small islands and optionally
    applying morphological operations.

    Parameters
    ----------
    la_fat_mask:
        Binary LA Fat mask (any dtype that can be cast to bool).
    config:
        Pipeline configuration controlling ``min_fat_island_volume_mm3``.
    spacing:
        Voxel spacing in mm (z, y, x).  Used to convert voxel counts
        to physical volumes.
    apply_opening:
        If True, apply binary morphological opening with a small
        spherical structuring element (radius 1) before island removal.
    apply_vessel_filling:
        If True, apply binary hole filling after island removal.

    Returns
    -------
    CleanupResult
    """
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]

    # Ensure input is boolean.
    mask = np.asarray(la_fat_mask, dtype=bool)

    # ---- Optional: morphological opening ------------------------------------
    opening_applied = False
    if apply_opening and np.any(mask):
        se = generate_binary_structure(rank=3, connectivity=1)
        mask = binary_opening(mask, structure=se)
        opening_applied = True

    # ---- Step 1: Connected components --------------------------------------
    # Use 26-connectivity for 3D (3x3x3 structuring element with all
    # face, edge, and corner neighbours).
    structure_26 = generate_binary_structure(rank=3, connectivity=3)
    labeled, n_features = connected_components(mask, structure=structure_26)

    if n_features == 0 or np.count_nonzero(mask) == 0:
        return CleanupResult(
            cleaned_mask=mask,
            islands_removed=0,
            island_volumes_mm3=[],
            total_removed_volume_mm3=0.0,
            morphological_opening_applied=opening_applied,
            vessel_filling_applied=False,
        )

    # ---- Step 2: Identify small islands ------------------------------------
    threshold_mm3 = config.min_fat_island_volume_mm3
    cleaned = mask.copy()
    island_volumes: list[float] = []
    removed_count = 0

    # Component labels start at 1.
    for comp_id in range(1, n_features + 1):
        comp_mask = labeled == comp_id
        n_voxels = np.count_nonzero(comp_mask)
        volume_mm3 = n_voxels * voxel_volume_mm3

        if volume_mm3 < threshold_mm3:
            cleaned[comp_mask] = False
            island_volumes.append(float(volume_mm3))
            removed_count += 1

    total_removed = sum(island_volumes)

    # ---- Optional: vessel filling (hole filling) ---------------------------
    filling_applied = False
    if apply_vessel_filling and np.any(cleaned):
        cleaned = binary_fill_holes(cleaned)
        filling_applied = True

    return CleanupResult(
        cleaned_mask=cleaned,
        islands_removed=removed_count,
        island_volumes_mm3=island_volumes,
        total_removed_volume_mm3=float(total_removed),
        morphological_opening_applied=opening_applied,
        vessel_filling_applied=filling_applied,
    )
