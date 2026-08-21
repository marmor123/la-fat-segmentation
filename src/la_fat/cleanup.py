"""Cleanup module for LA Fat Segmentation.

Post-processes the LA Fat mask by removing small islands (connected
components below a configurable physical mm³ volume threshold) and optionally
applying morphological opening and/or vessel filling (hole filling).
"""

from __future__ import annotations

import dataclasses
from typing import Union

import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_opening
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import label as connected_components

from la_fat.config import PipelineConfig
from la_fat.image_ops import GridGeometry


@dataclasses.dataclass(frozen=True)
class CleanupConfig:
    """Configuration for morphological cleanup of fat masks.

    Attributes
    ----------
    min_fat_island_volume_mm3:
        Minimum physical volume in mm³ for connected components to keep (default 50.0).
    apply_opening:
        Whether to apply binary morphological opening (default False).
    apply_vessel_filling:
        Whether to apply binary hole/vessel filling (default False).
    """

    min_fat_island_volume_mm3: float = 50.0
    apply_opening: bool = False
    apply_vessel_filling: bool = False

    @classmethod
    def from_pipeline_config(cls, cfg: object) -> CleanupConfig:
        """Construct CleanupConfig from a PipelineConfig or dict."""
        if isinstance(cfg, dict):
            return cls(
                min_fat_island_volume_mm3=float(cfg.get("min_fat_island_volume_mm3", 50.0)),
                apply_opening=bool(cfg.get("apply_opening", False)),
                apply_vessel_filling=bool(cfg.get("apply_vessel_filling", False)),
            )
        return cls(
            min_fat_island_volume_mm3=float(getattr(cfg, "min_fat_island_volume_mm3", 50.0)),
            apply_opening=bool(getattr(cfg, "apply_opening", False)),
            apply_vessel_filling=bool(getattr(cfg, "apply_vessel_filling", False)),
        )


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

    @property
    def total_removed_volume_ml(self) -> float:
        """Sum of all removed island volumes in mL."""
        return self.total_removed_volume_mm3 / 1000.0


def cleanup_la_fat_mask(
    la_fat_mask: np.ndarray,
    config: Union[CleanupConfig, PipelineConfig, None] = None,
    spacing: tuple[float, float, float] | None = None,
    geometry: GridGeometry | None = None,
    apply_opening: bool | None = None,
    apply_vessel_filling: bool | None = None,
) -> CleanupResult:
    """Clean the LA Fat mask by removing small islands and optionally
    applying morphological operations.

    Parameters
    ----------
    la_fat_mask:
        Binary LA Fat mask (any dtype that can be cast to bool).
    config:
        CleanupConfig or PipelineConfig controlling ``min_fat_island_volume_mm3``.
    spacing:
        Voxel spacing in mm (z, y, x). Used if geometry is not provided.
    geometry:
        GridGeometry defining voxel spacing and volume.
    apply_opening:
        Override for morphological opening.
    apply_vessel_filling:
        Override for binary hole filling.

    Returns
    -------
    CleanupResult
    """
    if geometry is not None:
        voxel_volume_mm3 = geometry.voxel_volume_ml * 1000.0
    elif spacing is not None:
        voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    else:
        voxel_volume_mm3 = 1.5 * 1.5 * 1.5

    # Resolve config
    if config is None:
        cfg = CleanupConfig()
    elif isinstance(config, CleanupConfig):
        cfg = config
    elif isinstance(config, PipelineConfig):
        cfg = CleanupConfig.from_pipeline_config(config)
    else:
        cfg = CleanupConfig()

    eff_opening = apply_opening if apply_opening is not None else cfg.apply_opening
    eff_filling = (
        apply_vessel_filling if apply_vessel_filling is not None else cfg.apply_vessel_filling
    )

    # Ensure input is boolean.
    mask = np.asarray(la_fat_mask, dtype=bool)

    # ---- Optional: morphological opening ------------------------------------
    opening_applied = False
    if eff_opening and np.any(mask):
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
            cleaned_mask=mask.astype(np.uint8),
            islands_removed=0,
            island_volumes_mm3=[],
            total_removed_volume_mm3=0.0,
            morphological_opening_applied=opening_applied,
            vessel_filling_applied=False,
        )

    # ---- Step 2: Identify small islands ------------------------------------
    threshold_mm3 = cfg.min_fat_island_volume_mm3
    cleaned = mask.copy()
    island_volumes: list[float] = []
    removed_count = 0

    # Component labels start at 1.
    for comp_id in range(1, n_features + 1):
        comp_mask = labeled == comp_id
        n_voxels = np.count_nonzero(comp_mask)
        volume_mm3 = float(n_voxels * voxel_volume_mm3)

        if volume_mm3 < threshold_mm3:
            cleaned[comp_mask] = False
            island_volumes.append(volume_mm3)
            removed_count += 1

    total_removed = sum(island_volumes)

    # ---- Optional: vessel filling (hole filling) ---------------------------
    filling_applied = False
    if eff_filling and np.any(cleaned):
        cleaned = binary_fill_holes(cleaned)
        filling_applied = True

    return CleanupResult(
        cleaned_mask=cleaned.astype(np.uint8),
        islands_removed=removed_count,
        island_volumes_mm3=island_volumes,
        total_removed_volume_mm3=float(total_removed),
        morphological_opening_applied=opening_applied,
        vessel_filling_applied=filling_applied,
    )
