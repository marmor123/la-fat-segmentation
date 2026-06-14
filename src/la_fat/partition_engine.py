"""Partition engine for LA Fat Segmentation.

Computes a distance-based partition of epicardial fat voxels to the
nearest of six anatomical anchor structures, replacing the four heuristic
steps from the legacy pipeline.

Domain
------
**Partition Anchors** = {LA, LV, RA, RV, Aorta, Pulmonary Artery} —
exactly 6 structures.  Pulmonary Veins are intentionally excluded (their
fat belongs to LA).

**Surface distance, not centroid distance** — each fat voxel is assigned
to the anchor whose **surface** is nearest, not whose centroid is
nearest.  This is the key design decision from ADR-0001.
"""

from __future__ import annotations

import dataclasses
import typing as t

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.ndimage import distance_transform_edt

from la_fat.anatomy import CANONICAL_ANCHORS, voxel_volume_ml
from la_fat.config import PipelineConfig

# 6-connectivity structuring element for surface extraction.
# A voxel is on the surface if it has at least one zero-valued face
# neighbour.
_SURFACE_STRUCTURE: np.ndarray = np.zeros((3, 3, 3), dtype=bool)
_SURFACE_STRUCTURE[1, 1, :] = True
_SURFACE_STRUCTURE[1, :, 1] = True
_SURFACE_STRUCTURE[:, 1, 1] = True


@dataclasses.dataclass(frozen=True)
class PartitionResult:
    """Result of partitioning epicardial fat by nearest anchor surface.

    Attributes
    ----------
    la_fat_mask:
        Binary mask of LA Fat (same shape as inputs).
    all_fat_mask:
        Binary mask of ALL epicardial fat (all anchors combined, for QA).
    anchor_assignments:
        Integer label map: 0=background, 1=LA, 2=LV, 3=RA, 4=RV, 5=Aorta,
        6=Pulmonary_Artery -- for every fat voxel inside the pericardium.
    anchor_volumes_ml:
        Volume in ml for each anchor (including excluded anchors as 0.0).
    anchor_shares:
        Percentage of total fat assigned to each anchor.
    unassigned_volume_ml:
        Fat voxels inside pericardium but too far from all anchors to
        assign (beyond *max_assign_distance_mm*).
    total_fat_volume_ml:
        Total epicardial fat volume (assigned + unassigned).
    excluded_anchors:
        Anchors excluded due to missing mask or below minimum volume.
    exclusion_reasons:
        Human-readable explanation for each excluded anchor.
    """

    la_fat_mask: np.ndarray
    all_fat_mask: np.ndarray
    anchor_assignments: np.ndarray
    anchor_volumes_ml: dict[str, float]
    anchor_shares: dict[str, float]
    unassigned_volume_ml: float
    total_fat_volume_ml: float
    excluded_anchors: list[str]
    exclusion_reasons: dict[str, str]


def partition_fat(
    ct_array: np.ndarray,
    pericardium_mask: np.ndarray,
    fat_hu_range: tuple[float, float],
    anchor_masks: dict[str, np.ndarray],
    config: PipelineConfig,
    spacing: tuple[float, float, float] = (1.5, 1.5, 1.5),
    max_assign_distance_mm: float = 30.0,
) -> PartitionResult:
    """Partition epicardial fat by assigning each fat voxel to the
    nearest anchor surface among the six canonical Partition Anchors.

    Parameters
    ----------
    ct_array:
        3D CT volume in Hounsfield Units (z, y, x).
    pericardium_mask:
        3D binary mask of the pericardium (same shape as *ct_array*).
    fat_hu_range:
        ``(hu_low, hu_high)`` -- the Hounsfield Unit range defining fat.
    anchor_masks:
        Dictionary mapping structure names to binary mask arrays.
        Only the six canonical Partition Anchors are used
        (LA, LV, RA, RV, Aorta, Pulmonary_Artery).
    config:
        Pipeline configuration controlling ``min_anchor_volume_ml``.
    spacing:
        Voxel spacing in mm (isotropic expected).
    max_assign_distance_mm:
        Maximum physical distance (mm) from an anchor surface for a
        fat voxel to be considered assigned.  Beyond this threshold
        the voxel is marked unassigned.

    Returns
    -------
    PartitionResult

    Raises
    ------
    ValueError
        If fewer than 2 valid anchors remain after filtering.
    """
    voxel_vol_ml = voxel_volume_ml(spacing)
    shape = ct_array.shape

    # ---- Step 1: Validate and filter anchors -------------------------------
    valid_anchors: list[str] = []
    excluded_anchors: list[str] = []
    exclusion_reasons: dict[str, str] = {}
    anchor_label: dict[str, int] = {}

    for anchor_name in CANONICAL_ANCHORS:
        if anchor_name not in anchor_masks:
            excluded_anchors.append(anchor_name)
            exclusion_reasons[anchor_name] = "mask not provided"
            continue

        mask = anchor_masks[anchor_name]
        if mask is None or np.count_nonzero(mask) == 0:
            excluded_anchors.append(anchor_name)
            exclusion_reasons[anchor_name] = "mask is empty"
            continue

        volume_ml = np.count_nonzero(mask) * voxel_vol_ml
        if volume_ml < config.min_anchor_volume_ml:
            excluded_anchors.append(anchor_name)
            exclusion_reasons[anchor_name] = (
                f"volume {volume_ml:.2f} ml < "
                f"{config.min_anchor_volume_ml:.1f} ml threshold"
            )
            continue

        valid_anchors.append(anchor_name)
        # Labels are 1-indexed based on position in the canonical list.
        anchor_label[anchor_name] = CANONICAL_ANCHORS.index(anchor_name) + 1

    if len(valid_anchors) < 2:
        raise ValueError(
            f"Cannot partition with {len(valid_anchors)} valid anchor(s): "
            "at least 2 Partition Anchors are required. "
            f"Excluded: {exclusion_reasons}"
        )

    # ---- Step 2: Compute anchor surfaces -----------------------------------
    surfaces: dict[str, np.ndarray] = {}
    for label in valid_anchors:
        mask = anchor_masks[label].astype(bool)
        surfaces[label] = _extract_surface(mask)

    # ---- Step 3: Compute distance transforms (one anchor at a time) --------
    min_dist = np.full(shape, np.inf, dtype=np.float64)
    best_label = np.zeros(shape, dtype=np.int32)  # 0 = unassigned

    for label in valid_anchors:
        surf = surfaces[label]
        # Inverted surface: distance_transform_edt computes distance from
        # each element to the nearest zero-valued element.  Since the
        # surface voxels are False in ~surf, they get distance 0, while
        # all other voxels get their physical distance to the nearest
        # surface voxel — exactly what we need.
        dist = distance_transform_edt(~surf, sampling=spacing)

        lbl = anchor_label[label]
        closer = dist < min_dist
        min_dist[closer] = dist[closer]
        best_label[closer] = lbl

    # ---- Step 4: Identify epicardial fat voxels ----------------------------
    hu_low, hu_high = fat_hu_range
    in_hu = (ct_array >= hu_low) & (ct_array <= hu_high)
    fat_mask = pericardium_mask.astype(bool) & in_hu

    # ---- Step 5: Assign each fat voxel to nearest anchor -------------------
    anchor_assignments = np.zeros(shape, dtype=np.int32)

    # Assign fat voxels within the distance threshold.
    assigned = fat_mask & (min_dist <= max_assign_distance_mm)
    anchor_assignments[assigned] = best_label[assigned]

    # Fat voxels beyond the threshold remain 0 (unassigned).

    # ---- Step 6: Compute statistics ----------------------------------------
    la_fat_mask = anchor_assignments == 1

    total_fat_voxels = np.count_nonzero(fat_mask)
    total_fat_volume = total_fat_voxels * voxel_vol_ml

    unassigned_voxels = np.count_nonzero(fat_mask & (anchor_assignments == 0))
    unassigned_volume = unassigned_voxels * voxel_vol_ml

    anchor_volumes_ml: dict[str, float] = {}
    anchor_shares: dict[str, float] = {}

    for anchor_name in valid_anchors:
        lbl = anchor_label[anchor_name]
        n_voxels = np.count_nonzero(anchor_assignments == lbl)
        vol = n_voxels * voxel_vol_ml
        anchor_volumes_ml[anchor_name] = vol
        share = (n_voxels / total_fat_voxels * 100.0) if total_fat_voxels > 0 else 0.0
        anchor_shares[anchor_name] = share

    # Include excluded anchors with zero volume/share.
    for anchor_name in excluded_anchors:
        anchor_volumes_ml[anchor_name] = 0.0
        anchor_shares[anchor_name] = 0.0

    return PartitionResult(
        la_fat_mask=la_fat_mask,
        all_fat_mask=fat_mask,
        anchor_assignments=anchor_assignments,
        anchor_volumes_ml=anchor_volumes_ml,
        anchor_shares=anchor_shares,
        unassigned_volume_ml=unassigned_volume,
        total_fat_volume_ml=total_fat_volume,
        excluded_anchors=excluded_anchors,
        exclusion_reasons=exclusion_reasons,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_surface(mask: np.ndarray) -> np.ndarray:
    """Extract surface voxels of a binary mask using 6-connectivity.

    A surface voxel is a non-zero voxel that has at least one zero-valued
    face neighbour.
    """
    eroded = binary_erosion(mask, structure=_SURFACE_STRUCTURE)
    return mask & ~eroded
