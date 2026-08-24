"""Deep Multi-Anchor Partition Engine for LA Fat Segmentation.

Part of Wayfinder Ticket 8 (Issue #38).
Computes a mathematically exact 3D Euclidean distance partition of epicardial
fat voxels across the six canonical Partition Anchors:
{LA, LV, RA, RV, Aorta, Pulmonary_Artery}.

Key Design Principles
----------------------
1. **Multi-Anchor Solid EDT**: Direct Euclidean Distance Transform to closed solid
   chamber volumes (``distance_transform_edt(~mask, sampling=spacing)``).
   Eliminates surface erosion loss, preserving 1-voxel thin interatrial septa
   and acute atrioventricular (AV) groove saddle concavities.
2. **Radial Distance Clamping**: Default ``max_assign_distance_mm = 35.0 mm``,
   capturing full physiological sulcus fat while rejecting far-field apical
   pericardial over-reach.
3. **Deep Geometry & Config Support**: Accepts ``GridGeometry`` or spacing tuple,
   typed ``PartitionConfig``, pre-computed ``fat_mask`` or raw ``(ct_array, fat_hu_range)``.
4. **Topological QA & Quality Flags**: Direct 3D 26-connectivity connected component
   auditing, computing primary component fraction, secondary island sizes, and
   emitting auditable ``QualityFlag`` instances.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import distance_transform_edt, label

from la_fat.anatomy import CANONICAL_ANCHORS, voxel_volume_ml
from la_fat.config import PipelineConfig
from la_fat.image_ops import GridGeometry
from la_fat.quality_flagger import QualityFlag, QualitySeverity

# 26-connectivity structuring element for 3D topological analysis
_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)


@dataclasses.dataclass(frozen=True)
class PartitionConfig:
    """Configuration for multi-anchor 3D Euclidean distance partitioning.

    Attributes
    ----------
    max_assign_distance_mm:
        Maximum physical distance (mm) from an anchor surface for a fat voxel
        to be assigned. Beyond this threshold, fat is marked unassigned (default 35.0 mm).
    min_anchor_volume_ml:
        Minimum volume in mL required for an anchor to participate in distance
        competition (default 0.5 mL).
    min_primary_component_fraction:
        Threshold for primary connected component purity ratio in LA fat mask (default 0.95).
    max_unassigned_share_pct:
        Maximum allowable unassigned fat percentage before raising a warning flag (default 20.0%).
    """

    max_assign_distance_mm: float = 35.0
    min_anchor_volume_ml: float = 0.5
    min_primary_component_fraction: float = 0.95
    max_unassigned_share_pct: float = 20.0

    @classmethod
    def from_pipeline_config(cls, cfg: object) -> PartitionConfig:
        """Construct PartitionConfig from PipelineConfig, dict, or existing instance."""
        if isinstance(cfg, cls):
            return cfg
        if isinstance(cfg, dict):
            return cls(
                max_assign_distance_mm=float(
                    cfg.get("max_assign_distance_mm", 35.0)
                ),
                min_anchor_volume_ml=float(
                    cfg.get("min_anchor_volume_ml", 0.5)
                ),
                min_primary_component_fraction=float(
                    cfg.get("min_primary_component_fraction", 0.95)
                ),
                max_unassigned_share_pct=float(
                    cfg.get("max_unassigned_share_pct", 20.0)
                ),
            )
        return cls(
            max_assign_distance_mm=float(
                getattr(cfg, "max_assign_distance_mm", 35.0)
            ),
            min_anchor_volume_ml=float(
                getattr(cfg, "min_anchor_volume_ml", 0.5)
            ),
            min_primary_component_fraction=float(
                getattr(cfg, "min_primary_component_fraction", 0.95)
            ),
            max_unassigned_share_pct=float(
                getattr(cfg, "max_unassigned_share_pct", 20.0)
            ),
        )


@dataclasses.dataclass(frozen=True)
class PartitionMetrics:
    """Quantitative topological and performance metrics for the partitioned fat mask.

    Attributes
    ----------
    la_fat_volume_ml:
        Total volume of fat assigned to Left Atrium in mL.
    total_fat_volume_ml:
        Total volume of epicardial fat inside pericardium in mL.
    la_fat_share_pct:
        Percentage of total epicardial fat assigned to LA.
    unassigned_volume_ml:
        Volume of epicardial fat beyond cutoff distance in mL.
    unassigned_share_pct:
        Percentage of total epicardial fat left unassigned.
    num_connected_components:
        Count of discrete 3D connected components in LA fat mask (26-connectivity).
    primary_component_volume_ml:
        Volume of the largest connected LA fat component in mL.
    primary_component_fraction:
        Ratio of primary component volume to total LA fat volume (1.0 = single solid mantle).
    secondary_component_max_ml:
        Volume of the second largest detached island in mL.
    execution_time_ms:
        Execution duration of the distance transform and partition step in milliseconds.
    """

    la_fat_volume_ml: float
    total_fat_volume_ml: float
    la_fat_share_pct: float
    unassigned_volume_ml: float
    unassigned_share_pct: float
    num_connected_components: int
    primary_component_volume_ml: float
    primary_component_fraction: float
    secondary_component_max_ml: float
    execution_time_ms: float


@dataclasses.dataclass(frozen=True)
class PartitionResult:
    """Result of partitioning epicardial fat by nearest anchor surface.

    Attributes
    ----------
    la_fat_mask:
        Binary mask of LA Fat (same 3D shape as inputs).
    all_fat_mask:
        Binary mask of ALL epicardial fat within the pericardium.
    anchor_assignments:
        Integer label map: 0=background/unassigned, 1=LA, 2=LV, 3=RA, 4=RV,
        5=Aorta, 6=Pulmonary_Artery.
    anchor_volumes_ml:
        Volume in mL for each anchor (including excluded anchors as 0.0).
    anchor_shares:
        Percentage of total fat assigned to each anchor.
    unassigned_volume_ml:
        Fat voxels inside pericardium beyond max_assign_distance_mm.
    total_fat_volume_ml:
        Total epicardial fat volume (assigned + unassigned).
    excluded_anchors:
        Anchors excluded due to missing mask or below minimum volume.
    exclusion_reasons:
        Human-readable explanation for each excluded anchor.
    metrics:
        Quantitative 3D topological metrics (components, purity, timing).
    quality_flags:
        List of auditable QualityFlag instances generated during partitioning.
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
    metrics: Optional[PartitionMetrics] = None
    quality_flags: list[QualityFlag] = dataclasses.field(default_factory=list)


def partition_fat(
    ct_array: Optional[np.ndarray] = None,
    pericardium_mask: Optional[np.ndarray] = None,
    fat_hu_range: Optional[tuple[float, float]] = None,
    anchor_masks: Optional[dict[str, np.ndarray]] = None,
    config: Optional[Union[PartitionConfig, PipelineConfig, dict]] = None,
    spacing: Optional[tuple[float, float, float]] = None,
    max_assign_distance_mm: Optional[float] = None,
    *,
    fat_mask: Optional[np.ndarray] = None,
    geometry: Optional[Union[GridGeometry, tuple[float, float, float]]] = None,
) -> PartitionResult:
    """Partition epicardial fat by assigning each fat voxel to the nearest
    anchor surface among the six canonical Partition Anchors via Solid EDT.

    Parameters
    ----------
    ct_array:
        Optional 3D CT volume in Hounsfield Units (z, y, x). Used if *fat_mask* is omitted.
    pericardium_mask:
        3D binary mask of the pericardial cavity. Required.
    fat_hu_range:
        ``(hu_low, hu_high)`` -- HU window defining fat if *ct_array* is used.
    anchor_masks:
        Dictionary mapping anchor names to binary mask arrays. Required.
        Supported keys: LA, LV, RA, RV, Aorta, Pulmonary_Artery.
    config:
        Optional PartitionConfig, PipelineConfig, or dict.
    spacing:
        Voxel spacing in mm (e.g. ``(1.5, 1.5, 1.5)``). Overridden by *geometry* if provided.
    max_assign_distance_mm:
        Optional override for maximum physical assignment distance in mm.
    fat_mask:
        Optional pre-computed 3D binary mask of fat voxels (e.g. from thresholding module).
    geometry:
        Optional GridGeometry instance or spacing tuple.

    Returns
    -------
    PartitionResult
        Complete partition output with masks, volume dictionaries, metrics, and quality flags.

    Raises
    ------
    ValueError
        If inputs are missing, Left Atrium anchor is missing/empty/below threshold,
        or fewer than 2 valid anchors remain.
    """
    t_start = time.perf_counter()

    if pericardium_mask is None:
        raise ValueError("pericardium_mask is required for fat partitioning.")
    if anchor_masks is None:
        raise ValueError("anchor_masks dictionary is required for fat partitioning.")

    # ---- 1. Resolve Geometry & Spacing --------------------------------------
    if geometry is not None:
        if isinstance(geometry, GridGeometry):
            eff_spacing = geometry.spacing
            voxel_vol_ml = geometry.voxel_volume_ml
        elif isinstance(geometry, tuple):
            eff_spacing = geometry
            voxel_vol_ml = (eff_spacing[0] * eff_spacing[1] * eff_spacing[2]) / 1000.0
        else:
            raise TypeError(f"Unsupported geometry type: {type(geometry)}")
    elif spacing is not None:
        eff_spacing = spacing
        voxel_vol_ml = (eff_spacing[0] * eff_spacing[1] * eff_spacing[2]) / 1000.0
    else:
        eff_spacing = (1.5, 1.5, 1.5)
        voxel_vol_ml = (eff_spacing[0] * eff_spacing[1] * eff_spacing[2]) / 1000.0

    # ---- 2. Resolve Configuration ------------------------------------------
    if config is None:
        eff_config = PartitionConfig()
    elif isinstance(config, PartitionConfig):
        eff_config = config
    else:
        eff_config = PartitionConfig.from_pipeline_config(config)

    if max_assign_distance_mm is not None:
        eff_config = dataclasses.replace(
            eff_config, max_assign_distance_mm=float(max_assign_distance_mm)
        )

    # ---- 3. Resolve Epicardial Fat Mask ------------------------------------
    if fat_mask is not None:
        eff_fat_bool = fat_mask.astype(bool)
    elif ct_array is not None and fat_hu_range is not None:
        hu_low, hu_high = fat_hu_range
        eff_fat_bool = (ct_array >= hu_low) & (ct_array <= hu_high)
    else:
        raise ValueError(
            "Either pre-computed fat_mask or both (ct_array, fat_hu_range) must be provided."
        )

    peri_bool = pericardium_mask.astype(bool)
    all_fat_mask = eff_fat_bool & peri_bool
    shape = pericardium_mask.shape

    # ---- 4. Validate and Filter Anchors ------------------------------------
    valid_anchors: list[str] = []
    excluded_anchors: list[str] = []
    exclusion_reasons: dict[str, str] = {}
    anchor_label_map: dict[str, int] = {}
    quality_flags: list[QualityFlag] = []

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

        vol_ml = np.count_nonzero(mask) * voxel_vol_ml
        if vol_ml < eff_config.min_anchor_volume_ml:
            excluded_anchors.append(anchor_name)
            exclusion_reasons[anchor_name] = (
                f"volume {vol_ml:.2f} ml < {eff_config.min_anchor_volume_ml:.1f} ml threshold"
            )
            continue

        valid_anchors.append(anchor_name)
        anchor_label_map[anchor_name] = CANONICAL_ANCHORS.index(anchor_name) + 1

    # Check LA validity (fatal if missing)
    if "LA" not in valid_anchors:
        reason = exclusion_reasons.get("LA", "missing from anchor_masks")
        raise ValueError(
            f"Cannot partition fat: Left Atrium (LA) anchor is invalid ({reason})."
        )

    # Check overall anchor count (fatal if < 2)
    if len(valid_anchors) < 2:
        raise ValueError(
            f"Cannot partition with {len(valid_anchors)} valid anchor(s): "
            f"at least 2 Partition Anchors are required. Excluded: {exclusion_reasons}"
        )

    # Emit quality flags for non-LA excluded anchors
    for exc in excluded_anchors:
        quality_flags.append(
            QualityFlag(
                severity=QualitySeverity.HIGH,
                concern=f"ANCHOR_EXCLUDED_{exc.upper()}",
                detail=f"Anchor '{exc}' excluded from partition: {exclusion_reasons[exc]}",
                flag_id=f"ANCHOR_EXCLUDED_{exc.upper()}",
                message=f"Anchor '{exc}' excluded: {exclusion_reasons[exc]}",
            )
        )

    # ---- 5. Multi-Anchor Solid EDT Computation -----------------------------
    min_dist = np.full(shape, np.inf, dtype=np.float64)
    best_label = np.zeros(shape, dtype=np.int32)

    # Invert (sx, sy, sz) to (sz, sy, sx) to match NumPy (z, y, x) array layout
    sampling_zyx = (float(eff_spacing[2]), float(eff_spacing[1]), float(eff_spacing[0]))

    for anchor_name in valid_anchors:
        mask = anchor_masks[anchor_name].astype(bool)
        # Direct exterior distance to solid mask with correct anisotropic voxel dimensions
        dist = distance_transform_edt(~mask, sampling=sampling_zyx)
        lbl = anchor_label_map[anchor_name]

        closer = dist < min_dist
        min_dist[closer] = dist[closer]
        best_label[closer] = lbl

    # ---- 6. Assign Fat Voxels with Distance Cutoff --------------------------
    anchor_assignments = np.zeros(shape, dtype=np.int32)
    assigned_mask = all_fat_mask & (min_dist <= eff_config.max_assign_distance_mm)
    anchor_assignments[assigned_mask] = best_label[assigned_mask]

    la_fat_mask = anchor_assignments == 1  # 1 = LA label

    # ---- 7. Compute Volume Quantifications & Statistics --------------------
    total_fat_voxels = int(np.count_nonzero(all_fat_mask))
    total_fat_volume = total_fat_voxels * voxel_vol_ml

    unassigned_voxels = int(np.count_nonzero(all_fat_mask & (anchor_assignments == 0)))
    unassigned_volume = unassigned_voxels * voxel_vol_ml
    unassigned_share_pct = (
        (unassigned_volume / total_fat_volume * 100.0) if total_fat_volume > 0 else 0.0
    )

    anchor_volumes_ml: dict[str, float] = {}
    anchor_shares: dict[str, float] = {}

    for anchor_name in CANONICAL_ANCHORS:
        if anchor_name in valid_anchors:
            lbl = anchor_label_map[anchor_name]
            n_vox = int(np.count_nonzero(anchor_assignments == lbl))
            vol = n_vox * voxel_vol_ml
            anchor_volumes_ml[anchor_name] = vol
            share = (vol / total_fat_volume * 100.0) if total_fat_volume > 0 else 0.0
            anchor_shares[anchor_name] = share
        else:
            anchor_volumes_ml[anchor_name] = 0.0
            anchor_shares[anchor_name] = 0.0

    la_fat_volume = anchor_volumes_ml["LA"]
    la_fat_share = anchor_shares["LA"]

    # ---- 8. 3D Connected Components & Topological Metrics ------------------
    labeled_cc, num_cc = label(la_fat_mask, structure=_CONNECTIVITY_26)
    la_fat_voxels = int(np.count_nonzero(la_fat_mask))

    if num_cc > 0 and la_fat_voxels > 0:
        counts = np.bincount(labeled_cc.ravel())[1:]
        sorted_counts = np.sort(counts)[::-1]
        primary_vox = int(sorted_counts[0])
        primary_ml = primary_vox * voxel_vol_ml
        primary_frac = primary_vox / la_fat_voxels
        sec_max_ml = (
            (float(sorted_counts[1]) * voxel_vol_ml)
            if len(sorted_counts) > 1
            else 0.0
        )
    else:
        primary_ml = 0.0
        primary_frac = 0.0
        sec_max_ml = 0.0

    runtime_ms = (time.perf_counter() - t_start) * 1000.0

    metrics = PartitionMetrics(
        la_fat_volume_ml=la_fat_volume,
        total_fat_volume_ml=total_fat_volume,
        la_fat_share_pct=la_fat_share,
        unassigned_volume_ml=unassigned_volume,
        unassigned_share_pct=unassigned_share_pct,
        num_connected_components=num_cc,
        primary_component_volume_ml=primary_ml,
        primary_component_fraction=primary_frac,
        secondary_component_max_ml=sec_max_ml,
        execution_time_ms=runtime_ms,
    )

    # ---- 9. Quality Flag Auditing ------------------------------------------
    if (
        metrics.primary_component_fraction < eff_config.min_primary_component_fraction
        and metrics.la_fat_volume_ml > 0.0
    ):
        quality_flags.append(
            QualityFlag(
                severity=QualitySeverity.MEDIUM,
                concern="FRAGMENTED_LA_FAT",
                detail=(
                    f"Primary connected component fraction ({metrics.primary_component_fraction:.1%}) "
                    f"< {eff_config.min_primary_component_fraction:.1%} threshold "
                    f"({num_cc} components, largest island = {sec_max_ml:.2f} mL)"
                ),
                actual_value=metrics.primary_component_fraction,
                threshold_value=eff_config.min_primary_component_fraction,
                flag_id="FRAGMENTED_LA_FAT",
                message=f"Fragmented LA fat mantle: {num_cc} components",
            )
        )

    if (
        unassigned_share_pct > eff_config.max_unassigned_share_pct
        and total_fat_volume > 0.0
    ):
        quality_flags.append(
            QualityFlag(
                severity=QualitySeverity.MEDIUM,
                concern="HIGH_UNASSIGNED_FAT",
                detail=(
                    f"Unassigned epicardial fat ({unassigned_share_pct:.1f}%) exceeds "
                    f"{eff_config.max_unassigned_share_pct:.1f}% threshold "
                    f"({unassigned_volume:.2f} mL unassigned beyond {eff_config.max_assign_distance_mm:.1f} mm)"
                ),
                actual_value=unassigned_share_pct,
                threshold_value=eff_config.max_unassigned_share_pct,
                flag_id="HIGH_UNASSIGNED_FAT",
                message=f"High unassigned fat: {unassigned_share_pct:.1f}%",
            )
        )

    return PartitionResult(
        la_fat_mask=la_fat_mask,
        all_fat_mask=all_fat_mask,
        anchor_assignments=anchor_assignments,
        anchor_volumes_ml=anchor_volumes_ml,
        anchor_shares=anchor_shares,
        unassigned_volume_ml=unassigned_volume,
        total_fat_volume_ml=total_fat_volume,
        excluded_anchors=excluded_anchors,
        exclusion_reasons=exclusion_reasons,
        metrics=metrics,
        quality_flags=quality_flags,
    )
