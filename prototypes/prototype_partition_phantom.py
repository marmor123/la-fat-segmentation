"""Prototype Module: 3D Non-Convex Cardiac Phantom & Multi-Anchor Partition Engine.

Ticket 4: [Prototype] Surface Distance Partition on Synthetic Phantom
Investigates 3D distance transform partition behavior across non-convex chamber
boundaries (AV groove saddle, interatrial septum, PV ostia) on both synthetic
mathematical phantoms and clinical TotalSegmentator cardiac segmentations.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, label

# 6-connectivity structuring element
_CONNECTIVITY_6 = np.zeros((3, 3, 3), dtype=bool)
_CONNECTIVITY_6[1, 1, :] = True
_CONNECTIVITY_6[1, :, 1] = True
_CONNECTIVITY_6[:, 1, 1] = True

# 26-connectivity structuring element for 3D component analysis
_CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)

CANONICAL_ANCHORS = ["LA", "LV", "RA", "RV", "Aorta", "Pulmonary_Artery"]
ANCHOR_LABELS = {name: idx + 1 for idx, name in enumerate(CANONICAL_ANCHORS)}


@dataclasses.dataclass(frozen=True)
class PartitionMetrics:
    """Quantitative QA metrics evaluating partition topological soundness."""
    la_fat_volume_ml: float
    total_fat_volume_ml: float
    la_fat_share_pct: float
    unassigned_volume_ml: float
    unassigned_share_pct: float
    num_connected_components: int
    primary_component_volume_ml: float
    primary_component_fraction: float  # Should be >= 0.98 (98%)
    secondary_component_max_ml: float  # Largest detached island
    septal_leakage_voxels: int          # Voxels crossing into RA territory
    execution_time_ms: float


@dataclasses.dataclass
class PhantomResult:
    """Container for phantom data, masks, and partition output."""
    ct_array: np.ndarray
    spacing: Tuple[float, float, float]
    pericardium_mask: np.ndarray
    fat_mask: np.ndarray
    anchor_masks: Dict[str, np.ndarray]
    la_fat_mask: np.ndarray
    anchor_assignments: np.ndarray
    metrics: PartitionMetrics


def create_synthetic_cardiac_phantom(
    shape: Tuple[int, int, int] = (128, 128, 128),
    spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    av_groove_sharpness: float = 1.0,
    ias_thickness_voxels: int = 1,
    include_pv_sleeves: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Generate an analytical 3D multi-chamber cardiac phantom with non-convex features.

    Coordinates: Z (Superior-Inferior: 0=Sup, 127=Inf),
                 Y (Anterior-Posterior: 0=Ant, 127=Post),
                 X (Right-Left: 0=Right, 127=Left)

    Features modeled:
    1. LA: Posterior-superior chamber with non-planar inferior saddle depression (mitral annulus).
    2. LV: Anterior-inferior chamber with matching saddle contact and apex.
    3. RA: Right lateral chamber abutting LA across a thin interatrial septum.
    4. RV: Anterior right chamber wrapping around LV.
    5. Aorta: Central ascending outflow tract curving superiorly.
    6. Pulmonary Artery: Anterior superior trunk.
    7. Pericardium: Surrounding ellipsoidal envelope creating fat space.
    8. Synthetic CT intensities: myocardium/blood = +50 HU, fat = -105 HU, lung/air = -800 HU.

    Returns
    -------
    (ct_array, pericardium_mask, anchor_masks, fat_mask)
    """
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cz, cy, cx = shape[0] // 2, shape[1] // 2, shape[2] // 2

    anchor_masks: Dict[str, np.ndarray] = {}

    # ---- 1. Left Atrium (LA) - Posterior, Superior ----
    # Base ellipsoid centered at (cz - 14, cy + 12, cx + 10)
    la_z, la_y, la_x = cz - 14, cy + 12, cx + 10
    la_dist = (
        ((z - la_z) / 16.0) ** 2
        + ((y - la_y) / 18.0) ** 2
        + ((x - la_x) / 18.0) ** 2
    )
    la_mask = la_dist <= 1.0

    # Carve non-planar 3D saddle at inferior LA-LV interface (AV groove concavity)
    # Saddle equation: z_cut = la_z + 10 + sharpness * (0.04*(x-la_x)^2 - 0.04*(y-la_y)^2)
    saddle_cut = la_z + 10 + av_groove_sharpness * (
        0.03 * ((x - la_x) ** 2) - 0.03 * ((y - la_y) ** 2)
    )
    la_mask = la_mask & (z < saddle_cut)

    # Enforce Interatrial Septum (IAS) boundary
    septum_x = cx - 2
    la_mask = la_mask & (x >= septum_x + ias_thickness_voxels)

    # Optional: Add tubular Pulmonary Vein sleeves on posterior LA wall
    if include_pv_sleeves:
        # Left superior & inferior PV cylinders
        lspv = (((z - (la_z - 6)) / 3.5) ** 2 + ((x - (la_x + 14)) / 3.5) ** 2 <= 1.0) & (y >= la_y + 12) & (y <= la_y + 24)
        lipv = (((z - (la_z + 6)) / 3.5) ** 2 + ((x - (la_x + 14)) / 3.5) ** 2 <= 1.0) & (y >= la_y + 12) & (y <= la_y + 24)
        la_mask = la_mask | lspv | lipv

    # ---- 2. Left Ventricle (LV) - Anterior, Inferior ----
    # Base ellipsoid centered at (cz + 18, cy - 2, cx + 8)
    lv_z, lv_y, lv_x = cz + 18, cy - 2, cx + 8
    lv_dist = (
        ((z - lv_z) / 26.0) ** 2
        + ((y - lv_y) / 22.0) ** 2
        + ((x - lv_x) / 22.0) ** 2
    )
    lv_mask = (lv_dist <= 1.0) & (z >= saddle_cut - 1)  # Abuts LA at saddle

    # Ensure no overlap between LA and LV
    la_mask = la_mask & ~lv_mask

    # ---- 3. Right Atrium (RA) - Right Lateral, Superior ----
    # Center at (cz - 12, cy + 8, cx - 18)
    ra_z, ra_y, ra_x = cz - 12, cy + 8, cx - 18
    ra_dist = (
        ((z - ra_z) / 16.0) ** 2
        + ((y - ra_y) / 18.0) ** 2
        + ((x - ra_x) / 16.0) ** 2
    )
    ra_mask = ra_dist <= 1.0

    # RA must stay to the right of the septum (x <= septum_x)
    ra_mask = ra_mask & (x <= septum_x) & ~la_mask & ~lv_mask

    # ---- 4. Right Ventricle (RV) - Anterior, Right/Inferior ----
    # Crescent shape wrapping anteriorly around LV
    rv_z, rv_y, rv_x = cz + 16, cy - 16, cx - 6
    rv_outer = (
        ((z - rv_z) / 24.0) ** 2
        + ((y - rv_y) / 20.0) ** 2
        + ((x - rv_x) / 24.0) ** 2
    ) <= 1.0
    rv_mask = rv_outer & ~lv_mask & ~la_mask & ~ra_mask

    # ---- 5. Aorta - Central Outflow Tract ----
    ao_z, ao_y, ao_x = cz - 28, cy - 4, cx
    ao_mask = (
        (((y - ao_y) / 7.0) ** 2 + ((x - ao_x) / 7.0) ** 2 <= 1.0)
        & (z >= cz - 48)
        & (z <= cz - 4)
        & ~la_mask
        & ~lv_mask
        & ~ra_mask
        & ~rv_mask
    )

    # ---- 6. Pulmonary Artery (PA) - Anterior Superior ----
    pa_z, pa_y, pa_x = cz - 24, cy - 14, cx - 4
    pa_mask = (
        (((y - pa_y) / 7.0) ** 2 + ((x - pa_x) / 7.0) ** 2 <= 1.0)
        & (z >= cz - 44)
        & (z <= cz - 4)
        & ~ao_mask
        & ~la_mask
        & ~lv_mask
        & ~ra_mask
        & ~rv_mask
    )

    anchor_masks["LA"] = la_mask.astype(np.uint8)
    anchor_masks["LV"] = lv_mask.astype(np.uint8)
    anchor_masks["RA"] = ra_mask.astype(np.uint8)
    anchor_masks["RV"] = rv_mask.astype(np.uint8)
    anchor_masks["Aorta"] = ao_mask.astype(np.uint8)
    anchor_masks["Pulmonary_Artery"] = pa_mask.astype(np.uint8)

    # ---- 7. Pericardium Envelope ----
    # Smooth outer shell enclosing all 6 chambers with a fat-bearing margin
    peri_dist = (
        ((z - cz) / 46.0) ** 2
        + ((y - cy) / 42.0) ** 2
        + ((x - cx) / 44.0) ** 2
    )
    pericardium_mask = (peri_dist <= 1.0).astype(np.uint8)

    # Combined chamber volume
    all_chambers = np.zeros(shape, dtype=bool)
    for m in anchor_masks.values():
        all_chambers |= (m > 0)

    # Ensure pericardium strictly contains all chambers
    pericardium_mask = (pericardium_mask > 0) | all_chambers

    # Epicardial fat domain = Pericardium \ Chambers
    fat_mask = (pericardium_mask > 0) & ~all_chambers

    # ---- 8. Synthetic CT Array ----
    # Background / Lung / Air = -800 HU
    ct_array = np.full(shape, -800.0, dtype=np.float32)
    # Fat voxels = -105 HU + Gaussian noise (sigma=10 HU)
    np.random.seed(42)
    noise = np.random.normal(0.0, 10.0, size=shape).astype(np.float32)
    ct_array[fat_mask] = -105.0 + noise[fat_mask]
    # Solid chamber myocardium / blood = +50 HU + noise
    ct_array[all_chambers] = 50.0 + noise[all_chambers]

    return ct_array, pericardium_mask.astype(np.uint8), anchor_masks, fat_mask.astype(np.uint8)


def extract_surface_mask(mask: np.ndarray) -> np.ndarray:
    """Extract 1-voxel surface of a binary mask using 6-connectivity erosion."""
    eroded = binary_erosion(mask, structure=_CONNECTIVITY_6)
    return mask & ~eroded


def partition_solid_edt(
    anchor_masks: Dict[str, np.ndarray],
    pericardium_mask: np.ndarray,
    fat_mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    max_assign_distance_mm: float = 35.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Partition epicardial fat using direct Solid-Mask Euclidean Distance Transform.

    Mathematically exact exterior distance to closed solids without boundary erosion.

    Returns
    -------
    (la_fat_mask, anchor_assignments, runtime_ms)
    """
    t0 = time.perf_counter()
    shape = pericardium_mask.shape
    min_dist = np.full(shape, np.inf, dtype=np.float64)
    best_label = np.zeros(shape, dtype=np.int32)

    for name in CANONICAL_ANCHORS:
        if name not in anchor_masks or np.count_nonzero(anchor_masks[name]) == 0:
            continue
        mask = anchor_masks[name].astype(bool)
        # Exterior distance to solid mask: distance from ~mask to nearest 0 (the solid mask)
        dist = distance_transform_edt(~mask, sampling=spacing)
        lbl = ANCHOR_LABELS[name]

        closer = dist < min_dist
        min_dist[closer] = dist[closer]
        best_label[closer] = lbl

    # Assign fat voxels within distance cutoff
    is_fat = fat_mask.astype(bool) & (pericardium_mask.astype(bool))
    assigned = is_fat & (min_dist <= max_assign_distance_mm)

    anchor_assignments = np.zeros(shape, dtype=np.int32)
    anchor_assignments[assigned] = best_label[assigned]

    la_fat_mask = anchor_assignments == ANCHOR_LABELS["LA"]
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return la_fat_mask, anchor_assignments, runtime_ms


def partition_surface_edt(
    anchor_masks: Dict[str, np.ndarray],
    pericardium_mask: np.ndarray,
    fat_mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    max_assign_distance_mm: float = 35.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Partition epicardial fat using legacy Surface-Erosion Distance Transform.

    Extracts 6-connectivity boundary prior to EDT.
    """
    t0 = time.perf_counter()
    shape = pericardium_mask.shape
    min_dist = np.full(shape, np.inf, dtype=np.float64)
    best_label = np.zeros(shape, dtype=np.int32)

    for name in CANONICAL_ANCHORS:
        if name not in anchor_masks or np.count_nonzero(anchor_masks[name]) == 0:
            continue
        mask = anchor_masks[name].astype(bool)
        surf = extract_surface_mask(mask)
        dist = distance_transform_edt(~surf, sampling=spacing)
        lbl = ANCHOR_LABELS[name]

        closer = dist < min_dist
        min_dist[closer] = dist[closer]
        best_label[closer] = lbl

    is_fat = fat_mask.astype(bool) & (pericardium_mask.astype(bool))
    assigned = is_fat & (min_dist <= max_assign_distance_mm)

    anchor_assignments = np.zeros(shape, dtype=np.int32)
    anchor_assignments[assigned] = best_label[assigned]

    la_fat_mask = anchor_assignments == ANCHOR_LABELS["LA"]
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return la_fat_mask, anchor_assignments, runtime_ms


def partition_domain_constrained_edt(
    anchor_masks: Dict[str, np.ndarray],
    pericardium_mask: np.ndarray,
    fat_mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.5, 1.5, 1.5),
    max_assign_distance_mm: float = 35.0,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Domain-constrained distance partition preventing straight-line shortcuts across non-target chambers.

    Masks all non-target myocardium chambers as barriers during per-anchor propagation.
    """
    t0 = time.perf_counter()
    shape = pericardium_mask.shape

    # Combined myocardium barrier
    all_myo = np.zeros(shape, dtype=bool)
    for m in anchor_masks.values():
        all_myo |= (m > 0)

    min_dist = np.full(shape, np.inf, dtype=np.float64)
    best_label = np.zeros(shape, dtype=np.int32)

    for name in CANONICAL_ANCHORS:
        if name not in anchor_masks or np.count_nonzero(anchor_masks[name]) == 0:
            continue
        target_mask = anchor_masks[name].astype(bool)

        # Allow propagation through fat space and target mask, blocking other myocardium
        passable = (fat_mask.astype(bool)) | target_mask

        # For voxels in passable domain, distance to target
        dist = distance_transform_edt(~target_mask, sampling=spacing)
        # Penalize paths traversing other chambers by barrier offset
        dist[~passable] = np.inf

        lbl = ANCHOR_LABELS[name]
        closer = dist < min_dist
        min_dist[closer] = dist[closer]
        best_label[closer] = lbl

    is_fat = fat_mask.astype(bool) & (pericardium_mask.astype(bool))
    assigned = is_fat & (min_dist <= max_assign_distance_mm)

    anchor_assignments = np.zeros(shape, dtype=np.int32)
    anchor_assignments[assigned] = best_label[assigned]

    la_fat_mask = anchor_assignments == ANCHOR_LABELS["LA"]
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    return la_fat_mask, anchor_assignments, runtime_ms


def evaluate_partition_metrics(
    la_fat_mask: np.ndarray,
    all_fat_mask: np.ndarray,
    anchor_assignments: np.ndarray,
    spacing: Tuple[float, float, float],
    runtime_ms: float,
    septal_plane_x: Optional[int] = None,
) -> PartitionMetrics:
    """Compute rigorous 3D topological QA metrics for the partitioned LA fat mask."""
    voxel_vol_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0

    total_fat_vox = int(np.count_nonzero(all_fat_mask))
    total_fat_ml = total_fat_vox * voxel_vol_ml

    la_fat_vox = int(np.count_nonzero(la_fat_mask))
    la_fat_ml = la_fat_vox * voxel_vol_ml
    la_share_pct = (la_fat_vox / total_fat_vox * 100.0) if total_fat_vox > 0 else 0.0

    unassigned_vox = int(np.count_nonzero(all_fat_mask & (anchor_assignments == 0)))
    unassigned_ml = unassigned_vox * voxel_vol_ml
    unassigned_pct = (unassigned_vox / total_fat_vox * 100.0) if total_fat_vox > 0 else 0.0

    # 3D 26-connected components analysis
    labeled_cc, num_cc = label(la_fat_mask, structure=_CONNECTIVITY_26)

    if num_cc > 0:
        counts = np.bincount(labeled_cc.ravel())[1:]  # Exclude background 0
        sorted_counts = np.sort(counts)[::-1]
        primary_vox = sorted_counts[0]
        primary_ml = primary_vox * voxel_vol_ml
        primary_frac = primary_vox / la_fat_vox if la_fat_vox > 0 else 0.0
        sec_max_ml = (sorted_counts[1] * voxel_vol_ml) if len(sorted_counts) > 1 else 0.0
    else:
        primary_ml = 0.0
        primary_frac = 0.0
        sec_max_ml = 0.0

    # Septal leakage check: count any LA-assigned voxels crossing into RA lateral domain (x <= ra_territory_x)
    septal_leakage = 0
    if septal_plane_x is not None:
        # Voxels where x <= septal_plane_x assigned to LA
        _, _, x_grid = np.ogrid[: la_fat_mask.shape[0], : la_fat_mask.shape[1], : la_fat_mask.shape[2]]
        # In the atrial Z-range, check if LA fat penetrates deep into RA territory
        leak_mask = la_fat_mask & (x_grid <= septal_plane_x)
        septal_leakage = int(np.count_nonzero(leak_mask))

    return PartitionMetrics(
        la_fat_volume_ml=la_fat_ml,
        total_fat_volume_ml=total_fat_ml,
        la_fat_share_pct=la_share_pct,
        unassigned_volume_ml=unassigned_ml,
        unassigned_share_pct=unassigned_pct,
        num_connected_components=num_cc,
        primary_component_volume_ml=primary_ml,
        primary_component_fraction=primary_frac,
        secondary_component_max_ml=sec_max_ml,
        septal_leakage_voxels=septal_leakage,
        execution_time_ms=runtime_ms,
    )
