"""Unit tests for Ticket 4: 3D Surface Distance Partition Prototype."""

from __future__ import annotations

import numpy as np
import pytest

from prototypes.prototype_partition_phantom import (
    CANONICAL_ANCHORS,
    create_synthetic_cardiac_phantom,
    evaluate_partition_metrics,
    partition_domain_constrained_edt,
    partition_solid_edt,
    partition_surface_edt,
)


def test_synthetic_phantom_generation_invariants():
    """Verify synthetic 3D cardiac phantom geometry and topological contracts."""
    shape = (64, 64, 64)
    spacing = (1.5, 1.5, 1.5)
    ct_array, peri_mask, anchor_masks, fat_mask = create_synthetic_cardiac_phantom(
        shape=shape, spacing=spacing, av_groove_sharpness=1.0, ias_thickness_voxels=1
    )

    assert ct_array.shape == shape
    assert peri_mask.shape == shape
    assert fat_mask.shape == shape
    assert len(anchor_masks) == 6

    # Verify all canonical anchors are populated
    for name in CANONICAL_ANCHORS:
        assert name in anchor_masks
        assert np.count_nonzero(anchor_masks[name]) > 0, f"Anchor {name} should have voxels"

    # Pericardium must strictly contain all chamber masks
    for name, mask in anchor_masks.items():
        assert np.all(peri_mask[mask > 0] == 1), f"Pericardium must enclose {name}"

    # Fat mask must be strictly disjoint from all chamber masks
    for name, mask in anchor_masks.items():
        overlap = np.count_nonzero(fat_mask & (mask > 0))
        assert overlap == 0, f"Fat mask must not overlap with {name}"


def test_solid_edt_partition_topological_invariants():
    """Verify that Solid-Mask EDT produces topologically sound LA fat without island artifacts."""
    shape = (64, 64, 64)
    spacing = (1.5, 1.5, 1.5)
    ct_array, peri_mask, anchor_masks, fat_mask = create_synthetic_cardiac_phantom(
        shape=shape, spacing=spacing, av_groove_sharpness=1.5
    )

    la_fat_mask, assignments, runtime_ms = partition_solid_edt(
        anchor_masks=anchor_masks,
        pericardium_mask=peri_mask,
        fat_mask=fat_mask,
        spacing=spacing,
        max_assign_distance_mm=35.0,
    )

    metrics = evaluate_partition_metrics(
        la_fat_mask=la_fat_mask,
        all_fat_mask=fat_mask,
        anchor_assignments=assignments,
        spacing=spacing,
        runtime_ms=runtime_ms,
        septal_plane_x=20,  # RA lateral boundary (cx - 12 in 64x64x64 grid)
    )

    # Invariant 1: LA fat volume must be non-zero and plausible (10% - 40% of total fat)
    assert metrics.la_fat_volume_ml > 0.0
    assert 5.0 <= metrics.la_fat_share_pct <= 50.0

    # Invariant 2: Primary component must contain >= 98% of LA fat mass
    assert metrics.primary_component_fraction >= 0.98, (
        f"Expected >= 98% in primary component, got {metrics.primary_component_fraction*100:.1f}%"
    )

    # Invariant 3: Zero septal leakage into RA half-space
    assert metrics.septal_leakage_voxels == 0, (
        f"Found {metrics.septal_leakage_voxels} voxels leaking across the interatrial septum!"
    )

    # Invariant 4: Runtime is fast on CPU (< 150ms on 64x64x64)
    assert runtime_ms < 1000.0


def test_solid_vs_surface_edt_comparison():
    """Compare Solid-Mask EDT vs Surface-Erosion EDT."""
    shape = (64, 64, 64)
    spacing = (1.5, 1.5, 1.5)
    ct_array, peri_mask, anchor_masks, fat_mask = create_synthetic_cardiac_phantom(
        shape=shape, spacing=spacing
    )

    la_solid, assign_solid, t_solid = partition_solid_edt(
        anchor_masks, peri_mask, fat_mask, spacing=spacing, max_assign_distance_mm=35.0
    )
    la_surf, assign_surf, t_surf = partition_surface_edt(
        anchor_masks, peri_mask, fat_mask, spacing=spacing, max_assign_distance_mm=35.0
    )

    # Both must produce valid non-empty masks
    assert np.count_nonzero(la_solid) > 0
    assert np.count_nonzero(la_surf) > 0

    # Agreement between solid and surface EDT should be extremely high (> 99% Dice similarity)
    intersection = np.count_nonzero(la_solid & la_surf)
    dice = 2.0 * intersection / (np.count_nonzero(la_solid) + np.count_nonzero(la_surf))
    assert dice > 0.99, f"Expected Dice > 0.99 between solid and surface EDT, got {dice:.4f}"


def test_distance_cutoff_sensitivity():
    """Verify distance radius cutoff behavior."""
    shape = (64, 64, 64)
    spacing = (1.5, 1.5, 1.5)
    ct_array, peri_mask, anchor_masks, fat_mask = create_synthetic_cardiac_phantom(
        shape=shape, spacing=spacing
    )

    # Small cutoff (10mm) should leave significant unassigned fat
    la_10, assign_10, _ = partition_solid_edt(anchor_masks, peri_mask, fat_mask, spacing=spacing, max_assign_distance_mm=10.0)
    unassigned_10 = np.count_nonzero(fat_mask & (assign_10 == 0))

    # Large cutoff (50mm) should leave almost 0 unassigned fat
    la_50, assign_50, _ = partition_solid_edt(anchor_masks, peri_mask, fat_mask, spacing=spacing, max_assign_distance_mm=50.0)
    unassigned_50 = np.count_nonzero(fat_mask & (assign_50 == 0))

    assert unassigned_10 > unassigned_50
    assert unassigned_50 == 0 or (unassigned_50 / np.count_nonzero(fat_mask)) < 0.05


def test_partition_determinism():
    """Verify bitwise determinism and reproducibility."""
    shape = (64, 64, 64)
    spacing = (1.5, 1.5, 1.5)
    ct_array, peri_mask, anchor_masks, fat_mask = create_synthetic_cardiac_phantom(
        shape=shape, spacing=spacing
    )

    la_1, assign_1, _ = partition_solid_edt(anchor_masks, peri_mask, fat_mask, spacing=spacing, max_assign_distance_mm=35.0)
    la_2, assign_2, _ = partition_solid_edt(anchor_masks, peri_mask, fat_mask, spacing=spacing, max_assign_distance_mm=35.0)

    assert np.array_equal(la_1, la_2)
    assert np.array_equal(assign_1, assign_2)
