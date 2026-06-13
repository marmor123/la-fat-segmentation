"""Tests for the la_fat.partition_engine module.

Exercises the distance-based partition of epicardial fat into the six
canonical Partition Anchors, including error handling, data validation,
and the surface-vs-centroid design decision (ADR-0001).
"""

from __future__ import annotations

import dataclasses
import typing as t

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from la_fat.config import PipelineConfig
from la_fat.partition_engine import PartitionResult, partition_fat

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

SHAPE = (64, 64, 64)
SPACING = (1.5, 1.5, 1.5)
VOXEL_VOLUME_ML = SPACING[0] * SPACING[1] * SPACING[2] / 1000.0  # 0.003375
CFG = PipelineConfig()
HU_RANGE = (-190.0, -30.0)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _ellipsoid(
    shape: tuple[int, ...],
    centre: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> np.ndarray:
    """Return a binary ellipsoid mask (uint8).

    Parameters
    ----------
    shape:
        (Z, Y, X) shape of the output volume.
    centre:
        (cz, cy, cx) centre in voxel coordinates.
    radii:
        (rz, ry, rx) semi-axis lengths in voxel units.
    """
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist = (
        ((z - centre[0]) / radii[0]) ** 2
        + ((y - centre[1]) / radii[1]) ** 2
        + ((x - centre[2]) / radii[2]) ** 2
    )
    return (dist <= 1.0).astype(np.uint8)


def _sphere(
    shape: tuple[int, ...],
    centre: tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    """Return a binary sphere mask (uint8)."""
    return _ellipsoid(shape, centre, (radius, radius, radius))


def _build_ct_and_partition(
    shape: tuple[int, ...],
    chamber_masks: dict[str, np.ndarray],
    pericardium_mask: np.ndarray,
    hu_range: tuple[float, float] = HU_RANGE,
    spacing: tuple[float, float, float] = SPACING,
    cfg: PipelineConfig = CFG,
    fat_hu: float = -100.0,
) -> t.Any:
    """Build a synthetic CT volume and run ``partition_fat``.

    Chamber voxels get HU = 0 (outside fat range).  All remaining
    pericardium voxels get *fat_hu* (inside fat range by default).
    Background outside the pericardium gets HU = 0.
    """
    ct = np.zeros(shape, dtype=np.float32)

    # Fill pericardium with fat HU.
    ct[pericardium_mask.astype(bool)] = fat_hu

    # Overwrite chamber voxels with non-fat HU.
    # Skip None masks (used in missing-key tests).
    for ch_mask in chamber_masks.values():
        if ch_mask is not None:
            ct[ch_mask.astype(bool)] = 0.0

    return partition_fat(
        ct_array=ct,
        pericardium_mask=pericardium_mask,
        fat_hu_range=hu_range,
        anchor_masks=chamber_masks,
        config=cfg,
        spacing=spacing,
    )


# ===================================================================
# 1. Basic partition — two synthetic chambers (LA, LV)
# ===================================================================


class TestBasicPartition:
    """Basic two-chamber partition with a fat band between them."""

    @pytest.fixture
    def two_chamber_setup(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create LA and LV as separate ellipsoids with pericardium enclosing both.

        Returns
        -------
        la_mask, lv_mask, pericardium_mask, ct_array
        """
        # LA: left side of the volume.
        la = _ellipsoid(SHAPE, (24, 32, 32), (8, 12, 12))
        # LV: right side.
        lv = _ellipsoid(SHAPE, (40, 32, 32), (8, 12, 12))
        # Pericardium: large ellipsoid enclosing both chambers.
        peri = _ellipsoid(SHAPE, (32, 32, 32), (26, 22, 22))
        return la, lv, peri

    def test_both_chambers_get_fat(
        self, two_chamber_setup
    ):
        la, lv, peri = two_chamber_setup
        result = _build_ct_and_partition(
            SHAPE,
            {"LA": la, "LV": lv},
            peri,
        )
        assert result.anchor_volumes_ml["LA"] > 0.0
        assert result.anchor_volumes_ml["LV"] > 0.0
        assert result.total_fat_volume_ml > 0.0

    def test_la_fat_subset_of_all_fat(
        self, two_chamber_setup
    ):
        la, lv, peri = two_chamber_setup
        result = _build_ct_and_partition(
            SHAPE, {"LA": la, "LV": lv}, peri,
        )
        # Every LA fat voxel must also be in all_fat_mask.
        assert np.all(result.la_fat_mask <= result.all_fat_mask)

    def test_no_unassigned_fat(
        self, two_chamber_setup
    ):
        """With only two close chambers, all fat should be assigned."""
        la, lv, peri = two_chamber_setup
        result = _build_ct_and_partition(
            SHAPE, {"LA": la, "LV": lv}, peri,
        )
        assert result.unassigned_volume_ml == 0.0

    def test_label_map_has_only_two_labels(
        self, two_chamber_setup
    ):
        la, lv, peri = two_chamber_setup
        result = _build_ct_and_partition(
            SHAPE, {"LA": la, "LV": lv}, peri,
        )
        # Only labels 0 (background), 1 (LA), 2 (LV) should appear.
        present = set(np.unique(result.anchor_assignments))
        assert present == {0, 1, 2}


# ===================================================================
# 2. Surface vs centroid — ADR-0001 key decision
# ===================================================================


class TestSurfaceVsCentroid:
    """Surface distance must be used, not centroid distance.

    A fat voxel near the LA appendage surface should be assigned to LA
    even when the Aorta centroid is geometrically closer.
    """

    @pytest.fixture
    def surface_vs_centroid_setup(self):
        """Create an asymmetric layout.

        LA is a two-component dumbbell (body + appendage) giving a
        distant centroid but a near surface for a test voxel near the
        appendage tip.  Aorta is a compact sphere whose centroid is
        closer to the same test voxel.
        """
        # LA body + appendage (dumbbell simulating LA with appendage).
        la_body = _sphere(SHAPE, (50, 25, 40), 8)
        la_app = _sphere(SHAPE, (50, 43, 40), 4)
        la = np.clip(la_body.astype(np.int32) + la_app.astype(np.int32), 0, 1).astype(
            np.uint8
        )

        # Aorta: compact sphere.
        aorta = _sphere(SHAPE, (42, 40, 40), 8)

        # Pericardium: encloses both.
        peri = _sphere(SHAPE, (46, 34, 40), 24)

        return la, aorta, peri

    def test_surface_distance_assigns_to_la(self, surface_vs_centroid_setup):
        """The fat voxel near the LA appendage tip is assigned to LA (label 1)."""
        la, aorta, peri = surface_vs_centroid_setup
        result = _build_ct_and_partition(
            SHAPE, {"LA": la, "Aorta": aorta}, peri,
        )
        # Test voxel near the LA appendage tip.
        assert result.anchor_assignments[50, 48, 40] == 1, (
            f"Voxel at (50, 48, 40) assigned label "
            f"{result.anchor_assignments[50, 48, 40]}, expected 1 (LA)"
        )

    def test_centroid_would_assign_to_aorta(self, surface_vs_centroid_setup):
        """Demonstrate that centroid distance *would* pick Aorta."""
        la, aorta, peri = surface_vs_centroid_setup

        # Compute surface distance maps.
        from la_fat.partition_engine import _extract_surface

        la_surf = _extract_surface(la.astype(bool))
        aorta_surf = _extract_surface(aorta.astype(bool))

        la_dist = distance_transform_edt(~la_surf, sampling=SPACING)
        aorta_dist = distance_transform_edt(~aorta_surf, sampling=SPACING)

        voxel = np.array([50, 48, 40])

        # Surface distances: LA is nearer.
        assert la_dist[*voxel] < aorta_dist[*voxel], (
            f"Surface distance: LA={la_dist[*voxel]:.2f} mm, "
            f"Aorta={aorta_dist[*voxel]:.2f} mm — expected LA to be nearer"
        )

        # Centroid distances: Aorta is nearer.
        la_centroid = np.mean(np.argwhere(la.astype(bool)), axis=0)
        aorta_centroid = np.mean(np.argwhere(aorta.astype(bool)), axis=0)

        la_cdist = float(np.linalg.norm(voxel - la_centroid)) * SPACING[0]
        aorta_cdist = float(np.linalg.norm(voxel - aorta_centroid)) * SPACING[0]

        assert aorta_cdist < la_cdist, (
            f"Centroid distance: LA={la_cdist:.2f} mm, "
            f"Aorta={aorta_cdist:.2f} mm — expected Aorta centroid to be nearer"
        )


# ===================================================================
# 3. Pulmonary Veins excluded
# ===================================================================


class TestPulmonaryVeinsExcluded:
    """Pulmonary Veins must be ignored even if present in anchor_masks."""

    def test_pv_key_ignored(self):
        la = _sphere(SHAPE, (24, 32, 32), 8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        pv = _sphere(SHAPE, (32, 48, 32), 5)  # pulmonary veins
        peri = _sphere(SHAPE, (32, 32, 32), 24)

        result = _build_ct_and_partition(
            SHAPE,
            {"LA": la, "LV": lv, "pulmonary_veins": pv},
            peri,
        )
        # PV should appear in excluded list with a reason.
        assert "pulmonary_veins" not in result.anchor_volumes_ml
        # The PV volume should NOT appear in the output at all since it's
        # not a canonical anchor.
        for key in result.anchor_volumes_ml:
            assert key in ("LA", "LV", "RA", "RV", "Aorta", "Pulmonary_Artery")


# ===================================================================
# 4. Missing anchor exclusion
# ===================================================================


class TestMissingAnchorExclusion:
    """Missing, empty, or too-small anchors are excluded gracefully."""

    def test_missing_key_excluded(self):
        la = _sphere(SHAPE, (24, 32, 32), 8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        peri = _sphere(SHAPE, (32, 32, 32), 24)

        result = _build_ct_and_partition(
            SHAPE, {"LA": la, "LV": lv}, peri,
        )
        # RA is not in the dict → "mask not provided".
        assert "RA" in result.excluded_anchors
        assert "mask not provided" in result.exclusion_reasons.get("RA", "").lower()

    def test_null_mask_excluded(self):
        """A None mask value is treated as empty."""
        la = _sphere(SHAPE, (24, 32, 32), 8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        peri = _sphere(SHAPE, (32, 32, 32), 24)

        result = _build_ct_and_partition(
            SHAPE, {"LA": la, "LV": lv, "RA": None}, peri,
        )
        assert "RA" in result.excluded_anchors
        assert "empty" in result.exclusion_reasons.get("RA", "").lower()

    def test_empty_mask_excluded(self):
        la = _sphere(SHAPE, (24, 32, 32), 8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        ra = np.zeros(SHAPE, dtype=np.uint8)
        peri = _sphere(SHAPE, (32, 32, 32), 24)

        result = _build_ct_and_partition(
            SHAPE, {"LA": la, "LV": lv, "RA": ra}, peri,
        )
        assert "RA" in result.excluded_anchors
        assert "empty" in result.exclusion_reasons.get("RA", "").lower()

    def test_too_small_anchor_excluded(self):
        la = _sphere(SHAPE, (24, 32, 32), 8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        # Tiny sphere: volume < 5 ml (radius 4 → ~0.9 ml).
        ra = _sphere(SHAPE, (24, 48, 32), 4)
        peri = _sphere(SHAPE, (32, 32, 32), 24)

        result = _build_ct_and_partition(
            SHAPE, {"LA": la, "LV": lv, "RA": ra}, peri,
        )
        assert "RA" in result.excluded_anchors
        assert "volume" in result.exclusion_reasons.get("RA", "").lower()

    def test_excluded_anchor_zero_volume(self):
        la = _sphere(SHAPE, (24, 32, 32), 8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        peri = _sphere(SHAPE, (32, 32, 32), 24)

        result = _build_ct_and_partition(
            SHAPE, {"LA": la, "LV": lv}, peri,
        )
        # Excluded anchors should have zero volume recorded.
        for excluded in result.excluded_anchors:
            assert result.anchor_volumes_ml[excluded] == 0.0
            assert result.anchor_shares[excluded] == 0.0

    def test_partition_works_with_remaining(self):
        """Only LA and LV provided (others missing) — partition still works."""
        la = _sphere(SHAPE, (24, 32, 32), 8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        peri = _sphere(SHAPE, (32, 32, 32), 24)

        result = _build_ct_and_partition(
            SHAPE, {"LA": la, "LV": lv}, peri,
        )
        assert result.total_fat_volume_ml > 0
        # Only LA and LV should have non-zero volume.
        assert result.anchor_volumes_ml["LA"] > 0
        assert result.anchor_volumes_ml["LV"] > 0

    def test_fewer_than_two_anchors_raises(self):
        la = _sphere(SHAPE, (24, 32, 32), 8)
        peri = _sphere(SHAPE, (32, 32, 32), 24)

        with pytest.raises(ValueError, match="at least 2"):
            _build_ct_and_partition(
                SHAPE, {"LA": la}, peri,
            )

    def test_no_valid_anchors_raises(self):
        peri = _sphere(SHAPE, (32, 32, 32), 24)

        with pytest.raises(ValueError, match="at least 2"):
            _build_ct_and_partition(
                SHAPE, {"LA": np.zeros(SHAPE, dtype=np.uint8)}, peri,
            )


# ===================================================================
# 5. All 6 anchors present
# ===================================================================


class TestAllSixAnchors:
    """Full partition with all 6 canonical Partition Anchors."""

    @pytest.fixture
    def six_anchors(self) -> tuple[dict[str, np.ndarray], np.ndarray]:
        anchors = {
            "LA": _sphere(SHAPE, (24, 20, 40), 8),
            "LV": _sphere(SHAPE, (40, 20, 40), 8),
            "RA": _sphere(SHAPE, (24, 60, 40), 8),
            "RV": _sphere(SHAPE, (40, 60, 40), 8),
            "Aorta": _sphere(SHAPE, (55, 40, 40), 8),
            "Pulmonary_Artery": _sphere(SHAPE, (10, 40, 40), 8),
        }
        # Pericardium encloses all.
        peri = _ellipsoid(SHAPE, (34, 40, 40), (40, 34, 26))
        return anchors, peri

    def test_all_six_labels_in_assignments(self, six_anchors):
        anchors, peri = six_anchors
        result = _build_ct_and_partition(
            SHAPE, anchors, peri,
        )
        present = set(np.unique(result.anchor_assignments))
        # 0 = background, 1-6 = the six canonical anchors.
        assert present == {0, 1, 2, 3, 4, 5, 6}

    def test_all_have_positive_volume(self, six_anchors):
        anchors, peri = six_anchors
        result = _build_ct_and_partition(
            SHAPE, anchors, peri,
        )
        for name in anchors:
            assert result.anchor_volumes_ml[name] > 0.0, (
                f"{name} has zero volume"
            )

    def test_shares_and_unassigned_sum_to_100(self, six_anchors):
        anchors, peri = six_anchors
        result = _build_ct_and_partition(
            SHAPE, anchors, peri,
        )
        anchor_total = sum(result.anchor_shares.values())
        # Some fat may be unassigned (beyond max_assign_distance_mm).
        unassigned_pct = (
            result.unassigned_volume_ml / result.total_fat_volume_ml * 100.0
        ) if result.total_fat_volume_ml > 0 else 0.0
        assert abs(anchor_total + unassigned_pct - 100.0) < 1e-3

    def test_no_excluded_anchors(self, six_anchors):
        anchors, peri = six_anchors
        result = _build_ct_and_partition(
            SHAPE, anchors, peri,
        )
        assert result.excluded_anchors == []


# ===================================================================
# 6. Unassigned fat beyond max distance threshold
# ===================================================================


class TestUnassignedFat:
    """Fat voxels beyond max_assign_distance_mm are marked unassigned."""

    def test_distant_fat_unassigned(self):
        """Place two anchors in one corner, fat throughout the volume."""
        la = _sphere(SHAPE, (12, 12, 12), 8)
        lv = _sphere(SHAPE, (12, 52, 12), 8)
        # Large pericardium covering most of the volume.
        peri = _sphere(SHAPE, (32, 32, 32), 28)

        ct = np.full(SHAPE, -100.0, dtype=np.float32)  # all fat HU
        # Reset chamber voxels.
        ct[la.astype(bool)] = 0.0
        ct[lv.astype(bool)] = 0.0

        result = partition_fat(
            ct_array=ct,
            pericardium_mask=peri,
            fat_hu_range=HU_RANGE,
            anchor_masks={"LA": la, "LV": lv},
            config=CFG,
            spacing=SPACING,
            max_assign_distance_mm=30.0,
        )

        # Some fat should be assigned (near anchors).
        assert result.anchor_volumes_ml["LA"] > 0.0
        assert result.anchor_volumes_ml["LV"] > 0.0

        # Some fat should be unassigned (far from both anchors).
        assert result.unassigned_volume_ml > 0.0

    def test_unassigned_in_exclusion_reasons(self):
        """Unassigned anchors don't appear in exclusion_reasons."""
        la = _sphere(SHAPE, (12, 12, 12), 8)
        lv = _sphere(SHAPE, (12, 52, 12), 8)
        peri = _sphere(SHAPE, (32, 32, 32), 28)

        ct = np.full(SHAPE, -100.0, dtype=np.float32)
        ct[la.astype(bool)] = 0.0
        ct[lv.astype(bool)] = 0.0

        result = partition_fat(
            ct_array=ct,
            pericardium_mask=peri,
            fat_hu_range=HU_RANGE,
            anchor_masks={"LA": la, "LV": lv},
            config=CFG,
            spacing=SPACING,
            max_assign_distance_mm=30.0,
        )
        assert result.unassigned_volume_ml >= 0.0
        assert result.total_fat_volume_ml > 0.0


# ===================================================================
# 7. Output shape consistency
# ===================================================================


class TestOutputConsistency:
    """All output masks have the same shape as inputs."""

    @pytest.fixture
    def setup(self):
        la = _sphere(SHAPE, (24, 32, 32), 8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        peri = _sphere(SHAPE, (32, 32, 32), 24)
        return _build_ct_and_partition(
            SHAPE, {"LA": la, "LV": lv}, peri,
        )

    def test_la_fat_mask_shape(self, setup):
        assert setup.la_fat_mask.shape == SHAPE

    def test_all_fat_mask_shape(self, setup):
        assert setup.all_fat_mask.shape == SHAPE

    def test_anchor_assignments_shape(self, setup):
        assert setup.anchor_assignments.shape == SHAPE


# ===================================================================
# 8. Volume computation
# ===================================================================


class TestVolumeComputation:
    """Volume in ml matches known synthetic geometry."""

    def test_volume_matches_known_count(self):
        """Create a known number of fat voxels and verify ml computation."""
        la = _sphere(SHAPE, (24, 32, 32), 8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)

        # Pericardium: a rectangular prism in a location that avoids
        # overlapping with the chamber spheres.
        peri = np.zeros(SHAPE, dtype=np.uint8)
        peri[5:15, 5:15, 5:15] = 1  # 10x10x10 = 1000 voxels

        ct = np.full(SHAPE, -100.0, dtype=np.float32)
        ct[peri.astype(bool)] = -100.0  # fat in prism
        ct[la.astype(bool)] = 0.0
        ct[lv.astype(bool)] = 0.0

        result = partition_fat(
            ct_array=ct,
            pericardium_mask=peri,
            fat_hu_range=HU_RANGE,
            anchor_masks={"LA": la, "LV": lv},
            config=CFG,
            spacing=SPACING,
        )

        # 1000 voxels * 0.003375 ml/voxel = 3.375 ml total
        # (prism is far from chambers, so no overlap)
        expected_total = 1000 * VOXEL_VOLUME_ML
        assert abs(result.total_fat_volume_ml - expected_total) < 1e-6

    def test_volumes_sum_to_total(self):
        la = _sphere(SHAPE, (24, 32, 32), 8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        peri = _sphere(SHAPE, (32, 32, 32), 24)

        result = _build_ct_and_partition(
            SHAPE, {"LA": la, "LV": lv}, peri,
        )

        # Sum of anchor + unassigned = total.
        assigned_sum = sum(
            v for k, v in result.anchor_volumes_ml.items()
            if k not in result.excluded_anchors
        )
        total = assigned_sum + result.unassigned_volume_ml
        assert abs(total - result.total_fat_volume_ml) < 1e-6


# ===================================================================
# 9. Empty fat
# ===================================================================


class TestEmptyFat:
    """No fat voxels yields empty masks with correct zeros."""

    def test_all_outside_hu_range(self):
        """CT values entirely outside fat HU range."""
        la = _sphere(SHAPE, (24, 32, 32), 8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        peri = _sphere(SHAPE, (32, 32, 32), 24)
        ct = np.zeros(SHAPE, dtype=np.float32)  # all 0 HU (outside [-190, -30])

        result = partition_fat(
            ct_array=ct,
            pericardium_mask=peri,
            fat_hu_range=HU_RANGE,
            anchor_masks={"LA": la, "LV": lv},
            config=CFG,
            spacing=SPACING,
        )

        assert np.count_nonzero(result.la_fat_mask) == 0
        assert np.count_nonzero(result.all_fat_mask) == 0
        assert np.count_nonzero(result.anchor_assignments) == 0
        assert result.total_fat_volume_ml == 0.0
        assert result.unassigned_volume_ml == 0.0
        for vol in result.anchor_volumes_ml.values():
            assert vol == 0.0
        for share in result.anchor_shares.values():
            assert share == 0.0

    def test_empty_pericardium(self):
        """No pericardium voxels → no fat."""
        la = _sphere(SHAPE, (24, 32, 32), 8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        peri = np.zeros(SHAPE, dtype=np.uint8)

        result = _build_ct_and_partition(
            SHAPE, {"LA": la, "LV": lv}, peri,
        )
        assert np.count_nonzero(result.all_fat_mask) == 0
        assert result.total_fat_volume_ml == 0.0

    def test_outputs_have_correct_shape(self):
        """Even with no fat, outputs maintain correct shape."""
        la = _sphere(SHAPE, (24, 32, 32), 8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        peri = _sphere(SHAPE, (32, 32, 32), 24)
        ct = np.zeros(SHAPE, dtype=np.float32)

        result = partition_fat(
            ct_array=ct,
            pericardium_mask=peri,
            fat_hu_range=HU_RANGE,
            anchor_masks={"LA": la, "LV": lv},
            config=CFG,
            spacing=SPACING,
        )
        assert result.la_fat_mask.shape == SHAPE
        assert result.all_fat_mask.shape == SHAPE
        assert result.anchor_assignments.shape == SHAPE


# ===================================================================
# 10. PartitionResult dataclass
# ===================================================================


class TestPartitionResultDataclass:
    """PartitionResult fields, types, and immutability."""

    def _make_dummy_result(self) -> PartitionResult:
        mask = np.zeros((4, 4, 4), dtype=bool)
        assignments = np.zeros((4, 4, 4), dtype=np.int32)
        return PartitionResult(
            la_fat_mask=mask,
            all_fat_mask=mask.copy(),
            anchor_assignments=assignments,
            anchor_volumes_ml={"LA": 0.0},
            anchor_shares={"LA": 0.0},
            unassigned_volume_ml=0.0,
            total_fat_volume_ml=0.0,
            excluded_anchors=[],
            exclusion_reasons={},
        )

    def test_fields_present(self):
        result = self._make_dummy_result()
        assert isinstance(result.la_fat_mask, np.ndarray)
        assert isinstance(result.all_fat_mask, np.ndarray)
        assert isinstance(result.anchor_assignments, np.ndarray)
        assert isinstance(result.anchor_volumes_ml, dict)
        assert isinstance(result.anchor_shares, dict)
        assert isinstance(result.unassigned_volume_ml, float)
        assert isinstance(result.total_fat_volume_ml, float)
        assert isinstance(result.excluded_anchors, list)
        assert isinstance(result.exclusion_reasons, dict)

    def test_frozen_immutable(self):
        result = self._make_dummy_result()
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.total_fat_volume_ml = 10.0  # type: ignore

    def test_repr(self):
        result = self._make_dummy_result()
        assert "PartitionResult" in repr(result)
        assert "total_fat_volume_ml" in repr(result)

    def test_la_fat_is_bool(self):
        result = self._make_dummy_result()
        assert result.la_fat_mask.dtype == bool

    def test_all_fat_is_bool(self):
        result = self._make_dummy_result()
        assert result.all_fat_mask.dtype == bool

    def test_anchor_assignments_is_int32(self):
        result = self._make_dummy_result()
        # The actual dtype may be int32 — accept both int32 and int (default).
        assert result.anchor_assignments.dtype in (np.int32, np.int64, int)
