"""Tests for the la_fat.partition_engine module.

Exercises the 3D distance-based partition of epicardial fat into the six
canonical Partition Anchors, including error handling, data validation,
surface-vs-centroid design decision (ADR-0001), Solid EDT thin septum preservation,
GridGeometry physical coordinates, PartitionConfig, PartitionMetrics,
and QualityFlag auditing.
"""

from __future__ import annotations

import dataclasses
import typing as t

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from la_fat.config import PipelineConfig
from la_fat.image_ops import GridGeometry
from la_fat.partition_engine import (
    PartitionConfig,
    PartitionMetrics,
    PartitionResult,
    partition_fat,
)
from la_fat.quality_flagger import QualityFlag, QualitySeverity

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

SHAPE = (64, 64, 64)
SPACING = (1.5, 1.5, 1.5)
VOXEL_VOLUME_ML = SPACING[0] * SPACING[1] * SPACING[2] / 1000.0  # 0.003375
CFG = PartitionConfig(min_anchor_volume_ml=0.5)
HU_RANGE = (-190.0, -30.0)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _ellipsoid(
    shape: tuple[int, ...],
    centre: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> np.ndarray:
    """Return a binary ellipsoid mask (uint8)."""
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
    cfg: t.Union[PipelineConfig, PartitionConfig] = CFG,
    fat_hu: float = -100.0,
    max_assign_distance_mm: t.Optional[float] = None,
) -> PartitionResult:
    """Build a synthetic CT volume and run ``partition_fat``."""
    ct = np.zeros(shape, dtype=np.float32)

    # Fill pericardium with fat HU.
    ct[pericardium_mask.astype(bool)] = fat_hu

    # Overwrite chamber voxels with non-fat HU.
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
        max_assign_distance_mm=max_assign_distance_mm,
    )


# ===================================================================
# 1. Basic partition — two synthetic chambers (LA, LV)
# ===================================================================


class TestBasicPartition:
    """Basic two-chamber partition with a fat band between them."""

    @pytest.fixture
    def two_chamber_setup(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # LA: left side of the volume.
        la = _ellipsoid(SHAPE, (24, 32, 32), (8, 12, 12))
        # LV: right side.
        lv = _ellipsoid(SHAPE, (40, 32, 32), (8, 12, 12))
        # Pericardium: large ellipsoid enclosing both chambers.
        peri = _ellipsoid(SHAPE, (32, 32, 32), (26, 22, 22))
        return la, lv, peri

    def test_both_chambers_get_fat(self, two_chamber_setup):
        la, lv, peri = two_chamber_setup
        result = _build_ct_and_partition(
            SHAPE,
            {"LA": la, "LV": lv},
            peri,
        )
        assert result.anchor_volumes_ml["LA"] > 0.0
        assert result.anchor_volumes_ml["LV"] > 0.0
        assert result.total_fat_volume_ml > 0.0

    def test_la_fat_subset_of_all_fat(self, two_chamber_setup):
        la, lv, peri = two_chamber_setup
        result = _build_ct_and_partition(
            SHAPE, {"LA": la, "LV": lv}, peri,
        )
        assert np.all(result.la_fat_mask <= result.all_fat_mask)

    def test_no_unassigned_fat_within_default_radius(self, two_chamber_setup):
        """With only two close chambers, all fat should be within 35 mm."""
        la, lv, peri = two_chamber_setup
        result = _build_ct_and_partition(
            SHAPE, {"LA": la, "LV": lv}, peri,
        )
        assert result.unassigned_volume_ml == 0.0

    def test_label_map_has_only_two_labels(self, two_chamber_setup):
        la, lv, peri = two_chamber_setup
        result = _build_ct_and_partition(
            SHAPE, {"LA": la, "LV": lv}, peri,
        )
        present = set(np.unique(result.anchor_assignments))
        assert present == {0, 1, 2}


# ===================================================================
# 2. Surface vs centroid — ADR-0001 key decision
# ===================================================================


class TestSurfaceVsCentroid:
    """Surface distance must be used, not centroid distance."""

    @pytest.fixture
    def surface_vs_centroid_setup(self):
        la_body = _sphere(SHAPE, (50, 25, 40), 8)
        la_app = _sphere(SHAPE, (50, 43, 40), 4)
        la = np.clip(la_body.astype(np.int32) + la_app.astype(np.int32), 0, 1).astype(
            np.uint8
        )
        ao = _sphere(SHAPE, (50, 43, 48), 5)
        peri = _sphere(SHAPE, (48, 35, 40), 22)
        return la, ao, peri

    def test_appendage_fat_assigned_to_la(self, surface_vs_centroid_setup):
        la, ao, peri = surface_vs_centroid_setup
        result = _build_ct_and_partition(
            SHAPE,
            {"LA": la, "Aorta": ao},
            peri,
        )
        # Point right next to appendage surface (closer to LA surface, but Aorta centroid is closer)
        test_z, test_y, test_x = 50, 48, 40
        assert result.all_fat_mask[test_z, test_y, test_x]
        assert result.anchor_assignments[test_z, test_y, test_x] == 1  # 1 = LA


# ===================================================================
# 3. All six canonical anchors
# ===================================================================


class TestAllSixAnchors:
    """Test full partition across all six canonical chambers."""

    @pytest.fixture
    def six_anchor_setup(self):
        anchors = {
            "LA": _sphere(SHAPE, (20, 25, 25), 6),
            "LV": _sphere(SHAPE, (44, 25, 25), 8),
            "RA": _sphere(SHAPE, (20, 40, 25), 6),
            "RV": _sphere(SHAPE, (44, 40, 25), 8),
            "Aorta": _sphere(SHAPE, (15, 32, 40), 5),
            "Pulmonary_Artery": _sphere(SHAPE, (15, 32, 16), 5),
        }
        peri = _sphere(SHAPE, (30, 32, 28), 28)
        return anchors, peri

    def test_all_six_anchors_get_positive_volume(self, six_anchor_setup):
        anchors, peri = six_anchor_setup
        result = _build_ct_and_partition(SHAPE, anchors, peri)
        for name in ["LA", "LV", "RA", "RV", "Aorta", "Pulmonary_Artery"]:
            assert result.anchor_volumes_ml[name] > 0.0
            assert result.anchor_shares[name] > 0.0

    def test_shares_sum_to_approx_100_percent(self, six_anchor_setup):
        anchors, peri = six_anchor_setup
        result = _build_ct_and_partition(SHAPE, anchors, peri)
        total_share = sum(result.anchor_shares.values())
        unassigned_pct = (
            result.unassigned_volume_ml / result.total_fat_volume_ml * 100.0
        )
        assert abs(total_share + unassigned_pct - 100.0) < 0.1

    def test_no_excluded_anchors(self, six_anchor_setup):
        anchors, peri = six_anchor_setup
        result = _build_ct_and_partition(SHAPE, anchors, peri)
        assert len(result.excluded_anchors) == 0


# ===================================================================
# 4. Unassigned fat and distance cutoff
# ===================================================================


class TestUnassignedFat:
    """Fat far from any anchor surface must remain unassigned."""

    def test_distant_fat_is_unassigned(self):
        la = _sphere(SHAPE, (10, 10, 10), 4)
        lv = _sphere(SHAPE, (10, 20, 10), 4)
        peri = np.zeros(SHAPE, dtype=np.uint8)
        peri[8:12, 8:22, 8:12] = 1
        # Add a distant fat pocket at the opposite corner (50, 50, 50)
        peri[50:54, 50:54, 50:54] = 1

        result = _build_ct_and_partition(
            SHAPE,
            {"LA": la, "LV": lv},
            peri,
            max_assign_distance_mm=10.0,
        )

        assert result.unassigned_volume_ml > 0.0
        assert np.all(result.anchor_assignments[50:54, 50:54, 50:54] == 0)


# ===================================================================
# 5. Missing / Invalid Anchor Handling & Error Policies
# ===================================================================


class TestAnchorIntegrityPolicies:
    """Missing or small anchors handling according to Ticket 8 specs."""

    def test_missing_la_raises_value_error(self):
        """LA missing is fatal."""
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        ra = _sphere(SHAPE, (20, 40, 25), 6)
        peri = _sphere(SHAPE, (32, 32, 32), 24)

        with pytest.raises(ValueError, match="Left Atrium"):
            _build_ct_and_partition(SHAPE, {"LV": lv, "RA": ra}, peri)

    def test_empty_la_raises_value_error(self):
        """Empty LA mask is fatal."""
        la = np.zeros(SHAPE, dtype=np.uint8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        peri = _sphere(SHAPE, (32, 32, 32), 24)

        with pytest.raises(ValueError, match="Left Atrium"):
            _build_ct_and_partition(SHAPE, {"LA": la, "LV": lv}, peri)

    def test_under_volume_la_raises_value_error(self):
        """LA below minimum volume threshold (0.5 mL) is fatal."""
        # 10 voxels = 0.03375 mL < 0.5 mL
        la = np.zeros(SHAPE, dtype=np.uint8)
        la[30, 30, 30:40] = 1
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        peri = _sphere(SHAPE, (32, 32, 32), 24)

        with pytest.raises(ValueError, match="Left Atrium"):
            _build_ct_and_partition(SHAPE, {"LA": la, "LV": lv}, peri)

    def test_fewer_than_two_anchors_raises_value_error(self):
        """LA alone without any other anchor cannot partition."""
        la = _sphere(SHAPE, (24, 32, 32), 8)
        peri = _sphere(SHAPE, (32, 32, 32), 24)

        with pytest.raises(ValueError, match="at least 2 Partition Anchors"):
            _build_ct_and_partition(SHAPE, {"LA": la}, peri)

    def test_missing_secondary_anchor_emits_quality_flag(self):
        """Missing non-LA anchor (e.g. Pulmonary_Artery) emits high quality flag."""
        la = _sphere(SHAPE, (20, 25, 25), 6)
        lv = _sphere(SHAPE, (44, 25, 25), 8)
        ra = _sphere(SHAPE, (20, 40, 25), 6)
        rv = _sphere(SHAPE, (44, 40, 25), 8)
        ao = _sphere(SHAPE, (15, 32, 40), 5)
        peri = _sphere(SHAPE, (30, 32, 28), 28)

        # PA missing
        result = _build_ct_and_partition(
            SHAPE,
            {"LA": la, "LV": lv, "RA": ra, "RV": rv, "Aorta": ao},
            peri,
        )

        assert "Pulmonary_Artery" in result.excluded_anchors
        assert result.anchor_volumes_ml["Pulmonary_Artery"] == 0.0
        assert any(
            qf.severity == QualitySeverity.HIGH
            and "PULMONARY_ARTERY" in qf.concern
            for qf in result.quality_flags
        )


# ===================================================================
# 6. PartitionConfig and Conversion
# ===================================================================


class TestPartitionConfig:
    """Test typed configuration and factory conversion."""

    def test_default_config(self):
        cfg = PartitionConfig()
        assert cfg.max_assign_distance_mm == 35.0
        assert cfg.min_anchor_volume_ml == 0.5
        assert cfg.min_primary_component_fraction == 0.95
        assert cfg.max_unassigned_share_pct == 20.0

    def test_from_pipeline_config(self):
        p_cfg = PipelineConfig(min_anchor_volume_ml=1.0)
        cfg = PartitionConfig.from_pipeline_config(p_cfg)
        assert cfg.min_anchor_volume_ml == 1.0
        assert cfg.max_assign_distance_mm == 35.0

    def test_from_dict(self):
        d = {"max_assign_distance_mm": 40.0, "min_anchor_volume_ml": 0.8}
        cfg = PartitionConfig.from_pipeline_config(d)
        assert cfg.max_assign_distance_mm == 40.0
        assert cfg.min_anchor_volume_ml == 0.8


# ===================================================================
# 7. GridGeometry Integration & Pre-Computed Fat Mask
# ===================================================================


class TestGridGeometryAndPrecomputedMask:
    """Test deep GridGeometry integration and precomputed fat_mask entrypoint."""

    def test_grid_geometry_input(self):
        la = _sphere(SHAPE, (24, 32, 32), 8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        peri = _sphere(SHAPE, (32, 32, 32), 24)
        fat = peri.copy()
        fat[la > 0] = 0
        fat[lv > 0] = 0

        geom = GridGeometry(
            shape_zyx=SHAPE,
            spacing=(1.0, 1.0, 2.0),  # 2.0 mm³ = 0.002 mL per voxel
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
        )

        result = partition_fat(
            fat_mask=fat,
            pericardium_mask=peri,
            anchor_masks={"LA": la, "LV": lv},
            geometry=geom,
        )

        assert result.metrics is not None
        assert result.metrics.execution_time_ms > 0.0
        # Voxel count * 0.002 mL
        expected_total = np.count_nonzero(fat) * 0.002
        assert pytest.approx(result.total_fat_volume_ml, rel=1e-3) == expected_total

    def test_spacing_tuple_geometry_input(self):
        la = _sphere(SHAPE, (24, 32, 32), 8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        peri = _sphere(SHAPE, (32, 32, 32), 24)
        fat = peri.copy()
        fat[la > 0] = 0
        fat[lv > 0] = 0

        result = partition_fat(
            fat_mask=fat,
            pericardium_mask=peri,
            anchor_masks={"LA": la, "LV": lv},
            geometry=(2.0, 2.0, 2.0),  # 8.0 mm³ = 0.008 mL
        )

        assert pytest.approx(result.total_fat_volume_ml, rel=1e-3) == np.count_nonzero(fat) * 0.008


# ===================================================================
# 8. 3D Topological QA Metrics & Quality Flags
# ===================================================================


class TestTopologicalQAMetricsAndQualityFlags:
    """Test 26-connectivity CC analysis and automated flag emission."""

    def test_solid_mantle_high_purity(self):
        la = _sphere(SHAPE, (24, 32, 32), 8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        peri = _sphere(SHAPE, (32, 32, 32), 24)

        result = _build_ct_and_partition(SHAPE, {"LA": la, "LV": lv}, peri)

        assert result.metrics is not None
        assert result.metrics.num_connected_components >= 1
        assert result.metrics.primary_component_fraction >= 0.95
        # No fragmentation flag
        assert not any(qf.concern == "FRAGMENTED_LA_FAT" for qf in result.quality_flags)

    def test_fragmented_fat_emits_quality_flag(self):
        la = _sphere(SHAPE, (24, 32, 32), 8)
        lv = _sphere(SHAPE, (40, 32, 32), 8)
        # Create disjoint fat pockets around LA
        fat_mask = np.zeros(SHAPE, dtype=np.uint8)
        fat_mask[16:20, 32, 32] = 1   # Small pocket 1
        fat_mask[28:32, 32, 32] = 1   # Small pocket 2 (equal size -> primary_frac = 0.50 < 0.95)
        peri = _sphere(SHAPE, (32, 32, 32), 24)

        result = partition_fat(
            fat_mask=fat_mask,
            pericardium_mask=peri,
            anchor_masks={"LA": la, "LV": lv},
            spacing=SPACING,
        )

        assert result.metrics.num_connected_components == 2
        assert result.metrics.primary_component_fraction < 0.95
        assert any(
            qf.concern == "FRAGMENTED_LA_FAT" and qf.severity == QualitySeverity.MEDIUM
            for qf in result.quality_flags
        )

    def test_high_unassigned_fat_emits_quality_flag(self):
        la = _sphere(SHAPE, (10, 10, 10), 4)
        lv = _sphere(SHAPE, (10, 20, 10), 4)
        peri = np.zeros(SHAPE, dtype=np.uint8)
        peri[8:12, 8:22, 8:12] = 1
        # Huge distant fat pocket (>50% of all fat)
        peri[40:56, 40:56, 40:56] = 1

        result = _build_ct_and_partition(
            SHAPE,
            {"LA": la, "LV": lv},
            peri,
            max_assign_distance_mm=10.0,
        )

        assert result.metrics.unassigned_share_pct > 20.0
        assert any(
            qf.concern == "HIGH_UNASSIGNED_FAT" and qf.severity == QualitySeverity.MEDIUM
            for qf in result.quality_flags
        )


# ===================================================================
# 9. Solid EDT Thin-Septum Preservation (Zero Septal Bleed)
# ===================================================================


class TestSolidEDTThinSeptumPreservation:
    """Verify Solid EDT maintains thin anatomical boundaries without erosion loss."""

    def test_one_voxel_thin_interatrial_septum(self):
        """LA and RA separated by a 1-voxel wall.

        Distance competition across the 1-voxel wall must not leak across the boundary.
        """
        shape = (40, 40, 40)
        la = np.zeros(shape, dtype=np.uint8)
        ra = np.zeros(shape, dtype=np.uint8)

        # LA on right (x >= 21)
        la[15:25, 15:25, 21:30] = 1
        # RA on left (x <= 19)
        ra[15:25, 15:25, 10:19] = 1
        # Septum at x = 20 (1 voxel wide)

        peri = np.zeros(shape, dtype=np.uint8)
        peri[10:30, 10:30, 8:32] = 1

        fat = peri.copy()
        fat[la > 0] = 0
        fat[ra > 0] = 0

        result = partition_fat(
            fat_mask=fat,
            pericardium_mask=peri,
            anchor_masks={"LA": la, "RA": ra},
            spacing=(1.0, 1.0, 1.0),
        )

        # Fat on the RA side (x < 20) must NEVER be assigned to LA
        ra_side_la_fat = np.count_nonzero(result.la_fat_mask[:, :, :20])
        assert ra_side_la_fat == 0, "Zero septal leakage onto RA side"


# ===================================================================
# 10. Dataclass Integrity and Immutability
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
        assert isinstance(result.quality_flags, list)

    def test_frozen_immutable(self):
        result = self._make_dummy_result()
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.total_fat_volume_ml = 10.0  # type: ignore

    def test_repr(self):
        result = self._make_dummy_result()
        assert "PartitionResult" in repr(result)
        assert "total_fat_volume_ml" in repr(result)
