"""Tests for the la_fat.cleanup module.

Exercises the post-processing cleanup of the LA Fat mask, including
small-island removal, morphological opening, and vessel filling.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy.ndimage import label as connected_components

from la_fat.config import PipelineConfig
from la_fat.cleanup import CleanupResult, cleanup_la_fat_mask

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

SHAPE = (32, 32, 32)
SPACING = (1.5, 1.5, 1.5)
VOXEL_VOLUME_MM3 = SPACING[0] * SPACING[1] * SPACING[2]  # 3.375
CFG = PipelineConfig()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sphere(
    shape: tuple[int, ...],
    centre: tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    """Return a binary sphere mask (bool)."""
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist = np.sqrt(
        (z - centre[0]) ** 2
        + (y - centre[1]) ** 2
        + (x - centre[2]) ** 2
    )
    return dist <= radius


def _voxel_count_for_volume(
    volume_mm3: float,
    spacing: tuple[float, float, float] = SPACING,
) -> int:
    """Compute how many voxels are needed to reach a given volume (mm³)."""
    voxel_vol = spacing[0] * spacing[1] * spacing[2]
    return max(1, int(np.ceil(volume_mm3 / voxel_vol)))


# ===================================================================
# 1. No cleanup needed — single large component
# ===================================================================


class TestNoCleanupNeeded:
    """LA fat mask with one large component — returned unchanged."""

    @pytest.fixture
    def large_sphere(self) -> np.ndarray:
        """A large sphere well above the island volume threshold."""
        return _make_sphere(SHAPE, (16, 16, 16), 12)

    def test_returns_identical_mask(self, large_sphere):
        result = cleanup_la_fat_mask(large_sphere, CFG, SPACING)
        assert np.array_equal(result.cleaned_mask, large_sphere)

    def test_islands_removed_zero(self, large_sphere):
        result = cleanup_la_fat_mask(large_sphere, CFG, SPACING)
        assert result.islands_removed == 0

    def test_island_volumes_empty(self, large_sphere):
        result = cleanup_la_fat_mask(large_sphere, CFG, SPACING)
        assert result.island_volumes_mm3 == []

    def test_total_removed_zero(self, large_sphere):
        result = cleanup_la_fat_mask(large_sphere, CFG, SPACING)
        assert result.total_removed_volume_mm3 == 0.0

    def test_morphological_opening_false(self, large_sphere):
        result = cleanup_la_fat_mask(large_sphere, CFG, SPACING)
        assert result.morphological_opening_applied is False

    def test_vessel_filling_false(self, large_sphere):
        result = cleanup_la_fat_mask(large_sphere, CFG, SPACING)
        assert result.vessel_filling_applied is False


# ===================================================================
# 2. Small island removal
# ===================================================================


class TestSmallIslandRemoval:
    """Mask with one large component + several small islands below threshold."""

    @pytest.fixture
    def mixed_mask(self) -> np.ndarray:
        mask = np.zeros(SHAPE, dtype=bool)
        # Large component: sphere with radius 10 (~4189 voxels, ~14138 mm³).
        mask |= _make_sphere(SHAPE, (16, 16, 16), 10)
        # Small islands: tiny clusters well below 100 mm³.
        # 5-voxel cluster: 5 * 3.375 = 16.875 mm³ (below 100 mm³).
        mask[2, 2, 2] = True
        mask[2, 2, 3] = True
        mask[2, 3, 2] = True
        mask[3, 2, 2] = True
        mask[3, 2, 3] = True
        # 10-voxel cluster: 10 * 3.375 = 33.75 mm³ (below 100 mm³).
        mask[28, 28, 28] = True
        mask[28, 28, 29] = True
        mask[28, 29, 28] = True
        mask[28, 29, 29] = True
        mask[28, 29, 30] = True
        mask[28, 30, 28] = True
        mask[28, 30, 29] = True
        mask[29, 28, 28] = True
        mask[29, 28, 29] = True
        mask[29, 29, 28] = True
        return mask

    def test_small_islands_removed(self, mixed_mask):
        result = cleanup_la_fat_mask(mixed_mask, CFG, SPACING)
        # Large component should remain.
        assert np.count_nonzero(result.cleaned_mask) > 0
        # The isolated small voxel clusters should be gone.
        assert result.cleaned_mask[2, 2, 2] == False
        assert result.cleaned_mask[28, 28, 28] == False

    def test_large_component_preserved(self, mixed_mask):
        result = cleanup_la_fat_mask(mixed_mask, CFG, SPACING)
        # The large sphere centre should still be present.
        assert result.cleaned_mask[16, 16, 16] == True

    def test_islands_removed_count_correct(self, mixed_mask):
        result = cleanup_la_fat_mask(mixed_mask, CFG, SPACING)
        assert result.islands_removed == 2

    def test_total_removed_volume_positive(self, mixed_mask):
        result = cleanup_la_fat_mask(mixed_mask, CFG, SPACING)
        # 15 voxels removed * 3.375 mm³/voxel = 50.625 mm³
        assert result.total_removed_volume_mm3 > 0.0

    def test_island_volumes_list_length(self, mixed_mask):
        result = cleanup_la_fat_mask(mixed_mask, CFG, SPACING)
        assert len(result.island_volumes_mm3) == 2


# ===================================================================
# 3. Threshold boundary
# ===================================================================


class TestThresholdBoundary:
    """Island exactly at threshold volume — not removed."""

    def test_island_at_threshold_not_removed(self):
        """Component with volume == min_fat_island_volume_mm3 stays."""
        threshold = CFG.min_fat_island_volume_mm3  # 100.0 mm³
        n_voxels = _voxel_count_for_volume(threshold, SPACING)

        mask = np.zeros(SHAPE, dtype=bool)
        # Fill exactly n_voxels at a known location.
        z, y, x = 5, 5, 5
        count = 0
        for dz in range(5):
            for dy in range(5):
                for dx in range(5):
                    if count < n_voxels:
                        mask[z + dz, y + dy, x + dx] = True
                        count += 1

        result = cleanup_la_fat_mask(mask, CFG, SPACING)
        # The component volume should be >= threshold, so it's kept.
        assert np.count_nonzero(result.cleaned_mask) == n_voxels
        assert result.islands_removed == 0

    def test_island_just_below_threshold_removed(self):
        """Component with volume just under threshold gets removed."""
        threshold = CFG.min_fat_island_volume_mm3  # 100.0 mm³
        n_voxels = _voxel_count_for_volume(threshold, SPACING) - 1
        n_voxels = max(1, n_voxels)

        mask = np.zeros(SHAPE, dtype=bool)
        count = 0
        for dz in range(5):
            for dy in range(5):
                for dx in range(5):
                    if count < n_voxels:
                        mask[5 + dz, 5 + dy, 5 + dx] = True
                        count += 1

        result = cleanup_la_fat_mask(mask, CFG, SPACING)
        assert np.count_nonzero(result.cleaned_mask) == 0
        assert result.islands_removed == 1


# ===================================================================
# 4. Empty mask
# ===================================================================


class TestEmptyMask:
    """All zeros input — no crash, all zeros output."""

    @pytest.fixture
    def empty_mask(self) -> np.ndarray:
        return np.zeros(SHAPE, dtype=bool)

    def test_output_is_all_zero(self, empty_mask):
        result = cleanup_la_fat_mask(empty_mask, CFG, SPACING)
        assert np.count_nonzero(result.cleaned_mask) == 0

    def test_no_islands_removed(self, empty_mask):
        result = cleanup_la_fat_mask(empty_mask, CFG, SPACING)
        assert result.islands_removed == 0

    def test_island_volumes_empty(self, empty_mask):
        result = cleanup_la_fat_mask(empty_mask, CFG, SPACING)
        assert result.island_volumes_mm3 == []

    def test_total_removed_zero(self, empty_mask):
        result = cleanup_la_fat_mask(empty_mask, CFG, SPACING)
        assert result.total_removed_volume_mm3 == 0.0

    def test_uint8_input_all_zero(self):
        """Also handles uint8 input correctly."""
        mask_u8 = np.zeros(SHAPE, dtype=np.uint8)
        result = cleanup_la_fat_mask(mask_u8, CFG, SPACING)
        assert np.count_nonzero(result.cleaned_mask) == 0


# ===================================================================
# 5. All small components
# ===================================================================


class TestAllSmallComponents:
    """Every component below threshold — all removed."""

    @pytest.fixture
    def all_small_mask(self) -> np.ndarray:
        mask = np.zeros(SHAPE, dtype=bool)
        # Several tiny clusters all well below threshold.
        placements = [
            (2, 2, 2),
            (2, 28, 2),
            (28, 2, 28),
            (15, 2, 15),
            (2, 15, 28),
        ]
        for z, y, x in placements:
            mask[z, y, x] = True
            mask[z, y + 1, x] = True
            mask[z + 1, y, x] = True
            mask[z + 1, y + 1, x] = True
        return mask

    def test_output_all_zero(self, all_small_mask):
        result = cleanup_la_fat_mask(all_small_mask, CFG, SPACING)
        assert np.count_nonzero(result.cleaned_mask) == 0

    def test_islands_removed_count(self, all_small_mask):
        result = cleanup_la_fat_mask(all_small_mask, CFG, SPACING)
        assert result.islands_removed == 5

    def test_total_removed_positive(self, all_small_mask):
        result = cleanup_la_fat_mask(all_small_mask, CFG, SPACING)
        assert result.total_removed_volume_mm3 > 0.0


# ===================================================================
# 6. Spatial metadata — output shape matches input
# ===================================================================


class TestSpatialMetadata:
    """Output mask has the same shape as input."""

    def test_same_shape_as_input(self):
        mask = _make_sphere(SHAPE, (16, 16, 16), 10)
        result = cleanup_la_fat_mask(mask, CFG, SPACING)
        assert result.cleaned_mask.shape == SHAPE

    def test_non_cubic_shape(self):
        shape = (20, 30, 40)
        mask = _make_sphere(shape, (10, 15, 20), 8)
        result = cleanup_la_fat_mask(mask, CFG, SPACING)
        assert result.cleaned_mask.shape == shape


# ===================================================================
# 7. Volume computation
# ===================================================================


class TestVolumeComputation:
    """Verify island volumes computed correctly from voxel counts and spacing."""

    def test_known_voxel_count_and_spacing(self):
        """Create component with known voxel count and verify volume."""
        mask = np.zeros(SHAPE, dtype=bool)
        # A 4x4x4 cube = 64 voxels.
        cube_vol_mm3 = 64 * VOXEL_VOLUME_MM3  # 64 * 3.375 = 216.0 mm³

        mask[10:14, 10:14, 10:14] = True  # 4*4*4 = 64 voxels

        result = cleanup_la_fat_mask(mask, CFG, SPACING)
        # 64 voxels * 3.375 = 216 mm³
        # This is above 100 mm³, so component is kept.
        assert result.islands_removed == 0
        assert np.array_equal(result.cleaned_mask, mask)

    def test_custom_spacing_volume(self):
        """Verify volume changes with different spacing."""
        mask = np.zeros(SHAPE, dtype=bool)
        # A 3x3x3 cube = 27 voxels.
        mask[10:13, 10:13, 10:13] = True  # 27 voxels

        # With spacing (1.0, 1.0, 1.0): volume = 27 mm³
        spacing_iso = (1.0, 1.0, 1.0)
        result_iso = cleanup_la_fat_mask(mask, CFG, spacing_iso)
        assert result_iso.islands_removed == 1
        assert abs(result_iso.total_removed_volume_mm3 - 27.0) < 1e-9

        # With spacing (2.0, 2.0, 2.0): volume = 27 * 8 = 216 mm³ (above 100).
        spacing_large = (2.0, 2.0, 2.0)
        result_large = cleanup_la_fat_mask(mask, CFG, spacing_large)
        assert result_large.islands_removed == 0
        assert result_large.total_removed_volume_mm3 == 0.0


# ===================================================================
# 8. CleanupResult dataclass
# ===================================================================


class TestCleanupResultDataclass:
    """CleanupResult fields, types, and immutability."""

    def _make_dummy_result(self) -> CleanupResult:
        return CleanupResult(
            cleaned_mask=np.zeros((4, 4, 4), dtype=bool),
            islands_removed=0,
            island_volumes_mm3=[],
            total_removed_volume_mm3=0.0,
            morphological_opening_applied=False,
            vessel_filling_applied=False,
        )

    def test_fields_present(self):
        result = self._make_dummy_result()
        assert isinstance(result.cleaned_mask, np.ndarray)
        assert isinstance(result.islands_removed, int)
        assert isinstance(result.island_volumes_mm3, list)
        assert isinstance(result.total_removed_volume_mm3, float)
        assert isinstance(result.morphological_opening_applied, bool)
        assert isinstance(result.vessel_filling_applied, bool)

    def test_frozen_immutable(self):
        result = self._make_dummy_result()
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.islands_removed = 5  # type: ignore

    def test_repr(self):
        result = self._make_dummy_result()
        assert "CleanupResult" in repr(result)
        assert "islands_removed" in repr(result)

    def test_cleaned_mask_is_bool(self):
        result = self._make_dummy_result()
        assert result.cleaned_mask.dtype == bool


# ===================================================================
# 9. Morphological opening (optional)
# ===================================================================


class TestMorphologicalOpening:
    """When enabled, morphological opening is applied to the mask."""

    @pytest.fixture
    def mask_with_noise(self) -> np.ndarray:
        """A clean sphere with some small protrusions."""
        mask = _make_sphere(SHAPE, (16, 16, 16), 10).copy()
        # Add a thin protrusion (noise).
        mask[5:8, 16, 16] = True
        return mask

    def test_opening_flag_set_when_enabled(self):
        """With opening enabled, flag should be True."""
        mask = _make_sphere(SHAPE, (16, 16, 16), 10)
        cfg = dataclasses.replace(CFG)
        result = cleanup_la_fat_mask(mask, cfg, SPACING, apply_opening=True)
        assert result.morphological_opening_applied is True

    def test_opening_flag_false_when_disabled(self, mask_with_noise):
        result = cleanup_la_fat_mask(mask_with_noise, CFG, SPACING, apply_opening=False)
        assert result.morphological_opening_applied is False

    def test_opening_flag_default_false(self, mask_with_noise):
        result = cleanup_la_fat_mask(mask_with_noise, CFG, SPACING)
        assert result.morphological_opening_applied is False


# ===================================================================
# 10. Vessel filling (optional)
# ===================================================================


class TestVesselFilling:
    """When enabled, vessel filling (hole filling) is applied."""

    @pytest.fixture
    def mask_with_hole(self) -> np.ndarray:
        """A mask with an internal hole."""
        outer = _make_sphere(SHAPE, (16, 16, 16), 12)
        inner = _make_sphere(SHAPE, (16, 16, 16), 4)
        return outer & ~inner  # Shell with a hole inside

    def test_filling_flag_set_when_enabled(self):
        mask = _make_sphere(SHAPE, (16, 16, 16), 10)
        result = cleanup_la_fat_mask(mask, CFG, SPACING, apply_vessel_filling=True)
        assert result.vessel_filling_applied is True

    def test_filling_flag_false_when_disabled(self, mask_with_hole):
        result = cleanup_la_fat_mask(mask_with_hole, CFG, SPACING, apply_vessel_filling=False)
        assert result.vessel_filling_applied is False

    def test_filling_flag_default_false(self, mask_with_hole):
        result = cleanup_la_fat_mask(mask_with_hole, CFG, SPACING)
        assert result.vessel_filling_applied is False
