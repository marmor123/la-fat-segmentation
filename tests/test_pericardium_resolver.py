"""Tests for the la_fat.pericardium_resolver module.

Exercises the normal TS-direct path and all fallback scenarios
(pericardium too small, missing, missing chambers, all chambers
missing, and the exact threshold boundary).
"""

import numpy as np
import pytest

from la_fat.config import PipelineConfig
from la_fat.pericardium_resolver import (
    PericardiumResult,
    resolve_pericardium,
)

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

SHAPE = (80, 80, 80)
SPACING = (1.5, 1.5, 1.5)
VOXEL_VOLUME_ML = SPACING[0] * SPACING[1] * SPACING[2] / 1000.0
CFG = PipelineConfig()  # min_pericardium_volume_ml = 50.0, dilation_mm = 5.0


def _sphere(
    shape: tuple[int, ...],
    centre: tuple[float, float, float],
    radius_voxels: float,
) -> np.ndarray:
    """Return a binary sphere mask (uint8).

    Parameters
    ----------
    shape:
        (Z, Y, X) shape of the output volume.
    centre:
        (cz, cy, cx) centre in voxel coordinates.
    radius_voxels:
        Radius in voxel units.
    """
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist_sq = (
        (z - centre[0]) ** 2
        + (y - centre[1]) ** 2
        + (x - centre[2]) ** 2
    )
    return (dist_sq <= radius_voxels**2).astype(np.uint8)


@pytest.fixture
def all_chambers() -> dict[str, np.ndarray]:
    """Return a dict with all 5 chamber masks plus pericardium."""
    # The large pericardium is a sphere of radius 20 voxels.
    # Volume = 4/3 * pi * 20^3 * 0.003375 = 33510 * 0.003375 = 113.1 ml
    return {
        "pericardium": _sphere(SHAPE, (40, 40, 40), 20),
        "LA": _sphere(SHAPE, (30, 35, 35), 8),
        "LV": _sphere(SHAPE, (30, 35, 55), 8),
        "RA": _sphere(SHAPE, (55, 35, 35), 8),
        "RV": _sphere(SHAPE, (55, 55, 35), 8),
        "Aorta": _sphere(SHAPE, (40, 50, 50), 6),
    }


@pytest.fixture
def small_pericardium_chambers() -> dict[str, np.ndarray]:
    """Pericardium is a small sphere (< 50 ml); all chambers present."""
    # Small pericardium: radius 10 voxels.
    # Volume = 4/3 * pi * 10^3 * 0.003375 = 4189 * 0.003375 = 14.1 ml
    return {
        "pericardium": _sphere(SHAPE, (40, 40, 40), 10),
        "LA": _sphere(SHAPE, (30, 35, 35), 8),
        "LV": _sphere(SHAPE, (30, 35, 55), 8),
        "RA": _sphere(SHAPE, (55, 35, 35), 8),
        "RV": _sphere(SHAPE, (55, 55, 35), 8),
        "Aorta": _sphere(SHAPE, (40, 50, 50), 6),
    }


@pytest.fixture
def no_pericardium_key() -> dict[str, np.ndarray]:
    """No ``pericardium`` key at all; chambers present."""
    return {
        "LA": _sphere(SHAPE, (30, 35, 35), 8),
        "LV": _sphere(SHAPE, (30, 35, 55), 8),
        "RA": _sphere(SHAPE, (55, 35, 35), 8),
        "RV": _sphere(SHAPE, (55, 55, 35), 8),
        "Aorta": _sphere(SHAPE, (40, 50, 50), 6),
    }


@pytest.fixture
def missing_some_chambers() -> dict[str, np.ndarray]:
    """Missing RA, RV, Aorta; only LA and LV available."""
    return {
        "pericardium": _sphere(SHAPE, (40, 40, 40), 10),
        "LA": _sphere(SHAPE, (30, 35, 35), 8),
        "LV": _sphere(SHAPE, (30, 35, 55), 8),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNormalPath:
    """Normal path: TS pericardium is adequate (volume >= 50 ml)."""

    def test_volume_sufficient(self, all_chambers: dict[str, np.ndarray]):
        result = resolve_pericardium(
            all_chambers, CFG, spacing=SPACING
        )
        assert result.fallback_triggered is False
        assert result.fallback_reason is None
        assert result.method == "ts_direct"

    def test_mask_returned_unchanged(
        self, all_chambers: dict[str, np.ndarray]
    ):
        result = resolve_pericardium(
            all_chambers, CFG, spacing=SPACING
        )
        # The mask should be the same object since no copy is made
        np.testing.assert_array_equal(
            result.mask, all_chambers["pericardium"].astype(bool)
        )

    def test_output_shape(self, all_chambers: dict[str, np.ndarray]):
        result = resolve_pericardium(
            all_chambers, CFG, spacing=SPACING
        )
        assert result.mask.shape == SHAPE

    def test_volume_ml_computed(self, all_chambers: dict[str, np.ndarray]):
        """Volume should be computed from voxel count and spacing."""
        result = resolve_pericardium(
            all_chambers, CFG, spacing=SPACING
        )
        # Sphere radius 20 voxels → ~113 ml
        assert result.volume_ml > 100.0
        assert result.volume_ml < 130.0


class TestFallbackTooSmall:
    """Fallback triggers when pericardium exists but is too small."""

    def test_fallback_triggers(
        self, small_pericardium_chambers: dict[str, np.ndarray]
    ):
        result = resolve_pericardium(
            small_pericardium_chambers, CFG, spacing=SPACING
        )
        assert result.fallback_triggered is True
        assert result.method == "convex_hull_fallback"
        assert result.fallback_reason is not None
        assert "threshold" in result.fallback_reason

    def test_output_larger_than_input(
        self, small_pericardium_chambers: dict[str, np.ndarray]
    ):
        result = resolve_pericardium(
            small_pericardium_chambers, CFG, spacing=SPACING
        )
        # The dilated hull should have more voxels than the small
        # pericardium alone.
        assert (
            np.count_nonzero(result.mask)
            > np.count_nonzero(
                small_pericardium_chambers["pericardium"]
            )
        )

    def test_output_shape(self, small_pericardium_chambers):
        result = resolve_pericardium(
            small_pericardium_chambers, CFG, spacing=SPACING
        )
        assert result.mask.shape == SHAPE


class TestFallbackMissingKey:
    """Fallback triggers when pericardium key is absent."""

    def test_fallback_triggers(
        self, no_pericardium_key: dict[str, np.ndarray]
    ):
        result = resolve_pericardium(
            no_pericardium_key, CFG, spacing=SPACING
        )
        assert result.fallback_triggered is True
        assert result.method == "convex_hull_fallback"
        assert result.fallback_reason is not None
        assert "not found" in result.fallback_reason

    def test_output_shape(self, no_pericardium_key):
        result = resolve_pericardium(
            no_pericardium_key, CFG, spacing=SPACING
        )
        assert result.mask.shape == SHAPE


class TestFallbackMissingChambers:
    """Fallback works even when some chamber masks are absent."""

    def test_fallback_triggers(
        self, missing_some_chambers: dict[str, np.ndarray]
    ):
        result = resolve_pericardium(
            missing_some_chambers, CFG, spacing=SPACING
        )
        assert result.fallback_triggered is True
        assert result.method == "convex_hull_fallback"

    def test_reason_mentions_used_chambers(
        self, missing_some_chambers: dict[str, np.ndarray]
    ):
        result = resolve_pericardium(
            missing_some_chambers, CFG, spacing=SPACING
        )
        assert result.fallback_reason is not None
        assert "LA" in result.fallback_reason
        assert "LV" in result.fallback_reason

    def test_output_shape(self, missing_some_chambers):
        result = resolve_pericardium(
            missing_some_chambers, CFG, spacing=SPACING
        )
        assert result.mask.shape == SHAPE


class TestAllChambersMissing:
    """Error is raised when no chamber masks exist."""

    def test_error_raised(self):
        ts_masks = {"pericardium": _sphere(SHAPE, (40, 40, 40), 5)}
        with pytest.raises(ValueError, match="no chamber masks found"):
            resolve_pericardium(ts_masks, CFG, spacing=SPACING)

    def test_error_raised_empty_dict(self):
        with pytest.raises(ValueError, match="no chamber masks found"):
            resolve_pericardium({}, CFG, spacing=SPACING)


class TestAllChambersEmpty:
    """Error is raised when all chamber masks are empty."""

    def test_error_raised(self):
        ts_masks = {
            "pericardium": _sphere(SHAPE, (40, 40, 40), 5),
            "LA": np.zeros(SHAPE, dtype=np.uint8),
            "LV": np.zeros(SHAPE, dtype=np.uint8),
            "RA": np.zeros(SHAPE, dtype=np.uint8),
            "RV": np.zeros(SHAPE, dtype=np.uint8),
            "Aorta": np.zeros(SHAPE, dtype=np.uint8),
        }
        with pytest.raises(
            ValueError, match="all available chamber masks are empty"
        ):
            resolve_pericardium(ts_masks, CFG, spacing=SPACING)


class TestThresholdBoundary:
    """Exact threshold boundary (>= 50.0 ml is normal path)."""

    def test_exactly_at_threshold(self):
        # Compute the radius needed for exactly 50.0 ml.
        # volume_ml = (4/3 * pi * r^3) * voxel_volume_ml
        # r = (50.0 / voxel_volume_ml / (4/3 * pi))^(1/3)
        target_voxels = 50.0 / VOXEL_VOLUME_ML
        radius = (target_voxels * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
        # radius ≈ 15.5 voxels → we round to nearest integer for
        # a discrete sphere on the grid.  ceil to be ≥ threshold.
        radius = int(np.ceil(radius))  # 16 voxels → ~57.9 ml

        masks = {
            "pericardium": _sphere(SHAPE, (40, 40, 40), radius),
            "LA": _sphere(SHAPE, (30, 35, 35), 8),
            "LV": _sphere(SHAPE, (30, 35, 55), 8),
            "RA": _sphere(SHAPE, (55, 35, 35), 8),
            "RV": _sphere(SHAPE, (55, 55, 35), 8),
            "Aorta": _sphere(SHAPE, (40, 50, 50), 6),
        }
        result = resolve_pericardium(masks, CFG, spacing=SPACING)
        assert result.fallback_triggered is False
        assert result.method == "ts_direct"


class TestOutputConsistency:
    """Output masks have consistent shapes."""

    def test_returns_bool_dtype(
        self, all_chambers: dict[str, np.ndarray]
    ):
        result = resolve_pericardium(
            all_chambers, CFG, spacing=SPACING
        )
        assert result.mask.dtype == bool

    def test_fallback_returns_bool_dtype(
        self, small_pericardium_chambers: dict[str, np.ndarray]
    ):
        result = resolve_pericardium(
            small_pericardium_chambers, CFG, spacing=SPACING
        )
        assert result.mask.dtype == bool


class TestConvexHullContainsChambers:
    """The fallback mask encloses all chamber voxels."""

    def test_all_chambers_inside(
        self, small_pericardium_chambers: dict[str, np.ndarray]
    ):
        result = resolve_pericardium(
            small_pericardium_chambers, CFG, spacing=SPACING
        )
        for key in ["LA", "LV", "RA", "RV", "Aorta"]:
            chamber_mask = small_pericardium_chambers[key].astype(bool)
            # Every chamber voxel should be inside the result mask
            assert np.all(result.mask[chamber_mask]), (
                f"Not all {key} voxels are inside the fallback mask"
            )

    def test_partial_chambers_inside(
        self, missing_some_chambers: dict[str, np.ndarray]
    ):
        result = resolve_pericardium(
            missing_some_chambers, CFG, spacing=SPACING
        )
        for key in ["LA", "LV"]:
            chamber_mask = missing_some_chambers[key].astype(bool)
            assert np.all(result.mask[chamber_mask]), (
                f"Not all {key} voxels are inside the fallback mask"
            )


class TestPericardiumResultDataclass:
    """PericardiumResult dataclass behaves correctly."""

    def test_repr(self):
        mask = np.zeros((4, 4, 4), dtype=bool)
        r = PericardiumResult(
            mask=mask,
            fallback_triggered=True,
            fallback_reason="test",
            method="convex_hull_fallback",
        )
        assert "fallback_triggered=True" in repr(r)
        assert "convex_hull_fallback" in repr(r)
