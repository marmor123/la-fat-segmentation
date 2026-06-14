"""Tests for the la_fat.fat_thresholder module."""

import numpy as np
import pytest

from la_fat.config import PipelineConfig
from la_fat.fat_thresholder import FatThresholdResult, compute_fat_threshold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ct_with_fat(
    shape: tuple[int, int, int],
    fat_mean: float,
    fat_sigma: float,
    fat_frac: float = 0.3,
    rng_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic CT volume with a known fat distribution inside a
    pericardium-mask region.

    The ROI is filled with two populations: fat (sub-0 HU, drawn from a
    normal distribution with the given mean/sigma) and tissue (positive
    HU, >0).  This ensures that sub-0 voxels are almost exclusively fat,
    so the Gaussian fit can recover the known parameters.

    Parameters
    ----------
    shape:
        3D shape of the volume.
    fat_mean:
        Mean HU of the fat component.
    fat_sigma:
        Standard deviation of the fat component.
    fat_frac:
        Fraction of the pericardium ROI that contains fat (sub-0 HU).
        The remaining ROI voxels are tissue (>0 HU).
    rng_seed:
        Random seed for reproducibility.

    Returns
    -------
    ct_array, pericardium_mask
    """
    rng = np.random.default_rng(rng_seed)
    ct = np.zeros(shape, dtype=np.float32)
    mask = np.zeros(shape, dtype=np.uint8)

    # Pick a central block as the pericardium ROI.
    sz, sy, sx = shape
    roi_region = (
        slice(sz // 4, 3 * sz // 4),
        slice(sy // 4, 3 * sy // 4),
        slice(sx // 4, 3 * sx // 4),
    )
    roi_size = (
        shape[0] // 2,
        shape[1] // 2,
        shape[2] // 2,
    )
    n_roi = roi_size[0] * roi_size[1] * roi_size[2]
    n_fat = int(n_roi * fat_frac)
    n_tissue = n_roi - n_fat

    # Build fat and tissue populations.  Tissue is strictly positive HU
    # so it does not contaminate the sub-zero fat distribution.
    fat = rng.normal(loc=fat_mean, scale=fat_sigma, size=n_fat)
    tissue = rng.uniform(low=100.0, high=500.0, size=n_tissue)
    all_voxels = np.concatenate([fat, tissue])
    rng.shuffle(all_voxels)

    ct[roi_region] = all_voxels.reshape(roi_size)
    mask[roi_region] = 1

    return ct, mask


class TestFatThresholdResultDataclass:
    """FatThresholdResult should store and expose expected fields."""

    def test_fields_present(self):
        result = FatThresholdResult(
            hu_low=-150.0,
            hu_high=-50.0,
            mean_hu=-100.0,
            sigma_hu=25.0,
            fallback_triggered=False,
            fallback_reason=None,
            method="gaussian_fit",
            num_voxels_fit=5000,
        )
        assert result.hu_low == -150.0
        assert result.hu_high == -50.0
        assert result.mean_hu == -100.0
        assert result.sigma_hu == 25.0
        assert result.fallback_triggered is False
        assert result.fallback_reason is None
        assert result.method == "gaussian_fit"
        assert result.num_voxels_fit == 5000

    def test_default_config(self):
        """Frozen dataclass — confirm no default surprises."""
        result = FatThresholdResult(
            hu_low=0.0,
            hu_high=0.0,
            mean_hu=0.0,
            sigma_hu=0.0,
            fallback_triggered=True,
            fallback_reason="test",
            method="fixed_fallback",
            num_voxels_fit=0,
        )
        assert isinstance(result.hu_low, float)
        assert isinstance(result.sigma_hu, float)
        assert isinstance(result.fallback_triggered, bool)
        assert isinstance(result.num_voxels_fit, int)
        assert isinstance(result.method, str)
        assert result.fallback_reason is not None


class TestNormalGaussianFit:
    """Fit should recover known mean and sigma within tolerance."""

    def test_recovers_known_parameters(self):
        cfg = PipelineConfig()
        ct, mask = _make_ct_with_fat(
            shape=(64, 64, 64),
            fat_mean=-110.0,
            fat_sigma=20.0,
            rng_seed=42,
        )
        result = compute_fat_threshold(ct, mask, cfg)

        assert result.method == "gaussian_fit"
        assert not result.fallback_triggered
        # Mean should be close to -110 (tolerance accounts for sampling variance)
        assert abs(result.mean_hu - (-110.0)) < 5.0
        # Sigma should be close to 20 (tolerance accounts for sampling variance)
        assert abs(result.sigma_hu - 20.0) < 5.0
        # Range should be mean ± multiplier*sigma, clamped to fallback
        m = cfg.gaussian_sigma_multiplier
        expected_low = max(result.mean_hu - m * result.sigma_hu, cfg.hu_fallback_low)
        expected_high = min(result.mean_hu + m * result.sigma_hu, cfg.hu_fallback_high)
        assert result.hu_low == pytest.approx(expected_low, abs=1e-6)
        assert result.hu_high == pytest.approx(expected_high, abs=1e-6)
        assert result.num_voxels_fit > 0

    def test_different_mean(self):
        """Fit should recover a different mean correctly."""
        cfg = PipelineConfig()
        ct, mask = _make_ct_with_fat(
            shape=(64, 64, 64),
            fat_mean=-80.0,
            fat_sigma=15.0,
            rng_seed=99,
        )
        result = compute_fat_threshold(ct, mask, cfg)

        assert result.method == "gaussian_fit"
        assert abs(result.mean_hu - (-80.0)) < 3.0
        assert abs(result.sigma_hu - 15.0) < 5.0


class TestClamping:
    """Output range should always be clamped to [hu_fallback_low, hu_fallback_high]."""

    def test_low_clamped(self):
        """Distribution centered far below fallback low (-190)."""
        cfg = PipelineConfig()
        ct, mask = _make_ct_with_fat(
            shape=(64, 64, 64),
            fat_mean=-250.0,
            fat_sigma=30.0,
            rng_seed=42,
        )
        result = compute_fat_threshold(ct, mask, cfg)

        # hu_low should be clamped to -190
        assert result.hu_low >= cfg.hu_fallback_low
        assert result.hu_low == pytest.approx(cfg.hu_fallback_low, abs=1e-6)

    def test_high_clamped(self):
        """Distribution centered near zero (-10 HU), high bound clamped."""
        cfg = PipelineConfig()
        ct, mask = _make_ct_with_fat(
            shape=(64, 64, 64),
            fat_mean=-10.0,
            fat_sigma=10.0,
            rng_seed=42,
        )
        result = compute_fat_threshold(ct, mask, cfg)

        # hu_high should be clamped to -30
        assert result.hu_high <= cfg.hu_fallback_high
        assert result.hu_high == pytest.approx(cfg.hu_fallback_high, abs=1e-6)

    def test_range_always_within_fallback(self):
        """For various means, output range should always be subset of fallback."""
        cfg = PipelineConfig()
        for mean in [-180.0, -150.0, -100.0, -60.0, -40.0]:
            ct, mask = _make_ct_with_fat(
                shape=(48, 48, 48),
                fat_mean=mean,
                fat_sigma=15.0,
                rng_seed=42,
            )
            result = compute_fat_threshold(ct, mask, cfg)
            assert result.hu_low >= cfg.hu_fallback_low
            assert result.hu_high <= cfg.hu_fallback_high
            assert result.hu_low < result.hu_high


class TestInsufficientVoxelsFallback:
    """Fewer than min_sub_zero_voxels_for_fit sub-0 voxels should trigger fallback."""

    def test_too_few_sub_zero_voxels(self):
        """ROI with < 1000 sub-0 voxels should fall back to fixed range."""
        cfg = PipelineConfig()
        shape = (16, 16, 16)
        ct = np.ones(shape, dtype=np.float32) * 200.0  # all positive
        mask = np.zeros(shape, dtype=np.uint8)
        # Small ROI with only a few sub-0 voxels.
        mask[4:8, 4:8, 4:8] = 1
        # Only 100 sub-0 voxels (well below 1000 threshold).
        ct[4:8, 4:8, 4:8] = -50.0
        # Leave most of the ROI positive.
        ct[6:8, 6:8, 6:8] = 300.0

        result = compute_fat_threshold(ct, mask, cfg)
        assert result.fallback_triggered
        assert result.method == "fixed_fallback"
        assert result.fallback_reason is not None
        assert "insufficient" in result.fallback_reason.lower()
        assert result.hu_low == cfg.hu_fallback_low
        assert result.hu_high == cfg.hu_fallback_high

    def test_all_positive_voxels_in_roi(self):
        """No sub-0 voxels at all should trigger fallback."""
        cfg = PipelineConfig()
        shape = (32, 32, 32)
        ct = np.ones(shape, dtype=np.float32) * 200.0
        mask = np.zeros(shape, dtype=np.uint8)
        mask[8:24, 8:24, 8:24] = 1

        result = compute_fat_threshold(ct, mask, cfg)
        assert result.fallback_triggered
        assert result.method == "fixed_fallback"
        assert result.num_voxels_fit == 0
        assert result.hu_low == cfg.hu_fallback_low
        assert result.hu_high == cfg.hu_fallback_high


class TestWideSigmaFallback:
    """Degenerate distributions with sigma > max_gaussian_sigma (100) should fall back."""

    def test_wide_distribution_falls_back(self):
        """A uniform-ish distribution across a huge range should trigger fallback."""
        cfg = PipelineConfig()
        shape = (32, 32, 32)
        ct = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[8:24, 8:24, 8:24] = 1

        # Create a nearly uniform distribution from -500 to -1
        rng = np.random.default_rng(42)
        roi_slice = (slice(8, 24), slice(8, 24), slice(8, 24))
        n_voxels = 16 * 16 * 16
        ct[roi_slice] = rng.uniform(
            low=-500.0, high=-1.0, size=n_voxels
        ).astype(np.float32).reshape(16, 16, 16)

        result = compute_fat_threshold(ct, mask, cfg)
        assert result.fallback_triggered
        assert result.method == "fixed_fallback"
        assert result.fallback_reason is not None
        assert "sigma" in result.fallback_reason.lower()
        assert result.hu_low == cfg.hu_fallback_low
        assert result.hu_high == cfg.hu_fallback_high


class TestReproducibility:
    """Same input should always produce identical output."""

    def test_deterministic_output(self):
        cfg = PipelineConfig()
        ct, mask = _make_ct_with_fat(
            shape=(48, 48, 48),
            fat_mean=-100.0,
            fat_sigma=25.0,
            rng_seed=42,
        )
        result1 = compute_fat_threshold(ct, mask, cfg)
        result2 = compute_fat_threshold(ct, mask, cfg)

        assert result1.hu_low == result2.hu_low
        assert result1.hu_high == result2.hu_high
        assert result1.mean_hu == result2.mean_hu
        assert result1.sigma_hu == result2.sigma_hu
        assert result1.fallback_triggered == result2.fallback_triggered
        assert result1.method == result2.method
        assert result1.num_voxels_fit == result2.num_voxels_fit

    def test_deterministic_fallback(self):
        """Even fallback results should be identical for same input."""
        cfg = PipelineConfig()
        shape = (16, 16, 16)
        ct = np.ones(shape, dtype=np.float32) * 200.0
        mask = np.zeros(shape, dtype=np.uint8)
        mask[4:8, 4:8, 4:8] = 1

        result1 = compute_fat_threshold(ct, mask, cfg)
        result2 = compute_fat_threshold(ct, mask, cfg)

        assert result1.hu_low == result2.hu_low
        assert result1.fallback_triggered == result2.fallback_triggered


class TestOutputTypes:
    """All result fields should have correct types."""

    def test_field_types_normal_path(self):
        cfg = PipelineConfig()
        ct, mask = _make_ct_with_fat(
            shape=(48, 48, 48),
            fat_mean=-100.0,
            fat_sigma=20.0,
            rng_seed=42,
        )
        result = compute_fat_threshold(ct, mask, cfg)

        assert isinstance(result.hu_low, float)
        assert isinstance(result.hu_high, float)
        assert isinstance(result.mean_hu, float)
        assert isinstance(result.sigma_hu, float)
        assert isinstance(result.fallback_triggered, bool)
        assert result.fallback_reason is None or isinstance(
            result.fallback_reason, str
        )
        assert isinstance(result.method, str)
        assert isinstance(result.num_voxels_fit, int)

    def test_field_types_fallback_path(self):
        cfg = PipelineConfig()
        shape = (16, 16, 16)
        ct = np.ones(shape, dtype=np.float32) * 200.0
        mask = np.zeros(shape, dtype=np.uint8)
        mask[4:8, 4:8, 4:8] = 1

        result = compute_fat_threshold(ct, mask, cfg)

        assert isinstance(result.hu_low, float)
        assert isinstance(result.hu_high, float)
        assert isinstance(result.mean_hu, float)
        assert isinstance(result.sigma_hu, float)
        assert isinstance(result.fallback_triggered, bool)
        assert isinstance(result.fallback_reason, str)
        assert isinstance(result.method, str)
        assert isinstance(result.num_voxels_fit, int)


class TestMethodField:
    """method field should be 'gaussian_fit' or 'fixed_fallback' as appropriate."""

    def test_normal_path_method(self):
        cfg = PipelineConfig()
        ct, mask = _make_ct_with_fat(
            shape=(48, 48, 48),
            fat_mean=-100.0,
            fat_sigma=20.0,
            rng_seed=42,
        )
        result = compute_fat_threshold(ct, mask, cfg)
        assert result.method == "gaussian_fit"

    def test_fallback_path_method(self):
        cfg = PipelineConfig()
        shape = (16, 16, 16)
        ct = np.ones(shape, dtype=np.float32) * 200.0
        mask = np.zeros(shape, dtype=np.uint8)
        mask[4:8, 4:8, 4:8] = 1

        result = compute_fat_threshold(ct, mask, cfg)
        assert result.method == "fixed_fallback"


class TestClampingFlags:
    """When output range is clamped to fallback bounds, the clamped_low /
    clamped_high flags should be set.  Clamping is NOT a fallback — the
    Gaussian fit succeeded, but tails were cut."""

    def test_low_clamped_flag_set(self):
        """Distribution far below -190 → clamped_low=True, still gaussian_fit."""
        cfg = PipelineConfig()
        ct, mask = _make_ct_with_fat(
            shape=(64, 64, 64),
            fat_mean=-250.0,
            fat_sigma=30.0,
            rng_seed=42,
        )
        result = compute_fat_threshold(ct, mask, cfg)
        assert result.method == "gaussian_fit"
        assert not result.fallback_triggered
        assert result.clamped_low is True
        assert result.hu_low == cfg.hu_fallback_low

    def test_high_clamped_flag_set(self):
        """Distribution near -25 mean, sigma 5 → upper bound -15, clamped to -30."""
        cfg = PipelineConfig()
        ct, mask = _make_ct_with_fat(
            shape=(64, 64, 64),
            fat_mean=-25.0,
            fat_sigma=5.0,
            rng_seed=42,
        )
        result = compute_fat_threshold(ct, mask, cfg)
        assert result.method == "gaussian_fit"
        assert not result.fallback_triggered
        assert result.clamped_high is True
        assert result.hu_high == cfg.hu_fallback_high

    def test_no_clamping_when_within_bounds(self):
        """Distribution well within fallback bounds → both flags False."""
        cfg = PipelineConfig()
        ct, mask = _make_ct_with_fat(
            shape=(64, 64, 64),
            fat_mean=-110.0,
            fat_sigma=20.0,
            rng_seed=42,
        )
        result = compute_fat_threshold(ct, mask, cfg)
        assert not result.fallback_triggered
        assert result.clamped_low is False
        assert result.clamped_high is False

    def test_fallback_path_both_false(self):
        """Full fallback → clamped flags should be False (not clamped, full
        replacement)."""
        cfg = PipelineConfig()
        shape = (16, 16, 16)
        ct = np.ones(shape, dtype=np.float32) * 200.0
        mask = np.zeros(shape, dtype=np.uint8)
        mask[4:8, 4:8, 4:8] = 1

        result = compute_fat_threshold(ct, mask, cfg)
        assert result.fallback_triggered
        assert result.clamped_low is False
        assert result.clamped_high is False


class TestSanityCheckFallback:
    """If clamping produces hu_low >= hu_high, fallback should trigger."""

    def test_inverted_range_triggers_fallback(self):
        """A degenerate distribution at exactly -190 HU with sigma = 0
        produces clamped low == clamped high == -190, triggering fallback."""
        cfg = PipelineConfig()
        shape = (32, 32, 32)
        ct = np.ones(shape, dtype=np.float32) * 200.0  # all positive background
        mask = np.zeros(shape, dtype=np.uint8)
        mask[8:24, 8:24, 8:24] = 1

        # All sub-0 voxels are exactly -190 HU => sigma = 0.
        # After clamping: hu_low = max(-190, -190) = -190,
        # hu_high = min(-190, -30) = -190 => low >= high.
        roi_slice = (slice(8, 24), slice(8, 24), slice(8, 24))
        ct[roi_slice] = -190.0

        result = compute_fat_threshold(ct, mask, cfg)
        assert result.fallback_triggered, (
            f"Expected fallback for inverted range, got hu_low={result.hu_low}, "
            f"hu_high={result.hu_high}"
        )
        assert result.method == "fixed_fallback"
        assert result.hu_low == cfg.hu_fallback_low
        assert result.hu_high == cfg.hu_fallback_high


class TestEdgeCases:
    """Edge-case inputs."""

    def test_empty_mask(self):
        """No pericardium mask voxels should produce zero sub-zero voxels -> fallback."""
        cfg = PipelineConfig()
        ct = np.random.default_rng(42).normal(size=(32, 32, 32)).astype(np.float32)
        mask = np.zeros((32, 32, 32), dtype=np.uint8)

        result = compute_fat_threshold(ct, mask, cfg)
        assert result.fallback_triggered
        assert result.method == "fixed_fallback"
        assert result.num_voxels_fit == 0

    def test_exactly_min_voxels(self):
        """Exactly min_sub_zero_voxels_for_fit sub-0 voxels should be sufficient."""
        cfg = PipelineConfig()
        shape = (32, 32, 32)
        ct = np.ones(shape, dtype=np.float32) * 200.0
        mask = np.zeros(shape, dtype=np.uint8)

        # Create exactly 1000 sub-0 voxels.
        n = cfg.min_sub_zero_voxels_for_fit
        rng = np.random.default_rng(42)
        flat_idx = rng.choice(np.prod(shape), size=n, replace=False)
        ct.flat[flat_idx] = rng.normal(loc=-100.0, scale=20.0, size=n).astype(
            np.float32
        )
        mask.flat[flat_idx] = 1

        result = compute_fat_threshold(ct, mask, cfg)
        # Should succeed — exactly at threshold.
        assert not result.fallback_triggered
        assert result.method == "gaussian_fit"
        assert result.num_voxels_fit == n
