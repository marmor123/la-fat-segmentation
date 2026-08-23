"""Unit tests for production la_fat.thresholding module.

Part of Wayfinder Ticket 7 (Issue #37).
Tests trimmed Gaussian peak fitting, prominence detection, upper-tail 0.0 HU clamping,
dual-window quantification, and deep 3D GridGeometry integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from la_fat.image_ops import GridGeometry
from la_fat.quality_flagger import QualitySeverity
from la_fat.thresholding import (
    GMMBayesResult,
    ThresholdConfig,
    ThresholdResult,
    compute_fat_threshold,
    create_fat_mask,
    fit_trimmed_gaussian,
    fit_two_component_gmm_bayes,
)


def test_ideal_gaussian_recovery() -> None:
    """Test convergence and parameter recovery on an ideal Gaussian distribution."""
    rng = np.random.default_rng(42)
    ground_truth_mu = -105.0
    ground_truth_sigma = 12.0
    n_samples = 15000

    voxels = rng.normal(loc=ground_truth_mu, scale=ground_truth_sigma, size=n_samples)
    vvol_ml = 0.003375  # 1.5mm isotropic

    result = fit_trimmed_gaussian(
        sub0_voxels=voxels,
        config=ThresholdConfig(),
        voxel_volume_ml=vvol_ml,
    )

    assert not result.is_fallback
    assert result.fitted_mu is not None
    assert result.fitted_sigma is not None
    assert abs(result.fitted_mu - ground_truth_mu) < 1.0
    assert abs(result.fitted_sigma - ground_truth_sigma) < 1.0
    assert result.hu_low == pytest.approx(ground_truth_mu - 2 * ground_truth_sigma, abs=2.0)
    assert result.hu_high == pytest.approx(ground_truth_mu + 2 * ground_truth_sigma, abs=2.0)
    assert result.voxel_count_evaluated == n_samples
    assert len(result.flags) == 0


def test_asymmetric_soft_tissue_shoulder_trimming() -> None:
    """Test that asymmetric soft tissue contamination ([-30, 0] HU) is trimmed."""
    rng = np.random.default_rng(123)
    # 10,000 true adipose voxels at -95 HU
    fat_voxels = rng.normal(loc=-95.0, scale=14.0, size=10000)
    # 4,000 partial-volume soft-tissue voxels in [-35, 0] HU
    soft_tissue_voxels = rng.uniform(low=-35.0, high=0.0, size=4000)

    combined = np.concatenate([fat_voxels, soft_tissue_voxels])

    result = fit_trimmed_gaussian(
        sub0_voxels=combined,
        config=ThresholdConfig(),
        voxel_volume_ml=0.003375,
    )

    assert not result.is_fallback
    assert result.fitted_mu is not None
    assert abs(result.fitted_mu - (-95.0)) < 1.5
    assert result.hu_high <= 0.0


def test_low_dose_noise_robustness() -> None:
    """Test convergence on broad, noisy distribution mimicking low-dose CT."""
    rng = np.random.default_rng(999)
    voxels = rng.normal(loc=-98.0, scale=26.0, size=12000)

    config = ThresholdConfig(wide_sigma_warn_hu=24.0)
    result = fit_trimmed_gaussian(sub0_voxels=voxels, config=config, voxel_volume_ml=0.001)

    assert not result.is_fallback
    assert result.fitted_sigma is not None
    assert abs(result.fitted_sigma - 26.0) < 2.0
    # Should emit wide sigma warning flag
    flag_ids = [f.flag_id or f.concern for f in result.flags]
    assert "FAT_WIDE_SIGMA_WARNING" in flag_ids
    warning_flag = next(
        f for f in result.flags if (f.flag_id or f.concern) == "FAT_WIDE_SIGMA_WARNING"
    )
    assert warning_flag.severity == QualitySeverity.LOW


def test_metal_outlier_spikes_robustness() -> None:
    """Test that isolated outlier spikes in the histogram do not misguide the peak."""
    rng = np.random.default_rng(456)
    voxels = rng.normal(loc=-100.0, scale=15.0, size=8000)
    # Spike at -140 HU (200 voxels in a single value)
    spike = np.full(200, -140.0)
    combined = np.concatenate([voxels, spike])

    result = fit_trimmed_gaussian(sub0_voxels=combined)
    assert not result.is_fallback
    assert result.fitted_mu is not None
    assert abs(result.fitted_mu - (-100.0)) < 1.5


def test_sparse_voxels_triggers_fallback() -> None:
    """Test fallback when sub-0 HU voxel count is below minimum threshold."""
    rng = np.random.default_rng(777)
    few_voxels = rng.normal(loc=-100.0, scale=15.0, size=150)  # < 500

    result = fit_trimmed_gaussian(
        sub0_voxels=few_voxels,
        config=ThresholdConfig(min_voxel_count=500),
        voxel_volume_ml=0.003375,
    )

    assert result.is_fallback
    assert "Insufficient sub-0 HU voxels" in (result.fallback_reason or "")
    assert result.hu_low == -190.0
    assert result.hu_high == -30.0
    assert result.fitted_mu is None

    flag = result.flags[0]
    assert flag.concern == "FAT_THRESHOLD_FALLBACK"
    assert flag.severity == QualitySeverity.HIGH


def test_monotonic_slope_triggers_fallback() -> None:
    """Test that a distribution with no prominent fat peak triggers standard fallback."""
    # Linearly decreasing distribution from 0 down to -250 HU
    rng = np.random.default_rng(888)
    ramp = rng.triangular(left=-250.0, mode=0.0, right=0.0, size=5000)

    result = fit_trimmed_gaussian(
        sub0_voxels=ramp,
        config=ThresholdConfig(plausible_mu_range=(-150.0, -50.0)),
    )

    assert result.is_fallback
    assert result.flags[0].severity == QualitySeverity.HIGH
    assert result.flags[0].concern == "FAT_THRESHOLD_FALLBACK"


def test_gmm_bayes_two_component_recovery() -> None:
    """Test that GMM Bayes identifies fat component and computes P(Fat|x)=0.5 boundary."""
    rng = np.random.default_rng(42)
    fat = rng.normal(loc=-95.0, scale=15.0, size=8000)
    soft = rng.normal(loc=-15.0, scale=12.0, size=4000)
    combined = np.concatenate([fat, soft])

    res = fit_two_component_gmm_bayes(combined)

    assert not res.is_fallback
    assert res.fitted_mu_fat is not None
    assert abs(res.fitted_mu_fat - (-95.0)) < 4.0
    assert res.hu_low <= -120.0
    assert -60.0 < res.hu_high < -20.0  # Bayes decision boundary between components


def test_gmm_bayes_sparse_fallback() -> None:
    """Test GMM Bayes fallback when voxel count is too low."""
    few = np.array([-100.0, -90.0, -80.0])
    res = fit_two_component_gmm_bayes(few, config=ThresholdConfig(min_voxel_count=500))
    assert res.is_fallback
    assert res.hu_low == -190.0
    assert res.hu_high == -30.0


def test_upper_clamping_at_zero_hu() -> None:
    """Test that adaptive upper threshold is clamped at clamping_max_hu (0.0 HU)."""
    rng = np.random.default_rng(321)
    # Center at -55 HU with sigma 35 HU -> raw upper is +15 HU
    voxels = rng.normal(loc=-55.0, scale=35.0, size=10000)

    config = ThresholdConfig(
        plausible_mu_range=(-150.0, -40.0),
        clamping_max_hu=0.0,
        wide_sigma_warn_hu=40.0,
    )
    result = fit_trimmed_gaussian(sub0_voxels=voxels, config=config)

    assert not result.is_fallback
    assert result.clamped_high
    assert result.hu_high == 0.0
    flag_ids = [f.concern for f in result.flags]
    assert "FAT_UPPER_BOUND_CLAMPED" in flag_ids


def test_lower_clamping_at_fallback_low() -> None:
    """Test that adaptive lower threshold is clamped at fallback_low_hu (-190.0 HU)."""
    rng = np.random.default_rng(654)
    # Center at -145 HU with sigma 30 HU -> raw lower is -205 HU
    voxels = rng.normal(loc=-145.0, scale=30.0, size=10000)

    config = ThresholdConfig(
        plausible_mu_range=(-160.0, -50.0),
        fallback_low_hu=-190.0,
        wide_sigma_warn_hu=40.0,
    )
    result = fit_trimmed_gaussian(sub0_voxels=voxels, config=config)

    assert not result.is_fallback
    assert result.clamped_low
    assert result.hu_low == -190.0
    flag_ids = [f.concern for f in result.flags]
    assert "FAT_LOWER_BOUND_CLAMPED" in flag_ids


def test_compute_fat_threshold_3d_volume_and_geometry() -> None:
    """Test compute_fat_threshold on 3D synthetic volume with GridGeometry."""
    shape = (30, 30, 30)
    ct_volume = np.full(shape, -1000.0, dtype=np.float32)  # Air background

    # Pericardium sphere in center
    pericardium_mask = np.zeros(shape, dtype=np.uint8)
    z, y, x = np.ogrid[:30, :30, :30]
    sphere_mask = (z - 15) ** 2 + (y - 15) ** 2 + (x - 15) ** 2 <= 8**2
    pericardium_mask[sphere_mask] = 1

    # Fill sphere with fat values
    rng = np.random.default_rng(101)
    n_sphere = int(np.sum(sphere_mask))
    ct_volume[sphere_mask] = rng.normal(loc=-102.0, scale=14.0, size=n_sphere)

    geo = GridGeometry(
        spacing=(1.5, 1.5, 1.5),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        shape_zyx=shape,
    )

    result = compute_fat_threshold(
        ct_volume=ct_volume,
        pericardium_mask=pericardium_mask,
        geometry=geo,
    )

    assert not result.is_fallback
    assert result.fitted_mu is not None
    assert abs(result.fitted_mu - (-102.0)) < 1.0
    assert result.fat_volume_adaptive_ml > 0.0
    assert result.fat_volume_adaptive_ml == pytest.approx(
        result.fat_voxel_count_adaptive * geo.voxel_volume_ml, rel=1e-5
    )


def test_create_fat_mask_adaptive_vs_conservative() -> None:
    """Test creating 3D masks with adaptive vs conservative options."""
    shape = (20, 20, 20)
    ct_volume = np.full(shape, 50.0, dtype=np.float32)  # soft tissue
    pericardium_mask = np.ones(shape, dtype=np.uint8)

    # Core fat: -100 HU
    ct_volume[5:10, 5:10, 5:10] = -100.0
    # Partial volume fat: -15 HU (in [-30, 0] HU)
    ct_volume[10:15, 10:15, 10:15] = -15.0

    threshold_res = ThresholdResult(
        hu_low=-120.0,
        hu_high=0.0,
        conservative_hu_low=-190.0,
        conservative_hu_high=-30.0,
        gmm_bayes_result=GMMBayesResult(hu_low=-130.0, hu_high=-35.0),
    )

    # Adaptive mask should include both core and partial volume
    mask_adapt = create_fat_mask(
        ct_volume, threshold_res, pericardium_mask=pericardium_mask, conservative=False
    )
    assert np.count_nonzero(mask_adapt[5:10, 5:10, 5:10]) == 125
    assert np.count_nonzero(mask_adapt[10:15, 10:15, 10:15]) == 125

    # Conservative mask should only include core fat <= -30 HU
    mask_cons = create_fat_mask(
        ct_volume, threshold_res, pericardium_mask=pericardium_mask, conservative=True
    )
    assert np.count_nonzero(mask_cons[5:10, 5:10, 5:10]) == 125
    assert np.count_nonzero(mask_cons[10:15, 10:15, 10:15]) == 0

    # GMM Bayes mask should use [-130, -35] HU window
    mask_gmm = create_fat_mask(
        ct_volume, threshold_res, pericardium_mask=pericardium_mask, gmm_bayes=True
    )
    assert np.count_nonzero(mask_gmm[5:10, 5:10, 5:10]) == 125
    assert np.count_nonzero(mask_gmm[10:15, 10:15, 10:15]) == 0


def test_resolution_invariance() -> None:
    """Test that volume quantification is mathematically invariant across voxel spacings."""
    rng = np.random.default_rng(555)

    # Target: 50.0 mL of fat at -100 HU
    target_vol_ml = 50.0

    # Grid 1: 1.5mm isotropic
    sp1 = (1.5, 1.5, 1.5)
    vvol1 = sp1[0] * sp1[1] * sp1[2] / 1000.0
    n1 = int(round(target_vol_ml / vvol1))
    voxels1 = rng.normal(loc=-100.0, scale=12.0, size=n1)

    res1 = fit_trimmed_gaussian(sub0_voxels=voxels1, voxel_volume_ml=vvol1)

    # Grid 2: Native 0.35x0.35x1.5mm
    sp2 = (0.35, 0.35, 1.5)
    vvol2 = sp2[0] * sp2[1] * sp2[2] / 1000.0
    n2 = int(round(target_vol_ml / vvol2))
    voxels2 = rng.normal(loc=-100.0, scale=12.0, size=n2)

    res2 = fit_trimmed_gaussian(sub0_voxels=voxels2, voxel_volume_ml=vvol2)

    # Theoretical 2-sigma Gaussian coverage is 95.45% of total volume (approx 47.73 mL)
    expected_2sigma_vol = target_vol_ml * 0.9545
    assert abs(res1.fat_volume_adaptive_ml - expected_2sigma_vol) < 1.0
    assert abs(res2.fat_volume_adaptive_ml - expected_2sigma_vol) < 1.0
    # Both resolutions converge within 1% of each other
    assert abs(res1.fat_volume_adaptive_ml - res2.fat_volume_adaptive_ml) < 0.2



def test_empty_pericardium_mask_error_handling() -> None:
    """Test compute_fat_threshold handles empty pericardium without crashing."""
    shape = (10, 10, 10)
    ct_volume = np.full(shape, -100.0, dtype=np.float32)
    empty_peri = np.zeros(shape, dtype=np.uint8)

    geo = GridGeometry(
        spacing=(1.5, 1.5, 1.5),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        shape_zyx=shape,
    )

    result = compute_fat_threshold(ct_volume, empty_peri, geo)
    assert result.is_fallback
    assert result.flags[0].severity == QualitySeverity.HIGH


def test_shape_mismatch_raises_value_error() -> None:
    """Test compute_fat_threshold raises ValueError on dimension mismatch."""
    ct_volume = np.zeros((10, 10, 10), dtype=np.float32)
    bad_peri = np.zeros((12, 10, 10), dtype=np.uint8)
    geo = GridGeometry(
        spacing=(1.5, 1.5, 1.5),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
        shape_zyx=(10, 10, 10),
    )

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_fat_threshold(ct_volume, bad_peri, geo)
