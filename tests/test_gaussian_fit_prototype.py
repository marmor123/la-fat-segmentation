"""Unit tests for the Trimmed-Gaussian Peak Fitting prototype (Ticket 3)."""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "prototypes")))

import numpy as np
import pytest
from prototype_gaussian_fit import (
    QualitySeverity,
    ThresholdConfig,
    ThresholdResult,
    fit_trimmed_gaussian_threshold,
)


def test_pure_gaussian_estimation():
    """Test that pure Gaussian parameters are recovered within 1.0 HU."""
    np.random.seed(123)
    voxels = np.random.normal(loc=-105.0, scale=12.0, size=50000)
    config = ThresholdConfig()

    result = fit_trimmed_gaussian_threshold(voxels, config=config, voxel_spacing_mm=(1.0, 1.0, 1.0))

    assert not result.is_fallback
    assert result.fitted_mu is not None
    assert result.fitted_sigma is not None
    assert abs(result.fitted_mu - (-105.0)) < 1.0
    assert abs(result.fitted_sigma - 12.0) < 1.0
    assert result.fat_voxel_count > 45000
    assert result.fat_volume_ml > 45.0


def test_asymmetric_distribution_robustness():
    """Test that an asymmetric muscle shoulder does not bias the fat peak."""
    np.random.seed(456)
    fat = np.random.normal(loc=-95.0, scale=13.0, size=40000)
    shoulder = -np.random.exponential(scale=25.0, size=60000)
    shoulder = shoulder[shoulder >= -250.0]
    voxels = np.concatenate([fat, shoulder])

    result = fit_trimmed_gaussian_threshold(voxels, voxel_spacing_mm=(1.0, 1.0, 1.0))

    assert not result.is_fallback
    assert result.fitted_mu is not None
    assert abs(result.fitted_mu - (-95.0)) < 2.5
    # The fitted high threshold should be tighter than -30 HU to exclude muscle
    assert result.hu_high < -30.0


def test_sparse_voxels_fallback():
    """Test that voxel count < 500 triggers high-severity fallback."""
    np.random.seed(789)
    voxels = np.random.normal(loc=-100.0, scale=15.0, size=200)

    result = fit_trimmed_gaussian_threshold(voxels, voxel_spacing_mm=(1.0, 1.0, 1.0))

    assert result.is_fallback
    assert result.hu_low == -190.0
    assert result.hu_high == -30.0
    assert any(f.flag_id == "FAT_THRESHOLD_FALLBACK" and f.severity == QualitySeverity.HIGH for f in result.flags)


def test_monotonic_slope_no_peak_fallback():
    """Test that pure non-fat slope without local mode triggers fallback."""
    np.random.seed(101)
    slope = -np.random.exponential(scale=15.0, size=50000)
    slope = slope[slope >= -250.0]

    result = fit_trimmed_gaussian_threshold(slope, voxel_spacing_mm=(1.0, 1.0, 1.0))

    assert result.is_fallback
    assert result.hu_low == -190.0
    assert result.hu_high == -30.0


def test_lower_tail_clamping():
    """Test that a broad fat distribution lower tail is safely clamped to -190 HU."""
    np.random.seed(202)
    # Fat centered at -125 HU with sigma=35 -> raw lower bound = -125 - 70 = -195 HU
    voxels = np.random.normal(loc=-125.0, scale=35.0, size=40000)
    config = ThresholdConfig(wide_sigma_warn_hu=25.0)

    result = fit_trimmed_gaussian_threshold(voxels, config=config, voxel_spacing_mm=(1.0, 1.0, 1.0))

    if not result.is_fallback:
        assert result.clamped_low
        assert result.hu_low == -190.0
        assert any(f.flag_id == "FAT_LOWER_BOUND_CLAMPED" for f in result.flags)


def test_volume_calculation_scaling():
    """Test that voxel spacing correctly scales volume in mL."""
    np.random.seed(303)
    voxels = np.random.normal(loc=-100.0, scale=10.0, size=10000)
    # 2x2x2 mm voxel = 8 mm^3 = 0.008 mL per voxel
    result = fit_trimmed_gaussian_threshold(voxels, voxel_spacing_mm=(2.0, 2.0, 2.0))

    expected_ml = result.fat_voxel_count * 0.008
    assert abs(result.fat_volume_ml - expected_ml) < 1e-4
