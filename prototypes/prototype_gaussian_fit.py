"""Prototype: Trimmed-Gaussian Peak Fitting for CT Adipose Thresholding.

Part of Wayfinder Ticket 3 (Issue #33).
Implements a robust, prominence-based peak detector, trimmed Gaussian curve fitter,
clamping against consensus clinical bounds [-190.0, -30.0] HU, and typed quality flagger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence, Tuple
import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.signal


class QualitySeverity(str, Enum):
    """Severity tier for quality audit flags."""
    HIGH = "HIGH"        # Pipeline failure / fallback / unphysiological
    MEDIUM = "MEDIUM"    # Atypical anatomy / potential contrast issue
    LOW = "LOW"          # Minor tail clamping / parameter spread


@dataclass(frozen=True)
class QualityFlag:
    """An auditable quality flag with severity and clinical description."""
    flag_id: str
    severity: QualitySeverity
    message: str
    metric_value: float | None = None
    threshold_value: float | None = None


@dataclass
class ThresholdConfig:
    """Configuration for adaptive Gaussian and fallback thresholding."""
    fallback_low_hu: float = -190.0
    fallback_high_hu: float = -30.0
    sigma_multiplier: float = 2.0
    smoothing_sigma_hu: float = 2.5
    min_voxel_count: int = 500
    peak_prominence_ratio: float = 0.003
    plausible_mu_range: Tuple[float, float] = (-150.0, -50.0)
    plausible_sigma_range: Tuple[float, float] = (5.0, 40.0)
    trim_window_left_hu: float = 35.0
    trim_window_right_hu: float = 30.0
    wide_sigma_warn_hu: float = 25.0


@dataclass
class ThresholdResult:
    """Complete result of threshold fitting and fat volume quantification."""
    hu_low: float
    hu_high: float
    fitted_mu: float | None = None
    fitted_sigma: float | None = None
    fitted_amplitude: float | None = None
    is_fallback: bool = False
    fallback_reason: str | None = None
    clamped_low: bool = False
    clamped_high: bool = False
    voxel_count_evaluated: int = 0
    fat_voxel_count: int = 0
    fat_volume_ml: float = 0.0
    flags: list[QualityFlag] = field(default_factory=list)

    @property
    def window(self) -> Tuple[float, float]:
        return self.hu_low, self.hu_high


def _gaussian_func(x: np.ndarray, a: float, mu: float, sigma: float) -> np.ndarray:
    """3-parameter Gaussian distribution."""
    return a * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


def fit_trimmed_gaussian_threshold(
    ct_voxels: np.ndarray,
    config: ThresholdConfig | None = None,
    voxel_spacing_mm: Tuple[float, float, float] | None = None,
) -> ThresholdResult:
    """Fit a trimmed Gaussian to sub-0 HU voxels and determine fat thresholds.

    Args:
        ct_voxels: 1D array of CT intensity values (HU), e.g. within pericardial envelope.
        config: Threshold configuration parameters.
        voxel_spacing_mm: Optional (dx, dy, dz) in mm to compute physical volume in mL.

    Returns:
        ThresholdResult with fitted/fallback window, parameters, flags, and volume.
    """
    if config is None:
        config = ThresholdConfig()

    flags: list[QualityFlag] = []
    voxel_volume_ml = 0.0
    if voxel_spacing_mm is not None:
        voxel_volume_ml = (voxel_spacing_mm[0] * voxel_spacing_mm[1] * voxel_spacing_mm[2]) / 1000.0

    # 1. Filter sub-0 HU candidates
    sub0 = ct_voxels[(ct_voxels >= -250.0) & (ct_voxels <= 0.0)]
    n_voxels = len(sub0)

    # Helper to return fallback
    def _make_fallback(reason: str, high_severity: bool = True) -> ThresholdResult:
        low = config.fallback_low_hu
        high = config.fallback_high_hu
        fat_count = int(np.sum((ct_voxels >= low) & (ct_voxels <= high)))
        fat_vol = fat_count * voxel_volume_ml
        
        fallback_flags = list(flags)
        fallback_flags.append(
            QualityFlag(
                flag_id="FAT_THRESHOLD_FALLBACK",
                severity=QualitySeverity.HIGH if high_severity else QualitySeverity.MEDIUM,
                message=f"Gaussian fit fell back to fixed standard window [{low:.1f}, {high:.1f}] HU. Reason: {reason}",
                metric_value=float(n_voxels),
                threshold_value=float(config.min_voxel_count),
            )
        )
        return ThresholdResult(
            hu_low=low,
            hu_high=high,
            is_fallback=True,
            fallback_reason=reason,
            voxel_count_evaluated=n_voxels,
            fat_voxel_count=fat_count,
            fat_volume_ml=fat_vol,
            flags=fallback_flags,
        )

    # Guard: insufficient voxels
    if n_voxels < config.min_voxel_count:
        return _make_fallback(f"Insufficient sub-0 HU voxels ({n_voxels} < {config.min_voxel_count})")

    # 2. Histogram binning (1 HU resolution)
    hist, bin_edges = np.histogram(sub0, bins=250, range=(-250.0, 0.0))
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # 3. Light smoothing for robust peak finding
    smoothed = scipy.ndimage.gaussian_filter1d(hist.astype(float), sigma=config.smoothing_sigma_hu)

    # 4. Topographical peak detection
    prominence_thresh = max(float(n_voxels) * config.peak_prominence_ratio, 10.0)
    peaks, props = scipy.signal.find_peaks(smoothed, prominence=prominence_thresh)

    # Filter peaks in plausible fat region
    fat_peaks = [p for p in peaks if config.plausible_mu_range[0] <= centers[p] <= config.plausible_mu_range[1]]
    if not fat_peaks:
        return _make_fallback("No prominent adipose peak detected in sub-0 HU distribution")

    # Select most prominent peak
    peak_proms = props["prominences"]
    best_peak_idx = fat_peaks[np.argmax([peak_proms[list(peaks).index(p)] for p in fat_peaks])]
    mode_hu = float(centers[best_peak_idx])

    # 5. Trim around mode to clip asymmetric soft tissue / partial volume shoulder
    trim_mask = (centers >= mode_hu - config.trim_window_left_hu) & (centers <= mode_hu + config.trim_window_right_hu)
    x_trim = centers[trim_mask]
    y_trim = smoothed[trim_mask]

    if len(x_trim) < 10:
        return _make_fallback("Trimmed fitting window contains too few bin points")

    # 6. Nonlinear least-squares Gaussian curve fit
    try:
        p0 = [float(y_trim.max()), mode_hu, 15.0]
        bounds_lower = [0.0, config.plausible_mu_range[0], config.plausible_sigma_range[0]]
        bounds_upper = [np.inf, config.plausible_mu_range[1], config.plausible_sigma_range[1]]

        popt, pcov = scipy.optimize.curve_fit(
            _gaussian_func,
            x_trim,
            y_trim,
            p0=p0,
            bounds=(bounds_lower, bounds_upper),
            maxfev=3000,
        )
    except Exception as ex:
        return _make_fallback(f"Gaussian curve_fit optimization failure: {ex}")

    a, mu, sigma = float(popt[0]), float(popt[1]), float(popt[2])

    # Check for NaN / Inf
    if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0):
        return _make_fallback("Fitted parameters contain NaN or non-positive sigma")

    # 7. Compute patient-specific window
    raw_low = mu - config.sigma_multiplier * sigma
    raw_high = mu + config.sigma_multiplier * sigma

    # 8. Clamping logic
    clamped_low = False
    clamped_high = False
    final_low = raw_low
    final_high = raw_high

    if raw_low < config.fallback_low_hu:
        final_low = config.fallback_low_hu
        clamped_low = True
        flags.append(
            QualityFlag(
                flag_id="FAT_LOWER_BOUND_CLAMPED",
                severity=QualitySeverity.LOW,
                message=f"Fitted lower threshold {raw_low:.1f} HU clamped to fallback {config.fallback_low_hu:.1f} HU",
                metric_value=raw_low,
                threshold_value=config.fallback_low_hu,
            )
        )

    if raw_high > config.fallback_high_hu:
        final_high = config.fallback_high_hu
        clamped_high = True
        flags.append(
            QualityFlag(
                flag_id="FAT_UPPER_BOUND_CLAMPED",
                severity=QualitySeverity.LOW,
                message=f"Fitted upper threshold {raw_high:.1f} HU clamped to fallback {config.fallback_high_hu:.1f} HU",
                metric_value=raw_high,
                threshold_value=config.fallback_high_hu,
            )
        )

    if sigma > config.wide_sigma_warn_hu:
        flags.append(
            QualityFlag(
                flag_id="FAT_WIDE_SIGMA_WARNING",
                severity=QualitySeverity.LOW,
                message=f"Fitted Gaussian sigma {sigma:.1f} HU exceeds warning threshold {config.wide_sigma_warn_hu:.1f} HU",
                metric_value=sigma,
                threshold_value=config.wide_sigma_warn_hu,
            )
        )

    # 9. Direct fat volume integration
    fat_mask = (ct_voxels >= final_low) & (ct_voxels <= final_high)
    fat_count = int(np.sum(fat_mask))
    fat_vol = fat_count * voxel_volume_ml

    return ThresholdResult(
        hu_low=final_low,
        hu_high=final_high,
        fitted_mu=mu,
        fitted_sigma=sigma,
        fitted_amplitude=a,
        is_fallback=False,
        fallback_reason=None,
        clamped_low=clamped_low,
        clamped_high=clamped_high,
        voxel_count_evaluated=n_voxels,
        fat_voxel_count=fat_count,
        fat_volume_ml=fat_vol,
        flags=flags,
    )
