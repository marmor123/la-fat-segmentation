"""Production Fat Thresholding and Trimmed Gaussian Peak Fitting Module.

Part of Wayfinder Ticket 7 (Issue #37).
Provides deep, pure-function intensity thresholding for Cardiac CT epicardial
adipose tissue (EAT), featuring:
1. Prominence-based peak detection on sub-0 HU pericardial voxel distributions.
2. Asymmetric tail trimming around mode to reject soft-tissue partial volume shoulders.
3. Levenberg-Marquardt Gaussian curve fitting with strict parameter bounds.
4. Upper-tail partial volume clamping at 0.0 HU (retaining the 1-2 voxel boundary layer).
5. Dual-window and dual-volume computation (Adaptive vs. Conservative [-190, -30] HU).
6. Typed QualitySeverity and QualityFlag auditability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Union
import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.signal

from la_fat.image_ops import GridGeometry
from la_fat.quality_flagger import QualityFlag, QualitySeverity


@dataclass(frozen=True)
class ThresholdConfig:
    """Configuration for adaptive Gaussian, GMM Bayes, and fallback thresholding.

    Attributes
    ----------
    fallback_low_hu:
        Lower bound of consensus fallback window in HU (default -190.0).
    fallback_high_hu:
        Upper bound of consensus fallback window in HU (default -30.0).
    clamping_max_hu:
        Upper ceiling for adaptive threshold in HU (default 0.0).
    sigma_multiplier:
        Number of standard deviations around mean for adaptive window (default 2.0).
    smoothing_sigma_hu:
        Gaussian kernel standard deviation for histogram smoothing in HU (default 2.5).
    min_voxel_count:
        Minimum number of sub-0 HU voxels required for Gaussian fitting (default 500).
    peak_prominence_ratio:
        Relative height fraction required for peak prominence (default 0.003).
    plausible_mu_range:
        Plausible range for Gaussian center in HU (default (-150.0, -50.0)).
    plausible_sigma_range:
        Plausible range for Gaussian width in HU (default (5.0, 40.0)).
    trim_window_left_hu:
        Window width to the left of mode for Gaussian fitting in HU (default 35.0).
    trim_window_right_hu:
        Window width to the right of mode for Gaussian fitting in HU (default 30.0).
    wide_sigma_warn_hu:
        Threshold above which a wide sigma quality flag is raised in HU (default 25.0).
    """

    fallback_low_hu: float = -190.0
    fallback_high_hu: float = -30.0
    clamping_max_hu: float = 0.0
    sigma_multiplier: float = 2.0
    smoothing_sigma_hu: float = 2.5
    min_voxel_count: int = 500
    peak_prominence_ratio: float = 0.003
    plausible_mu_range: Tuple[float, float] = (-150.0, -50.0)
    plausible_sigma_range: Tuple[float, float] = (5.0, 40.0)
    trim_window_left_hu: float = 35.0
    trim_window_right_hu: float = 30.0
    wide_sigma_warn_hu: float = 25.0

    @classmethod
    def from_pipeline_config(cls, cfg: object) -> ThresholdConfig:
        """Construct a ThresholdConfig from a PipelineConfig or dict."""
        if isinstance(cfg, dict):
            return cls(
                fallback_low_hu=float(cfg.get("fat_hu_low", -190.0)),
                fallback_high_hu=float(cfg.get("fat_hu_high", -30.0)),
                clamping_max_hu=float(cfg.get("fat_clamping_max_hu", 0.0)),
                sigma_multiplier=float(cfg.get("fat_sigma_multiplier", 2.0)),
                smoothing_sigma_hu=float(cfg.get("fat_smoothing_sigma_hu", 2.5)),
                min_voxel_count=int(cfg.get("min_fat_voxels", 500)),
                peak_prominence_ratio=float(cfg.get("fat_peak_prominence_ratio", 0.003)),
                wide_sigma_warn_hu=float(cfg.get("fat_wide_sigma_warn_hu", 25.0)),
            )
        return cls(
            fallback_low_hu=float(getattr(cfg, "fat_hu_low", -190.0)),
            fallback_high_hu=float(getattr(cfg, "fat_hu_high", -30.0)),
            clamping_max_hu=float(getattr(cfg, "fat_clamping_max_hu", 0.0)),
            sigma_multiplier=float(getattr(cfg, "fat_sigma_multiplier", 2.0)),
            smoothing_sigma_hu=float(getattr(cfg, "fat_smoothing_sigma_hu", 2.5)),
            min_voxel_count=int(getattr(cfg, "min_fat_voxels", 500)),
            peak_prominence_ratio=float(getattr(cfg, "fat_peak_prominence_ratio", 0.003)),
            wide_sigma_warn_hu=float(getattr(cfg, "fat_wide_sigma_warn_hu", 25.0)),
        )


@dataclass(frozen=True)
class GMMBayesResult:
    """Result of two-component Gaussian Mixture Model with Bayes decision boundary.

    Attributes
    ----------
    hu_low:
        Lower bound of fat threshold in HU.
    hu_high:
        Bayes decision boundary threshold where P(Fat|x) == 0.5 in HU.
    fitted_mu_fat:
        Mean of adipose Gaussian component in HU.
    fitted_sigma_fat:
        Standard deviation of adipose Gaussian component in HU.
    weight_fat:
        Mixture weight of adipose component (0 to 1).
    fitted_mu_soft:
        Mean of soft-tissue/partial volume Gaussian component in HU.
    fitted_sigma_soft:
        Standard deviation of soft-tissue component in HU.
    weight_soft:
        Mixture weight of soft-tissue component (0 to 1).
    is_fallback:
        True if GMM fitting failed and fallback window was used.
    fallback_reason:
        Reason string if fallback was triggered.
    """

    hu_low: float = -190.0
    hu_high: float = -30.0
    fitted_mu_fat: float | None = None
    fitted_sigma_fat: float | None = None
    weight_fat: float | None = None
    fitted_mu_soft: float | None = None
    fitted_sigma_soft: float | None = None
    weight_soft: float | None = None
    is_fallback: bool = False
    fallback_reason: str | None = None


@dataclass(frozen=True)
class ThresholdResult:
    """Complete result of threshold fitting and fat volume quantification.

    Attributes
    ----------
    hu_low:
        Adaptive lower threshold in HU (Trimmed Gaussian).
    hu_high:
        Adaptive upper threshold in HU (Trimmed Gaussian, clamped at clamping_max_hu).
    conservative_hu_low:
        Standard clinical lower threshold (-190.0 HU).
    conservative_hu_high:
        Standard clinical upper threshold (-30.0 HU).
    fitted_mu:
        Fitted Gaussian center (mean) in HU, or None if fallback.
    fitted_sigma:
        Fitted Gaussian width (std dev) in HU, or None if fallback.
    fitted_amplitude:
        Fitted Gaussian peak amplitude, or None if fallback.
    is_fallback:
        True if Gaussian fitting failed and standard fallback was used.
    fallback_reason:
        Detailed explanation if fallback was triggered.
    clamped_low:
        True if the lower bound was clamped to fallback_low_hu.
    clamped_high:
        True if the upper bound was clamped to clamping_max_hu.
    voxel_count_evaluated:
        Number of candidate sub-0 HU voxels evaluated.
    fat_voxel_count_adaptive:
        Number of voxels inside the adaptive window [hu_low, hu_high].
    fat_voxel_count_conservative:
        Number of voxels inside the conservative window [-190.0, -30.0] HU.
    fat_volume_adaptive_ml:
        Quantified physical fat volume in mL using adaptive window.
    fat_volume_conservative_ml:
        Quantified physical fat volume in mL using conservative window.
    fat_voxel_count_gmm_bayes:
        Number of voxels inside the GMM Bayes decision boundary window.
    fat_volume_gmm_bayes_ml:
        Quantified physical fat volume in mL using GMM Bayes decision boundary.
    gmm_bayes_result:
        Optional detailed result from two-component GMM Bayes model.
    flags:
        List of auditable quality flags with severity tiers.
    """

    hu_low: float
    hu_high: float
    conservative_hu_low: float = -190.0
    conservative_hu_high: float = -30.0
    fitted_mu: float | None = None
    fitted_sigma: float | None = None
    fitted_amplitude: float | None = None
    is_fallback: bool = False
    fallback_reason: str | None = None
    clamped_low: bool = False
    clamped_high: bool = False
    voxel_count_evaluated: int = 0
    fat_voxel_count_adaptive: int = 0
    fat_voxel_count_conservative: int = 0
    fat_volume_adaptive_ml: float = 0.0
    fat_volume_conservative_ml: float = 0.0
    fat_voxel_count_gmm_bayes: int = 0
    fat_volume_gmm_bayes_ml: float = 0.0
    gmm_bayes_result: GMMBayesResult | None = None
    flags: list[QualityFlag] = field(default_factory=list)

    @property
    def window(self) -> Tuple[float, float]:
        """Adaptive window (hu_low, hu_high)."""
        return self.hu_low, self.hu_high

    @property
    def conservative_window(self) -> Tuple[float, float]:
        """Conservative standard window (-190.0, -30.0)."""
        return self.conservative_hu_low, self.conservative_hu_high

    @property
    def gmm_bayes_window(self) -> Tuple[float, float]:
        """GMM Bayes decision boundary window."""
        if self.gmm_bayes_result is not None:
            return self.gmm_bayes_result.hu_low, self.gmm_bayes_result.hu_high
        return self.conservative_hu_low, self.conservative_hu_high

    @property
    def fat_voxel_count(self) -> int:
        """Alias for fat_voxel_count_adaptive."""
        return self.fat_voxel_count_adaptive

    @property
    def fat_volume_ml(self) -> float:
        """Alias for fat_volume_adaptive_ml."""
        return self.fat_volume_adaptive_ml


def _gaussian_func(x: np.ndarray, a: float, mu: float, sigma: float) -> np.ndarray:
    """3-parameter Gaussian distribution function."""
    return a * np.exp(-((x - mu) ** 2) / (2.0 * (sigma ** 2)))


def fit_two_component_gmm_bayes(
    sub0_voxels: np.ndarray,
    config: ThresholdConfig | None = None,
    random_state: int = 42,
) -> GMMBayesResult:
    """Fit a two-component Gaussian Mixture Model and determine Bayes decision boundary.

    Fits two Gaussian distributions (adipose tissue + partial volume/soft tissue)
    and computes the upper decision threshold where posterior P(Fat | x) == 0.5.

    Parameters
    ----------
    sub0_voxels:
        1D array of CT intensity values (HU) within the pericardium (<= 0 HU).
    config:
        Thresholding configuration parameters.
    random_state:
        Random seed for reproducible EM initialization.

    Returns
    -------
    GMMBayesResult
        Fitted component means, std devs, mixture weights, and Bayes threshold.
    """
    from sklearn.mixture import GaussianMixture

    if config is None:
        config = ThresholdConfig()

    arr = np.asarray(sub0_voxels, dtype=np.float32).flatten()
    filtered = arr[(arr >= -250.0) & (arr <= 0.0)]
    n_voxels = int(len(filtered))

    if n_voxels < config.min_voxel_count:
        return GMMBayesResult(
            hu_low=config.fallback_low_hu,
            hu_high=config.fallback_high_hu,
            is_fallback=True,
            fallback_reason=f"Insufficient sub-0 HU voxels ({n_voxels} < {config.min_voxel_count})",
        )

    if len(filtered) > 100000:
        rng = np.random.default_rng(random_state)
        sample = rng.choice(filtered, size=100000, replace=False)
    else:
        sample = filtered

    X = sample.reshape(-1, 1)
    init_means = np.array([[-95.0], [-25.0]])

    try:
        gmm = GaussianMixture(
            n_components=2,
            means_init=init_means,
            covariance_type="full",
            max_iter=200,
            random_state=random_state,
        )
        gmm.fit(X)
    except Exception as ex:
        return GMMBayesResult(
            hu_low=config.fallback_low_hu,
            hu_high=config.fallback_high_hu,
            is_fallback=True,
            fallback_reason=f"GMM fitting exception: {ex}",
        )

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()

    fat_idx = int(np.argmin(means))
    soft_idx = int(np.argmax(means))

    mu_fat, sigma_fat = float(means[fat_idx]), float(stds[fat_idx])
    mu_soft, sigma_soft = float(means[soft_idx]), float(stds[soft_idx])
    w_fat, w_soft = float(weights[fat_idx]), float(weights[soft_idx])

    # Find crossing where fat posterior drops below 0.5 between mu_fat and 0.0 HU
    grid_hu = np.linspace(-250.0, 0.0, 2501)
    probs = gmm.predict_proba(grid_hu.reshape(-1, 1))
    fat_probs = probs[:, fat_idx]

    valid_crossings = grid_hu[(grid_hu > mu_fat) & (fat_probs < 0.5)]
    if len(valid_crossings) > 0:
        bayes_high = float(valid_crossings[0])
    else:
        bayes_high = min(0.0, mu_fat + config.sigma_multiplier * sigma_fat)

    raw_low = mu_fat - config.sigma_multiplier * sigma_fat
    bayes_low = max(config.fallback_low_hu, raw_low)
    bayes_high = min(config.clamping_max_hu, bayes_high)

    return GMMBayesResult(
        hu_low=bayes_low,
        hu_high=bayes_high,
        fitted_mu_fat=mu_fat,
        fitted_sigma_fat=sigma_fat,
        weight_fat=w_fat,
        fitted_mu_soft=mu_soft,
        fitted_sigma_soft=sigma_soft,
        weight_soft=w_soft,
        is_fallback=False,
    )


def fit_trimmed_gaussian(
    sub0_voxels: np.ndarray,
    config: ThresholdConfig | None = None,
    voxel_volume_ml: float = 0.0,
) -> ThresholdResult:
    """Fit a trimmed Gaussian to sub-0 HU CT voxels and determine fat thresholds.

    Parameters
    ----------
    sub0_voxels:
        1D array of CT intensity values (HU) within the pericardium (typically <= 0 HU).
    config:
        Thresholding configuration parameters.
    voxel_volume_ml:
        Physical volume of a single voxel in mL (cm³) for volume integration.

    Returns
    -------
    ThresholdResult
        Fitted thresholds, Gaussian parameters, flags, and dual-window volumes.
    """
    if config is None:
        config = ThresholdConfig()

    flags: list[QualityFlag] = []

    # 1. Filter sub-0 HU candidates
    arr = np.asarray(sub0_voxels, dtype=np.float32)
    filtered = arr[(arr >= -250.0) & (arr <= 0.0)]
    n_voxels = int(len(filtered))

    def _make_fallback(reason: str, high_severity: bool = True) -> ThresholdResult:
        low = config.fallback_low_hu
        high = config.fallback_high_hu
        # Count within input array
        fat_count_cons = int(np.sum((arr >= low) & (arr <= high)))
        fat_vol_cons = fat_count_cons * voxel_volume_ml
        # In fallback, adaptive matches conservative
        fat_count_adapt = fat_count_cons
        fat_vol_adapt = fat_vol_cons

        fallback_flags = list(flags)
        fallback_flags.append(
            QualityFlag(
                severity=QualitySeverity.HIGH if high_severity else QualitySeverity.MEDIUM,
                concern="FAT_THRESHOLD_FALLBACK",
                detail=(
                    f"Gaussian fit fell back to fixed standard window [{low:.1f}, {high:.1f}] HU. "
                    f"Reason: {reason}"
                ),
                threshold_value=float(config.min_voxel_count),
                actual_value=float(n_voxels),
            )
        )
        return ThresholdResult(
            hu_low=low,
            hu_high=high,
            conservative_hu_low=low,
            conservative_hu_high=high,
            is_fallback=True,
            fallback_reason=reason,
            voxel_count_evaluated=n_voxels,
            fat_voxel_count_adaptive=fat_count_adapt,
            fat_voxel_count_conservative=fat_count_cons,
            fat_volume_adaptive_ml=fat_vol_adapt,
            fat_volume_conservative_ml=fat_vol_cons,
            flags=fallback_flags,
        )

    # Guard: insufficient voxels
    if n_voxels < config.min_voxel_count:
        return _make_fallback(
            f"Insufficient sub-0 HU voxels ({n_voxels} < {config.min_voxel_count})"
        )

    # 2. Histogram binning (1 HU resolution)
    hist, bin_edges = np.histogram(filtered, bins=250, range=(-250.0, 0.0))
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # 3. Light smoothing for robust peak detection
    smoothed = scipy.ndimage.gaussian_filter1d(
        hist.astype(float), sigma=config.smoothing_sigma_hu
    )

    # 4. Topographical peak detection
    prominence_thresh = max(
        min(float(n_voxels) * config.peak_prominence_ratio, float(smoothed.max()) * 0.05),
        5.0,
    )
    peaks, props = scipy.signal.find_peaks(smoothed, prominence=prominence_thresh)

    # Filter peaks in plausible fat region
    fat_peaks = [
        p
        for p in peaks
        if config.plausible_mu_range[0] <= centers[p] <= config.plausible_mu_range[1]
    ]
    if not fat_peaks:
        return _make_fallback("No prominent adipose peak detected in sub-0 HU distribution")

    # Select most prominent peak
    peak_proms = props["prominences"]
    best_peak_idx = fat_peaks[np.argmax([peak_proms[list(peaks).index(p)] for p in fat_peaks])]
    mode_hu = float(centers[best_peak_idx])

    # 5. Trim around mode to clip asymmetric soft tissue / partial volume shoulder
    trim_mask = (centers >= mode_hu - config.trim_window_left_hu) & (
        centers <= mode_hu + config.trim_window_right_hu
    )
    x_trim = centers[trim_mask]
    y_trim = smoothed[trim_mask]

    if len(x_trim) < 10:
        return _make_fallback("Trimmed fitting window contains too few bin points")

    # 6. Nonlinear least-squares Gaussian curve fit
    try:
        p0 = [float(y_trim.max()), mode_hu, 15.0]
        bounds_lower = [0.0, config.plausible_mu_range[0], config.plausible_sigma_range[0]]
        bounds_upper = [np.inf, config.plausible_mu_range[1], config.plausible_sigma_range[1]]

        popt, _ = scipy.optimize.curve_fit(
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
                severity=QualitySeverity.LOW,
                concern="FAT_LOWER_BOUND_CLAMPED",
                detail=(
                    f"Fitted lower threshold {raw_low:.1f} HU clamped to "
                    f"fallback {config.fallback_low_hu:.1f} HU"
                ),
                threshold_value=config.fallback_low_hu,
                actual_value=raw_low,
            )
        )

    # Clamping upper bound at clamping_max_hu (0.0 HU)
    if raw_high > config.clamping_max_hu:
        final_high = config.clamping_max_hu
        clamped_high = True
        flags.append(
            QualityFlag(
                severity=QualitySeverity.LOW,
                concern="FAT_UPPER_BOUND_CLAMPED",
                detail=(
                    f"Fitted upper threshold {raw_high:.1f} HU clamped to "
                    f"ceiling {config.clamping_max_hu:.1f} HU"
                ),
                threshold_value=config.clamping_max_hu,
                actual_value=raw_high,
            )
        )

    if sigma > config.wide_sigma_warn_hu:
        flags.append(
            QualityFlag(
                severity=QualitySeverity.LOW,
                concern="FAT_WIDE_SIGMA_WARNING",
                detail=(
                    f"Fitted Gaussian sigma {sigma:.1f} HU exceeds warning "
                    f"threshold {config.wide_sigma_warn_hu:.1f} HU"
                ),
                threshold_value=config.wide_sigma_warn_hu,
                actual_value=sigma,
            )
        )

    # 9. Direct fat volume integration
    fat_count_adapt = int(np.sum((arr >= final_low) & (arr <= final_high)))
    fat_vol_adapt = fat_count_adapt * voxel_volume_ml

    fat_count_cons = int(
        np.sum((arr >= config.fallback_low_hu) & (arr <= config.fallback_high_hu))
    )
    fat_vol_cons = fat_count_cons * voxel_volume_ml

    return ThresholdResult(
        hu_low=final_low,
        hu_high=final_high,
        conservative_hu_low=config.fallback_low_hu,
        conservative_hu_high=config.fallback_high_hu,
        fitted_mu=mu,
        fitted_sigma=sigma,
        fitted_amplitude=a,
        is_fallback=False,
        fallback_reason=None,
        clamped_low=clamped_low,
        clamped_high=clamped_high,
        voxel_count_evaluated=n_voxels,
        fat_voxel_count_adaptive=fat_count_adapt,
        fat_voxel_count_conservative=fat_count_cons,
        fat_volume_adaptive_ml=fat_vol_adapt,
        fat_volume_conservative_ml=fat_vol_cons,
        flags=flags,
    )


def compute_fat_threshold(
    ct_volume: np.ndarray,
    pericardium_mask: np.ndarray,
    geometry: GridGeometry,
    config: ThresholdConfig | None = None,
) -> ThresholdResult:
    """Compute adaptive fat thresholds within the 3D pericardial envelope.

    Parameters
    ----------
    ct_volume:
        3D NumPy array of CT Hounsfield Units (shape z, y, x).
    pericardium_mask:
        3D binary mask of the pericardium envelope (same shape as ct_volume).
    geometry:
        Spatial grid geometry defining voxel spacing and origin.
    config:
        Optional thresholding configuration parameters.

    Returns
    -------
    ThresholdResult
        Fitted thresholds, flags, and physical volume quantifications.
    """
    ct_arr = np.asarray(ct_volume, dtype=np.float32)
    peri_arr = np.asarray(pericardium_mask, dtype=bool)

    if ct_arr.shape != peri_arr.shape:
        raise ValueError(
            f"Shape mismatch: ct_volume {ct_arr.shape} vs pericardium_mask {peri_arr.shape}"
        )
    if ct_arr.shape != geometry.shape_zyx:
        raise ValueError(
            f"Shape mismatch: ct_volume {ct_arr.shape} vs geometry.shape_zyx {geometry.shape_zyx}"
        )

    # Extract pericardial voxels
    if not np.any(peri_arr):
        if config is None:
            config = ThresholdConfig()
        return ThresholdResult(
            hu_low=config.fallback_low_hu,
            hu_high=config.fallback_high_hu,
            conservative_hu_low=config.fallback_low_hu,
            conservative_hu_high=config.fallback_high_hu,
            is_fallback=True,
            fallback_reason="Empty or non-existent pericardium mask provided",
            voxel_count_evaluated=0,
            flags=[
                QualityFlag(
                    severity=QualitySeverity.HIGH,
                    concern="FAT_THRESHOLD_FALLBACK",
                    detail="Pericardium mask is empty (0 voxels).",
                    threshold_value=float(config.min_voxel_count),
                    actual_value=0.0,
                )
            ],
        )

    peri_voxels = ct_arr[peri_arr]
    trimmed_res = fit_trimmed_gaussian(
        sub0_voxels=peri_voxels,
        config=config,
        voxel_volume_ml=geometry.voxel_volume_ml,
    )
    gmm_res = fit_two_component_gmm_bayes(
        sub0_voxels=peri_voxels,
        config=config,
    )
    gmm_count = int(np.sum((peri_voxels >= gmm_res.hu_low) & (peri_voxels <= gmm_res.hu_high)))
    gmm_vol = gmm_count * geometry.voxel_volume_ml

    return ThresholdResult(
        hu_low=trimmed_res.hu_low,
        hu_high=trimmed_res.hu_high,
        conservative_hu_low=trimmed_res.conservative_hu_low,
        conservative_hu_high=trimmed_res.conservative_hu_high,
        fitted_mu=trimmed_res.fitted_mu,
        fitted_sigma=trimmed_res.fitted_sigma,
        fitted_amplitude=trimmed_res.fitted_amplitude,
        is_fallback=trimmed_res.is_fallback,
        fallback_reason=trimmed_res.fallback_reason,
        clamped_low=trimmed_res.clamped_low,
        clamped_high=trimmed_res.clamped_high,
        voxel_count_evaluated=trimmed_res.voxel_count_evaluated,
        fat_voxel_count_adaptive=trimmed_res.fat_voxel_count_adaptive,
        fat_voxel_count_conservative=trimmed_res.fat_voxel_count_conservative,
        fat_volume_adaptive_ml=trimmed_res.fat_volume_adaptive_ml,
        fat_volume_conservative_ml=trimmed_res.fat_volume_conservative_ml,
        fat_voxel_count_gmm_bayes=gmm_count,
        fat_volume_gmm_bayes_ml=gmm_vol,
        gmm_bayes_result=gmm_res,
        flags=trimmed_res.flags,
    )


def create_fat_mask(
    ct_volume: np.ndarray,
    threshold_result: ThresholdResult,
    pericardium_mask: np.ndarray | None = None,
    conservative: bool = False,
    gmm_bayes: bool = False,
) -> np.ndarray:
    """Create a 3D binary fat mask from CT volume and threshold result.

    Parameters
    ----------
    ct_volume:
        3D NumPy array of CT Hounsfield Units.
    threshold_result:
        Result containing fitted or fallback threshold windows.
    pericardium_mask:
        Optional 3D binary mask of pericardium to restrict fat extraction.
    conservative:
        If True, uses conservative [-190.0, -30.0] HU window.
    gmm_bayes:
        If True, uses GMM Bayes decision boundary window.

    Returns
    -------
    np.ndarray
        Binary uint8 mask (0 or 1) of identified fat voxels.
    """
    ct_arr = np.asarray(ct_volume, dtype=np.float32)
    if gmm_bayes:
        low, high = threshold_result.gmm_bayes_window
    elif conservative:
        low, high = threshold_result.conservative_window
    else:
        low, high = threshold_result.window

    fat_mask = (ct_arr >= low) & (ct_arr <= high)
    if pericardium_mask is not None:
        peri_arr = np.asarray(pericardium_mask, dtype=bool)
        if peri_arr.shape != ct_arr.shape:
            raise ValueError(
                f"Shape mismatch: ct_volume {ct_arr.shape} vs pericardium_mask {peri_arr.shape}"
            )
        fat_mask &= peri_arr

    return fat_mask.astype(np.uint8)
