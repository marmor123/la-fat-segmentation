"""Fat thresholder module for LA Fat Segmentation.

Provides the single function ``compute_fat_threshold`` which fits a
single Gaussian to the sub-0 HU voxel distribution within the
Pericardium ROI and returns the fat HU range.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy import stats as sp_stats

from la_fat.config import PipelineConfig


@dataclasses.dataclass(frozen=True)
class FatThresholdResult:
    """Result of fitting a Gaussian to the sub-0 HU voxel distribution.

    Attributes
    ----------
    hu_low:
        Lower bound of the fat HU range.
    hu_high:
        Upper bound of the fat HU range.
    mean_hu:
        Fitted Gaussian mean (or 0.0 if fallback).
    sigma_hu:
        Fitted Gaussian standard deviation (or 0.0 if fallback).
    fallback_triggered:
        Whether the fixed fallback range was used.
    fallback_reason:
        Human-readable explanation if fallback was triggered, else None.
    method:
        ``"gaussian_fit"`` if the fit succeeded, ``"fixed_fallback"`` otherwise.
    num_voxels_fit:
        Number of sub-0 HU voxels used for fitting.
    """

    hu_low: float
    hu_high: float
    mean_hu: float
    sigma_hu: float
    fallback_triggered: bool
    fallback_reason: str | None
    method: str
    num_voxels_fit: int


def compute_fat_threshold(
    ct_array: np.ndarray,
    pericardium_mask: np.ndarray,
    config: PipelineConfig,
) -> FatThresholdResult:
    """Compute per-patient fat HU threshold by fitting a single Gaussian
    to the sub-0 HU voxels within the pericardium ROI.

    Parameters
    ----------
    ct_array:
        3D CT volume in Hounsfield Units (z, y, x).
    pericardium_mask:
        3D binary mask of the pericardium (same shape as *ct_array*).
        Non-zero values indicate pericardial voxels.
    config:
        Pipeline configuration controlling fallback bounds, minimum voxel
        count, and sigma multiplier.

    Returns
    -------
    FatThresholdResult
        The fitted or fallback fat HU range.
    """
    # ---- Extract pericardium voxels ------------------------------------------
    roi_voxels: np.ndarray = ct_array[pericardium_mask > 0]

    # ---- Filter to sub-0 HU --------------------------------------------------
    sub_zero: np.ndarray = roi_voxels[roi_voxels < 0]
    num_voxels: int = int(sub_zero.size)

    # ---- Check minimum voxels ------------------------------------------------
    if num_voxels < config.min_sub_zero_voxels_for_fit:
        return FatThresholdResult(
            hu_low=config.hu_fallback_low,
            hu_high=config.hu_fallback_high,
            mean_hu=0.0,
            sigma_hu=0.0,
            fallback_triggered=True,
            fallback_reason=(
                f"insufficient sub-0 voxels: {num_voxels} < "
                f"{config.min_sub_zero_voxels_for_fit}"
            ),
            method="fixed_fallback",
            num_voxels_fit=num_voxels,
        )

    # ---- Fit Gaussian (MLE — deterministic) ----------------------------------
    # scipy returns float32 if input is float32; cast to Python float so the
    # type annotations (and isinstance checks) match.
    _mean: float
    _sigma: float
    _mean, _sigma = sp_stats.norm.fit(sub_zero)
    mean_hu: float = float(_mean)
    sigma_hu: float = float(_sigma)

    # ---- Check sigma ---------------------------------------------------------
    if sigma_hu > config.max_gaussian_sigma:
        return FatThresholdResult(
            hu_low=config.hu_fallback_low,
            hu_high=config.hu_fallback_high,
            mean_hu=mean_hu,
            sigma_hu=sigma_hu,
            fallback_triggered=True,
            fallback_reason=(
                f"sigma too large: {sigma_hu:.2f} > "
                f"{config.max_gaussian_sigma}"
            ),
            method="fixed_fallback",
            num_voxels_fit=num_voxels,
        )

    # ---- Compute range -------------------------------------------------------
    hu_low: float = mean_hu - config.gaussian_sigma_multiplier * sigma_hu
    hu_high: float = mean_hu + config.gaussian_sigma_multiplier * sigma_hu

    # ---- Clamp to fallback bounds --------------------------------------------
    hu_low = max(hu_low, config.hu_fallback_low)
    hu_high = min(hu_high, config.hu_fallback_high)

    # ---- Sanity check --------------------------------------------------------
    if hu_low >= hu_high:
        return FatThresholdResult(
            hu_low=config.hu_fallback_low,
            hu_high=config.hu_fallback_high,
            mean_hu=mean_hu,
            sigma_hu=sigma_hu,
            fallback_triggered=True,
            fallback_reason=(
                f"clamped range inverted: low={hu_low} >= high={hu_high}"
            ),
            method="fixed_fallback",
            num_voxels_fit=num_voxels,
        )

    return FatThresholdResult(
        hu_low=hu_low,
        hu_high=hu_high,
        mean_hu=mean_hu,
        sigma_hu=sigma_hu,
        fallback_triggered=False,
        fallback_reason=None,
        method="gaussian_fit",
        num_voxels_fit=num_voxels,
    )
