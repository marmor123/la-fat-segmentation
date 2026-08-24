"""Pericardium resolver for the LA Fat Segmentation pipeline.

Given a dict of TotalSegmentator (TS) masks, this module always
returns a valid pericardium binary mask — either directly from TS
(when the mask volume is adequate) or via a convex-hull fallback
constructed from chamber masks.
"""

from __future__ import annotations

import dataclasses
import typing as t

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.spatial import ConvexHull

from la_fat.anatomy import voxel_volume_ml
from la_fat.config import PipelineConfig


@dataclasses.dataclass(frozen=True)
class PericardiumResult:
    """Result of the pericardium resolution process.

    Attributes
    ----------
    mask:
        Binary pericardium mask with the same shape as the input masks.
    fallback_triggered:
        ``True`` when the convex-hull fallback was used.
    fallback_reason:
        Human-readable explanation when *fallback_triggered* is
        ``True``; ``None`` otherwise.
    method:
        Either ``"ts_direct"`` or ``"convex_hull_fallback"``.
    """

    mask: np.ndarray
    fallback_triggered: bool
    fallback_reason: str | None
    method: str
    volume_ml: float = 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_pericardium(
    ts_masks: dict[str, np.ndarray],
    config: PipelineConfig,
    spacing: tuple[float, float, float] = (1.5, 1.5, 1.5),
) -> PericardiumResult:
    """Return a valid pericardium mask, using a fallback if needed.

    Parameters
    ----------
    ts_masks:
        Dictionary mapping structure names to binary mask arrays
        (typically the output of TotalSegmentator).  Masks are
        expected to be on the isotropic grid produced by the
        preprocessor.
    config:
        Pipeline configuration that provides the minimum pericardium
        volume (ml) and the dilation distance (mm) for the fallback.
    spacing:
        Voxel spacing in mm.  Because the preprocessor resamples to
        isotropic resolution the default ``(1.5, 1.5, 1.5)`` is
        appropriate in most cases.

    Returns
    -------
    PericardiumResult

    Raises
    ------
    ValueError
        If no pericardium can be produced (all chamber masks are
        missing or empty).
    """
    vox_vol_ml = voxel_volume_ml(spacing)
    chamber_keys = ["LA", "LV", "RA", "RV", "Aorta"]

    # ---- Normal path: use TS pericardium directly ----
    if "pericardium" in ts_masks:
        peri = ts_masks["pericardium"]
        volume_ml = np.count_nonzero(peri) * vox_vol_ml
        if volume_ml >= config.min_pericardium_volume_ml:
            return PericardiumResult(
                mask=peri.astype(bool),
                fallback_triggered=False,
                fallback_reason=None,
                method="ts_direct",
                volume_ml=float(volume_ml),
            )
        reason = (
            f"TS pericardium volume {volume_ml:.1f} ml < "
            f"{config.min_pericardium_volume_ml:.1f} ml threshold"
        )
    else:
        reason = "TS pericardium mask not found in ts_masks"

    # ---- Fallback: convex hull of available chambers ----
    available = [k for k in chamber_keys if k in ts_masks]
    if not available:
        raise ValueError(
            "Cannot produce pericardium: no chamber masks found in ts_masks. "
            "At least one of {LA, LV, RA, RV, Aorta} is required."
        )

    # Build the union of non-empty chamber masks.
    # We use the first available mask's shape as the reference.
    shape: tuple[int, ...] = ts_masks[available[0]].shape
    union = np.zeros(shape, dtype=bool)
    used_keys: list[str] = []
    for k in available:
        mask = ts_masks[k]
        if mask is not None and np.any(mask):
            union |= mask.astype(bool)
            used_keys.append(k)

    if not np.any(union):
        raise ValueError(
            "Cannot produce pericardium: all available chamber masks "
            f"are empty.  Available keys: {available}"
        )

    # Collect voxel coordinates from the union and compute the convex
    # hull facets.  Each facet is stored as *equations* of the form
    # ``[A, B, C, D]`` where a point ``(x, y, z)`` is on the interior
    # side when ``A*x + B*y + C*z + D <= 0`` (subject to a small
    # tolerance for floating-point precision on the boundary).
    points = np.argwhere(union).astype(np.float64)
    try:
        hull = ConvexHull(points)
    except Exception as exc:
        raise ValueError(
            f"Failed to compute convex hull from chamber voxels: {exc}"
        ) from exc

    # Rasterise the hull into a binary volume (z-slice by z-slice to
    # keep memory usage bounded).
    hull_mask = np.zeros(shape, dtype=bool)
    normals = hull.equations[:, :3]     # (n_facets, 3)
    offsets = hull.equations[:, 3]      # (n_facets,)
    eps = 1e-7  # tolerance for boundary voxels
    for z in range(shape[0]):
        yy, xx = np.meshgrid(
            np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
        )
        zz = np.full_like(yy, z)
        pts = np.column_stack([zz.ravel(), yy.ravel(), xx.ravel()])
        # Signed distance to each facet (positive = outside).
        distances = pts @ normals.T + offsets  # (N, n_facets)
        hull_mask[z] = np.all(distances <= eps, axis=1).reshape(
            shape[1], shape[2]
        )

    # Dilate the hull by the configured physical margin (in mm), correctly accounting for anisotropic spacing
    sampling_zyx = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
    dist_from_hull = distance_transform_edt(~hull_mask, sampling=sampling_zyx)
    dilated = dist_from_hull <= config.pericardium_dilation_mm

    # Build the reason string.
    fallback_reason = reason
    if len(used_keys) < len(chamber_keys):
        missing_keys = [k for k in chamber_keys if k not in used_keys]
        parts = []
        for k in missing_keys:
            if k in ts_masks:
                parts.append(f"{k}=empty")
            else:
                parts.append(f"{k}=missing")
        if parts:
            fallback_reason += (
                f" (chambers used: {', '.join(used_keys)}; "
                f"{'; '.join(parts)})"
            )
    else:
        fallback_reason += (
            f" (chambers used: {', '.join(used_keys)})"
        )

    fallback_volume_ml = float(np.count_nonzero(dilated) * vox_vol_ml)
    return PericardiumResult(
        mask=dilated,
        fallback_triggered=True,
        fallback_reason=fallback_reason,
        method="convex_hull_fallback",
        volume_ml=fallback_volume_ml,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ball_structure(radius_voxels: float) -> np.ndarray:
    """Return a 3D binary ball-shaped structuring element.

    The element has shape ``(2*ceil(radius)+1)`` along each axis;
    voxels whose centre distance from the centre element is **at most**
    *radius_voxels* are included.

    Parameters
    ----------
    radius_voxels:
        Radius of the ball in voxel units (may be fractional).
    """
    if radius_voxels <= 0.0:
        return np.ones((1, 1, 1), dtype=bool)

    radius = int(np.ceil(radius_voxels))
    size = 2 * radius + 1
    centre = np.array([radius, radius, radius], dtype=float)
    z, y, x = np.ogrid[:size, :size, :size]
    dist = np.sqrt(
        (z - centre[0]) ** 2
        + (y - centre[1]) ** 2
        + (x - centre[2]) ** 2
    )
    return dist <= radius_voxels
