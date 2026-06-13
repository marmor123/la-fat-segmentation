"""Preprocessor module for LA Fat Segmentation.

Provides the single function ``resample_to_isotropic`` which resamples
a 3D CT volume (NIfTI) to isotropic voxel spacing using SimpleITK.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import SimpleITK as sitk


@dataclasses.dataclass(frozen=True)
class ResampleResult:
    """Result of resampling a CT volume to isotropic spacing.

    Attributes
    ----------
    ct_array:
        The resampled 3D volume as a numpy array (z, y, x) — i.e.
        the same axis order as SimpleITK internally uses.
    spacing:
        The new isotropic spacing ``(sx, sy, sz)`` in mm.
    origin:
        The spatial origin ``(ox, oy, oz)`` in mm.
    direction:
        3×3 direction cosine matrix.
    original_spacing:
        The spacing of the input volume before resampling.
    original_shape:
        The shape of the input volume before resampling.
    """

    ct_array: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: np.ndarray
    original_spacing: tuple[float, float, float]
    original_shape: tuple[int, int, int]


def resample_to_isotropic(
    ct_path: str,
    target_spacing_mm: float = 1.5,
) -> ResampleResult:
    """Resample a 3D CT volume to isotropic voxel spacing.

    Uses SimpleITK with linear interpolation.  Non-orthonormal direction
    cosines (e.g. oblique acquisitions) are handled correctly because
    SimpleITK carries the full direction matrix through the resampling
    transform.

    Parameters
    ----------
    ct_path:
        Path to a NIfTI (``.nii`` or ``.nii.gz``) file containing the
        CT volume.
    target_spacing_mm:
        Desired isotropic spacing in millimetres.  Defaults to 1.5 mm.

    Returns
    -------
    ResampleResult
        The resampled volume together with spatial metadata.

    Raises
    ------
    RuntimeError
        If the image cannot be read or resampled.
    """
    # ---- Read input ---------------------------------------------------------
    image = sitk.ReadImage(ct_path)

    original_spacing: tuple[float, float, float] = image.GetSpacing()
    original_shape: tuple[int, int, int] = image.GetSize()
    original_origin: tuple[float, float, float] = image.GetOrigin()
    original_direction = np.array(image.GetDirection()).reshape(3, 3)

    target_spacing: tuple[float, float, float] = (
        target_spacing_mm,
        target_spacing_mm,
        target_spacing_mm,
    )

    # ---- Compute output size ------------------------------------------------
    # new_size = ceil(old_size * old_spacing / new_spacing)
    new_size: tuple[int, int, int] = tuple(
        int(math.ceil(sz * osp / target_spacing_mm))
        for sz, osp in zip(original_shape, original_spacing, strict=True)
    )

    # ---- Resample -----------------------------------------------------------
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetOutputDirection(original_direction.ravel().tolist())
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    # Use identity transform — direction/origin/spacing in the filter
    # already define the mapping from fixed to moving image coordinates.
    resampled: sitk.Image = resampler.Execute(image)

    # ---- Extract result -----------------------------------------------------
    ct_array: np.ndarray = sitk.GetArrayFromImage(resampled)
    out_spacing: tuple[float, float, float] = resampled.GetSpacing()
    out_origin: tuple[float, float, float] = resampled.GetOrigin()
    out_direction: np.ndarray = np.array(
        resampled.GetDirection()
    ).reshape(3, 3)

    return ResampleResult(
        ct_array=ct_array,
        spacing=out_spacing,
        origin=out_origin,
        direction=out_direction,
        original_spacing=original_spacing,
        original_shape=original_shape,
    )
