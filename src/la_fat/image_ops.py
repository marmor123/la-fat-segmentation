"""Deep Image Operations and Grid Geometry Seam for LA Fat Segmentation.

Provides fundamental primitives for 3D spatial grid operations, isotropic
resampling, reference-locked grid matching, and NIfTI/SimpleITK coordinate
transformations.
"""

from __future__ import annotations

import dataclasses
import math
import os
import pathlib
from typing import Union

import numpy as np
import SimpleITK as sitk

__all__ = [
    "GridGeometry",
    "ResampleResult",
    "apply_grid_geometry",
    "get_grid_geometry",
    "resample_to_isotropic",
    "resample_to_reference",
]


# ---------------------------------------------------------------------------
# Interpolator Mapping
# ---------------------------------------------------------------------------

_INTERPOLATOR_MAP: dict[str, int] = {
    "nearest": sitk.sitkNearestNeighbor,
    "nearest_neighbor": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "bspline": sitk.sitkBSpline,
    "gaussian": sitk.sitkGaussian,
    "label_gaussian": sitk.sitkLabelGaussian,
}


def _resolve_interpolator(
    interpolator: int | str | None,
    is_label: bool,
) -> int:
    """Resolve an interpolator argument to a SimpleITK constant."""
    if interpolator is None:
        return sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    if isinstance(interpolator, int):
        return interpolator
    interp_key = str(interpolator).strip().lower()
    if interp_key in _INTERPOLATOR_MAP:
        return _INTERPOLATOR_MAP[interp_key]
    raise ValueError(
        f"Unknown interpolator: {interpolator}. "
        f"Supported: {list(_INTERPOLATOR_MAP.keys())} or SimpleITK integer constants."
    )


# ---------------------------------------------------------------------------
# Grid Geometry Dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GridGeometry:
    """Immutable spatial geometry of a 3D medical image volume.

    Attributes
    ----------
    spacing:
        Voxel spacing ``(sx, sy, sz)`` in millimetres (x, y, z order).
    origin:
        World origin coordinates ``(ox, oy, oz)`` in mm (SimpleITK LPS).
    direction:
        3×3 direction cosine matrix in SimpleITK LPS convention.
    shape_zyx:
        Voxel array dimensions ``(dim_z, dim_y, dim_x)`` matching NumPy indexing.
    """

    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: np.ndarray
    shape_zyx: tuple[int, int, int]

    def __post_init__(self) -> None:
        if len(self.spacing) != 3 or any(s <= 0 for s in self.spacing):
            raise ValueError(f"Invalid voxel spacing: {self.spacing}. Must be 3 positive floats.")
        if len(self.origin) != 3:
            raise ValueError(f"Invalid origin: {self.origin}. Must be 3 floats.")
        dir_arr = np.asarray(self.direction, dtype=np.float64)
        if dir_arr.shape != (3, 3):
            raise ValueError(f"Invalid direction matrix shape: {dir_arr.shape}. Must be (3, 3).")
        if len(self.shape_zyx) != 3 or any(d <= 0 for d in self.shape_zyx):
            raise ValueError(f"Invalid shape_zyx: {self.shape_zyx}. Must be 3 positive integers.")
        # Ensure direction array is immutable in frozen dataclass
        dir_arr.flags.writeable = False
        object.__setattr__(self, "direction", dir_arr)

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        """Voxel array dimensions in (dim_x, dim_y, dim_z) order matching SimpleITK GetSize()."""
        return (self.shape_zyx[2], self.shape_zyx[1], self.shape_zyx[0])

    @property
    def voxel_volume_ml(self) -> float:
        """Physical volume of a single voxel in mL (cm³)."""
        return float(self.spacing[0] * self.spacing[1] * self.spacing[2] / 1000.0)

    @property
    def total_volume_ml(self) -> float:
        """Total physical volume of the bounding grid box in mL."""
        total_voxels = int(self.shape_zyx[0] * self.shape_zyx[1] * self.shape_zyx[2])
        return float(total_voxels * self.voxel_volume_ml)

    def to_affine(self) -> np.ndarray:
        """Construct the 4×4 affine matrix from direction, spacing, and origin.

        Returns
        -------
        np.ndarray
            4×4 affine transformation matrix ``[R @ diag(s) | t; 0 0 0 1]``.
        """
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = self.direction @ np.diag(self.spacing)
        affine[:3, 3] = self.origin
        return affine

    def to_ras_affine(self) -> np.ndarray:
        """Construct standard NIfTI RAS 4×4 affine matrix.

        Converts SimpleITK native LPS coordinates to standard NIfTI RAS coordinates
        using the transformation ``LPS_from_RAS = diag(-1, -1, 1)``.
        """
        lps_from_ras = np.diag([-1.0, -1.0, 1.0])
        ras_direction = lps_from_ras @ self.direction
        ras_origin = lps_from_ras @ np.array(self.origin, dtype=np.float64)
        affine = np.eye(4, dtype=np.float64)
        affine[:3, :3] = ras_direction @ np.diag(self.spacing)
        affine[:3, 3] = ras_origin
        return affine

    @classmethod
    def from_affine(
        cls,
        affine: np.ndarray,
        shape_zyx: tuple[int, int, int],
    ) -> GridGeometry:
        """Construct GridGeometry from a 4×4 affine matrix and array shape."""
        aff = np.asarray(affine, dtype=np.float64)
        if aff.shape != (4, 4):
            raise ValueError(f"Invalid affine shape: {aff.shape}. Expected (4, 4).")
        origin = (float(aff[0, 3]), float(aff[1, 3]), float(aff[2, 3]))
        rot_scale = aff[:3, :3]
        spacing_vec = np.linalg.norm(rot_scale, axis=0)
        spacing = (float(spacing_vec[0]), float(spacing_vec[1]), float(spacing_vec[2]))
        # Normalized direction columns
        inv_spacing = np.diag(1.0 / np.where(spacing_vec > 0, spacing_vec, 1.0))
        direction = rot_scale @ inv_spacing
        return cls(
            spacing=spacing,
            origin=origin,
            direction=direction,
            shape_zyx=shape_zyx,
        )

    @classmethod
    def from_sitk_image(cls, image: sitk.Image) -> GridGeometry:
        """Extract GridGeometry from a SimpleITK Image instance."""
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        direction = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)
        size_xyz = image.GetSize()
        shape_zyx = (int(size_xyz[2]), int(size_xyz[1]), int(size_xyz[0]))
        return cls(
            spacing=spacing,
            origin=origin,
            direction=direction,
            shape_zyx=shape_zyx,
        )

    def to_sitk_image(self, array_zyx: np.ndarray) -> sitk.Image:
        """Convert a 3D NumPy array into a SimpleITK Image with this geometry applied."""
        arr = np.asarray(array_zyx)
        if arr.shape != self.shape_zyx:
            raise ValueError(
                f"Array shape {arr.shape} does not match GridGeometry shape_zyx {self.shape_zyx}."
            )
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing(self.spacing)
        img.SetOrigin(self.origin)
        img.SetDirection(self.direction.ravel().tolist())
        return img


# ---------------------------------------------------------------------------
# Resample Result Dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ResampleResult:
    """Result of resampling a 3D image volume.

    Attributes
    ----------
    array:
        Resampled 3D array in ``(z, y, x)`` order.
    geometry:
        Spatial grid geometry of the resampled output.
    original_geometry:
        Spatial grid geometry of the input before resampling.
    """

    array: np.ndarray
    geometry: GridGeometry
    original_geometry: GridGeometry

    @property
    def ct_array(self) -> np.ndarray:
        """Alias for array for backwards compatibility with preprocessor."""
        return self.array

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Output voxel spacing ``(sx, sy, sz)`` in mm."""
        return self.geometry.spacing

    @property
    def origin(self) -> tuple[float, float, float]:
        """Output origin coordinates ``(ox, oy, oz)`` in mm."""
        return self.geometry.origin

    @property
    def direction(self) -> np.ndarray:
        """Output 3×3 direction cosine matrix."""
        return self.geometry.direction

    @property
    def shape_zyx(self) -> tuple[int, int, int]:
        """Output array shape ``(dim_z, dim_y, dim_x)``."""
        return self.geometry.shape_zyx

    @property
    def shape(self) -> tuple[int, int, int]:
        """Array shape ``(dim_z, dim_y, dim_x)``."""
        return self.geometry.shape_zyx

    @property
    def original_spacing(self) -> tuple[float, float, float]:
        """Input voxel spacing before resampling."""
        return self.original_geometry.spacing

    @property
    def original_shape(self) -> tuple[int, int, int]:
        """Input shape in ``(dim_x, dim_y, dim_z)`` order (SimpleITK GetSize)."""
        return self.original_geometry.shape_xyz

    @property
    def original_shape_zyx(self) -> tuple[int, int, int]:
        """Input shape in ``(dim_z, dim_y, dim_x)`` NumPy order."""
        return self.original_geometry.shape_zyx

    @property
    def affine(self) -> np.ndarray:
        """4×4 affine matrix of the resampled volume."""
        return self.geometry.to_affine()

    @property
    def affine_4x4(self) -> np.ndarray:
        """4×4 affine matrix of the resampled volume."""
        return self.geometry.to_affine()

    @property
    def sitk_image(self) -> sitk.Image:
        """Underlying SimpleITK Image representation of the resampled volume."""
        return self.geometry.to_sitk_image(self.array)


# ---------------------------------------------------------------------------
# Helper Functions: Image Conversion & Metadata
# ---------------------------------------------------------------------------


def _ensure_sitk_image(
    image_or_path: Union[str, os.PathLike, sitk.Image, np.ndarray],
    geometry: GridGeometry | None = None,
    is_label: bool = False,
) -> tuple[sitk.Image, GridGeometry]:
    """Coerce various input types into a SimpleITK Image and its GridGeometry."""
    if isinstance(image_or_path, (str, os.PathLike, pathlib.Path)):
        path_str = str(image_or_path)
        if not os.path.isfile(path_str):
            raise FileNotFoundError(f"Image file not found: {path_str}")
        img = sitk.ReadImage(
            path_str,
            sitk.sitkUInt8 if is_label else sitk.sitkFloat32,
        )
        return img, GridGeometry.from_sitk_image(img)

    if isinstance(image_or_path, sitk.Image):
        return image_or_path, GridGeometry.from_sitk_image(image_or_path)

    if isinstance(image_or_path, np.ndarray):
        arr = image_or_path
        if arr.ndim != 3:
            raise ValueError(f"Array must be 3-dimensional (z, y, x). Got ndim={arr.ndim}.")
        if geometry is None:
            # Default isotropic 1mm spacing and identity orientation
            geometry = GridGeometry(
                spacing=(1.0, 1.0, 1.0),
                origin=(0.0, 0.0, 0.0),
                direction=np.eye(3),
                shape_zyx=arr.shape,
            )
        img = geometry.to_sitk_image(arr)
        return img, geometry

    raise TypeError(
        f"Unsupported image input type: {type(image_or_path)}. "
        "Expected str, Path, SimpleITK.Image, or np.ndarray."
    )


def apply_grid_geometry(
    array_zyx: np.ndarray,
    geometry: GridGeometry,
) -> sitk.Image:
    """Create a SimpleITK Image from a 3D NumPy array with specified GridGeometry."""
    return geometry.to_sitk_image(array_zyx)


def get_grid_geometry(
    image_or_path: Union[str, os.PathLike, sitk.Image, np.ndarray],
    geometry: GridGeometry | None = None,
) -> GridGeometry:
    """Extract spatial GridGeometry from a file header or in-memory image."""
    if isinstance(image_or_path, (str, os.PathLike, pathlib.Path)):
        path_str = str(image_or_path)
        if not os.path.isfile(path_str):
            raise FileNotFoundError(f"Image file not found: {path_str}")
        reader = sitk.ImageFileReader()
        reader.SetFileName(path_str)
        reader.ReadImageInformation()
        spacing = reader.GetSpacing()
        origin = reader.GetOrigin()
        direction = np.array(reader.GetDirection(), dtype=np.float64).reshape(3, 3)
        size_xyz = reader.GetSize()
        shape_zyx = (int(size_xyz[2]), int(size_xyz[1]), int(size_xyz[0]))
        return GridGeometry(
            spacing=spacing,
            origin=origin,
            direction=direction,
            shape_zyx=shape_zyx,
        )
    _, geo = _ensure_sitk_image(image_or_path, geometry=geometry)
    return geo


# ---------------------------------------------------------------------------
# Core Operations: Resample to Isotropic
# ---------------------------------------------------------------------------


def resample_to_isotropic(
    image_or_path: Union[str, os.PathLike, sitk.Image, np.ndarray],
    target_spacing_mm: float = 1.5,
    is_label: bool = False,
    geometry: GridGeometry | None = None,
    default_value: float | None = None,
    interpolator: int | str | None = None,
) -> ResampleResult:
    """Resample a 3D volume (CT or label mask) to isotropic voxel spacing.

    Parameters
    ----------
    image_or_path:
        Input CT or mask as a file path, SimpleITK Image, or NumPy array.
    target_spacing_mm:
        Target isotropic spacing in millimetres (e.g. 1.5 mm).
    is_label:
        If True, treats input as a discrete integer mask: defaults to
        ``sitkNearestNeighbor``, ``0`` default value, and ``np.uint8`` output.
        If False, treats input as continuous CT intensities: defaults to
        ``sitkLinear``, strict ``-1000.0 HU`` air padding, and ``np.float32`` output.
    geometry:
        GridGeometry required if input is a raw NumPy array.
    default_value:
        Override value for out-of-bounds voxels. Defaults to ``-1000.0`` for CT
        and ``0.0`` for masks.
    interpolator:
        Override interpolator (e.g. ``'linear'``, ``'nearest'``, ``'bspline'``).

    Returns
    -------
    ResampleResult
        The resampled array and spatial metadata.
    """
    if target_spacing_mm <= 0:
        raise ValueError(f"target_spacing_mm must be positive, got {target_spacing_mm}")

    sitk_img, orig_geo = _ensure_sitk_image(
        image_or_path,
        geometry=geometry,
        is_label=is_label,
    )

    interp = _resolve_interpolator(interpolator, is_label=is_label)
    if default_value is None:
        def_val = 0.0 if is_label else -1000.0
    else:
        def_val = float(default_value)

    original_shape_xyz = sitk_img.GetSize()
    original_spacing_xyz = sitk_img.GetSpacing()

    target_spacing = (
        float(target_spacing_mm),
        float(target_spacing_mm),
        float(target_spacing_mm),
    )

    new_size_xyz = tuple(
        int(math.ceil(sz * osp / target_spacing_mm))
        for sz, osp in zip(original_shape_xyz, original_spacing_xyz, strict=True)
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(sitk_img.GetOrigin())
    resampler.SetOutputDirection(sitk_img.GetDirection())
    resampler.SetSize(new_size_xyz)
    resampler.SetInterpolator(interp)
    resampler.SetDefaultPixelValue(def_val)
    resampler.SetOutputPixelType(sitk.sitkUInt8 if is_label else sitk.sitkFloat32)

    resampled_img = resampler.Execute(sitk_img)

    out_array = sitk.GetArrayFromImage(resampled_img)
    if is_label:
        out_array = out_array.astype(np.uint8)
    else:
        out_array = out_array.astype(np.float32)

    out_geo = GridGeometry.from_sitk_image(resampled_img)

    return ResampleResult(
        array=out_array,
        geometry=out_geo,
        original_geometry=orig_geo,
    )


# ---------------------------------------------------------------------------
# Core Operations: Resample to Reference
# ---------------------------------------------------------------------------


def resample_to_reference(
    moving_or_path: Union[str, os.PathLike, sitk.Image, np.ndarray],
    reference_or_path: Union[str, os.PathLike, sitk.Image, np.ndarray],
    is_label: bool = False,
    moving_geometry: GridGeometry | None = None,
    reference_geometry: GridGeometry | None = None,
    default_value: float | None = None,
    interpolator: int | str | None = None,
) -> ResampleResult:
    """Resample a moving volume onto the exact grid of a reference volume.

    Uses SimpleITK's reference-locked resampling (``SetReferenceImage``) to ensure
    identical spatial dimensions, spacing, origin, and direction matrix without
    coordinate drift.

    Parameters
    ----------
    moving_or_path:
        Moving CT volume or mask to resample.
    reference_or_path:
        Reference volume defining target grid geometry (e.g. raw 512×512 native CT).
    is_label:
        If True, uses ``sitkNearestNeighbor`` and ``0`` default padding.
        If False, uses ``sitkLinear`` and ``-1000.0 HU`` air padding.
    moving_geometry:
        Required if ``moving_or_path`` is a raw NumPy array.
    reference_geometry:
        Required if ``reference_or_path`` is a raw NumPy array.
    default_value:
        Override value for out-of-bounds voxels.
    interpolator:
        Override interpolator.

    Returns
    -------
    ResampleResult
        The moving volume resampled onto the exact reference grid.
    """
    moving_img, orig_geo = _ensure_sitk_image(
        moving_or_path,
        geometry=moving_geometry,
        is_label=is_label,
    )
    ref_img, _ = _ensure_sitk_image(
        reference_or_path,
        geometry=reference_geometry,
        is_label=False,
    )

    interp = _resolve_interpolator(interpolator, is_label=is_label)
    if default_value is None:
        def_val = 0.0 if is_label else -1000.0
    else:
        def_val = float(default_value)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_img)
    resampler.SetInterpolator(interp)
    resampler.SetDefaultPixelValue(def_val)
    resampler.SetOutputPixelType(sitk.sitkUInt8 if is_label else sitk.sitkFloat32)

    resampled_img = resampler.Execute(moving_img)

    out_array = sitk.GetArrayFromImage(resampled_img)
    if is_label:
        out_array = out_array.astype(np.uint8)
    else:
        out_array = out_array.astype(np.float32)

    out_geo = GridGeometry.from_sitk_image(resampled_img)

    return ResampleResult(
        array=out_array,
        geometry=out_geo,
        original_geometry=orig_geo,
    )
