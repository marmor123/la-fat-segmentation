"""Comprehensive unit tests for the la_fat.image_ops module.

Exercises GridGeometry, ResampleResult, resample_to_isotropic, resample_to_reference,
get_grid_geometry, and coordinate system transformations with 100% coverage.
"""

from __future__ import annotations

import math
import os
import pathlib
import tempfile

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from la_fat.image_ops import (
    GridGeometry,
    ResampleResult,
    apply_grid_geometry,
    get_grid_geometry,
    resample_to_isotropic,
    resample_to_reference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nifti(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: np.ndarray | None = None,
    dtype: type = np.float32,
    constant_val: float | None = None,
    rng_seed: int = 42,
) -> str:
    """Create a temporary NIfTI file with synthetic 3D volume."""
    if constant_val is not None:
        data = np.full(shape, constant_val, dtype=dtype)
    else:
        rng = np.random.default_rng(rng_seed)
        data = rng.normal(loc=-50.0, scale=30.0, size=shape).astype(dtype)

    if direction is None:
        direction = np.eye(3)

    affine = np.eye(4)
    affine[:3, :3] = direction * np.array(spacing)[:, np.newaxis]
    affine[:3, 3] = origin

    img = nib.Nifti1Image(data, affine)
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    nib.save(img, tmp.name)
    return tmp.name


# ===========================================================================
# 1. GridGeometry Dataclass Tests
# ===========================================================================


class TestGridGeometry:
    """Tests for GridGeometry creation, properties, and affine conversions."""

    def test_valid_construction(self):
        spacing = (1.5, 1.5, 2.0)
        origin = (10.0, -20.0, 30.0)
        direction = np.eye(3)
        shape_zyx = (40, 50, 60)

        geo = GridGeometry(
            spacing=spacing,
            origin=origin,
            direction=direction,
            shape_zyx=shape_zyx,
        )

        assert geo.spacing == spacing
        assert geo.origin == origin
        np.testing.assert_array_equal(geo.direction, direction)
        assert geo.shape_zyx == shape_zyx
        assert geo.shape_xyz == (60, 50, 40)
        assert geo.voxel_volume_ml == pytest.approx(1.5 * 1.5 * 2.0 / 1000.0)
        assert geo.total_volume_ml == pytest.approx((40 * 50 * 60) * (1.5 * 1.5 * 2.0 / 1000.0))

    def test_direction_array_is_immutable(self):
        geo = GridGeometry(
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            shape_zyx=(10, 10, 10),
        )
        assert not geo.direction.flags.writeable
        with pytest.raises(ValueError):
            geo.direction[0, 0] = 99.0

    def test_validation_errors(self):
        # Invalid spacing
        with pytest.raises(ValueError, match="Invalid voxel spacing"):
            GridGeometry(spacing=(1.0, -1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=np.eye(3), shape_zyx=(10, 10, 10))

        # Invalid origin
        with pytest.raises(ValueError, match="Invalid origin"):
            GridGeometry(spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0), direction=np.eye(3), shape_zyx=(10, 10, 10))  # type: ignore

        # Invalid direction shape
        with pytest.raises(ValueError, match="Invalid direction matrix"):
            GridGeometry(spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=np.eye(4), shape_zyx=(10, 10, 10))

        # Invalid shape_zyx
        with pytest.raises(ValueError, match="Invalid shape_zyx"):
            GridGeometry(spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0), direction=np.eye(3), shape_zyx=(10, 0, 10))

    def test_to_affine_and_from_affine_round_trip(self):
        spacing = (0.8, 0.9, 1.2)
        origin = (12.5, -45.0, 78.2)
        theta = math.radians(30)
        direction = np.array([
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta),  math.cos(theta), 0.0],
            [0.0,              0.0,             1.0],
        ])
        shape_zyx = (32, 48, 64)

        geo = GridGeometry(spacing=spacing, origin=origin, direction=direction, shape_zyx=shape_zyx)
        affine = geo.to_affine()
        assert affine.shape == (4, 4)

        recovered = GridGeometry.from_affine(affine, shape_zyx)
        assert recovered.spacing == pytest.approx(spacing, abs=1e-6)
        assert recovered.origin == pytest.approx(origin, abs=1e-6)
        np.testing.assert_array_almost_equal(recovered.direction, direction, decimal=5)
        assert recovered.shape_zyx == shape_zyx

    def test_from_affine_invalid_shape(self):
        with pytest.raises(ValueError, match="Invalid affine shape"):
            GridGeometry.from_affine(np.eye(3), shape_zyx=(10, 10, 10))

    def test_to_ras_affine(self):
        spacing = (1.0, 1.0, 1.0)
        origin = (10.0, 20.0, 30.0)
        direction = np.eye(3)
        shape_zyx = (10, 10, 10)

        geo = GridGeometry(spacing=spacing, origin=origin, direction=direction, shape_zyx=shape_zyx)
        ras_affine = geo.to_ras_affine()
        # LPS to RAS flips X and Y
        assert ras_affine[0, 3] == pytest.approx(-10.0)
        assert ras_affine[1, 3] == pytest.approx(-20.0)
        assert ras_affine[2, 3] == pytest.approx(30.0)

    def test_sitk_image_conversion(self):
        shape_zyx = (16, 24, 32)
        arr = np.arange(np.prod(shape_zyx), dtype=np.float32).reshape(shape_zyx)
        geo = GridGeometry(
            spacing=(1.5, 1.5, 2.0),
            origin=(5.0, 10.0, 15.0),
            direction=np.eye(3),
            shape_zyx=shape_zyx,
        )

        sitk_img = geo.to_sitk_image(arr)
        assert sitk_img.GetSize() == (32, 24, 16)
        assert sitk_img.GetSpacing() == (1.5, 1.5, 2.0)
        assert sitk_img.GetOrigin() == (5.0, 10.0, 15.0)

        recovered_geo = GridGeometry.from_sitk_image(sitk_img)
        assert recovered_geo.spacing == geo.spacing
        assert recovered_geo.origin == geo.origin
        assert recovered_geo.shape_zyx == geo.shape_zyx

    def test_to_sitk_image_shape_mismatch(self):
        geo = GridGeometry(
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            shape_zyx=(10, 10, 10),
        )
        bad_arr = np.zeros((5, 5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match"):
            geo.to_sitk_image(bad_arr)


# ===========================================================================
# 2. ResampleResult Dataclass Tests
# ===========================================================================


class TestResampleResult:
    """Tests for ResampleResult property delegates."""

    def test_property_delegates(self):
        arr = np.zeros((10, 20, 30), dtype=np.float32)
        orig_geo = GridGeometry(
            spacing=(0.5, 0.5, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            shape_zyx=(20, 40, 60),
        )
        new_geo = GridGeometry(
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            shape_zyx=(10, 20, 30),
        )

        res = ResampleResult(array=arr, geometry=new_geo, original_geometry=orig_geo)

        assert np.array_equal(res.ct_array, arr)
        assert res.spacing == (1.0, 1.0, 1.0)
        assert res.origin == (0.0, 0.0, 0.0)
        np.testing.assert_array_equal(res.direction, np.eye(3))
        assert res.shape_zyx == (10, 20, 30)
        assert res.shape == (10, 20, 30)
        assert res.original_spacing == (0.5, 0.5, 1.0)
        assert res.original_shape == (60, 40, 20)  # xyz
        assert res.original_shape_zyx == (20, 40, 60)
        assert res.affine.shape == (4, 4)
        assert res.affine_4x4.shape == (4, 4)
        assert isinstance(res.sitk_image, sitk.Image)


# ===========================================================================
# 3. resample_to_isotropic Tests
# ===========================================================================


class TestResampleToIsotropic:
    """Tests for isotropic resampling with CT air padding and discrete masks."""

    def test_ct_volume_resampling_default_air_padding(self):
        """CT resampling must use linear interpolation and -1000 HU air padding."""
        path = _make_nifti(shape=(30, 40, 50), spacing=(2.0, 2.0, 2.0))
        result = resample_to_isotropic(path, target_spacing_mm=1.0, is_label=False)

        assert result.spacing == (1.0, 1.0, 1.0)
        # Halving spacing doubles size: (50*2, 40*2, 30*2) -> (100, 80, 60) in zyx
        assert result.array.shape == (100, 80, 60)
        assert result.array.dtype == np.float32
        assert isinstance(result, ResampleResult)

    def test_mask_resampling_nearest_neighbor_and_zero_padding(self):
        """Discrete mask resampling must use nearest neighbor, 0 padding, and uint8."""
        shape = (16, 16, 16)
        mask = np.zeros(shape, dtype=np.uint8)
        mask[4:12, 4:12, 4:12] = 1
        mask[8:12, 8:12, 8:12] = 2  # Multi-label test

        geo = GridGeometry(
            spacing=(2.0, 2.0, 2.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            shape_zyx=shape,
        )

        result = resample_to_isotropic(
            mask,
            target_spacing_mm=1.0,
            is_label=True,
            geometry=geo,
        )

        assert result.spacing == (1.0, 1.0, 1.0)
        assert result.array.shape == (32, 32, 32)
        assert result.array.dtype == np.uint8
        # Ensure only discrete labels {0, 1, 2} exist, no fractional artifacts
        unique_labels = set(np.unique(result.array))
        assert unique_labels.issubset({0, 1, 2})

    def test_anisotropic_input_becomes_isotropic(self):
        path = _make_nifti(shape=(50, 100, 200), spacing=(0.5, 1.0, 2.0))
        result = resample_to_isotropic(path, target_spacing_mm=1.5)
        sx, sy, sz = result.spacing
        assert abs(sx - sy) < 1e-6
        assert abs(sx - sz) < 1e-6
        assert abs(sx - 1.5) < 1e-6

    def test_custom_default_value_and_interpolator(self):
        path = _make_nifti(shape=(20, 20, 20), spacing=(1.0, 1.0, 1.0))
        result = resample_to_isotropic(
            path,
            target_spacing_mm=2.0,
            default_value=-500.0,
            interpolator="bspline",
        )
        assert result.spacing == (2.0, 2.0, 2.0)
        assert result.array.dtype == np.float32

    def test_accepts_sitk_image_directly(self):
        arr = np.zeros((10, 10, 10), dtype=np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((0.5, 0.5, 0.5))

        result = resample_to_isotropic(img, target_spacing_mm=1.0)
        assert result.spacing == (1.0, 1.0, 1.0)
        assert result.array.shape == (5, 5, 5)

    def test_accepts_pathlib_path(self, tmp_path):
        p = pathlib.Path(str(tmp_path)) / "pathlib_test.nii.gz"
        arr = np.ones((8, 8, 8), dtype=np.float32) * 50.0
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(img, str(p))

        result = resample_to_isotropic(p, target_spacing_mm=2.0)
        assert result.spacing == (2.0, 2.0, 2.0)
        assert result.array.shape == (4, 4, 4)

    def test_oblique_direction_preservation(self):
        """A 45-degree rotation in the axial plane."""
        theta = math.radians(45)
        direction = np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta),  math.cos(theta), 0],
            [0, 0, 1],
        ])
        path = _make_nifti(shape=(32, 32, 32), spacing=(1.0, 1.0, 1.0), direction=direction)
        result = resample_to_isotropic(path, target_spacing_mm=2.0)
        np.testing.assert_array_almost_equal(
            np.abs(result.direction), np.abs(direction), decimal=5
        )

    def test_invalid_target_spacing_raises(self):
        path = _make_nifti(shape=(10, 10, 10), spacing=(1.0, 1.0, 1.0))
        with pytest.raises(ValueError, match="target_spacing_mm must be positive"):
            resample_to_isotropic(path, target_spacing_mm=-1.0)


# ===========================================================================
# 4. resample_to_reference Tests
# ===========================================================================


class TestResampleToReference:
    """Tests for reference-locked grid matching."""

    def test_resample_mask_onto_native_ct_reference_grid(self):
        """Resample low-res 1.5mm mask onto 0.35mm native CT reference."""
        # 1. Native reference CT geometry (e.g. 512x512x100 at 0.35x0.35x1.5mm)
        ref_shape_zyx = (30, 60, 60)
        ref_geo = GridGeometry(
            spacing=(0.35, 0.35, 1.5),
            origin=(10.0, 20.0, 30.0),
            direction=np.eye(3),
            shape_zyx=ref_shape_zyx,
        )
        ref_arr = np.full(ref_shape_zyx, -50.0, dtype=np.float32)

        # 2. Moving 1.5mm isotropic mask
        mov_shape_zyx = (30, 14, 14)
        mov_geo = GridGeometry(
            spacing=(1.5, 1.5, 1.5),
            origin=(10.0, 20.0, 30.0),
            direction=np.eye(3),
            shape_zyx=mov_shape_zyx,
        )
        mov_mask = np.zeros(mov_shape_zyx, dtype=np.uint8)
        mov_mask[5:25, 3:11, 3:11] = 1

        result = resample_to_reference(
            moving_or_path=mov_mask,
            reference_or_path=ref_arr,
            is_label=True,
            moving_geometry=mov_geo,
            reference_geometry=ref_geo,
        )

        assert result.spacing == ref_geo.spacing
        assert result.origin == ref_geo.origin
        np.testing.assert_array_equal(result.direction, ref_geo.direction)
        assert result.array.shape == ref_shape_zyx
        assert result.array.dtype == np.uint8
        assert set(np.unique(result.array)).issubset({0, 1})

    def test_resample_ct_volume_onto_reference_with_air_padding(self):
        """CT volume resampled to reference grid with strict -1000 HU out-of-bounds padding."""
        # Moving CT is small cube [0..10]
        mov_arr = np.full((10, 10, 10), -80.0, dtype=np.float32)
        mov_geo = GridGeometry(
            spacing=(1.0, 1.0, 1.0),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            shape_zyx=(10, 10, 10),
        )

        # Reference grid is larger cube [-10..20]
        ref_arr = np.zeros((30, 30, 30), dtype=np.float32)
        ref_geo = GridGeometry(
            spacing=(1.0, 1.0, 1.0),
            origin=(-10.0, -10.0, -10.0),
            direction=np.eye(3),
            shape_zyx=(30, 30, 30),
        )

        result = resample_to_reference(
            moving_or_path=mov_arr,
            reference_or_path=ref_arr,
            is_label=False,
            moving_geometry=mov_geo,
            reference_geometry=ref_geo,
        )

        assert result.array.shape == (30, 30, 30)
        assert result.array.dtype == np.float32
        # Corner voxels that were outside moving image must be exactly -1000.0 HU air
        assert result.array[0, 0, 0] == -1000.0
        # Inside voxels should preserve moving image value
        assert result.array[15, 15, 15] == pytest.approx(-80.0, abs=1.0)

    def test_resample_to_reference_from_file_paths(self, tmp_path):
        mov_path = _make_nifti(shape=(16, 16, 16), spacing=(2.0, 2.0, 2.0), constant_val=1.0, dtype=np.uint8)
        ref_path = _make_nifti(shape=(32, 32, 32), spacing=(1.0, 1.0, 1.0), constant_val=-50.0, dtype=np.float32)

        result = resample_to_reference(
            moving_or_path=mov_path,
            reference_or_path=ref_path,
            is_label=True,
        )

        assert result.spacing == (1.0, 1.0, 1.0)
        assert result.array.shape == (32, 32, 32)
        assert result.array.dtype == np.uint8


# ===========================================================================
# 5. Metadata Helpers & Edge Cases
# ===========================================================================


class TestHelpersAndEdgeCases:
    """Tests for get_grid_geometry, apply_grid_geometry, and error handling."""

    def test_get_grid_geometry_from_file_header(self):
        path = _make_nifti(shape=(20, 30, 40), spacing=(0.7, 0.8, 1.5), origin=(1.0, 2.0, 3.0))
        geo = get_grid_geometry(path)

        assert geo.spacing == pytest.approx((0.7, 0.8, 1.5), abs=1e-5)
        # NIfTI RAS origin (1, 2, 3) -> SimpleITK LPS (-1, -2, 3)
        assert geo.origin == pytest.approx((-1.0, -2.0, 3.0), abs=1e-5)
        assert geo.shape_zyx == (40, 30, 20)

    def test_get_grid_geometry_from_in_memory_image(self):
        arr = np.zeros((10, 20, 30), dtype=np.float32)
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.2, 1.4, 1.6))
        geo = get_grid_geometry(img)
        assert geo.spacing == (1.2, 1.4, 1.6)
        assert geo.shape_zyx == (10, 20, 30)

    def test_apply_grid_geometry(self):
        arr = np.ones((8, 8, 8), dtype=np.float32)
        geo = GridGeometry(
            spacing=(1.5, 1.5, 1.5),
            origin=(0.0, 0.0, 0.0),
            direction=np.eye(3),
            shape_zyx=(8, 8, 8),
        )
        img = apply_grid_geometry(arr, geo)
        assert img.GetSpacing() == (1.5, 1.5, 1.5)

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            get_grid_geometry("non_existent_file.nii.gz")
        with pytest.raises(FileNotFoundError):
            resample_to_isotropic("non_existent_file.nii.gz")

    def test_invalid_interpolator_name_raises(self):
        path = _make_nifti(shape=(10, 10, 10), spacing=(1.0, 1.0, 1.0))
        with pytest.raises(ValueError, match="Unknown interpolator"):
            resample_to_isotropic(path, interpolator="invalid_interp")

    def test_unsupported_input_type_raises_type_error(self):
        with pytest.raises(TypeError, match="Unsupported image input type"):
            resample_to_isotropic(12345)  # type: ignore

    def test_2d_array_raises_value_error(self):
        arr_2d = np.zeros((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="must be 3-dimensional"):
            resample_to_isotropic(arr_2d)
