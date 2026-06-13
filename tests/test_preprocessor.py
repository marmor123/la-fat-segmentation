"""Tests for the la_fat.preprocessor module."""

import math
import tempfile

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from la_fat.preprocessor import ResampleResult, resample_to_isotropic


# SimpleITK converts NIfTI RAS coordinates to its native LPS convention.
# This means the direction and origin reported by SimpleITK differ from
# what nibabel stores.  The conversion matrix from RAS to LPS is:
#   LPS_from_RAS = diag(-1, -1, 1)
# All expected-value calculations below account for this.


def _make_nifti(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    direction: np.ndarray | None = None,
    rng_seed: int = 42,
) -> str:
    """Create a temporary NIfTI file with a synthetic 3D volume.

    Returns the absolute path to the file.
    """
    rng = np.random.default_rng(rng_seed)
    data = rng.normal(size=shape).astype(np.float32)

    if direction is None:
        direction = np.eye(3)

    affine = np.eye(4)
    affine[:3, :3] = direction * np.array(spacing)[:, np.newaxis]
    affine[:3, 3] = origin

    img = nib.Nifti1Image(data, affine)
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    nib.save(img, tmp.name)
    return tmp.name


class TestResampleOutputSpacing:
    """Output spacing should match the target spacing."""

    def test_output_spacing_equals_target(self):
        path = _make_nifti(shape=(64, 64, 64), spacing=(1.0, 1.0, 1.0))
        result = resample_to_isotropic(path, target_spacing_mm=1.5)
        assert result.spacing == (1.5, 1.5, 1.5)

    def test_anisotropic_input_becomes_isotropic(self):
        path = _make_nifti(shape=(50, 100, 200), spacing=(0.5, 1.0, 2.0))
        result = resample_to_isotropic(path, target_spacing_mm=1.0)
        sx, sy, sz = result.spacing
        assert abs(sx - sy) < 1e-6
        assert abs(sx - sz) < 1e-6
        assert abs(sx - 1.0) < 1e-6


class TestResampleOutputShape:
    """Output shape should change proportionally to spacing ratio."""

    def test_shape_changes_correctly(self):
        path = _make_nifti(shape=(30, 40, 50), spacing=(2.0, 2.0, 2.0))
        result = resample_to_isotropic(path, target_spacing_mm=1.0)
        # Halving spacing doubles dimensions:
        #   new_x = ceil(30*2/1)=60, new_y=ceil(40*2/1)=80, new_z=ceil(50*2/1)=100
        # sitk.GetArrayFromImage returns (z, y, x), hence (100, 80, 60)
        assert result.ct_array.shape == (100, 80, 60)

    def test_coarser_spacing_reduces_size(self):
        path = _make_nifti(shape=(100, 100, 100), spacing=(0.5, 0.5, 0.5))
        result = resample_to_isotropic(path, target_spacing_mm=1.0)
        # Doubling spacing should roughly halve dimensions
        assert result.ct_array.shape == (50, 50, 50)

    def test_already_isotropic_preserves_shape(self):
        path = _make_nifti(shape=(64, 64, 64), spacing=(1.0, 1.0, 1.0))
        result = resample_to_isotropic(path, target_spacing_mm=1.0)
        assert result.ct_array.shape == (64, 64, 64)


class TestResampleSpatialMetadata:
    """Origin and direction should be preserved."""

    def test_origin_preserved(self):
        origin = (42.0, -13.0, 7.5)
        path = _make_nifti(
            shape=(32, 32, 32), spacing=(1.0, 1.0, 1.0), origin=origin
        )
        result = resample_to_isotropic(path, target_spacing_mm=2.0)
        # SimpleITK converts RAS to LPS: x_LPS = -x_RAS, y_LPS = -y_RAS
        expected = (-42.0, 13.0, 7.5)
        assert result.origin == pytest.approx(expected, abs=1e-6)

    def test_direction_preserved_identity(self):
        path = _make_nifti(shape=(32, 32, 32), spacing=(1.0, 1.0, 1.0))
        result = resample_to_isotropic(path, target_spacing_mm=1.0)
        # RAS identity direction becomes LPS: diag(-1, -1, 1)
        expected = np.array([[-1.0, 0.0, 0.0],
                             [0.0, -1.0, 0.0],
                             [0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(result.direction, expected)

    def test_original_spacing_and_shape_recorded(self):
        path = _make_nifti(shape=(40, 50, 60), spacing=(0.7, 1.2, 2.5))
        result = resample_to_isotropic(path, target_spacing_mm=1.5)
        assert result.original_spacing == pytest.approx((0.7, 1.2, 2.5))
        assert result.original_shape == (40, 50, 60)

    def test_return_type(self):
        path = _make_nifti(shape=(16, 16, 16), spacing=(1.0, 1.0, 1.0))
        result = resample_to_isotropic(path, target_spacing_mm=1.0)
        assert isinstance(result, ResampleResult)
        assert isinstance(result.ct_array, np.ndarray)
        assert isinstance(result.spacing, tuple)
        assert isinstance(result.origin, tuple)
        assert isinstance(result.direction, np.ndarray)


class TestResampleDirectionCosines:
    """Non-orthonormal / realistic direction cosines must be handled."""

    def test_oblique_direction(self):
        """A 45-degree rotation in the axial plane."""
        theta = math.radians(45)
        direction = np.array(
            [
                [math.cos(theta), -math.sin(theta), 0],
                [math.sin(theta), math.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        path = _make_nifti(
            shape=(32, 32, 32),
            spacing=(1.0, 1.0, 1.0),
            direction=direction,
        )
        result = resample_to_isotropic(path, target_spacing_mm=2.0)
        # Direction matrix should be preserved (up to sign)
        np.testing.assert_array_almost_equal(
            np.abs(result.direction), np.abs(direction), decimal=5
        )
        assert result.ct_array.ndim == 3

    def test_negative_determinant_direction(self):
        """Direction with negative determinant (flipped axis).

        The NIfTI stores a direction where the x-axis is flipped (RAS
        with x pointing Left).  SimpleITK converts to LPS so the
        output direction = LPS_from_RAS @ D_RAS.
        """
        direction_ras = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        path = _make_nifti(
            shape=(32, 32, 32),
            spacing=(1.0, 1.0, 1.0),
            direction=direction_ras,
        )
        result = resample_to_isotropic(path, target_spacing_mm=1.0)
        # Expected: LPS_from_RAS @ D_RAS = diag(-1,-1,1) @ [[-1,0,0],[0,1,0],[0,0,1]]
        #         = [[1,0,0],[0,-1,0],[0,0,1]]
        expected = np.array([[1.0, 0.0, 0.0],
                             [0.0, -1.0, 0.0],
                             [0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(result.direction, expected)
