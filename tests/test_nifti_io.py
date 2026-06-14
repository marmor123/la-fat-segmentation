"""Tests for the la_fat.nifti_io module.

Exercises unified NIfTI save/load with round-trip verification of
array data and spatial metadata.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from la_fat.nifti_io import load_nifti, save_nifti


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_3d_array(shape: tuple[int, int, int]) -> np.ndarray:
    """Return a simple 3D test array with known values."""
    data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    return data


# ===========================================================================
# 1. Round-trip: save then load preserves array data and spatial metadata
# ===========================================================================


class TestRoundTrip:
    """save_nifti + load_nifti round-trip preserves all information."""

    def test_round_trip_preserves_array_data(self, tmp_path):
        """Array values are unchanged after save+load."""
        shape = (10, 12, 14)
        original = _make_3d_array(shape)
        path = os.path.join(str(tmp_path), "test.nii.gz")
        spacing = (1.5, 1.5, 2.0)
        origin = (10.0, 20.0, 30.0)
        direction = np.eye(3)

        save_nifti(original, path, spacing=spacing, origin=origin, direction=direction)
        loaded, affine = load_nifti(path)

        assert loaded.shape == shape
        assert np.array_equal(loaded, original)

    def test_round_trip_preserves_spacing(self, tmp_path):
        """Spacing metadata is preserved through the round trip."""
        array = _make_3d_array((8, 8, 8))
        path = os.path.join(str(tmp_path), "spacing.nii.gz")
        spacing = (2.0, 1.5, 3.0)
        origin = (0.0, 0.0, 0.0)
        direction = np.eye(3)

        save_nifti(array, path, spacing=spacing, origin=origin, direction=direction)
        loaded, affine = load_nifti(path)

        # Affine's diagonal should encode spacing (for identity direction)
        assert abs(affine[0, 0]) == pytest.approx(spacing[0], abs=1e-6)
        assert abs(affine[1, 1]) == pytest.approx(spacing[1], abs=1e-6)
        assert abs(affine[2, 2]) == pytest.approx(spacing[2], abs=1e-6)

    def test_round_trip_preserves_origin(self, tmp_path):
        """Origin metadata is preserved through the round trip."""
        array = _make_3d_array((8, 8, 8))
        path = os.path.join(str(tmp_path), "origin.nii.gz")
        spacing = (1.5, 1.5, 1.5)
        origin = (42.0, -13.0, 77.0)
        direction = np.eye(3)

        save_nifti(array, path, spacing=spacing, origin=origin, direction=direction)
        loaded, affine = load_nifti(path)

        # Origin is the translation part of the affine
        assert affine[0, 3] == pytest.approx(origin[0], abs=1e-6)
        assert affine[1, 3] == pytest.approx(origin[1], abs=1e-6)
        assert affine[2, 3] == pytest.approx(origin[2], abs=1e-6)

    def test_round_trip_preserves_direction(self, tmp_path):
        """Non-identity direction matrix is preserved."""
        array = _make_3d_array((8, 8, 8))
        path = os.path.join(str(tmp_path), "direction.nii.gz")
        spacing = (1.5, 1.5, 1.5)
        origin = (0.0, 0.0, 0.0)
        # A 90-degree rotation about z axis in the x-y plane
        direction = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ])

        save_nifti(array, path, spacing=spacing, origin=origin, direction=direction)
        loaded, affine = load_nifti(path)

        # Check the direction part of the affine (diag(spacing) removed)
        loaded_direction = affine[:3, :3] @ np.linalg.inv(np.diag(spacing))
        # The signs might flip due to coordinate conventions
        assert np.allclose(np.abs(loaded_direction), np.abs(direction), atol=1e-6)


# ===========================================================================
# 2. Default values for optional parameters
# ===========================================================================


class TestDefaultParameters:
    """Optional parameters default correctly."""

    def test_default_origin_is_zero(self, tmp_path):
        """When origin is not provided, it defaults to (0, 0, 0)."""
        array = _make_3d_array((4, 4, 4))
        path = os.path.join(str(tmp_path), "default_origin.nii.gz")

        save_nifti(array, path, spacing=(1.0, 1.0, 1.0))
        loaded, affine = load_nifti(path)

        assert affine[0, 3] == pytest.approx(0.0, abs=1e-6)
        assert affine[1, 3] == pytest.approx(0.0, abs=1e-6)
        assert affine[2, 3] == pytest.approx(0.0, abs=1e-6)

    def test_default_direction_is_identity(self, tmp_path):
        """When direction is not provided, it defaults to identity."""
        array = _make_3d_array((4, 4, 4))
        path = os.path.join(str(tmp_path), "default_direction.nii.gz")

        save_nifti(array, path, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0))
        loaded, affine = load_nifti(path)

        # With identity direction and isotropic spacing, affine should be diag-like
        assert abs(affine[0, 1]) < 1e-6
        assert abs(affine[1, 0]) < 1e-6
        assert abs(affine[0, 2]) < 1e-6


# ===========================================================================
# 3. Path handling
# ===========================================================================


class TestPathHandling:
    """Functions accept both str and Path objects."""

    def test_accepts_pathlib_path(self, tmp_path):
        """Can pass a pathlib.Path instead of str."""
        import pathlib
        array = _make_3d_array((4, 4, 4))
        p = pathlib.Path(str(tmp_path)) / "pathlib_test.nii.gz"

        save_nifti(array, p, spacing=(1.0, 1.0, 1.0))
        assert os.path.isfile(p)

        loaded, affine = load_nifti(p)
        assert loaded.shape == (4, 4, 4)

    def test_creates_parent_directories(self, tmp_path):
        """save_nifti creates parent directories if they don't exist."""
        array = _make_3d_array((4, 4, 4))
        deep_path = os.path.join(str(tmp_path), "a", "b", "c", "deep.nii.gz")

        save_nifti(array, deep_path, spacing=(1.0, 1.0, 1.0))
        assert os.path.isfile(deep_path)

    def test_load_nifti_raises_on_missing_file(self, tmp_path):
        """load_nifti raises FileNotFoundError for non-existent file."""
        missing = os.path.join(str(tmp_path), "does_not_exist.nii.gz")
        with pytest.raises(FileNotFoundError):
            load_nifti(missing)


# ===========================================================================
# 4. Binary mask save/load
# ===========================================================================


class TestBinaryMask:
    """Round-trip works specifically with binary mask data."""

    def test_binary_mask_round_trip(self, tmp_path):
        """A boolean mask is correctly saved and loaded."""
        shape = (16, 16, 16)
        mask = np.zeros(shape, dtype=bool)
        mask[4:12, 4:12, 4:12] = True
        path = os.path.join(str(tmp_path), "mask.nii.gz")

        save_nifti(mask.astype(np.uint8), path, spacing=(1.5, 1.5, 1.5))
        loaded, affine = load_nifti(path)

        assert loaded.shape == shape
        assert loaded.dtype == np.uint8
        assert np.array_equal(loaded, mask.astype(np.uint8))
