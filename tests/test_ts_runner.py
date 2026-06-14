"""Tests for the la_fat.ts_runner module.

Tests are written to run without TotalSegmentator installed.  The
``_run_totalsegmentator`` function is mocked so the test environment
does not need TS or a GPU.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk

from la_fat.config import PipelineConfig
from la_fat.ts_runner import (
    TsPrecomputeResult,
    _compute_volume_ml,
    _resample_mask_to_isotropic,
    extract_patient_id,
    is_ts_available,
    resolve_ts_mask_path,
    run_ts_precompute,
    TS_STRUCTURE_NAMES,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_nifti(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    data: np.ndarray | None = None,
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> str:
    """Create a temporary NIfTI file with the given spatial metadata.

    Returns the absolute path.  The caller is responsible for cleanup.
    """
    if data is None:
        data = np.ones(shape, dtype=np.uint8)

    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    affine[:3, 3] = origin

    img = nib.Nifti1Image(data, affine)
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    nib.save(img, tmp.name)
    return tmp.name


def _make_fake_ts_output(
    output_dir: str,
    structure_names: dict[str, str] | None = None,
    shape: tuple[int, int, int] = (16, 16, 16),
    spacing: tuple[float, float, float] = (1.5, 1.5, 1.5),
    voxel_value: int = 1,
) -> dict[str, str]:
    """Populate *output_dir* with fake TS output mask files.

    Returns a mapping of domain name → file path for the masks
    that were created.
    """
    if structure_names is None:
        structure_names = TS_STRUCTURE_NAMES

    os.makedirs(output_dir, exist_ok=True)

    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]

    result: dict[str, str] = {}
    half = tuple(s // 2 for s in shape)
    for domain_name, ts_stem in structure_names.items():
        data = np.zeros(shape, dtype=np.uint8)
        # 4×4×4 cube of ones in the centre
        data[
            half[0] - 2 : half[0] + 2,
            half[1] - 2 : half[1] + 2,
            half[2] - 2 : half[2] + 2,
        ] = voxel_value

        path = os.path.join(output_dir, f"{ts_stem}.nii.gz")
        nib.save(nib.Nifti1Image(data, affine), path)
        result[domain_name] = path

    return result


def _get_ct_fixture(tmp_path: str) -> str:
    """Return the path to a minimal valid NIfTI that acts as the "raw CT"."""
    path = os.path.join(tmp_path, "patient_001.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.int16), np.eye(4)), path)
    return path


# =============================================================================
# TsPrecomputeResult
# =============================================================================


class TestTsPrecomputeResult:
    """TsPrecomputeResult dataclass."""

    def test_create_with_all_fields(self):
        result = TsPrecomputeResult(
            patient_id="ABC123",
            output_dir="/tmp/output/ABC123",
            masks_saved={"LA": "/tmp/output/ABC123/ABC123_LA.nii.gz"},
            mask_volumes_ml={"LA": 12.5},
            errors=[],
            total_runtime_seconds=45.2,
        )
        assert result.patient_id == "ABC123"
        assert result.output_dir == "/tmp/output/ABC123"
        assert result.masks_saved["LA"] == "/tmp/output/ABC123/ABC123_LA.nii.gz"
        assert result.mask_volumes_ml["LA"] == 12.5
        assert result.errors == []
        assert result.total_runtime_seconds == 45.2

    def test_immutable(self):
        result = TsPrecomputeResult(
            patient_id="ABC123",
            output_dir="/tmp/out",
            masks_saved={},
            mask_volumes_ml={},
            errors=[],
            total_runtime_seconds=0.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.patient_id = "changed"  # type: ignore[misc]

    def test_repr_contains_class_name(self):
        result = TsPrecomputeResult(
            patient_id="ABC123",
            output_dir="/tmp/out",
            masks_saved={},
            mask_volumes_ml={},
            errors=[],
            total_runtime_seconds=0.0,
        )
        assert "TsPrecomputeResult" in repr(result)

    def test_with_errors(self):
        result = TsPrecomputeResult(
            patient_id="XYZ",
            output_dir="/tmp/out",
            masks_saved={"LA": "/tmp/out/LA.nii.gz"},
            mask_volumes_ml={"LA": 5.0},
            errors=["Pulmonary Veins", "Pericardium"],
            total_runtime_seconds=30.0,
        )
        assert len(result.errors) == 2
        assert "Pulmonary Veins" in result.errors
        assert result.total_runtime_seconds == 30.0

    def test_empty_masks_saved(self):
        result = TsPrecomputeResult(
            patient_id="EMPTY",
            output_dir="/tmp/out",
            masks_saved={},
            mask_volumes_ml={},
            errors=[],
            total_runtime_seconds=0.0,
        )
        assert result.masks_saved == {}
        assert result.mask_volumes_ml == {}


# =============================================================================
# extract_patient_id
# =============================================================================


class TestExtractPatientId:
    """Patient ID extraction from CT file paths."""

    def test_nii_gz(self):
        assert extract_patient_id("/data/raw/001.nii.gz") == "001"

    def test_nii(self):
        assert extract_patient_id("/data/raw/patient42.nii") == "patient42"

    def test_no_directory(self):
        assert extract_patient_id("scan_001.nii.gz") == "scan_001"

    def test_windows_path(self):
        assert extract_patient_id("C:\\data\\raw\\ABC.nii.gz") == "ABC"

    def test_no_extension(self):
        assert extract_patient_id("/data/raw/patient") == "patient"

    def test_other_extension_is_kept(self):
        # Edge case: file with non-standard extension
        assert extract_patient_id("/data/raw/001.nrrd") == "001.nrrd"


# =============================================================================
# _compute_volume_ml
# =============================================================================


class TestComputeVolumeMl:
    """Volume computation from binary masks."""

    def test_known_volume(self):
        """64 voxels at 1.5 mm spacing → 64 × 3.375 / 1000 = 0.216 ml."""
        shape = (10, 10, 10)
        data = np.zeros(shape, dtype=np.uint8)
        data[3:7, 3:7, 3:7] = 1  # 64 voxels

        mask = sitk.GetImageFromArray(data)
        mask.SetSpacing((1.5, 1.5, 1.5))

        volume = _compute_volume_ml(mask, 1.5)
        expected = 64 * (1.5**3) / 1000.0
        assert volume == pytest.approx(expected, abs=1e-6)

    def test_empty_mask_returns_zero(self):
        data = np.zeros((10, 10, 10), dtype=np.uint8)
        mask = sitk.GetImageFromArray(data)
        mask.SetSpacing((1.5, 1.5, 1.5))
        assert _compute_volume_ml(mask, 1.5) == 0.0

    def test_single_voxel(self):
        """1 voxel at 2.0 mm spacing → 1 × 8.0 / 1000 = 0.008 ml."""
        data = np.zeros((5, 5, 5), dtype=np.uint8)
        data[2, 2, 2] = 1

        mask = sitk.GetImageFromArray(data)
        mask.SetSpacing((2.0, 2.0, 2.0))

        assert _compute_volume_ml(mask, 2.0) == pytest.approx(0.008, abs=1e-6)

    def test_large_mask(self):
        """1000 voxels at 1.0 mm spacing → 1.0 ml."""
        data = np.ones((10, 10, 10), dtype=np.uint8)
        mask = sitk.GetImageFromArray(data)
        mask.SetSpacing((1.0, 1.0, 1.0))
        assert _compute_volume_ml(mask, 1.0) == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# _resample_mask_to_isotropic
# =============================================================================


class TestResampleMaskToIsotropic:
    """Internal mask resampling helper."""

    def test_output_spacing_matches_target(self):
        path = _make_nifti(shape=(20, 20, 20), spacing=(2.0, 2.0, 2.0))
        result = _resample_mask_to_isotropic(path, target_spacing_mm=1.5)
        sx, sy, sz = result.GetSpacing()
        assert abs(sx - 1.5) < 1e-6
        assert abs(sy - 1.5) < 1e-6
        assert abs(sz - 1.5) < 1e-6

    def test_already_isotropic_preserves_shape(self):
        path = _make_nifti(shape=(16, 16, 16), spacing=(1.5, 1.5, 1.5))
        result = _resample_mask_to_isotropic(path, target_spacing_mm=1.5)
        assert result.GetSize() == (16, 16, 16)

    def test_binary_values_preserved(self):
        """Only 0 and 1 should appear after nearest-neighbour resampling."""
        shape = (10, 10, 10)
        data = np.zeros(shape, dtype=np.uint8)
        data[3:6, 3:6, 3:6] = 1

        path = _make_nifti(shape=shape, spacing=(2.0, 2.0, 2.0), data=data)
        result = _resample_mask_to_isotropic(path, target_spacing_mm=2.0)

        array = sitk.GetArrayFromImage(result)
        assert set(np.unique(array)).issubset({0, 1})

    def test_returns_sitk_image(self):
        path = _make_nifti(shape=(8, 8, 8), spacing=(2.0, 2.0, 2.0))
        result = _resample_mask_to_isotropic(path, target_spacing_mm=2.0)
        assert isinstance(result, sitk.Image)


# =============================================================================
# run_ts_precompute (mocked TS)
# =============================================================================


class TestRunTsPrecompute:
    """Integration-level tests with a mocked TotalSegmentator."""

    @pytest.fixture
    def ct_path(self, tmp_path):
        return _get_ct_fixture(str(tmp_path))

    @pytest.fixture
    def mock_ts_all_success(self, monkeypatch):
        """Mock _run_totalsegmentator to create all 8 TS masks."""
        from la_fat import ts_runner as _tr

        def _fake_run(ct_path_arg: str, output_dir: str) -> None:
            _make_fake_ts_output(output_dir, structure_names=_tr.TS_STRUCTURE_NAMES)

        monkeypatch.setattr("la_fat.ts_runner._run_totalsegmentator", _fake_run)
        return _fake_run

    # ── output directory & naming ────────────────────────────────────────

    def test_output_directory_created(self, ct_path, tmp_path, mock_ts_all_success):
        config = PipelineConfig()
        result = run_ts_precompute(ct_path, str(tmp_path), config)

        assert os.path.isdir(result.output_dir)
        assert result.patient_id == "patient_001"

    def test_patient_dir_is_under_output_dir(self, ct_path, tmp_path, mock_ts_all_success):
        config = PipelineConfig()
        result = run_ts_precompute(ct_path, str(tmp_path), config)

        expected = os.path.join(str(tmp_path), "patient_001")
        assert result.output_dir == expected

    def test_all_eight_structures_saved(self, ct_path, tmp_path, mock_ts_all_success):
        config = PipelineConfig()
        result = run_ts_precompute(ct_path, str(tmp_path), config)

        assert set(result.masks_saved.keys()) == set(TS_STRUCTURE_NAMES.keys())
        assert len(result.masks_saved) == 8

    def test_mask_files_exist_on_disk(self, ct_path, tmp_path, mock_ts_all_success):
        config = PipelineConfig()
        result = run_ts_precompute(ct_path, str(tmp_path), config)

        for path in result.masks_saved.values():
            assert os.path.isfile(path), f"Missing mask file: {path}"

    def test_mask_files_are_valid_nifti(self, ct_path, tmp_path, mock_ts_all_success):
        config = PipelineConfig()
        result = run_ts_precompute(ct_path, str(tmp_path), config)

        for path in result.masks_saved.values():
            img = nib.load(path)
            assert isinstance(img, nib.Nifti1Image)

    def test_mask_naming_convention(self, ct_path, tmp_path, mock_ts_all_success):
        """Masks follow ``{patient_id}_{structure}.nii.gz``."""
        config = PipelineConfig()
        result = run_ts_precompute(ct_path, str(tmp_path), config)

        for domain_name, path in result.masks_saved.items():
            basename = os.path.basename(path)
            expected = f"patient_001_{domain_name}.nii.gz"
            assert basename == expected, f"Unexpected filename: {basename}"

    # ── volumes ──────────────────────────────────────────────────────────

    def test_volume_computed_for_each_mask(self, ct_path, tmp_path, mock_ts_all_success):
        config = PipelineConfig()
        result = run_ts_precompute(ct_path, str(tmp_path), config)

        assert set(result.mask_volumes_ml.keys()) == set(result.masks_saved.keys())
        for volume in result.mask_volumes_ml.values():
            assert volume > 0

    def test_volume_approximates_expected(self, ct_path, tmp_path, mock_ts_all_success):
        """Fake masks have a 4×4×4 cube of ones at 1.5 mm → ~0.216 ml."""
        config = PipelineConfig()
        result = run_ts_precompute(ct_path, str(tmp_path), config)

        for domain_name in result.mask_volumes_ml:
            vol = result.mask_volumes_ml[domain_name]
            # 64 voxels × 1.5³ mm³/voxel ÷ 1000 = 0.216
            assert vol == pytest.approx(0.216, abs=0.01)

    # ── errors & edge cases ──────────────────────────────────────────────

    def test_no_errors_when_all_succeed(self, ct_path, tmp_path, mock_ts_all_success):
        config = PipelineConfig()
        result = run_ts_precompute(ct_path, str(tmp_path), config)
        assert result.errors == []

    def test_some_masks_missing_reported_as_errors(
        self, ct_path, tmp_path, monkeypatch
    ):
        """Only create 2 masks, expect the other 6 in errors."""

        def _fake_partial(ct_path_arg: str, output_dir: str) -> None:
            subset = {"LA": "heart_atrium_left", "LV": "heart_ventricle_left"}
            _make_fake_ts_output(output_dir, structure_names=subset)

        monkeypatch.setattr("la_fat.ts_runner._run_totalsegmentator", _fake_partial)

        config = PipelineConfig()
        result = run_ts_precompute(ct_path, str(tmp_path), config)

        assert len(result.masks_saved) == 2
        assert "LA" in result.masks_saved
        assert "LV" in result.masks_saved
        assert len(result.errors) == 6
        assert "RA" in result.errors
        assert "Pericardium" in result.errors

    def test_ts_failure_propagates(self, ct_path, tmp_path, monkeypatch):
        """When _run_totalsegmentator raises, the exception bubbles up."""

        def _fake_crash(ct_path_arg: str, output_dir: str) -> None:
            raise RuntimeError("TS crashed")

        monkeypatch.setattr("la_fat.ts_runner._run_totalsegmentator", _fake_crash)

        config = PipelineConfig()
        with pytest.raises(RuntimeError, match="TS crashed"):
            run_ts_precompute(ct_path, str(tmp_path), config)

    def test_corrupted_mask_does_not_crash_pipeline(
        self, ct_path, tmp_path, monkeypatch
    ):
        """A corrupt mask file should be reported as an error, not crash."""

        def _fake_corrupt(ct_path_arg: str, output_dir: str) -> None:
            os.makedirs(output_dir, exist_ok=True)
            # Create a non-NIfTI file at a TS structure path
            bad_path = os.path.join(output_dir, "heart_atrium_left.nii.gz")
            with open(bad_path, "w") as f:
                f.write("not a NIfTI file")

        monkeypatch.setattr("la_fat.ts_runner._run_totalsegmentator", _fake_corrupt)

        config = PipelineConfig()
        result = run_ts_precompute(ct_path, str(tmp_path), config)

        assert len(result.masks_saved) == 0
        assert "LA" in result.errors

    def test_runtime_positive(self, ct_path, tmp_path, mock_ts_all_success):
        config = PipelineConfig()
        result = run_ts_precompute(ct_path, str(tmp_path), config)
        assert result.total_runtime_seconds > 0


# =============================================================================
# is_ts_available
# =============================================================================


class TestIsTsAvailable:
    """Check for TS availability returns a bool."""

    def test_returns_bool(self):
        assert isinstance(is_ts_available(), bool)


# =============================================================================
# Real TS integration (skip unless TS + GPU are available)
# =============================================================================

def _has_gpu() -> bool:
    """Return True if torch reports a CUDA-capable GPU."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


_no_ts = not is_ts_available() or not _has_gpu()


@pytest.mark.skipif(_no_ts, reason="TotalSegmentator or GPU not available")
class TestRealTsIntegration:
    """End-to-end tests requiring TotalSegmentator (GPU)."""

    def test_precompute_runs_without_error(self, tmp_path):
        """Run on a synthetic CT; TS output may be empty but should not crash."""
        ct_path = os.path.join(str(tmp_path), "test_scan.nii.gz")
        # Use a sufficiently large scan so TS can process it
        data = np.ones((200, 200, 200), dtype=np.int16) * -1000
        data[50:150, 50:150, 50:150] = 0
        nib.save(nib.Nifti1Image(data, np.eye(4)), ct_path)

        config = PipelineConfig()
        result = run_ts_precompute(ct_path, str(tmp_path), config)

        assert result.patient_id == "test_scan"
        assert os.path.isdir(result.output_dir)


# =============================================================================
# resolve_ts_mask_path
# =============================================================================


class TestResolveTsMaskPath:
    """Path resolution for TS output masks."""

    def _make_v2_mask(self, output_dir: str, patient_id: str, name: str) -> str:
        """Create a v2-style mask file (with patient_id prefix)."""
        path = os.path.join(output_dir, f"{patient_id}_{name}.nii.gz")
        nib.save(
            nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)),
            path,
        )
        return path

    def _make_v1_mask(self, output_dir: str, ts_stem: str) -> str:
        """Create a v1-style mask file (TS native name, no patient prefix)."""
        path = os.path.join(output_dir, f"{ts_stem}.nii.gz")
        nib.save(
            nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)),
            path,
        )
        return path

    def test_v2_naming_found(self, tmp_path):
        """Resolves v2-style filenames (patient_id + name)."""
        d = str(tmp_path)
        self._make_v2_mask(d, "PAT001", "LA")
        result = resolve_ts_mask_path(d, "PAT001", "LA")
        assert result is not None
        assert os.path.isfile(result)
        assert "PAT001_LA.nii.gz" in result

    def test_v2_naming_with_space(self, tmp_path):
        """Resolves v2-style with spaces in the name (Pulmonary Artery)."""
        d = str(tmp_path)
        self._make_v2_mask(d, "PAT001", "Pulmonary Artery")
        result = resolve_ts_mask_path(d, "PAT001", "Pulmonary_Artery")
        assert result is not None
        assert "Pulmonary Artery" in result

    def test_v1_native_fallback(self, tmp_path):
        """Falls back to v1 TS native filename."""
        d = str(tmp_path)
        self._make_v1_mask(d, "heart_atrium_left")
        result = resolve_ts_mask_path(d, "PAT001", "LA")
        assert result is not None
        assert "heart_atrium_left.nii.gz" in result

    def test_prefers_v2_over_v1(self, tmp_path):
        """v2 naming is preferred when both exist."""
        d = str(tmp_path)
        v2_path = self._make_v2_mask(d, "PAT001", "LA")
        self._make_v1_mask(d, "heart_atrium_left")
        result = resolve_ts_mask_path(d, "PAT001", "LA")
        assert result == v2_path

    def test_returns_none_when_not_found(self, tmp_path):
        """Returns None when no mask file exists."""
        d = str(tmp_path)
        result = resolve_ts_mask_path(d, "PAT001", "LA")
        assert result is None

    def test_uncompressed_nii_extension(self, tmp_path):
        """Works with .nii (uncompressed) extension."""
        d = str(tmp_path)
        path = os.path.join(d, "PAT001_LA.nii")
        nib.save(
            nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)),
            path,
        )
        result = resolve_ts_mask_path(d, "PAT001", "LA")
        assert result is not None
        assert result.endswith(".nii")

    def test_v1_uncompressed_extension(self, tmp_path):
        """Works with .nii (uncompressed) extension for v1 native names."""
        d = str(tmp_path)
        path = os.path.join(d, "heart_atrium_left.nii")
        nib.save(
            nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.uint8), np.eye(4)),
            path,
        )
        result = resolve_ts_mask_path(d, "PAT001", "LA")
        assert result is not None
        assert result.endswith(".nii")

    def test_pulmonary_artery_v2(self, tmp_path):
        """Pulmonary_Artery with underscore maps to v2 name with space."""
        d = str(tmp_path)
        self._make_v2_mask(d, "PAT001", "Pulmonary Artery")
        result = resolve_ts_mask_path(d, "PAT001", "Pulmonary_Artery")
        assert result is not None
        assert "Pulmonary Artery" in result

    def test_pulmonary_artery_v1(self, tmp_path):
        """Pulmonary_Artery falls back to v1 native name."""
        d = str(tmp_path)
        self._make_v1_mask(d, "pulmonary_artery")
        result = resolve_ts_mask_path(d, "PAT001", "Pulmonary_Artery")
        assert result is not None
        assert "pulmonary_artery" in result

    def test_pulmonary_veins_v2(self, tmp_path):
        """Pulmonary_Veins maps to v2 name with space."""
        d = str(tmp_path)
        self._make_v2_mask(d, "PAT001", "Pulmonary Veins")
        result = resolve_ts_mask_path(d, "PAT001", "Pulmonary_Veins")
        assert result is not None

    def test_pulmonary_veins_v1(self, tmp_path):
        """Pulmonary_Veins falls back to v1 native name."""
        d = str(tmp_path)
        self._make_v1_mask(d, "pulmonary_vein")
        result = resolve_ts_mask_path(d, "PAT001", "Pulmonary_Veins")
        assert result is not None
