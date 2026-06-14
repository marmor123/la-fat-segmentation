"""Integration tests for the la_fat.pipeline module.

Exercises the full ``run_fat_extraction_pipeline`` function end-to-end
using synthetic NIfTI data written to a temporary directory.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile

import nibabel as nib
import numpy as np
import pytest

from la_fat.config import PipelineConfig
from la_fat.pipeline import PipelineResult, run_fat_extraction_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SHAPE = (32, 32, 32)
SPACING_MM = 1.5
VOXEL_VOL_ML = SPACING_MM ** 3 / 1000.0  # 0.003375

# Structure filenames as saved by TS Pre-Compute runner.
STRUCTURE_FILES: dict[str, str] = {
    "LA": "LA",
    "LV": "LV",
    "RA": "RA",
    "RV": "RV",
    "Aorta": "Aorta",
    "Pulmonary_Artery": "Pulmonary Artery",
    "Pericardium": "Pericardium",
    "Pulmonary_Veins": "Pulmonary Veins",
}

ANCHOR_KEYS: list[str] = [
    "LA", "LV", "RA", "RV", "Aorta", "Pulmonary_Artery",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sphere(
    shape: tuple[int, ...],
    centre: tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    """Return a binary sphere mask (bool)."""
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist = np.sqrt(
        (z - centre[0]) ** 2
        + (y - centre[1]) ** 2
        + (x - centre[2]) ** 2
    )
    return dist <= radius


def _make_ellipsoid(
    shape: tuple[int, ...],
    centre: tuple[float, float, float],
    radii: tuple[float, float, float],
) -> np.ndarray:
    """Return a binary ellipsoid mask (bool)."""
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist = (
        ((z - centre[0]) / radii[0]) ** 2
        + ((y - centre[1]) / radii[1]) ** 2
        + ((x - centre[2]) / radii[2]) ** 2
    )
    return dist <= 1.0


def _write_nifti(array: np.ndarray, path: str, spacing_mm: float = SPACING_MM) -> str:
    """Write a numpy array as a NIfTI file (compressed).

    Sets the affine so that SimpleITK reads ``spacing_mm`` as the
    isotropic voxel spacing.  This avoids shape changes when the
    pipeline resamples to the same spacing.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    affine = np.diag([spacing_mm, spacing_mm, spacing_mm, 1.0])
    img = nib.Nifti1Image(array, affine=affine)
    nib.save(img, path)
    return path


def _create_synthetic_ct(
    raw_dir: str,
    patient_id: str,
    shape: tuple[int, ...] = SHAPE,
    spacing_mm: float = SPACING_MM,
) -> str:
    """Create a synthetic CT volume with realistic HU values.

    - Background (outside body): -1000 HU (air)
    - Pericardium region (centre sphere ~radius 16): soft tissue baseline
      mixed with fat and positive HU values.

    Returns the path to the created NIfTI file.
    """
    ct = np.full(shape, -1000.0, dtype=np.float32)

    # Body region inside pericardium: soft tissue baseline
    body_mask = _make_sphere(shape, (16, 16, 16), 16)
    ct[body_mask] = 0.0

    # Fat region (sub-0 HU)
    fat_mask = _make_sphere(shape, (16, 16, 16), 12)
    ct[fat_mask] = -80.0

    # Small high-intensity region (blood pool / bone)
    bright_mask = _make_sphere(shape, (16, 16, 16), 6)
    ct[bright_mask] = 60.0

    path = os.path.join(raw_dir, f"{patient_id}.nii.gz")
    return _write_nifti(ct, path, spacing_mm=spacing_mm)


def _create_synthetic_masks(
    intermediate_dir: str,
    patient_id: str,
    shape: tuple[int, ...] = SHAPE,
    pericardium_radius: float = 16.0,
    min_anchor_volume: bool = True,
) -> dict[str, str]:
    """Create synthetic TS masks on disk.

    Returns a dict mapping structure name to file path.
    """
    paths: dict[str, str] = {}

    # Pericardium: a large sphere in the centre.
    peri_mask = _make_sphere(shape, (16, 16, 16), pericardium_radius)
    _write_nifti(
        peri_mask.astype(np.uint8),
        os.path.join(intermediate_dir, f"{patient_id}_Pericardium.nii.gz"),
    )
    paths["Pericardium"] = os.path.join(
        intermediate_dir, f"{patient_id}_Pericardium.nii.gz",
    )

    # Anatomical anchors at various positions.
    anchor_positions: dict[str, tuple[float, float, float]] = {
        "LA": (12, 16, 16),
        "LV": (20, 16, 16),
        "RA": (12, 16, 12),
        "RV": (20, 16, 12),
        "Aorta": (16, 16, 22),
        "Pulmonary_Artery": (16, 16, 10),
    }

    anchor_radius = 8.0 if min_anchor_volume else 2.0

    for name, pos in anchor_positions.items():
        mask = _make_sphere(shape, pos, anchor_radius)
        filename_stem = STRUCTURE_FILES[name]
        _write_nifti(
            mask.astype(np.uint8),
            os.path.join(intermediate_dir, f"{patient_id}_{filename_stem}.nii.gz"),
        )
        paths[name] = os.path.join(
            intermediate_dir, f"{patient_id}_{filename_stem}.nii.gz",
        )

    # Pulmonary Veins (not used by pipeline but good to include)
    pv = _make_ellipsoid(shape, (14, 16, 14), (2, 4, 2))
    _write_nifti(
        pv.astype(np.uint8),
        os.path.join(intermediate_dir, f"{patient_id}_Pulmonary Veins.nii.gz"),
    )
    paths["Pulmonary_Veins"] = os.path.join(
        intermediate_dir, f"{patient_id}_Pulmonary Veins.nii.gz",
    )

    return paths


def _create_full_synthetic_dataset(
    base_dir: str,
    patient_id: str,
    pericardium_radius: float = 16.0,
    spacing_mm: float = SPACING_MM,
) -> tuple[str, str]:
    """Create both CT and TS masks for a synthetic patient.

    Returns (raw_ct_path, intermediate_dir).
    """
    raw_dir = os.path.join(base_dir, "raw")
    intermediate_dir = os.path.join(base_dir, "intermediate", patient_id)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)

    _create_synthetic_ct(raw_dir, patient_id, spacing_mm=spacing_mm)
    _create_synthetic_masks(
        intermediate_dir, patient_id, pericardium_radius=pericardium_radius,
    )

    raw_ct_path = os.path.join(raw_dir, f"{patient_id}.nii.gz")
    return raw_ct_path, intermediate_dir


def _make_config(
    data_dir: str,
    output_dir: str,
    spacing_mm: float = SPACING_MM,
) -> PipelineConfig:
    """Create a PipelineConfig with overridden paths."""
    return PipelineConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        spacing_mm=spacing_mm,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineEndToEnd:
    """End-to-end integration tests with synthetic data."""

    def test_pipeline_runs_successfully(self, tmp_path):
        """Pipeline runs end-to-end with synthetic data and returns success."""
        patient_id = "SYNTH001"
        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        config = _make_config(data_dir, output_dir)

        _create_full_synthetic_dataset(data_dir, patient_id)

        result = run_fat_extraction_pipeline(patient_id, config=config)

        assert result.success, f"Pipeline failed: {result.errors}"
        assert result.patient_id == patient_id
        assert result.pericardium_result is not None
        assert result.fat_threshold_result is not None
        assert result.partition_result is not None
        assert result.cleanup_result is not None
        assert result.total_runtime_seconds > 0

    def test_output_directory_created(self, tmp_path):
        """After successful run, the patient output directory exists with
        expected files."""
        patient_id = "SYNTH002"
        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        config = _make_config(data_dir, output_dir)

        _create_full_synthetic_dataset(data_dir, patient_id)
        result = run_fat_extraction_pipeline(patient_id, config=config)

        assert result.success
        patient_out = os.path.join(output_dir, patient_id)
        assert os.path.isdir(patient_out)

        # Expected output files
        assert os.path.isfile(os.path.join(patient_out, "la_fat_mask.nii.gz"))
        assert os.path.isfile(os.path.join(patient_out, "quality_flags.json"))

        # Dashboard files should exist
        if result.dashboard_output is not None:
            assert os.path.isfile(result.dashboard_output.slice_gallery_path)
            assert os.path.isfile(result.dashboard_output.summary_table_path)

        # Verify quality flags JSON is valid
        with open(os.path.join(patient_out, "quality_flags.json")) as f:
            flags_data = json.load(f)
        assert isinstance(flags_data, list)

    def test_la_fat_mask_saved(self, tmp_path):
        """LA fat mask is saved as a valid NIfTI file."""
        patient_id = "SYNTH003"
        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        config = _make_config(data_dir, output_dir)

        _create_full_synthetic_dataset(data_dir, patient_id)
        result = run_fat_extraction_pipeline(patient_id, config=config)

        assert result.success
        mask_path = os.path.join(output_dir, patient_id, "la_fat_mask.nii.gz")
        assert os.path.isfile(mask_path)

        # Verify it can be read back as a NIfTI
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        assert mask_data.shape == SHAPE
        assert mask_data.dtype == np.uint8 or mask_data.dtype == np.float64

    def test_pipeline_result_dataclass(self, tmp_path):
        """PipelineResult dataclass has correct fields and types."""
        patient_id = "SYNTH004"
        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        config = _make_config(data_dir, output_dir)

        _create_full_synthetic_dataset(data_dir, patient_id)
        result = run_fat_extraction_pipeline(patient_id, config=config)

        assert isinstance(result, PipelineResult)
        assert isinstance(result.patient_id, str)
        assert isinstance(result.success, bool)
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert isinstance(result.total_runtime_seconds, float)

    def test_re_running_is_idempotent(self, tmp_path):
        """Running the pipeline twice with same inputs overwrites outputs
        without error."""
        patient_id = "SYNTH005"
        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        config = _make_config(data_dir, output_dir)

        _create_full_synthetic_dataset(data_dir, patient_id)

        # First run
        result1 = run_fat_extraction_pipeline(patient_id, config=config)
        assert result1.success

        mask_path = os.path.join(output_dir, patient_id, "la_fat_mask.nii.gz")
        assert os.path.isfile(mask_path)
        mtime1 = os.path.getmtime(mask_path)

        # Second run (immediately after)
        result2 = run_fat_extraction_pipeline(patient_id, config=config)
        assert result2.success

        assert os.path.isfile(mask_path)
        # File should have been overwritten (mtime may be same if fast,
        # but the key assertion is no crash)
        assert result2.total_runtime_seconds > 0

    def test_warnings_collected_on_fallback(self, tmp_path):
        """When pericardium is too small, a warning is recorded but the
        pipeline continues."""
        patient_id = "SYNTH006"
        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        # Use a high min_pericardium_volume so the normal pericardium triggers fallback
        config = PipelineConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            min_pericardium_volume_ml=9999.0,  # unrealistically high
        )

        _create_full_synthetic_dataset(data_dir, patient_id, pericardium_radius=13.0)

        result = run_fat_extraction_pipeline(patient_id, config=config)

        assert result.success or len(result.errors) == 0
        # Should have fallback warning
        fallback_warnings = [
            w for w in result.warnings
            if "Pericardium fallback" in w
        ]
        assert len(fallback_warnings) > 0, (
            f"Expected pericardium fallback warning, got warnings: {result.warnings}"
        )

    def test_config_from_yaml_is_respected(self, tmp_path):
        """Running with a custom config YAML applies the config values."""
        patient_id = "SYNTH007"
        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        custom_spacing = 2.0  # different from default 1.5

        # Create config YAML
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        config_path = str(config_dir / "custom.yaml")
        with open(config_path, "w") as f:
            f.write(f"spacing_mm: {custom_spacing}\n")
            f.write(f"data_dir: {data_dir}\n")
            f.write(f"output_dir: {output_dir}\n")

        # Create CT at the custom spacing so resampling doesn't change shape
        _create_full_synthetic_dataset(
            data_dir, patient_id, spacing_mm=custom_spacing,
        )

        result = run_fat_extraction_pipeline(
            patient_id, config_path=config_path,
        )

        assert result.success, f"Pipeline failed: {result.errors}"

    def test_pipeline_outputs_la_fat_volume(self, tmp_path):
        """Pipeline result includes a plausible LA fat volume."""
        patient_id = "SYNTH008"
        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        config = _make_config(data_dir, output_dir)

        _create_full_synthetic_dataset(data_dir, patient_id)
        result = run_fat_extraction_pipeline(patient_id, config=config)

        assert result.success
        assert result.partition_result is not None
        la_vol = result.partition_result.anchor_volumes_ml.get("LA", 0.0)
        assert la_vol >= 0.0
        # With our synthetic data, some LA fat should be assigned
        assert result.partition_result.total_fat_volume_ml > 0


class TestPipelineErrorHandling:
    """Tests for error-handling paths in the pipeline."""

    def test_missing_ts_masks(self, tmp_path):
        """Pipeline handles missing TS masks gracefully."""
        patient_id = "MISSING001"
        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        config = _make_config(data_dir, output_dir)

        # Only create CT, no masks
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        _create_synthetic_ct(raw_dir, patient_id)

        result = run_fat_extraction_pipeline(patient_id, config=config)

        assert not result.success
        assert len(result.errors) > 0
        # Should mention masks not found
        error_text = " ".join(result.errors).lower()
        assert "mask" in error_text or "not found" in error_text

    def test_missing_ct(self, tmp_path):
        """Pipeline handles missing raw CT gracefully."""
        patient_id = "MISSING002"
        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        config = _make_config(data_dir, output_dir)

        # Neither CT nor masks exist
        result = run_fat_extraction_pipeline(patient_id, config=config)

        assert not result.success
        assert len(result.errors) > 0
        # Should mention CT not found
        error_text = " ".join(result.errors).lower()
        assert "ct" in error_text or "not found" in error_text

    def test_partial_masks(self, tmp_path):
        """Pipeline runs with only some TS masks present (partial data)."""
        patient_id = "PARTIAL001"
        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        config = _make_config(data_dir, output_dir)

        # Create CT
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        _create_synthetic_ct(raw_dir, patient_id)

        # Create only pericardium and a few anchor masks
        intermediate_dir = os.path.join(data_dir, "intermediate", patient_id)
        os.makedirs(intermediate_dir, exist_ok=True)

        peri_mask = _make_sphere(SHAPE, (16, 16, 16), 13)
        _write_nifti(
            peri_mask.astype(np.uint8),
            os.path.join(intermediate_dir, f"{patient_id}_Pericardium.nii.gz"),
        )

        # Only create LA and LV masks
        for name, pos in [("LA", (12, 16, 16)), ("LV", (20, 16, 16))]:
            mask = _make_sphere(SHAPE, pos, 5)
            filename_stem = STRUCTURE_FILES[name]
            _write_nifti(
                mask.astype(np.uint8),
                os.path.join(intermediate_dir, f"{patient_id}_{filename_stem}.nii.gz"),
            )

        result = run_fat_extraction_pipeline(patient_id, config=config)

        # Should succeed or at least not crash — partition needs >= 2 anchors
        if result.success:
            assert result.partition_result is not None
        else:
            # If partition fails (< 2 valid anchors), it should still be graceful
            assert len(result.errors) > 0

    def test_pipeline_never_crashes(self, tmp_path):
        """Pipeline never raises an exception — always returns PipelineResult."""
        patient_id = "NOCRASH"
        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        config = _make_config(data_dir, output_dir)

        # No data at all
        result = run_fat_extraction_pipeline(patient_id, config=config)

        # This is the key assertion: it should return a PipelineResult, not raise
        assert isinstance(result, PipelineResult)
        assert not result.success


class TestMeshExtraction:
    """Tests for the mesh extraction pipeline step."""

    def test_mesh_paths_in_result(self):
        """Verify PipelineResult has mesh_paths field with default None."""
        result = PipelineResult(
            patient_id="test",
            success=True,
            partition_result=None,
            fat_threshold_result=None,
            pericardium_result=None,
            cleanup_result=None,
            quality_flags=[],
            dashboard_output=None,
            errors=[],
            warnings=[],
            total_runtime_seconds=0.0,
        )
        assert hasattr(result, "mesh_paths")
        assert result.mesh_paths is None

    def test_mesh_extraction_step_creates_mesh_dirs(self, tmp_path):
        """Running the full pipeline creates mesh directories with .ply files."""
        patient_id = "MESHTEST"
        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        config = _make_config(data_dir, output_dir)

        _create_full_synthetic_dataset(data_dir, patient_id)
        result = run_fat_extraction_pipeline(patient_id, config=config)

        assert result.success, f"Pipeline failed: {result.errors}"
        assert result.mesh_paths is not None

        meshes_root = os.path.join(output_dir, patient_id, "meshes")
        assert os.path.isdir(os.path.join(meshes_root, "step2_anchors"))
        assert os.path.isdir(os.path.join(meshes_root, "step5_partition"))
        assert os.path.isdir(os.path.join(meshes_root, "step7_final"))

        # Verify .ply files exist in each subdirectory
        step2_plys = glob.glob(os.path.join(meshes_root, "step2_anchors", "*.ply"))
        assert len(step2_plys) > 0
        step5_plys = glob.glob(os.path.join(meshes_root, "step5_partition", "*.ply"))
        assert len(step5_plys) > 0
        step7_plys = glob.glob(os.path.join(meshes_root, "step7_final", "*.ply"))
        assert len(step7_plys) > 0

    def test_pipeline_succeeds_when_mesh_extraction_fails(self, tmp_path, monkeypatch):
        """Pipeline continues gracefully if mesh extraction raises an exception."""
        def failing_extract(*args, **kwargs):
            raise RuntimeError("Simulated mesh extraction failure")

        monkeypatch.setattr(
            "la_fat.pipeline.extract_interactive_meshes",
            failing_extract,
        )

        patient_id = "FAILMESH"
        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        config = _make_config(data_dir, output_dir)

        _create_full_synthetic_dataset(data_dir, patient_id)
        result = run_fat_extraction_pipeline(patient_id, config=config)

        # Pipeline did NOT crash — returns PipelineResult
        assert isinstance(result, PipelineResult)
        # mesh_paths is None since extraction failed
        assert result.mesh_paths is None
        # The mesh extraction error is recorded
        mesh_errors = [e for e in result.errors if "Mesh extraction" in e]
        assert len(mesh_errors) > 0
        # Pipeline continued to later steps (dashboard, flag saving)
        assert result.dashboard_output is not None
        # success=False because errors list is non-empty
        assert not result.success


class TestPipelineCLI:
    """Tests for the CLI entry point (run_pipeline.py)."""

    def test_cli_help(self):
        """``python run_pipeline.py --help`` returns exit code 0 and shows
        usage."""
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "run_pipeline.py",
        )
        result = subprocess.run(
            [sys.executable, script_path, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower()
        assert "--patient" in result.stdout or "--patient" in result.stderr

    def test_cli_missing_patient_exits_error(self):
        """Running the CLI without --patient exits with non-zero."""
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "run_pipeline.py",
        )
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        # Should mention --patient is required
        err_text = (result.stdout + result.stderr).lower()
        assert "patient" in err_text

    def test_cli_accepts_data_output_dir(self, tmp_path):
        """CLI accepts --data-dir and --output-dir arguments."""
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "run_pipeline.py",
        )
        data_dir = str(tmp_path / "cli_data")
        output_dir = str(tmp_path / "cli_outputs")
        patient_id = "CLI001"

        # Create synthetic data
        _create_full_synthetic_dataset(str(tmp_path / "cli_data"), patient_id)

        result = subprocess.run(
            [
                sys.executable, script_path,
                "--patient", patient_id,
                "--data-dir", data_dir,
                "--output-dir", output_dir,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        # Output directory should exist
        assert os.path.isdir(os.path.join(output_dir, patient_id))
        assert os.path.isfile(
            os.path.join(output_dir, patient_id, "la_fat_mask.nii.gz"),
        )

    def test_cli_with_config(self, tmp_path):
        """CLI accepts --config argument."""
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "run_pipeline.py",
        )
        data_dir = str(tmp_path / "cfg_data")
        output_dir = str(tmp_path / "cfg_outputs")
        patient_id = "CLI002"

        # Create config YAML
        config_dir = tmp_path / "cli_configs"
        config_dir.mkdir()
        config_path = str(config_dir / "cfg.yaml")
        with open(config_path, "w") as f:
            f.write(f"data_dir: {data_dir}\n")
            f.write(f"output_dir: {output_dir}\n")
            f.write("spacing_mm: 2.0\n")

        _create_full_synthetic_dataset(
            str(tmp_path / "cfg_data"), patient_id, spacing_mm=2.0,
        )

        result = subprocess.run(
            [
                sys.executable, script_path,
                "--patient", patient_id,
                "--config", config_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert os.path.isdir(os.path.join(output_dir, patient_id))
