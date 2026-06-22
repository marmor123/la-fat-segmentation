"""Tests for the la_fat.batch_pipeline module.

Exercises the batch processing wrapper that discovers CT scans and
runs the pipeline sequentially, skipping already-processed patients.
"""

from __future__ import annotations

import json
import os

import pytest

from la_fat.config import PipelineConfig
from la_fat.pipeline import PipelineResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _touch(path: str) -> str:
    """Create an empty file at *path*, creating parent directories."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("")
    return path


def _write_result_json(output_dir: str, patient_id: str) -> str:
    """Write a minimal pipeline_result.json for a patient."""
    path = os.path.join(output_dir, patient_id, "pipeline_result.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "patient_id": patient_id,
            "la_fat_volume_ml": 12.34,
            "total_fat_volume_ml": 20.0,
            "pericardium_volume_ml": 100.0,
            "unassigned_volume_ml": 1.0,
            "unassigned_fat_pct": 5.0,
            "anchor_volumes_ml": {"LA": 12.34},
            "quality_flags": [],
            "fat_hu_range": {"__tuple__": True, "items": [-190.0, -30.0]},
            "voxel_volume_ml": 0.003375,
            "excluded_anchors": [],
            "islands_removed": 0,
            "total_removed_volume_mm3": 0.0,
            "warnings": [],
            "errors": [],
        }, f)
    return path


# ---------------------------------------------------------------------------
# Tests: File discovery
# ---------------------------------------------------------------------------


class TestFileDiscovery:
    """Discovery of CT scan files in data/raw/."""

    def test_discovers_nii_gz_files(self, tmp_path):
        """Finds all .nii.gz files in data/raw/."""
        from la_fat.batch_pipeline import _discover_ct_files

        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        _touch(str(raw_dir / "patient1.nii.gz"))
        _touch(str(raw_dir / "patient2.nii.gz"))
        _touch(str(raw_dir / "notes.txt"))  # non-NIfTI — ignored

        files = _discover_ct_files(str(tmp_path / "data"))

        assert len(files) == 2
        assert any("patient1.nii.gz" in f for f in files)
        assert any("patient2.nii.gz" in f for f in files)

    def test_discovers_nii_files(self, tmp_path):
        """Finds .nii files alongside .nii.gz."""
        from la_fat.batch_pipeline import _discover_ct_files

        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        _touch(str(raw_dir / "scan_a.nii"))
        _touch(str(raw_dir / "scan_b.nii.gz"))

        files = _discover_ct_files(str(tmp_path / "data"))

        assert len(files) == 2

    def test_returns_empty_list_when_no_ct_files(self, tmp_path):
        """Returns empty list when raw/ exists but has no NIfTI files."""
        from la_fat.batch_pipeline import _discover_ct_files

        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        _touch(str(raw_dir / "readme.txt"))

        files = _discover_ct_files(str(tmp_path / "data"))

        assert files == []

    def test_returns_empty_when_raw_dir_missing(self, tmp_path):
        """Returns empty list when raw/ directory does not exist."""
        from la_fat.batch_pipeline import _discover_ct_files

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        # No raw/ subdirectory

        files = _discover_ct_files(str(data_dir))

        assert files == []

    def test_sorted_order(self, tmp_path):
        """Files are returned in sorted order for deterministic processing."""
        from la_fat.batch_pipeline import _discover_ct_files

        raw_dir = tmp_path / "data" / "raw"
        raw_dir.mkdir(parents=True)
        _touch(str(raw_dir / "zebra.nii.gz"))
        _touch(str(raw_dir / "alpha.nii.gz"))
        _touch(str(raw_dir / "beta.nii.gz"))

        files = _discover_ct_files(str(tmp_path / "data"))

        names = [os.path.basename(f) for f in files]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Tests: Skip logic
# ---------------------------------------------------------------------------


class TestSkipLogic:
    """Already-processed patients are skipped."""

    def test_patient_with_result_is_skipped(self, tmp_path):
        """A patient with an existing pipeline_result.json is identified as
        completed."""
        from la_fat.batch_pipeline import _is_completed

        output_dir = str(tmp_path / "outputs")
        _write_result_json(output_dir, "DONE001")

        assert _is_completed(output_dir, "DONE001") is True

    def test_patient_without_result_is_not_skipped(self, tmp_path):
        """A patient without pipeline_result.json is not completed."""
        from la_fat.batch_pipeline import _is_completed

        output_dir = str(tmp_path / "outputs")
        os.makedirs(os.path.join(output_dir, "NEW001"), exist_ok=True)

        assert _is_completed(output_dir, "NEW001") is False

    def test_patient_without_output_dir_is_not_skipped(self, tmp_path):
        """A patient with no output directory at all is not completed."""
        from la_fat.batch_pipeline import _is_completed

        output_dir = str(tmp_path / "outputs")
        # No directory for this patient

        assert _is_completed(output_dir, "NEW002") is False


# ---------------------------------------------------------------------------
# Tests: Pipeline invocation
# ---------------------------------------------------------------------------


class TestBatchPipelineInvocation:
    """The batch wrapper calls run_fat_extraction_pipeline for new patients."""

    def test_calls_pipeline_for_each_new_patient(self, tmp_path, monkeypatch):
        """Each discovered patient without existing results triggers a
        pipeline call."""
        from la_fat.batch_pipeline import run_batch_pipeline

        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir)
        _touch(os.path.join(raw_dir, "PAT1.nii.gz"))
        _touch(os.path.join(raw_dir, "PAT2.nii.gz"))

        config = PipelineConfig(data_dir=data_dir, output_dir=output_dir)

        called_ids: list[str] = []

        def _fake_pipeline(patient_id, config=None, config_path=None):
            called_ids.append(patient_id)
            return PipelineResult(
                patient_id=patient_id,
                success=True,
                partition_result=None,
                fat_threshold_result=None,
                pericardium_result=None,
                cleanup_result=None,
                quality_flags=[],
                dashboard_output=None,
                errors=[],
                warnings=[],
                total_runtime_seconds=1.0,
            )

        monkeypatch.setattr(
            "la_fat.batch_pipeline.run_fat_extraction_pipeline",
            _fake_pipeline,
        )

        summary = run_batch_pipeline(data_dir=data_dir, output_dir=output_dir)

        assert called_ids == ["PAT1", "PAT2"]
        assert summary["total"] == 2
        assert summary["succeeded"] == 2
        assert summary["failed"] == 0
        assert summary["skipped"] == 0

    def test_skips_completed_patients(self, tmp_path, monkeypatch):
        """Patients with existing results are skipped, new ones are processed."""
        from la_fat.batch_pipeline import run_batch_pipeline

        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir)
        _touch(os.path.join(raw_dir, "OLD.nii.gz"))
        _touch(os.path.join(raw_dir, "NEW.nii.gz"))

        # Pre-mark OLD as completed
        _write_result_json(output_dir, "OLD")

        called_ids: list[str] = []

        def _fake_pipeline(patient_id, config=None, config_path=None):
            called_ids.append(patient_id)
            return PipelineResult(
                patient_id=patient_id,
                success=True,
                partition_result=None,
                fat_threshold_result=None,
                pericardium_result=None,
                cleanup_result=None,
                quality_flags=[],
                dashboard_output=None,
                errors=[],
                warnings=[],
                total_runtime_seconds=1.0,
            )

        monkeypatch.setattr(
            "la_fat.batch_pipeline.run_fat_extraction_pipeline",
            _fake_pipeline,
        )

        summary = run_batch_pipeline(data_dir=data_dir, output_dir=output_dir)

        assert called_ids == ["NEW"]
        assert summary["skipped"] == 1

    def test_all_skipped_when_nothing_new(self, tmp_path, monkeypatch):
        """When all patients are already processed, no pipeline calls happen."""
        from la_fat.batch_pipeline import run_batch_pipeline

        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir)
        _touch(os.path.join(raw_dir, "DONE1.nii.gz"))
        _touch(os.path.join(raw_dir, "DONE2.nii.gz"))

        _write_result_json(output_dir, "DONE1")
        _write_result_json(output_dir, "DONE2")

        called_ids: list[str] = []

        def _fake_pipeline(patient_id, config=None, config_path=None):
            called_ids.append(patient_id)
            return PipelineResult(
                patient_id=patient_id,
                success=True,
                partition_result=None,
                fat_threshold_result=None,
                pericardium_result=None,
                cleanup_result=None,
                quality_flags=[],
                dashboard_output=None,
                errors=[],
                warnings=[],
                total_runtime_seconds=1.0,
            )

        monkeypatch.setattr(
            "la_fat.batch_pipeline.run_fat_extraction_pipeline",
            _fake_pipeline,
        )

        summary = run_batch_pipeline(data_dir=data_dir, output_dir=output_dir)

        assert called_ids == []
        assert summary["skipped"] == 2
        assert summary["total"] == 2


# ---------------------------------------------------------------------------
# Tests: Error resilience
# ---------------------------------------------------------------------------


class TestBatchErrorResilience:
    """Individual patient failures do not halt batch processing."""

    def test_continues_after_patient_failure(self, tmp_path, monkeypatch):
        """When one patient fails, remaining patients are still processed."""
        from la_fat.batch_pipeline import run_batch_pipeline

        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir)
        _touch(os.path.join(raw_dir, "GOOD1.nii.gz"))
        _touch(os.path.join(raw_dir, "BAD.nii.gz"))
        _touch(os.path.join(raw_dir, "GOOD2.nii.gz"))

        called_ids: list[str] = []

        def _fake_pipeline(patient_id, config=None, config_path=None):
            called_ids.append(patient_id)
            if patient_id == "BAD":
                return PipelineResult(
                    patient_id=patient_id,
                    success=False,
                    partition_result=None,
                    fat_threshold_result=None,
                    pericardium_result=None,
                    cleanup_result=None,
                    quality_flags=[],
                    dashboard_output=None,
                    errors=["Simulated failure"],
                    warnings=[],
                    total_runtime_seconds=0.5,
                )
            return PipelineResult(
                patient_id=patient_id,
                success=True,
                partition_result=None,
                fat_threshold_result=None,
                pericardium_result=None,
                cleanup_result=None,
                quality_flags=[],
                dashboard_output=None,
                errors=[],
                warnings=[],
                total_runtime_seconds=1.0,
            )

        monkeypatch.setattr(
            "la_fat.batch_pipeline.run_fat_extraction_pipeline",
            _fake_pipeline,
        )

        summary = run_batch_pipeline(data_dir=data_dir, output_dir=output_dir)

        assert called_ids == ["BAD", "GOOD1", "GOOD2"]  # sorted order
        assert summary["total"] == 3
        assert summary["succeeded"] == 2
        assert summary["failed"] == 1

    def test_continues_after_pipeline_raises_exception(self, tmp_path, monkeypatch):
        """When the pipeline raises an exception, the batch continues."""
        from la_fat.batch_pipeline import run_batch_pipeline

        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir)
        _touch(os.path.join(raw_dir, "CRASH.nii.gz"))
        _touch(os.path.join(raw_dir, "SURVIVOR.nii.gz"))

        called_ids: list[str] = []

        def _fake_pipeline(patient_id, config=None, config_path=None):
            called_ids.append(patient_id)
            if patient_id == "CRASH":
                raise RuntimeError("KABOOM")
            return PipelineResult(
                patient_id=patient_id,
                success=True,
                partition_result=None,
                fat_threshold_result=None,
                pericardium_result=None,
                cleanup_result=None,
                quality_flags=[],
                dashboard_output=None,
                errors=[],
                warnings=[],
                total_runtime_seconds=1.0,
            )

        monkeypatch.setattr(
            "la_fat.batch_pipeline.run_fat_extraction_pipeline",
            _fake_pipeline,
        )

        summary = run_batch_pipeline(data_dir=data_dir, output_dir=output_dir)

        assert called_ids == ["CRASH", "SURVIVOR"]
        assert summary["failed"] == 1
        assert summary["succeeded"] == 1

    def test_failed_patients_in_summary(self, tmp_path, monkeypatch):
        """The summary includes the list of failed patient IDs."""
        from la_fat.batch_pipeline import run_batch_pipeline

        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir)
        _touch(os.path.join(raw_dir, "FAIL1.nii.gz"))

        def _fake_pipeline(patient_id, config=None, config_path=None):
            return PipelineResult(
                patient_id=patient_id,
                success=False,
                partition_result=None,
                fat_threshold_result=None,
                pericardium_result=None,
                cleanup_result=None,
                quality_flags=[],
                dashboard_output=None,
                errors=["Something went wrong"],
                warnings=[],
                total_runtime_seconds=0.0,
            )

        monkeypatch.setattr(
            "la_fat.batch_pipeline.run_fat_extraction_pipeline",
            _fake_pipeline,
        )

        summary = run_batch_pipeline(data_dir=data_dir, output_dir=output_dir)

        assert "FAIL1" in summary["failed_ids"]


# ---------------------------------------------------------------------------
# Tests: Config propagation
# ---------------------------------------------------------------------------


class TestConfigPropagation:
    """Configuration is forwarded to the pipeline."""

    def test_config_passed_to_pipeline(self, tmp_path, monkeypatch):
        """The batch wrapper passes config to each pipeline call."""
        from la_fat.batch_pipeline import run_batch_pipeline

        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir)
        _touch(os.path.join(raw_dir, "PATIENT.nii.gz"))

        config = PipelineConfig(
            data_dir=data_dir,
            output_dir=output_dir,
            spacing_mm=2.0,  # non-default
        )

        received_configs: list[PipelineConfig] = []

        def _fake_pipeline(patient_id, config=None, config_path=None):
            received_configs.append(config)
            return PipelineResult(
                patient_id=patient_id,
                success=True,
                partition_result=None,
                fat_threshold_result=None,
                pericardium_result=None,
                cleanup_result=None,
                quality_flags=[],
                dashboard_output=None,
                errors=[],
                warnings=[],
                total_runtime_seconds=1.0,
            )

        monkeypatch.setattr(
            "la_fat.batch_pipeline.run_fat_extraction_pipeline",
            _fake_pipeline,
        )

        run_batch_pipeline(config=config)

        assert len(received_configs) == 1
        assert received_configs[0].spacing_mm == 2.0


# ---------------------------------------------------------------------------
# Tests: No scans case
# ---------------------------------------------------------------------------


class TestEmptyInput:
    """Graceful handling when no scans are found."""

    def test_no_scans_returns_empty_summary(self, tmp_path):
        """When no CT files exist, returns a summary with zeros."""
        from la_fat.batch_pipeline import run_batch_pipeline

        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        summary = run_batch_pipeline(data_dir=data_dir, output_dir=output_dir)

        assert summary["total"] == 0
        assert summary["succeeded"] == 0
        assert summary["failed"] == 0
        assert summary["skipped"] == 0
