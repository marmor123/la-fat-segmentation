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

    def _create_masks(self, data_dir: str, patient_id: str) -> None:
        """Create minimal TS mask files so TS pre-compute is skipped."""
        mask_dir = os.path.join(data_dir, "intermediate", patient_id)
        os.makedirs(mask_dir, exist_ok=True)
        _touch(os.path.join(mask_dir, f"{patient_id}_LA.nii.gz"))

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
        # Pre-create masks so TS pre-compute is skipped
        self._create_masks(data_dir, "PAT1")
        self._create_masks(data_dir, "PAT2")

        config = PipelineConfig(data_dir=data_dir, output_dir=output_dir)

        called_ids: list[str] = []

        def _fake_pipeline(patient_id, config=None, config_path=None):
            called_ids.append(patient_id)
            return PipelineResult(
                patient_id=patient_id,
                success=True,
                partition_result=None,

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
        self._create_masks(data_dir, "NEW")

        # Pre-mark OLD as completed
        _write_result_json(output_dir, "OLD")

        called_ids: list[str] = []

        def _fake_pipeline(patient_id, config=None, config_path=None):
            called_ids.append(patient_id)
            return PipelineResult(
                patient_id=patient_id,
                success=True,
                partition_result=None,

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

    def _create_masks(self, data_dir: str, patient_id: str) -> None:
        """Create minimal TS mask files so TS pre-compute is skipped."""
        mask_dir = os.path.join(data_dir, "intermediate", patient_id)
        os.makedirs(mask_dir, exist_ok=True)
        _touch(os.path.join(mask_dir, f"{patient_id}_LA.nii.gz"))

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
        self._create_masks(data_dir, "GOOD1")
        self._create_masks(data_dir, "BAD")
        self._create_masks(data_dir, "GOOD2")

        called_ids: list[str] = []

        def _fake_pipeline(patient_id, config=None, config_path=None):
            called_ids.append(patient_id)
            if patient_id == "BAD":
                return PipelineResult(
                    patient_id=patient_id,
                    success=False,
                    partition_result=None,
    
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
        self._create_masks(data_dir, "CRASH")
        self._create_masks(data_dir, "SURVIVOR")

        called_ids: list[str] = []

        def _fake_pipeline(patient_id, config=None, config_path=None):
            called_ids.append(patient_id)
            if patient_id == "CRASH":
                raise RuntimeError("KABOOM")
            return PipelineResult(
                patient_id=patient_id,
                success=True,
                partition_result=None,

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
        self._create_masks(data_dir, "FAIL1")

        def _fake_pipeline(patient_id, config=None, config_path=None):
            return PipelineResult(
                patient_id=patient_id,
                success=False,
                partition_result=None,

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
        # Pre-create masks so TS pre-compute is skipped
        mask_dir = os.path.join(data_dir, "intermediate", "PATIENT")
        os.makedirs(mask_dir, exist_ok=True)
        _touch(os.path.join(mask_dir, "PATIENT_LA.nii.gz"))

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


# ---------------------------------------------------------------------------
# Tests: TS pre-compute integration
# ---------------------------------------------------------------------------


class TestTsPrecomputeIntegration:
    """TS pre-compute runs automatically when masks don't exist."""

    def test_masks_exist_returns_true_when_masks_present(self, tmp_path):
        """_masks_exist returns True when mask files exist."""
        from la_fat.batch_pipeline import _masks_exist

        intermediate_dir = str(tmp_path / "intermediate")
        patient_dir = os.path.join(intermediate_dir, "PAT1")
        os.makedirs(patient_dir)
        _touch(os.path.join(patient_dir, "PAT1_LA.nii.gz"))
        _touch(os.path.join(patient_dir, "PAT1_Pericardium.nii.gz"))

        assert _masks_exist(intermediate_dir, "PAT1") is True

    def test_masks_exist_returns_false_when_no_masks(self, tmp_path):
        """_masks_exist returns False when no .nii.gz files exist."""
        from la_fat.batch_pipeline import _masks_exist

        intermediate_dir = str(tmp_path / "intermediate")
        patient_dir = os.path.join(intermediate_dir, "PAT2")
        os.makedirs(patient_dir)

        assert _masks_exist(intermediate_dir, "PAT2") is False

    def test_masks_exist_returns_false_when_dir_missing(self, tmp_path):
        """_masks_exist returns False when patient dir doesn't exist."""
        from la_fat.batch_pipeline import _masks_exist

        intermediate_dir = str(tmp_path / "intermediate")
        os.makedirs(intermediate_dir, exist_ok=True)

        assert _masks_exist(intermediate_dir, "NOPAT") is False

    def test_ts_called_when_masks_missing(self, tmp_path, monkeypatch):
        """TS pre-compute is called when no masks exist for a patient."""
        from la_fat.batch_pipeline import run_batch_pipeline
        from la_fat.ts_runner import TsPrecomputeResult

        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir)
        _touch(os.path.join(raw_dir, "NEWPT.nii.gz"))

        ts_calls: list[str] = []

        def _fake_ts(ct_path, output_dir, config):
            ts_calls.append(ct_path)
            return TsPrecomputeResult(
                patient_id="NEWPT",
                output_dir="",
                masks_saved={"LA": "/fake/LA.nii.gz"},
                mask_volumes_ml={"LA": 10.0},
                errors=[],
                total_runtime_seconds=5.0,
            )

        pipeline_calls: list[str] = []

        def _fake_pipeline(patient_id, config=None, config_path=None):
            pipeline_calls.append(patient_id)
            return PipelineResult(
                patient_id=patient_id,
                success=True,
                partition_result=None,

                pericardium_result=None,
                cleanup_result=None,
                quality_flags=[],
                dashboard_output=None,
                errors=[],
                warnings=[],
                total_runtime_seconds=1.0,
            )

        monkeypatch.setattr(
            "la_fat.batch_pipeline.run_ts_precompute",
            _fake_ts,
        )
        monkeypatch.setattr(
            "la_fat.batch_pipeline.run_fat_extraction_pipeline",
            _fake_pipeline,
        )

        summary = run_batch_pipeline(data_dir=data_dir, output_dir=output_dir)

        assert len(ts_calls) == 1, f"TS should be called once, got {len(ts_calls)}"
        assert len(pipeline_calls) == 1
        assert summary["succeeded"] == 1

    def test_ts_skipped_when_masks_exist(self, tmp_path, monkeypatch):
        """TS pre-compute is skipped when masks already exist."""
        from la_fat.batch_pipeline import run_batch_pipeline

        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir)
        _touch(os.path.join(raw_dir, "HASMASK.nii.gz"))

        # Pre-create mask files
        intermediate_dir = os.path.join(data_dir, "intermediate", "HASMASK")
        os.makedirs(intermediate_dir)
        _touch(os.path.join(intermediate_dir, "HASMASK_LA.nii.gz"))

        ts_calls: list[str] = []

        def _fake_ts(ct_path, output_dir, config):
            ts_calls.append(ct_path)
            from la_fat.ts_runner import TsPrecomputeResult
            return TsPrecomputeResult(
                patient_id="HASMASK", output_dir="",
                masks_saved={"LA": "/fake/LA.nii.gz"},
                mask_volumes_ml={"LA": 10.0},
                errors=[], total_runtime_seconds=1.0,
            )

        pipeline_calls: list[str] = []

        def _fake_pipeline(patient_id, config=None, config_path=None):
            pipeline_calls.append(patient_id)
            return PipelineResult(
                patient_id=patient_id,
                success=True,
                partition_result=None,

                pericardium_result=None,
                cleanup_result=None,
                quality_flags=[],
                dashboard_output=None,
                errors=[],
                warnings=[],
                total_runtime_seconds=1.0,
            )

        monkeypatch.setattr(
            "la_fat.batch_pipeline.run_ts_precompute",
            _fake_ts,
        )
        monkeypatch.setattr(
            "la_fat.batch_pipeline.run_fat_extraction_pipeline",
            _fake_pipeline,
        )

        summary = run_batch_pipeline(data_dir=data_dir, output_dir=output_dir)

        assert len(ts_calls) == 0, f"TS should be skipped, got {len(ts_calls)} calls"
        assert len(pipeline_calls) == 1

    def test_ts_failure_skips_pipeline(self, tmp_path, monkeypatch):
        """When TS pre-compute fails, the pipeline is not called for that patient."""
        from la_fat.batch_pipeline import run_batch_pipeline

        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir)
        _touch(os.path.join(raw_dir, "TSFAIL.nii.gz"))

        def _fake_ts(ct_path, output_dir, config):
            raise RuntimeError("TS crashed!")

        pipeline_calls: list[str] = []

        def _fake_pipeline(patient_id, config=None, config_path=None):
            pipeline_calls.append(patient_id)
            return PipelineResult(
                patient_id=patient_id, success=True,
                partition_result=None,
                pericardium_result=None, cleanup_result=None,
                quality_flags=[], dashboard_output=None,
                errors=[], warnings=[], total_runtime_seconds=1.0,
            )

        monkeypatch.setattr(
            "la_fat.batch_pipeline.run_ts_precompute",
            _fake_ts,
        )
        monkeypatch.setattr(
            "la_fat.batch_pipeline.run_fat_extraction_pipeline",
            _fake_pipeline,
        )

        summary = run_batch_pipeline(data_dir=data_dir, output_dir=output_dir)

        assert len(pipeline_calls) == 0, "Pipeline should not be called after TS failure"
        assert summary["failed"] == 1
        assert "TSFAIL" in summary["failed_ids"]

    def test_mixed_masks_some_have_some_dont(self, tmp_path, monkeypatch):
        """Patients with masks skip TS; those without get TS first."""
        from la_fat.batch_pipeline import run_batch_pipeline
        from la_fat.ts_runner import TsPrecomputeResult

        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir)
        _touch(os.path.join(raw_dir, "HAS.nii.gz"))
        _touch(os.path.join(raw_dir, "NEEDS.nii.gz"))

        # HAS already has masks
        intermediate_dir = os.path.join(data_dir, "intermediate")
        os.makedirs(os.path.join(intermediate_dir, "HAS"))
        _touch(os.path.join(intermediate_dir, "HAS", "HAS_LA.nii.gz"))

        # NEEDS has no masks (dir doesn't exist)

        ts_calls: list[str] = []

        def _fake_ts(ct_path, output_dir, config):
            ts_calls.append(ct_path)
            return TsPrecomputeResult(
                patient_id="NEEDS", output_dir="",
                masks_saved={"LA": "/fake/LA.nii.gz"},
                mask_volumes_ml={"LA": 10.0},
                errors=[], total_runtime_seconds=1.0,
            )

        pipeline_calls: list[str] = []

        def _fake_pipeline(patient_id, config=None, config_path=None):
            pipeline_calls.append(patient_id)
            return PipelineResult(
                patient_id=patient_id, success=True,
                partition_result=None,
                pericardium_result=None, cleanup_result=None,
                quality_flags=[], dashboard_output=None,
                errors=[], warnings=[], total_runtime_seconds=1.0,
            )

        monkeypatch.setattr(
            "la_fat.batch_pipeline.run_ts_precompute",
            _fake_ts,
        )
        monkeypatch.setattr(
            "la_fat.batch_pipeline.run_fat_extraction_pipeline",
            _fake_pipeline,
        )

        summary = run_batch_pipeline(data_dir=data_dir, output_dir=output_dir)

        assert len(ts_calls) == 1, f"TS should be called once for NEEDS, got {len(ts_calls)}"
        assert len(pipeline_calls) == 2, f"Pipeline should run for both, got {len(pipeline_calls)}"
        assert summary["succeeded"] == 2


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
