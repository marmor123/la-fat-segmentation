"""Tests for the la_fat.pipeline_result module.

Exercises PipelineResultData dataclass, save_pipeline_result, and
load_pipeline_result round-trip serialization.
"""

from __future__ import annotations

import json
import os

import pytest

from la_fat.pipeline_result import (
    PipelineResultData,
    load_pipeline_result,
    save_pipeline_result,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_result() -> PipelineResultData:
    """Return a representative PipelineResultData for testing."""
    return PipelineResultData(
        patient_id="TEST001",
        la_fat_volume_ml=12.34,
        total_fat_volume_ml=56.78,
        pericardium_volume_ml=200.0,
        unassigned_volume_ml=1.23,
        unassigned_fat_pct=2.17,
        anchor_volumes_ml={
            "LA": 12.34,
            "LV": 15.0,
            "RA": 8.5,
            "RV": 10.2,
            "Aorta": 6.1,
            "Pulmonary_Artery": 4.64,
        },
        quality_flags=[
            {
                "severity": "low",
                "concern": "islands_cleaned",
                "detail": "1 small island(s) removed",
                "threshold_value": None,
                "actual_value": 1.0,
            },
        ],
        fat_hu_range=(-190.0, -30.0),
        voxel_volume_ml=0.003375,
        excluded_anchors=[],
        islands_removed=1,
        total_removed_volume_mm3=10.0,
        warnings=["Pericardium fallback triggered"],
        errors=[],
    )


# ===================================================================
# 1. Dataclass construction and immutability
# ===================================================================


class TestPipelineResultDataDataclass:
    """PipelineResultData fields, types, and immutability."""

    def test_fields_present(self, sample_result):
        assert sample_result.patient_id == "TEST001"
        assert sample_result.la_fat_volume_ml == 12.34
        assert sample_result.total_fat_volume_ml == 56.78
        assert sample_result.pericardium_volume_ml == 200.0
        assert sample_result.unassigned_volume_ml == 1.23
        assert sample_result.unassigned_fat_pct == 2.17
        assert sample_result.anchor_volumes_ml["LA"] == 12.34
        assert sample_result.fat_hu_range == (-190.0, -30.0)
        assert sample_result.voxel_volume_ml == 0.003375
        assert sample_result.islands_removed == 1
        assert sample_result.total_removed_volume_mm3 == 10.0
        assert sample_result.warnings == ["Pericardium fallback triggered"]
        assert sample_result.errors == []

    def test_quality_flags_are_dicts(self, sample_result):
        for flag in sample_result.quality_flags:
            assert isinstance(flag, dict)

    def test_frozen_immutable(self, sample_result):
        with pytest.raises(AttributeError):
            sample_result.patient_id = "OTHER"  # type: ignore

    def test_fat_hu_range_is_tuple(self, sample_result):
        assert isinstance(sample_result.fat_hu_range, tuple)


# ===================================================================
# 2. Save / Load round-trip
# ===================================================================


class TestSaveLoadRoundTrip:
    """save_pipeline_result + load_pipeline_result round-trip preserves all fields."""

    def test_round_trip_preserves_all_fields(self, sample_result, tmp_path):
        """All fields survive a save-then-load cycle unchanged."""
        save_pipeline_result(sample_result, str(tmp_path))

        loaded = load_pipeline_result(str(tmp_path))

        assert loaded == sample_result

    def test_round_trip_preserves_empty_fields(self, tmp_path):
        """Round-trip works with empty lists and zero values."""
        minimal = PipelineResultData(
            patient_id="MINIMAL",
            la_fat_volume_ml=0.0,
            total_fat_volume_ml=0.0,
            pericardium_volume_ml=0.0,
            unassigned_volume_ml=0.0,
            unassigned_fat_pct=0.0,
            anchor_volumes_ml={},
            quality_flags=[],
            fat_hu_range=(0.0, 0.0),
            voxel_volume_ml=0.0,
            excluded_anchors=[],
            islands_removed=0,
            total_removed_volume_mm3=0.0,
            warnings=[],
            errors=[],
        )

        save_pipeline_result(minimal, str(tmp_path))
        loaded = load_pipeline_result(str(tmp_path))

        assert loaded == minimal

    def test_round_trip_preserves_excluded_anchors(self, tmp_path):
        """excluded_anchors and quality_flags with values survive round-trip."""
        result = PipelineResultData(
            patient_id="EXCL",
            la_fat_volume_ml=5.0,
            total_fat_volume_ml=20.0,
            pericardium_volume_ml=150.0,
            unassigned_volume_ml=2.0,
            unassigned_fat_pct=10.0,
            anchor_volumes_ml={"LA": 5.0, "LV": 8.0},
            quality_flags=[
                {
                    "severity": "high",
                    "concern": "anchor_excluded",
                    "detail": "RA excluded",
                    "threshold_value": 5.0,
                    "actual_value": None,
                },
            ],
            fat_hu_range=(-200.0, -20.0),
            voxel_volume_ml=0.003375,
            excluded_anchors=["RA", "RV"],
            islands_removed=2,
            total_removed_volume_mm3=25.5,
            warnings=["Anchor RA excluded"],
            errors=["Something failed"],
        )

        save_pipeline_result(result, str(tmp_path))
        loaded = load_pipeline_result(str(tmp_path))

        assert loaded == result

    def test_save_creates_json_file(self, sample_result, tmp_path):
        """save_pipeline_result creates pipeline_result.json in output_dir."""
        save_pipeline_result(sample_result, str(tmp_path))

        json_path = os.path.join(str(tmp_path), "pipeline_result.json")
        assert os.path.isfile(json_path)

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["patient_id"] == "TEST001"

    def test_load_missing_file_raises(self, tmp_path):
        """load_pipeline_result raises FileNotFoundError when no result file."""
        with pytest.raises(FileNotFoundError):
            load_pipeline_result(str(tmp_path))

    def test_load_corrupt_file_raises(self, tmp_path):
        """load_pipeline_result raises ValueError on corrupt JSON."""
        json_path = os.path.join(str(tmp_path), "pipeline_result.json")
        os.makedirs(str(tmp_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            f.write("not valid json")

        with pytest.raises(ValueError, match="Failed to parse"):
            load_pipeline_result(str(tmp_path))


# ===================================================================
# 3. Field type preservation
# ===================================================================


class TestFieldTypePreservation:
    """Fields retain their Python types after round-trip."""

    def test_numeric_types_preserved(self, sample_result, tmp_path):
        """Float fields remain float after round-trip."""
        save_pipeline_result(sample_result, str(tmp_path))
        loaded = load_pipeline_result(str(tmp_path))

        assert isinstance(loaded.la_fat_volume_ml, float)
        assert isinstance(loaded.total_fat_volume_ml, float)
        assert isinstance(loaded.pericardium_volume_ml, float)
        assert isinstance(loaded.unassigned_volume_ml, float)
        assert isinstance(loaded.unassigned_fat_pct, float)
        assert isinstance(loaded.voxel_volume_ml, float)
        assert isinstance(loaded.islands_removed, int)
        assert isinstance(loaded.total_removed_volume_mm3, float)

    def test_list_types_preserved(self, sample_result, tmp_path):
        """List fields remain list after round-trip."""
        save_pipeline_result(sample_result, str(tmp_path))
        loaded = load_pipeline_result(str(tmp_path))

        assert isinstance(loaded.quality_flags, list)
        assert isinstance(loaded.excluded_anchors, list)
        assert isinstance(loaded.warnings, list)
        assert isinstance(loaded.errors, list)

    def test_dict_types_preserved(self, sample_result, tmp_path):
        """Dict fields remain dict after round-trip."""
        save_pipeline_result(sample_result, str(tmp_path))
        loaded = load_pipeline_result(str(tmp_path))

        assert isinstance(loaded.anchor_volumes_ml, dict)

    def test_tuple_types_preserved(self, sample_result, tmp_path):
        """Tuple fields remain tuple after round-trip."""
        save_pipeline_result(sample_result, str(tmp_path))
        loaded = load_pipeline_result(str(tmp_path))

        assert isinstance(loaded.fat_hu_range, tuple)
        assert len(loaded.fat_hu_range) == 2
