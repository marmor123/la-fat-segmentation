"""Tests for pipeline types (PipelineArtifacts, SurfaceSpec, ViewportPreset)."""

from __future__ import annotations

import numpy as np
import pytest

from la_fat.pipeline_types import PipelineArtifacts, SurfaceSpec, ViewportPreset


# ===================================================================
# PipelineArtifacts
# ===================================================================


class TestPipelineArtifactsConstruction:
    """PipelineArtifacts can be constructed and its fields accessed."""

    def test_minimal_construction(self):
        """Construct with minimal valid data and verify field access."""
        from dataclasses import dataclass

        @dataclass
        class FakePartitionResult:
            anchor_assignments: np.ndarray
            all_fat_mask: np.ndarray

        @dataclass
        class FakeCleanupResult:
            cleaned_mask: np.ndarray

        shape = (8, 8, 8)
        artifacts = PipelineArtifacts(
            anchor_masks={
                "LA": np.zeros(shape, dtype=bool),
                "LV": np.zeros(shape, dtype=bool),
            },
            pericardium_mask=np.ones(shape, dtype=bool),
            partition_result=FakePartitionResult(
                anchor_assignments=np.zeros(shape, dtype=np.int32),
                all_fat_mask=np.zeros(shape, dtype=bool),
            ),
            cleanup_result=FakeCleanupResult(
                cleaned_mask=np.zeros(shape, dtype=bool),
            ),
            spacing=(1.5, 1.5, 1.5),
        )

        assert isinstance(artifacts, PipelineArtifacts)
        assert "LA" in artifacts.anchor_masks
        assert artifacts.anchor_masks["LA"].shape == shape
        assert artifacts.pericardium_mask.shape == shape
        assert artifacts.spacing == (1.5, 1.5, 1.5)

    def test_frozen_cannot_be_mutated(self):
        """PipelineArtifacts is frozen -- attribute assignment raises TypeError."""
        from dataclasses import dataclass

        @dataclass
        class FakePR:
            anchor_assignments: np.ndarray
            all_fat_mask: np.ndarray

        @dataclass
        class FakeCR:
            cleaned_mask: np.ndarray

        shape = (4, 4, 4)
        artifacts = PipelineArtifacts(
            anchor_masks={"LA": np.zeros(shape, dtype=bool)},
            pericardium_mask=np.zeros(shape, dtype=bool),
            partition_result=FakePR(
                anchor_assignments=np.zeros(shape, dtype=np.int32),
                all_fat_mask=np.zeros(shape, dtype=bool),
            ),
            cleanup_result=FakeCR(cleaned_mask=np.zeros(shape, dtype=bool)),
            spacing=(1.0, 1.0, 1.0),
        )

        with pytest.raises(AttributeError):
            artifacts.anchor_masks = {}  # type: ignore[misc]

    def test_field_types_are_preserved(self):
        """Fields retain their declared types after construction."""
        from dataclasses import dataclass

        @dataclass
        class FakePR:
            anchor_assignments: np.ndarray
            all_fat_mask: np.ndarray

        @dataclass
        class FakeCR:
            cleaned_mask: np.ndarray

        shape = (10, 10, 10)
        artifacts = PipelineArtifacts(
            anchor_masks={"X": np.ones(shape, dtype=bool)},
            pericardium_mask=np.zeros(shape, dtype=bool),
            partition_result=FakePR(
                anchor_assignments=np.ones(shape, dtype=np.int32),
                all_fat_mask=np.ones(shape, dtype=bool),
            ),
            cleanup_result=FakeCR(cleaned_mask=np.zeros(shape, dtype=bool)),
            spacing=(2.0, 2.0, 2.0),
        )

        assert isinstance(artifacts.anchor_masks, dict)
        assert isinstance(artifacts.pericardium_mask, np.ndarray)
        assert isinstance(artifacts.spacing, tuple)


# ===================================================================
# SurfaceSpec
# ===================================================================


class TestSurfaceSpecConstruction:
    """SurfaceSpec can be constructed and its fields accessed."""

    def test_minimal_construction(self):
        """Construct with only required fields."""
        spec = SurfaceSpec(
            color=(1.0, 0.0, 0.0),
            opacity=0.5,
            label="Test Surface",
        )
        assert spec.color == (1.0, 0.0, 0.0)
        assert spec.opacity == 0.5
        assert spec.label == "Test Surface"
        assert spec.show_edges is False  # default
        assert spec.style == "surface"  # default

    def test_full_construction(self):
        """Construct with all fields including optional ones."""
        spec = SurfaceSpec(
            color=(0.0, 0.5, 1.0),
            opacity=0.8,
            label="Full Spec",
            show_edges=True,
            style="wireframe",
        )
        assert spec.color == (0.0, 0.5, 1.0)
        assert spec.opacity == 0.8
        assert spec.label == "Full Spec"
        assert spec.show_edges is True
        assert spec.style == "wireframe"

    def test_frozen_cannot_be_mutated(self):
        """SurfaceSpec is frozen -- attribute assignment raises TypeError."""
        spec = SurfaceSpec(color=(0.0, 1.0, 0.0), opacity=0.3, label="Frozen")
        with pytest.raises(AttributeError):
            spec.opacity = 0.9  # type: ignore[misc]

    def test_defaults_are_applied(self):
        """Optional fields get their default values."""
        spec = SurfaceSpec(color=(0.5, 0.5, 0.5), opacity=1.0, label="Defaults")
        assert spec.show_edges is False
        assert spec.style == "surface"


# ===================================================================
# ViewportPreset
# ===================================================================


class TestViewportPresetConstruction:
    """ViewportPreset can be constructed and its fields accessed."""

    def test_minimal_construction(self):
        """Construct with only required fields."""
        preset = ViewportPreset(name="Show All", label="Show All")
        assert preset.name == "Show All"
        assert preset.label == "Show All"
        assert preset.button_type == "default"
        assert preset.hide is None
        assert preset.show_only is None

    def test_with_hide(self):
        """Construct with hide list."""
        preset = ViewportPreset(
            name="Anchors Only",
            label="Anchors Only",
            button_type="primary",
            hide=["Pericardium"],
        )
        assert preset.name == "Anchors Only"
        assert preset.hide == ["Pericardium"]
        assert preset.show_only is None

    def test_with_show_only(self):
        """Construct with show_only list."""
        preset = ViewportPreset(
            name="Pericardium Only",
            label="Pericardium Only",
            button_type="warning",
            show_only=["Pericardium"],
        )
        assert preset.show_only == ["Pericardium"]
        assert preset.hide is None

    def test_frozen_cannot_be_mutated(self):
        """ViewportPreset is frozen -- attribute assignment raises TypeError."""
        preset = ViewportPreset(name="Test", label="Test")
        with pytest.raises(AttributeError):
            preset.name = "Other"  # type: ignore[misc]

    def test_optional_defaults(self):
        """Optional fields default correctly."""
        preset = ViewportPreset(name="Hide All", label="Hide All")
        assert preset.button_type == "default"
        assert preset.hide is None
        assert preset.show_only is None
