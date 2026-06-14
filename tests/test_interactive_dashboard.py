"""Tests for the la_fat.interactive_dashboard module.

Exercises patient discovery, data loading, the interactive dashboard
construction, and the Step 7 3D viewport.  Panel UI rendering and
PyVista 3D rendering are not tested — only the data/logic layer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from la_fat.interactive_dashboard import (
    PatientSummary,
    _build_step7_viewport,
    discover_patients,
)


# ---------------------------------------------------------------------------
# Test: empty directory
# ---------------------------------------------------------------------------


class TestDiscoverPatientsEmptyDir:
    """Scanning an empty or non-existent output directory."""

    def test_discover_patients_empty_dir_returns_empty_list(self, tmp_path):
        """An empty output directory yields an empty patient list with no crash."""
        patients = discover_patients(str(tmp_path))
        assert patients == []


# ---------------------------------------------------------------------------
# Test: complete patient
# ---------------------------------------------------------------------------


class TestDiscoverPatientsComplete:
    """Detecting a patient with the full mesh output."""

    def _make_complete_patient(self, root: Path, patient_id: str) -> None:
        """Create a fake patient directory with all expected files."""
        patient_dir = root / patient_id
        meshes_dir = patient_dir / "meshes" / "step7_final"
        meshes_dir.mkdir(parents=True, exist_ok=True)
        (meshes_dir / "LA_fat.ply").touch()

    def test_discover_patients_discovers_complete_patient(self, tmp_path):
        """A patient with meshes/step7_final/LA_fat.ply is found as complete."""
        self._make_complete_patient(tmp_path, "patient_001")

        patients = discover_patients(str(tmp_path))
        assert len(patients) == 1
        assert patients[0].patient_id == "patient_001"
        assert patients[0].status == "complete"


# ---------------------------------------------------------------------------
# Test: partial patient
# ---------------------------------------------------------------------------


class TestDiscoverPatientsPartial:
    """Detecting a patient with some meshes but no final step."""

    def _make_partial_patient(self, root: Path, patient_id: str) -> None:
        """Create a fake patient directory without step7_final."""
        patient_dir = root / patient_id
        meshes_dir = patient_dir / "meshes" / "step2_anchors"
        meshes_dir.mkdir(parents=True, exist_ok=True)
        (meshes_dir / "LA.ply").touch()
        (meshes_dir / "LV.ply").touch()

    def test_discover_patients_discovers_partial_patient(self, tmp_path):
        """A patient with meshes but no step7_final is found as partial."""
        self._make_partial_patient(tmp_path, "patient_002")

        patients = discover_patients(str(tmp_path))
        assert len(patients) == 1
        assert patients[0].patient_id == "patient_002"
        assert patients[0].status == "partial"


# ---------------------------------------------------------------------------
# Test: quality flag severity
# ---------------------------------------------------------------------------


class TestDiscoverPatientsQualityFlags:
    """Correctly determining severity from quality_flags.json."""

    def _make_patient_with_flags(
        self, root: Path, patient_id: str, flags: list[dict]
    ) -> None:
        """Create a fake patient directory with quality_flags.json."""
        patient_dir = root / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)
        flags_path = patient_dir / "quality_flags.json"
        with open(flags_path, "w", encoding="utf-8") as f:
            json.dump(flags, f)

    def test_discover_patients_reads_quality_flags_severity(self, tmp_path):
        """Highest severity from quality_flags.json is reflected in summary."""
        flags = [
            {"severity": "low", "concern": "islands_cleaned", "detail": "..."},
            {"severity": "medium", "concern": "lv_exceeds_la", "detail": "..."},
            {"severity": "high", "concern": "anchor_excluded", "detail": "..."},
        ]
        self._make_patient_with_flags(tmp_path, "patient_003", flags)

        patients = discover_patients(str(tmp_path))
        assert len(patients) == 1
        assert patients[0].severity == "high"

    def test_discover_patients_no_flags_returns_none_severity(self, tmp_path):
        """Patient with empty quality_flags.json has severity 'none'."""
        self._make_patient_with_flags(tmp_path, "patient_004", [])

        patients = discover_patients(str(tmp_path))
        assert len(patients) == 1
        assert patients[0].severity == "none"

    def test_discover_patients_missing_flags_file_returns_none_severity(
        self, tmp_path
    ):
        """Patient without quality_flags.json has severity 'none'."""
        patient_dir = tmp_path / "patient_005"
        patient_dir.mkdir(parents=True, exist_ok=True)

        patients = discover_patients(str(tmp_path))
        assert len(patients) == 1
        assert patients[0].severity == "none"


# ---------------------------------------------------------------------------
# Test: missing summary.csv
# ---------------------------------------------------------------------------


class TestDiscoverPatientsMissingSummary:
    """Handling patients without summary.csv."""

    def _make_patient_no_summary(self, root: Path, patient_id: str) -> None:
        """Create a fake patient directory without summary.csv."""
        patient_dir = root / patient_id
        patient_dir.mkdir(parents=True, exist_ok=True)

    def test_discover_patients_missing_summary_csv_handles_gracefully(
        self, tmp_path
    ):
        """A patient without summary.csv still appears with N/A volumes."""
        self._make_patient_no_summary(tmp_path, "patient_006")

        patients = discover_patients(str(tmp_path))
        assert len(patients) == 1
        assert patients[0].patient_id == "patient_006"
        assert patients[0].status == "partial"
        assert patients[0].la_fat_volume_ml is None
        assert patients[0].total_epicardial_volume_ml is None


# ---------------------------------------------------------------------------
# Test: non-patient directories
# ---------------------------------------------------------------------------


class TestDiscoverPatientsIgnoresNonPatientDirs:
    """Files in output_dir are ignored; only directories are scanned."""

    def test_discover_patients_ignores_non_patient_dirs(self, tmp_path):
        """Regular files in the output directory are not treated as patients."""
        # Create a real patient dir
        patient_dir = tmp_path / "patient_007"
        patient_dir.mkdir(parents=True, exist_ok=True)

        # Create a regular file at the root level
        (tmp_path / "output.log").touch()
        (tmp_path / "summary.csv").touch()

        patients = discover_patients(str(tmp_path))
        # Only the directory should be found
        assert len(patients) == 1
        assert patients[0].patient_id == "patient_007"


# ===================================================================
# Helpers for 3D viewport tests
# ===================================================================


def _make_all_meshes(root: Path, patient_id: str) -> str:
    """Create synthetic PLY files for all three steps via mesh_extractor.

    Returns the patient directory path.
    """
    from la_fat.mesh_extractor import extract_interactive_meshes
    from tests.test_mesh_extractor import _make_pipeline_state

    state = _make_pipeline_state()
    patient_dir = root / patient_id
    patient_dir.mkdir(parents=True, exist_ok=True)
    extract_interactive_meshes(state, str(patient_dir))
    return str(patient_dir)


# ===================================================================
# Test: Step 2 (Anchors) 3D viewport
# ===================================================================


class TestStep2Viewport:
    """Exercises ``_build_step2_viewport`` — the Anchors + Pericardium 3D viewport."""

    def test_step2_viewport_loads_ply_files(self, tmp_path):
        """_build_step2_viewport returns a Panel layout when PLY files exist."""
        pyvista = pytest.importorskip("pyvista", reason="PyVista not installed")

        from la_fat.interactive_dashboard import _build_step2_viewport

        patient_dir = _make_all_meshes(tmp_path, "test_step2_patient_001")
        result = _build_step2_viewport(patient_dir)

        import panel as pn

        assert isinstance(result, pn.Column)
        # A populated viewport has more than the empty-card fallback
        assert len(result) >= 2

    def test_step2_viewport_handles_missing_files(self, tmp_path):
        """_build_step2_viewport shows placeholder when step meshes missing."""
        from la_fat.interactive_dashboard import _build_step2_viewport

        patient_dir = tmp_path / "empty_patient"
        patient_dir.mkdir(parents=True, exist_ok=True)

        result = _build_step2_viewport(str(patient_dir))

        import panel as pn

        assert isinstance(result, pn.Column)
        md_panes = [c for c in result if isinstance(c, pn.pane.Markdown)]
        texts = [str(md.object) for md in md_panes]
        assert any("not available" in t for t in texts)

    def test_step2_viewport_has_seven_checkboxes(self, tmp_path):
        """_build_step2_viewport creates checkboxes for all 7 surfaces."""
        pyvista = pytest.importorskip("pyvista", reason="PyVista not installed")

        from la_fat.interactive_dashboard import _build_step2_viewport

        patient_dir = _make_all_meshes(tmp_path, "test_step2_patient_002")
        result = _build_step2_viewport(patient_dir)

        assert hasattr(result, "_checkboxes")
        assert len(result._checkboxes) == 7  # 6 anchors + 1 pericardium

    def test_step2_viewport_has_four_preset_buttons(self, tmp_path):
        """_build_step2_viewport creates all four preset buttons."""
        pyvista = pytest.importorskip("pyvista", reason="PyVista not installed")

        from la_fat.interactive_dashboard import _build_step2_viewport

        patient_dir = _make_all_meshes(tmp_path, "test_step2_patient_003")
        result = _build_step2_viewport(patient_dir)

        assert hasattr(result, "_preset_handlers")
        assert set(result._preset_handlers.keys()) == {
            "Show All",
            "Hide All",
            "Anchors Only",
            "Pericardium Only",
        }


# ===================================================================
# Test: Step 5 (Partition) 3D viewport
# ===================================================================


class TestStep5Viewport:
    """Exercises ``_build_step5_viewport`` — the Fat Partition 3D viewport."""

    def test_step5_viewport_loads_ply_files(self, tmp_path):
        """_build_step5_viewport returns a Panel layout when PLY files exist."""
        pyvista = pytest.importorskip("pyvista", reason="PyVista not installed")

        from la_fat.interactive_dashboard import _build_step5_viewport

        patient_dir = _make_all_meshes(tmp_path, "test_step5_patient_001")
        result = _build_step5_viewport(patient_dir)

        import panel as pn

        assert isinstance(result, pn.Column)
        assert len(result) >= 2

    def test_step5_viewport_handles_missing_files(self, tmp_path):
        """_build_step5_viewport shows placeholder when step meshes missing."""
        from la_fat.interactive_dashboard import _build_step5_viewport

        patient_dir = tmp_path / "empty_patient"
        patient_dir.mkdir(parents=True, exist_ok=True)

        result = _build_step5_viewport(str(patient_dir))

        import panel as pn

        assert isinstance(result, pn.Column)
        md_panes = [c for c in result if isinstance(c, pn.pane.Markdown)]
        texts = [str(md.object) for md in md_panes]
        assert any("not available" in t for t in texts)

    def test_step5_viewport_has_seven_checkboxes(self, tmp_path):
        """_build_step5_viewport creates checkboxes for all 7 surfaces."""
        pyvista = pytest.importorskip("pyvista", reason="PyVista not installed")

        from la_fat.interactive_dashboard import _build_step5_viewport

        patient_dir = _make_all_meshes(tmp_path, "test_step5_patient_002")
        result = _build_step5_viewport(patient_dir)

        assert hasattr(result, "_checkboxes")
        assert len(result._checkboxes) == 7  # 6 fat surfaces + 1 pericardium

    def test_step5_viewport_has_four_preset_buttons(self, tmp_path):
        """_build_step5_viewport creates all four preset buttons."""
        pyvista = pytest.importorskip("pyvista", reason="PyVista not installed")

        from la_fat.interactive_dashboard import _build_step5_viewport

        patient_dir = _make_all_meshes(tmp_path, "test_step5_patient_003")
        result = _build_step5_viewport(patient_dir)

        assert hasattr(result, "_preset_handlers")
        assert set(result._preset_handlers.keys()) == {
            "Show All",
            "Hide All",
            "Fat Only",
            "Pericardium Only",
        }


# ===================================================================
# Test: Step 2 preset button behavior
# ===================================================================


class TestStep2Presets:
    """Exercises preset button behavior on the Step 2 viewport."""

    def test_step2_anchors_only_preset(self, tmp_path):
        """Anchors Only preset hides Pericardium and shows all anchors."""
        pyvista = pytest.importorskip("pyvista", reason="PyVista not installed")

        from la_fat.interactive_dashboard import _build_step2_viewport

        patient_dir = _make_all_meshes(tmp_path, "test_preset_patient_001")
        result = _build_step2_viewport(patient_dir)

        # All checkboxes start as True
        assert result._checkboxes["Pericardium"].value is True

        # Trigger Anchors Only preset
        result._preset_handlers["Anchors Only"](None)

        # Pericardium should now be unchecked
        assert result._checkboxes["Pericardium"].value is False

        # All anchor checkboxes should remain True
        for name in ["LA", "LV", "RA", "RV", "Aorta", "Pulmonary_Artery"]:
            assert result._checkboxes[name].value is True

    def test_step2_pericardium_only_preset(self, tmp_path):
        """Pericardium Only preset shows only Pericardium."""
        pyvista = pytest.importorskip("pyvista", reason="PyVista not installed")

        from la_fat.interactive_dashboard import _build_step2_viewport

        patient_dir = _make_all_meshes(tmp_path, "test_preset_patient_002")
        result = _build_step2_viewport(patient_dir)

        # Trigger Pericardium Only preset
        result._preset_handlers["Pericardium Only"](None)

        # Pericardium should be checked
        assert result._checkboxes["Pericardium"].value is True

        # All anchor checkboxes should be False
        for name in ["LA", "LV", "RA", "RV", "Aorta", "Pulmonary_Artery"]:
            assert result._checkboxes[name].value is False

    def test_step2_show_all_restores_visibility(self, tmp_path):
        """Show All preset restores all checkboxes to True."""
        pyvista = pytest.importorskip("pyvista", reason="PyVista not installed")

        from la_fat.interactive_dashboard import _build_step2_viewport

        patient_dir = _make_all_meshes(tmp_path, "test_preset_patient_003")
        result = _build_step2_viewport(patient_dir)

        # First hide all
        result._preset_handlers["Hide All"](None)
        for cb in result._checkboxes.values():
            assert cb.value is False

        # Then show all
        result._preset_handlers["Show All"](None)
        for cb in result._checkboxes.values():
            assert cb.value is True

    def test_step2_hide_all_hides_everything(self, tmp_path):
        """Hide All preset sets all checkboxes to False."""
        pyvista = pytest.importorskip("pyvista", reason="PyVista not installed")

        from la_fat.interactive_dashboard import _build_step2_viewport

        patient_dir = _make_all_meshes(tmp_path, "test_preset_patient_004")
        result = _build_step2_viewport(patient_dir)

        result._preset_handlers["Hide All"](None)

        for cb in result._checkboxes.values():
            assert cb.value is False


# ===================================================================
# Test: Step 7 refactored viewport (backward compatibility)
# ===================================================================


class TestStep7ViewportRefactored:
    """Exercises ``_build_step7_viewport`` after refactoring to shared helper."""

    def test_step7_viewport_still_works(self, tmp_path):
        """_build_step7_viewport still returns a valid Panel layout after refactor."""
        pyvista = pytest.importorskip("pyvista", reason="PyVista not installed")

        patient_dir = _make_all_meshes(tmp_path, "test_step7_refactored")
        result = _build_step7_viewport(patient_dir)

        import panel as pn

        assert isinstance(result, pn.Column)
        assert len(result) >= 2

    def test_step7_viewport_has_checkboxes(self, tmp_path):
        """_build_step7_viewport exposes checkboxes after refactor."""
        pyvista = pytest.importorskip("pyvista", reason="PyVista not installed")

        patient_dir = _make_all_meshes(tmp_path, "test_step7_checkboxes")
        result = _build_step7_viewport(patient_dir)

        assert hasattr(result, "_checkboxes")
        assert len(result._checkboxes) == 3  # LA_chamber, Pericardium, LA_fat


# ===================================================================
# Test: _get_high_flags helper
# ===================================================================


class TestGetHighFlags:
    """Exercises ``_get_high_flags`` — extracting high-severity quality flags."""

    def test_get_high_flags_returns_high_flags_only(self, tmp_path):
        """Only high-severity flags are returned by _get_high_flags."""
        patient_dir = tmp_path / "patient_high"
        patient_dir.mkdir(parents=True)
        flags_path = patient_dir / "quality_flags.json"
        flags = [
            {"severity": "high", "concern": "anchor_excluded", "detail": "LA anchor excluded"},
            {"severity": "medium", "concern": "lv_exceeds_la", "detail": "LV larger than LA"},
            {"severity": "low", "concern": "islands_cleaned", "detail": "..."},
        ]
        with open(flags_path, "w", encoding="utf-8") as f:
            json.dump(flags, f)

        from la_fat.interactive_dashboard import _get_high_flags

        result = _get_high_flags(str(patient_dir))
        assert len(result) == 1
        assert result[0]["concern"] == "anchor_excluded"

    def test_get_high_flags_empty_when_no_high_flags(self, tmp_path):
        """Only medium/low flags produce an empty list."""
        patient_dir = tmp_path / "patient_med"
        patient_dir.mkdir(parents=True)
        flags_path = patient_dir / "quality_flags.json"
        flags = [
            {"severity": "medium", "concern": "lv_exceeds_la", "detail": "..."},
            {"severity": "low", "concern": "islands_cleaned", "detail": "..."},
        ]
        with open(flags_path, "w", encoding="utf-8") as f:
            json.dump(flags, f)

        from la_fat.interactive_dashboard import _get_high_flags

        result = _get_high_flags(str(patient_dir))
        assert result == []

    def test_get_high_flags_empty_when_no_flags_file(self, tmp_path):
        """Missing quality_flags.json returns empty list."""
        patient_dir = tmp_path / "patient_no_flags"
        patient_dir.mkdir(parents=True)

        from la_fat.interactive_dashboard import _get_high_flags

        result = _get_high_flags(str(patient_dir))
        assert result == []


# ===================================================================
# Test: _check_step_available helper
# ===================================================================


class TestCheckStepAvailable:
    """Exercises ``_check_step_available`` — checking step directory existence."""

    def test_step_available_returns_true(self, tmp_path):
        """Existing meshes directory returns True."""
        patient_dir = tmp_path / "patient_ok"
        meshes_dir = patient_dir / "meshes" / "step2_anchors"
        meshes_dir.mkdir(parents=True)

        from la_fat.interactive_dashboard import _check_step_available

        assert _check_step_available(str(patient_dir), "step2_anchors") is True

    def test_step_missing_returns_false(self, tmp_path):
        """Missing meshes directory returns False."""
        patient_dir = tmp_path / "patient_missing"
        patient_dir.mkdir(parents=True)

        from la_fat.interactive_dashboard import _check_step_available

        assert _check_step_available(str(patient_dir), "step2_anchors") is False


# ===================================================================
# Test: Missing step directory shows placeholder
# ===================================================================


class TestMissingStep:
    """Missing step directory shows a placeholder card instead of crashing."""

    def test_missing_step5_dir_shows_placeholder(self, tmp_path):
        """_build_step5_viewport shows placeholder when step5 dir missing."""
        from la_fat.interactive_dashboard import _build_step5_viewport

        patient_dir = tmp_path / "partial_patient"
        patient_dir.mkdir(parents=True)
        result = _build_step5_viewport(str(patient_dir))

        import panel as pn

        assert isinstance(result, pn.Column)
        md_panes = [c for c in result if isinstance(c, pn.pane.Markdown)]
        texts = [str(md.object) for md in md_panes]
        assert any("not available" in t for t in texts)

    def test_missing_step2_dir_shows_placeholder(self, tmp_path):
        """_build_step2_viewport shows placeholder when step2 dir missing."""
        from la_fat.interactive_dashboard import _build_step2_viewport

        patient_dir = tmp_path / "partial_patient"
        patient_dir.mkdir(parents=True)
        result = _build_step2_viewport(str(patient_dir))

        import panel as pn

        assert isinstance(result, pn.Column)
        md_panes = [c for c in result if isinstance(c, pn.pane.Markdown)]
        texts = [str(md.object) for md in md_panes]
        assert any("not available" in t for t in texts)


# ===================================================================
# Test: Missing surface checkbox disabled
# ===================================================================


class TestMissingSurfaceCheckbox:
    """Missing .ply files produce disabled checkboxes with tooltip."""

    def test_missing_surface_checkbox_disabled(self, tmp_path):
        """One missing .ply produces disabled checkbox for that surface."""
        pyvista = pytest.importorskip("pyvista", reason="PyVista not installed")

        from la_fat.interactive_dashboard import _build_step2_viewport

        patient_dir = tmp_path / "missing_surface"
        meshes_dir = patient_dir / "meshes" / "step2_anchors"
        meshes_dir.mkdir(parents=True)

        # Create only LA.ply, leave LV.ply missing
        from la_fat.mesh_extractor import extract_interactive_meshes
        from tests.test_mesh_extractor import _make_pipeline_state

        state = _make_pipeline_state()
        extract_interactive_meshes(state, str(patient_dir))

        # Delete LV.ply to simulate missing surface
        lv_ply = meshes_dir / "LV.ply"
        if lv_ply.exists():
            lv_ply.unlink()

        result = _build_step2_viewport(str(patient_dir))
        assert hasattr(result, "_checkboxes")
        assert result._checkboxes["LV"].disabled is True
        assert result._checkboxes["LV"].value is False
        assert result._checkboxes["LA"].disabled is False
        assert result._checkboxes["LA"].value is True


# ===================================================================
# Test: Corrupt .ply file handling
# ===================================================================


class TestCorruptPLY:
    """Corrupt .ply files are handled gracefully without crashing."""

    def test_corrupt_ply_does_not_crash(self, tmp_path):
        """Corrupt PLY file is silently omitted, no crash."""
        pyvista = pytest.importorskip("pyvista", reason="PyVista not installed")

        from la_fat.interactive_dashboard import _build_step2_viewport

        patient_dir = tmp_path / "corrupt_patient"
        meshes_dir = patient_dir / "meshes" / "step2_anchors"
        meshes_dir.mkdir(parents=True)

        # Create a text file masquerading as .ply
        bad_ply = meshes_dir / "LA.ply"
        with open(bad_ply, "w", encoding="utf-8") as f:
            f.write("not a ply file\n")
        bad_ply2 = meshes_dir / "LV.ply"
        with open(bad_ply2, "w", encoding="utf-8") as f:
            f.write("also not a ply file\n")

        result = _build_step2_viewport(str(patient_dir))

        import panel as pn

        assert isinstance(result, pn.Column)
        # Should either show empty card or have disabled checkboxes
        if hasattr(result, "_checkboxes"):
            for cb in result._checkboxes.values():
                assert cb.disabled is True
        else:
            md_panes = [c for c in result if isinstance(c, pn.pane.Markdown)]
            texts = [str(md.object) for md in md_panes]
            assert any("No meshes" in t for t in texts)

    def test_corrupt_ply_disables_checkbox(self, tmp_path):
        """Corrupt PLY results in disabled checkbox with tooltip icon."""
        pyvista = pytest.importorskip("pyvista", reason="PyVista not installed")

        from la_fat.interactive_dashboard import _build_step2_viewport

        patient_dir = tmp_path / "corrupt_surface"
        meshes_dir = patient_dir / "meshes" / "step2_anchors"
        meshes_dir.mkdir(parents=True)

        # Create one valid PLY and one corrupt
        from la_fat.mesh_extractor import extract_interactive_meshes
        from tests.test_mesh_extractor import _make_pipeline_state

        state = _make_pipeline_state()
        extract_interactive_meshes(state, str(patient_dir))

        # Corrupt RA.ply
        ra_ply = meshes_dir / "RA.ply"
        with open(ra_ply, "w", encoding="utf-8") as f:
            f.write("corrupted content")

        result = _build_step2_viewport(str(patient_dir))
        assert hasattr(result, "_checkboxes")
        assert result._checkboxes["RA"].disabled is True
        assert result._checkboxes["RA"].value is False
        # Other surfaces remain available
        assert result._checkboxes["LA"].disabled is False
        assert result._checkboxes["LA"].value is True


# ===================================================================
# Test: Empty output directory
# ===================================================================


class TestEmptyOutputDir:
    """Empty output directory shows meaningful message."""

    def test_empty_output_dir_shows_message(self, tmp_path):
        """Empty output dir produces a centered 'No patients found' message."""
        from la_fat.interactive_dashboard import create_dashboard

        result = create_dashboard(str(tmp_path))

        import panel as pn

        assert isinstance(result, pn.Column)
        md_panes = [c for c in result if isinstance(c, pn.pane.Markdown)]
        texts = [str(md.object) for md in md_panes]
        assert any("No patients found" in t for t in texts)
