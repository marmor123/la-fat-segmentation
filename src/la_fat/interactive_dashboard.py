"""Interactive dashboard module for LA Fat Segmentation.

Provides a Panel-based interactive QA dashboard with sidebar, patient
list, key numbers, quality flags, and three 3D viewports for Steps 2
(Anchors), 5 (Partition), and 7 (Final LA Fat).
"""

from __future__ import annotations

import dataclasses
import json
import os
import typing as t

from la_fat.pipeline_result import load_pipeline_result


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Severity ordering for computing highest severity from a list of flags.
_SEVERITY_ORDER: dict[str, int] = {"high": 0, "medium": 1, "low": 2}


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PatientSummary:
    """Summary of a single patient discovered in the output directory.

    Attributes
    ----------
    patient_id:
        Directory name of the patient.
    processing_date:
        ISO-formatted timestamp of the most recent file modification time
        in the patient directory, or ``""`` if the directory is empty.
    severity:
        Highest flag severity from quality flags:
        ``"high"`` / ``"medium"`` / ``"low"`` / ``"none"``.
    status:
        ``"complete"`` if ``meshes/step7_final/LA_fat.ply`` exists,
        otherwise ``"partial"``.
    la_fat_volume_ml:
        LA fat volume from PipelineResultData, or ``None`` if unavailable.
    total_epicardial_volume_ml:
        Total epicardial fat volume from PipelineResultData, or ``None`` if
        unavailable.
    quality_flags:
        Parsed quality flags from PipelineResultData, or empty list.
    """

    patient_id: str
    processing_date: str
    severity: str
    status: str
    la_fat_volume_ml: float | None
    total_epicardial_volume_ml: float | None
    quality_flags: list[dict[str, t.Any]]


# ---------------------------------------------------------------------------
# Patient discovery
# ---------------------------------------------------------------------------


def discover_patients(output_dir: str) -> list[PatientSummary]:
    """Scan *output_dir* for patient subdirectories and return summaries.

    A patient is considered **complete** when it contains
    ``meshes/step7_final/LA_fat.ply``.  Otherwise it is **partial**.

    Parameters
    ----------
    output_dir:
        Root directory containing per-patient subdirectories.

    Returns
    -------
    list[PatientSummary]
        One entry per discovered patient directory.  Empty list when
        *output_dir* is empty or does not exist.
    """
    if not os.path.isdir(output_dir):
        return []

    patients: list[PatientSummary] = []

    for entry in sorted(os.listdir(output_dir)):
        patient_dir = os.path.join(output_dir, entry)
        if not os.path.isdir(patient_dir):
            continue

        patient_id = entry

        # Determine status: complete or partial
        final_mesh = os.path.join(
            patient_dir, "meshes", "step7_final", "LA_fat.ply"
        )
        status = "complete" if os.path.isfile(final_mesh) else "partial"

        # Compute processing_date from most recent mtime
        processing_date = _get_latest_mtime(patient_dir)

        # Load PipelineResultData for typed access to quality flags, volumes, etc.
        # If pipeline_result.json is not available, fall back gracefully with
        # None/empty defaults so old output directories still appear in the list.
        try:
            result_data = load_pipeline_result(patient_dir)
            quality_flags = result_data.quality_flags
            la_vol = result_data.la_fat_volume_ml
            total_epi_vol = result_data.total_fat_volume_ml
        except (FileNotFoundError, ValueError):
            quality_flags = []
            la_vol = None
            total_epi_vol = None

        # Compute highest severity from quality flags
        severity = "none"
        for flag in quality_flags:
            sev = flag.get("severity", "none")
            if sev in _SEVERITY_ORDER and (
                severity == "none"
                or _SEVERITY_ORDER.get(sev, 99) < _SEVERITY_ORDER.get(severity, 99)
            ):
                severity = sev

        patients.append(
            PatientSummary(
                patient_id=patient_id,
                processing_date=processing_date,
                severity=severity,
                status=status,
                la_fat_volume_ml=la_vol if la_vol is not None else None,
                total_epicardial_volume_ml=total_epi_vol if total_epi_vol is not None else None,
                quality_flags=quality_flags,
            )
        )

    return patients


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_latest_mtime(directory: str) -> str:
    """Return the most recent modification time in *directory* as ISO string.

    Returns ``""`` if the directory contains no files or cannot be read.
    """
    latest: float = 0.0
    try:
        for dirpath, _dirnames, filenames in os.walk(directory):
            for fname in filenames:
                fpath = os.path.join(dirpath, fname)
                try:
                    mtime = os.path.getmtime(fpath)
                    if mtime > latest:
                        latest = mtime
                except OSError:
                    continue
    except OSError:
        return ""

    if latest == 0.0:
        return ""

    from datetime import datetime, timezone

    return datetime.fromtimestamp(latest, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Dashboard error helpers
# ---------------------------------------------------------------------------


def _get_high_flags(patient_dir: str) -> list[dict[str, t.Any]]:
    """Read ``quality_flags.json`` and return only high-severity flags.

    Parameters
    ----------
    patient_dir:
        Path to a patient's output directory.

    Returns
    -------
    list[dict[str, Any]]
        List of high-severity quality flags, or empty list if the file is
        missing or contains no high-severity flags.
    """
    flags_path = os.path.join(patient_dir, "quality_flags.json")
    if not os.path.isfile(flags_path):
        return []
    try:
        with open(flags_path, encoding="utf-8") as f:
            flags: list[dict[str, t.Any]] = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    return [f for f in flags if f.get("severity") == "high"]


def _check_step_available(patient_dir: str, step_name: str) -> bool:
    """Check if a pipeline step's meshes directory exists.

    Parameters
    ----------
    patient_dir:
        Path to a patient's output directory.
    step_name:
        Subdirectory under ``meshes/`` (e.g. ``"step2_anchors"``).

    Returns
    -------
    bool
        ``True`` if the meshes directory for *step_name* exists and is
        a directory.
    """
    meshes_dir = os.path.join(patient_dir, "meshes", step_name)
    return os.path.isdir(meshes_dir)


# ===================================================================
# Dashboard (Panel-based)
# ===================================================================

try:
    import panel as pn
except ImportError:  # pragma: no cover
    pn = None  # type: ignore[assignment]


_SEVERITY_COLORS: dict[str, str] = {
    "high": "#e74c3c",
    "medium": "#f39c12",
    "low": "#2ecc71",
    "none": "#888888",
}


# ---------------------------------------------------------------------------
# Shared 3D viewport builder
# ---------------------------------------------------------------------------


def _build_step_viewport(
    patient_dir: str,
    step_name: str,
    step_display_name: str,
    surface_specs: dict[str, dict[str, t.Any]],
    presets: list[dict[str, t.Any]] | None = None,
    on_pane_created: t.Callable[[t.Any], None] | None = None,
) -> pn.Column:  # type: ignore[return-value]
    """Build a 3D viewport for a pipeline step.

    Loads PLY meshes from ``patient_dir/meshes/<step_name>/`` and returns a
    Panel layout containing a PyVista 3D viewer, toggle controls for each
    surface, and preset buttons.

    Parameters
    ----------
    patient_dir:
        Path to a patient's output directory.
    step_name:
        Subdirectory under ``meshes/`` (e.g. ``"step2_anchors"``).
    step_display_name:
        Heading shown in the card header.
    surface_specs:
        Mapping of surface name to ``{"color", "opacity", "label",
        "show_edges" (optional), "style" (optional)}``.
    presets:
        List of preset button specs.  Each spec has ``"name"``, ``"label"``,
        ``"button_type"``, and either ``"hide"`` (list of names to hide) or
        ``"show_only"`` (list of names to show).  ``"Show All"`` and
        ``"Hide All"`` are built-in.
    on_pane_created:
        Optional callback invoked with the VTK pane each time it is created
        (used for camera sync registration).

    Returns
    -------
    pn.Column
        A Panel column with the PyVista viewport, controls, and presets.
        Exposes ``._checkboxes`` and ``._preset_handlers`` for testing.
    """
    import glob

    meshes_dir = os.path.join(patient_dir, "meshes", step_name)

    # Early exit when the meshes directory is empty or missing.
    if not os.path.isdir(meshes_dir) or not glob.glob(
        os.path.join(meshes_dir, "*.ply")
    ):
        return _empty_card(step_display_name, "No meshes available for this patient")

    import pyvista as pv

    # Load available meshes — skip files that are missing or fail to load.
    loaded_meshes: dict[str, pv.PolyData] = {}
    for name in surface_specs:
        ply_path = os.path.join(meshes_dir, f"{name}.ply")
        if os.path.isfile(ply_path):
            try:
                mesh = pv.read(ply_path)
                if mesh.n_points > 0:
                    loaded_meshes[name] = mesh
            except Exception:
                pass

    if not loaded_meshes:
        return _empty_card(step_display_name, "No meshes available for this patient")

    # ---- Checkboxes -------------------------------------------------------
    checkboxes: dict[str, pn.widgets.Checkbox] = {}
    for name in surface_specs:
        available = name in loaded_meshes
        cb = pn.widgets.Checkbox(  # type: ignore[call-overload]
            name=surface_specs[name]["label"],
            value=available,
            disabled=not available,
        )
        checkboxes[name] = cb

    # ---- Preset buttons ---------------------------------------------------
    preset_buttons: list[pn.widgets.Button] = []
    preset_handlers: dict[str, t.Callable[[t.Any], None]] = {}

    if presets:
        for preset in presets:
            btn = pn.widgets.Button(  # type: ignore[call-overload]
                name=preset["label"],
                button_type=preset.get("button_type", "default"),
                width=100,
            )

            preset_name = preset["name"]
            hide_list = preset.get("hide")
            show_only_list = preset.get("show_only")

            if hide_list is not None:
                hidden = set(hide_list)

                def _handler(
                    _event: t.Any = None, _hidden: set[str] = hidden
                ) -> None:
                    for cb_name, cb in checkboxes.items():
                        if not cb.disabled:
                            cb.value = cb_name not in _hidden

            elif show_only_list is not None:
                shown = set(show_only_list)

                def _handler(
                    _event: t.Any = None, _shown: set[str] = shown
                ) -> None:
                    for cb_name, cb in checkboxes.items():
                        if not cb.disabled:
                            cb.value = cb_name in _shown

            elif preset_name == "Show All":

                def _handler(_event: t.Any = None) -> None:
                    for cb in checkboxes.values():
                        if not cb.disabled:
                            cb.value = True

            elif preset_name == "Hide All":

                def _handler(_event: t.Any = None) -> None:
                    for cb in checkboxes.values():
                        if not cb.disabled:
                            cb.value = False

            else:
                continue  # unknown preset — skip

            btn.on_click(_handler)
            preset_buttons.append(btn)
            preset_handlers[preset_name] = _handler

    # ---- Reactive VTK pane ------------------------------------------------
    @pn.depends(*(cb.param.value for cb in checkboxes.values()))  # type: ignore[arg-type]
    def _get_vtk_pane(*visibilities: bool) -> pn.pane.VTK:  # type: ignore[valid-type]
        """Rebuild the PyVista plotter with only the checked surfaces."""
        plotter = pv.Plotter(notebook=False, off_screen=True)
        for (name, _cb), visible in zip(checkboxes.items(), visibilities):
            if visible and name in loaded_meshes:
                config = surface_specs[name]
                mesh = loaded_meshes[name]
                plotter.add_mesh(
                    mesh,
                    color=config["color"],
                    opacity=config["opacity"],
                    show_edges=config.get("show_edges", False),
                    style=config.get("style", "surface"),
                    name=name,
                )
        plotter.camera_position = "xy"
        pane = pn.pane.VTK(
            plotter.ren_win,
            height=500,
            sizing_mode="stretch_width",
        )
        # Keep a reference to prevent garbage collection of the plotter.
        pane._plotter_ref = plotter  # type: ignore[attr-defined]
        if on_pane_created is not None:
            on_pane_created(pane)
        return pane

    vtk_pane = _get_vtk_pane

    # ---- Checkbox rows with colored dots ----------------------------------
    checkbox_rows: list[pn.Row] = []
    for name, cb in checkboxes.items():
        r, g, b = surface_specs[name]["color"]
        hex_color = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        dot = pn.pane.HTML(
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'border-radius:50%;background:{hex_color};'
            f'margin-right:6px;"></span>',
            width=20,
            sizing_mode="fixed",
        )
        row_items: list[t.Any] = [dot, cb]
        if cb.disabled:
            row_items.append(
                pn.widgets.TooltipIcon(value="mesh not available", width=16)  # type: ignore[call-overload]
            )
        checkbox_rows.append(pn.Row(*row_items, sizing_mode="stretch_width"))

    controls = pn.Column(
        pn.Row(*preset_buttons, sizing_mode="stretch_width"),
        *checkbox_rows,
        styles={"padding": "4px 0"},
        sizing_mode="stretch_width",
    )

    column = pn.Column(
        pn.pane.Markdown(
            f"### {step_display_name}",
            styles={"margin": "0 0 8px 0"},
        ),
        vtk_pane,
        controls,
        styles={
            "background": "#2d2d44",
            "border-radius": "8px",
            "border": "1px solid #3a3a55",
            "padding": "12px",
        },
        sizing_mode="stretch_width",
    )

    # Expose internals for testing.
    column._checkboxes = checkboxes  # type: ignore[attr-defined]
    column._preset_handlers = preset_handlers  # type: ignore[attr-defined]
    column._loaded_meshes = loaded_meshes  # type: ignore[attr-defined]

    return column


# ---------------------------------------------------------------------------
# Step 2 — Anchors + Pericardium 3D viewport
# ---------------------------------------------------------------------------


def _build_step2_viewport(patient_dir: str) -> pn.Column:  # type: ignore[return-value]
    """Build the Step 2 (Anchors + Pericardium) 3D viewport.

    Loads PLY meshes from ``patient_dir/meshes/step2_anchors/`` and returns a
    Panel layout with toggle controls for each cardiac anchor surface.

    Parameters
    ----------
    patient_dir:
        Path to a patient's output directory.

    Returns
    -------
    pn.Column
        A Panel column with the PyVista viewport, checkboxes, and presets.
    """
    if not _check_step_available(patient_dir, "step2_anchors"):
        return _empty_card(
            "Anchors + Pericardium",
            "Step 2 — Anchors + Pericardium not available: meshes directory not found",
        )

    from la_fat.qa_dashboard import ANCHOR_COLORS, PERICARDIUM_COLOR

    surface_specs: dict[str, dict[str, t.Any]] = {}
    for name, color in ANCHOR_COLORS.items():
        surface_specs[name] = {
            "color": color,
            "opacity": 0.5,
            "label": name,
        }
    surface_specs["Pericardium"] = {
        "color": PERICARDIUM_COLOR,
        "opacity": 0.1,
        "label": "Pericardium",
        "show_edges": True,
        "style": "wireframe",
    }

    presets: list[dict[str, t.Any]] = [
        {"name": "Show All", "label": "Show All", "button_type": "primary"},
        {"name": "Hide All", "label": "Hide All", "button_type": "warning"},
        {
            "name": "Anchors Only",
            "label": "Anchors Only",
            "button_type": "default",
            "hide": ["Pericardium"],
        },
        {
            "name": "Pericardium Only",
            "label": "Pericardium Only",
            "button_type": "default",
            "show_only": ["Pericardium"],
        },
    ]

    return _build_step_viewport(
        patient_dir,
        "step2_anchors",
        "Anchors + Pericardium",
        surface_specs,
        presets=presets,
    )


# ---------------------------------------------------------------------------
# Step 5 — Fat Partition 3D viewport
# ---------------------------------------------------------------------------


def _build_step5_viewport(patient_dir: str) -> pn.Column:  # type: ignore[return-value]
    """Build the Step 5 (Fat Partition) 3D viewport.

    Loads PLY meshes from ``patient_dir/meshes/step5_partition/`` and returns
    a Panel layout with toggle controls for each anchor's fat surface.

    Parameters
    ----------
    patient_dir:
        Path to a patient's output directory.

    Returns
    -------
    pn.Column
        A Panel column with the PyVista viewport, checkboxes, and presets.
    """
    if not _check_step_available(patient_dir, "step5_partition"):
        return _empty_card(
            "Fat Partition",
            "Step 5 — Fat Partition not available: meshes directory not found",
        )

    from la_fat.qa_dashboard import ANCHOR_COLORS, PERICARDIUM_COLOR

    surface_specs: dict[str, dict[str, t.Any]] = {}
    for name, color in ANCHOR_COLORS.items():
        surface_specs[name] = {
            "color": color,
            "opacity": 0.7,
            "label": f"{name} Fat",
        }
    surface_specs["Pericardium"] = {
        "color": PERICARDIUM_COLOR,
        "opacity": 0.1,
        "label": "Pericardium",
        "show_edges": True,
        "style": "wireframe",
    }

    presets: list[dict[str, t.Any]] = [
        {"name": "Show All", "label": "Show All", "button_type": "primary"},
        {"name": "Hide All", "label": "Hide All", "button_type": "warning"},
        {
            "name": "Fat Only",
            "label": "Fat Only",
            "button_type": "default",
            "hide": ["Pericardium"],
        },
        {
            "name": "Pericardium Only",
            "label": "Pericardium Only",
            "button_type": "default",
            "show_only": ["Pericardium"],
        },
    ]

    return _build_step_viewport(
        patient_dir,
        "step5_partition",
        "Fat Partition",
        surface_specs,
        presets=presets,
    )


# ---------------------------------------------------------------------------
# Step 7 — Final LA Fat 3D viewport (refactored onto shared helper)
# ---------------------------------------------------------------------------


def _build_step7_viewport(patient_dir: str) -> pn.Column:  # type: ignore[return-value]
    """Build the Step 7 (Final LA Fat) 3D viewport with mask toggles.

    Loads PLY meshes from ``patient_dir/meshes/step7_final/`` and returns a
    Panel layout containing a PyVista 3D viewer and toggle controls for each
    surface.

    Delegates to :func:`_build_step_viewport` with Step 7 surface specs.

    Parameters
    ----------
    patient_dir:
        Path to a patient's output directory.

    Returns
    -------
    pn.Column
        A Panel column with the PyVista viewport, checkboxes, and
        Show All / Hide All buttons.
    """
    if not _check_step_available(patient_dir, "step7_final"):
        return _empty_card(
            "Final LA Fat",
            "Step 7 — Final LA Fat not available: meshes directory not found",
        )

    from la_fat.qa_dashboard import ANCHOR_COLORS, LA_FAT_COLOR_3D, PERICARDIUM_COLOR

    surface_specs: dict[str, dict[str, t.Any]] = {
        "LA_chamber": {
            "color": ANCHOR_COLORS["LA"],
            "opacity": 0.5,
            "label": "LA Chamber",
        },
        "Pericardium": {
            "color": PERICARDIUM_COLOR,
            "opacity": 0.1,
            "label": "Pericardium",
        },
        "LA_fat": {
            "color": LA_FAT_COLOR_3D,
            "opacity": 0.85,
            "label": "LA Fat",
        },
    }

    presets: list[dict[str, t.Any]] = [
        {"name": "Show All", "label": "Show All", "button_type": "primary"},
        {"name": "Hide All", "label": "Hide All", "button_type": "warning"},
    ]

    result = _build_step_viewport(
        patient_dir,
        "step7_final",
        "Final LA Fat",
        surface_specs,
        presets=presets,
    )

    # Check for zero LA fat (valid mesh with fewer than 4 vertices).
    if (
        hasattr(result, "_loaded_meshes")
        and "LA_fat" in result._loaded_meshes
        and result._loaded_meshes["LA_fat"].n_points < 4
    ):
        result.append(
            pn.pane.Markdown(
                "**No LA Fat detected.**",
                styles={"color": "#f39c12", "font-weight": "bold", "margin-top": "8px"},
            )
        )

    return result


# ---------------------------------------------------------------------------
# Camera sync implementation
# ---------------------------------------------------------------------------

# Shared registry of VTK panes for all three viewports, updated each time a
# pane is (re-)created.  Each entry is a single-element list so that
# references can be updated from closures.
_VTK_PANE_REGISTRY: dict[str, list[t.Any]] = {
    "step2": [None],
    "step5": [None],
    "step7": [None],
}

# Guard flag to prevent recursive syncs when the periodic callback
# programmatically sets a pane's camera.
_camera_syncing: bool = False


def _get_default_camera() -> dict[str, t.Any]:
    """Return a default camera dict (isometric view from xy quadrant)."""
    return {
        "position": (300.0, 300.0, 200.0),
        "focalPoint": (0.0, 0.0, 0.0),
        "viewUp": (0.0, 0.0, 1.0),
    }


def _reset_all_cameras() -> None:
    """Reset all registered VTK panes to the default camera position."""
    default_cam = _get_default_camera()
    for ref_list in _VTK_PANE_REGISTRY.values():
        pane = ref_list[0]
        if pane is not None:
            try:
                pane.camera = default_cam
            except Exception:
                pass


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS: str = """
:root {
  --bg-dark: #1a1a2e;
  --bg-card: #2d2d44;
  --text-light: #e0e0e0;
  --text-muted: #999999;
  --accent: #4a90d9;
  --border: #3a3a55;
}
body, .bk-root {
  background-color: var(--bg-dark) !important;
  color: var(--text-light) !important;
}
.patient-row {
  display: flex; align-items: center; padding: 8px 12px;
  cursor: pointer; border-radius: 6px; transition: background 0.2s;
}
.patient-row:hover { background: #3a3a55; }
.patient-row.selected { background: var(--accent); color: #fff; }
.severity-dot {
  width: 12px; height: 12px; border-radius: 50%; display: inline-block;
  margin-right: 10px; flex-shrink: 0;
}
.patient-info { flex: 1; }
.patient-id { font-weight: 600; font-size: 14px; }
.patient-date { font-size: 11px; color: var(--text-muted); }
.partial-badge {
  background: #e74c3c; color: #fff; font-size: 10px; font-weight: bold;
  padding: 2px 6px; border-radius: 8px; margin-left: 8px;
}
.viewport-card {
  background: var(--bg-card); border-radius: 8px; padding: 20px;
  border: 1px solid var(--border); min-height: 200px;
  display: flex; align-items: center; justify-content: center;
  color: var(--text-muted); font-size: 18px;
}
.flag-row {
  padding: 6px 0; border-bottom: 1px solid var(--border);
  font-size: 13px;
}
.flag-high { border-left: 3px solid #e74c3c; padding-left: 8px; }
.flag-medium { border-left: 3px solid #f39c12; padding-left: 8px; }
.flag-low { border-left: 3px solid #2ecc71; padding-left: 8px; }
.flag-concern { font-weight: 600; }
.flag-detail { color: var(--text-muted); font-size: 11px; }
"""


# ---------------------------------------------------------------------------
# Dashboard top-level constructor
# ---------------------------------------------------------------------------


def create_dashboard(output_dir: str) -> pn.Column:  # type: ignore[return-value]
    """Create the interactive Panel dashboard for LA Fat analysis.

    Builds a dark-themed dashboard with:
    - Collapsible sidebar containing patient list, key numbers, quality
      flags, and camera sync controls
    - Main area with three 3D viewports stacked vertically: Anchors
      (Step 2), Fat Partition (Step 5), and Final LA Fat (Step 7)

    Parameters
    ----------
    output_dir:
        Root directory containing per-patient subdirectories.

    Returns
    -------
    pn.Column
        A Panel Column (or template) ready for ``.servable()`` or
        ``.show()``.
    """
    if pn is None:
        raise ImportError("Panel is required. Install with: pip install panel")

    pn.extension("vtk", sizing_mode="stretch_width")

    patients = discover_patients(output_dir)

    # ── Empty output directory ─────────────────────────────────────────────
    if not patients:
        return pn.Column(
            pn.pane.Markdown(
                "# No patients found.",
                styles={
                    "text-align": "center",
                    "margin-top": "40px",
                    "font-size": "24px",
                    "color": "#e0e0e0",
                },
            ),
            pn.pane.Markdown(
                "Run the pipeline first to generate patient data.",
                styles={
                    "text-align": "center",
                    "color": "#999",
                    "font-size": "16px",
                },
            ),
            styles={
                "background": "#1a1a2e",
                "height": "100vh",
                "display": "flex",
                "align-items": "center",
                "justify-content": "center",
            },
        )

    # ------------------------------------------------------------------
    # Reactive state
    # ------------------------------------------------------------------
    selected_idx: pn.rx = pn.rx(-1)  # type: ignore[valid-type]
    sidebar_collapsed: pn.rx = pn.rx(False)  # type: ignore[valid-type]
    camera_sync_enabled: pn.rx = pn.rx(False)  # type: ignore[valid-type]
    _banner_visible: pn.rx = pn.rx(True)  # type: ignore[valid-type]

    # ------------------------------------------------------------------
    # Camera sync machinery
    # ------------------------------------------------------------------
    _sync_guard: dict[str, bool] = {"active": False}

    def _make_on_pane_created(step_key: str) -> t.Callable[[t.Any], None]:
        """Return a callback that registers a VTK pane in the registry."""

        def _on_pane_created(pane: t.Any) -> None:
            _VTK_PANE_REGISTRY[step_key][0] = pane

        return _on_pane_created

    def _camera_sync_callback() -> None:
        """Periodic callback: sync all viewport cameras when enabled."""
        if _sync_guard["active"]:
            return
        if not getattr(camera_sync_enabled, "rx", camera_sync_enabled).value:  # type: ignore[union-attr]
            return

        # Collect non-None panes
        panes: dict[str, t.Any] = {}
        for key, ref_list in _VTK_PANE_REGISTRY.items():
            if ref_list[0] is not None:
                panes[key] = ref_list[0]

        if len(panes) < 2:
            return

        # Use the first available pane as reference
        ref_key = next(iter(panes))
        ref_pane = panes[ref_key]
        try:
            ref_camera = getattr(ref_pane, "camera", None)
        except Exception:
            return
        if ref_camera is None:
            return

        # Apply reference camera to all other panes
        _sync_guard["active"] = True
        try:
            for other_key, pane in panes.items():
                if other_key != ref_key:
                    try:
                        pane.camera = ref_camera
                    except Exception:
                        pass
        finally:
            _sync_guard["active"] = False

    # Start the periodic camera sync callback (fires every 500 ms).
    _sync_cb_handle = pn.state.add_periodic_callback(  # type: ignore[attr-defined]
        _camera_sync_callback, period=500
    )

    # Collapse toggle
    def _toggle_sidebar(_event: t.Any = None) -> None:
        sidebar_collapsed.rx.value = not sidebar_collapsed.rx.value  # type: ignore[attr-defined]

    collapse_btn = pn.widgets.Button(
        name="☰",
        button_type="light",
        width=40,
        styles={"background": "transparent", "color": "#e0e0e0", "border": "none", "font-size": "18px"},
    )
    collapse_btn.on_click(_toggle_sidebar)

    # Camera sync toggle
    sync_toggle = pn.widgets.Toggle(  # type: ignore[call-overload]
        name="Sync Cameras",
        value=False,
        button_type="primary",
        width=150,
    )

    def _on_sync_toggle(event: t.Any) -> None:
        camera_sync_enabled.rx.value = event.new  # type: ignore[attr-defined]

    sync_toggle.param.watch(_on_sync_toggle, "value")

    # Reset all cameras button
    reset_cam_btn = pn.widgets.Button(  # type: ignore[call-overload]
        name="Reset All Cameras",
        button_type="default",
        width=150,
    )
    reset_cam_btn.on_click(lambda _event: _reset_all_cameras())

    # Patient list
    def _select_patient(idx: int) -> None:
        selected_idx.rx.value = idx  # type: ignore[attr-defined]
        _banner_visible.rx.value = True  # type: ignore[attr-defined]

    def _build_patient_list() -> pn.Column:
        """Build the clickable patient list with severity dots."""
        rows: list[pn.Row] = []

        for i, pat in enumerate(patients):
            color = _SEVERITY_COLORS.get(pat.severity, _SEVERITY_COLORS["none"])
            date_str = pat.processing_date[:10] if pat.processing_date else ""

            dot = pn.pane.HTML(
                f'<span class="severity-dot" style="background:{color};"></span>',
                width=25,
                sizing_mode="fixed",
            )

            badge = ""
            if pat.status == "partial":
                badge = '<span class="partial-badge">partial</span>'

            info = pn.pane.HTML(
                f'<div class="patient-info">'
                f'<div class="patient-id">{pat.patient_id}</div>'
                f'<div class="patient-date">{date_str}</div>'
                f"</div>"
                f"{badge}",
                sizing_mode="stretch_width",
            )

            btn = pn.widgets.Button(
                button_type="light",
                css_classes=["patient-row"],
                sizing_mode="stretch_width",
                styles={"background": "transparent", "padding": "0", "border": "none"},
            )
            btn.on_click(lambda _event, idx=i: _select_patient(idx))

            row = pn.Row(
                dot,
                pn.Column(info, btn, sizing_mode="stretch_width"),
                sizing_mode="stretch_width",
                styles={"margin": "2px 0"},
            )
            rows.append(row)

        return pn.Column(*rows, sizing_mode="stretch_width", scroll=False)

    # Key Numbers Card
    def _key_numbers_card_fn(idx: int) -> pn.Column:
        """Build the key numbers card for the selected patient."""
        if idx < 0 or idx >= len(patients):
            return _empty_card("Key Numbers", "Select a patient to view key numbers.")

        pat = patients[idx]
        la_vol_str = f"{pat.la_fat_volume_ml:.2f}" if pat.la_fat_volume_ml is not None else "N/A"
        epi_vol_str = (
            f"{pat.total_epicardial_volume_ml:.2f}"
            if pat.total_epicardial_volume_ml is not None
            else "N/A"
        )

        # Count flags by severity
        high = sum(1 for f in pat.quality_flags if f.get("severity") == "high")
        med = sum(1 for f in pat.quality_flags if f.get("severity") == "medium")
        low = sum(1 for f in pat.quality_flags if f.get("severity") == "low")

        html = f"""
        <div style="padding:8px 0;">
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
            <div style="font-size:12px; color:#999;">Patient ID</div>
            <div style="font-size:14px; font-weight:600;">{pat.patient_id}</div>
            <div style="font-size:12px; color:#999;">LA Fat Vol.</div>
            <div style="font-size:14px;">{la_vol_str} ml</div>
            <div style="font-size:12px; color:#999;">Total Epicardial Fat</div>
            <div style="font-size:14px;">{epi_vol_str} ml</div>
            <div style="font-size:12px; color:#999;">Quality Flags</div>
            <div style="font-size:14px;">
              <span style="color:#e74c3c;">{high} high</span>
              &nbsp;
              <span style="color:#f39c12;">{med} med</span>
              &nbsp;
              <span style="color:#2ecc71;">{low} low</span>
            </div>
          </div>
        </div>
        """
        return pn.Column(
            pn.pane.Markdown("### Key Numbers", styles={"margin": "0 0 4px 0"}),
            pn.pane.HTML(html, sizing_mode="stretch_width"),
            styles={
                "background": "#2d2d44",
                "border-radius": "8px",
                "padding": "12px",
                "border": "1px solid #3a3a55",
            },
        )

    key_numbers_card = pn.bind(_key_numbers_card_fn, selected_idx)

    def _quality_flags_panel_fn(idx: int) -> pn.Column:
        """Build the quality flags panel for the selected patient."""
        if idx < 0 or idx >= len(patients):
            return _empty_card("Quality Flags", "Select a patient to view flags.")

        pat = patients[idx]

        if not pat.quality_flags:
            return _build_card(
                "Quality Flags",
                [
                    pn.pane.Markdown(
                        "_No flags_",
                        styles={"color": "#999", "font-style": "italic"},
                    )
                ],
            )

        # Sort flags: high -> medium -> low
        severity_order = {"high": 0, "medium": 1, "low": 2}
        sorted_flags = sorted(
            pat.quality_flags,
            key=lambda f: severity_order.get(f.get("severity", "low"), 99),
        )

        items: list[pn.Column] = []
        for flag in sorted_flags:
            sev = flag.get("severity", "low")
            concern = flag.get("concern", "")
            detail = flag.get("detail", "")
            threshold = flag.get("threshold_value")
            actual = flag.get("actual_value")

            detail_lines = [f"<span class='flag-detail'>{detail}</span>"]
            if threshold is not None:
                detail_lines.append(
                    f"<span class='flag-detail'>Threshold: {threshold}</span>"
                )
            if actual is not None:
                detail_lines.append(
                    f"<span class='flag-detail'>Actual: {actual}</span>"
                )

            items.append(
                pn.Column(
                    pn.pane.HTML(
                        f"<div class='flag-row flag-{sev}'>"
                        f"<div class='flag-concern'>{concern}</div>"
                        + "".join(detail_lines)
                        + "</div>",
                        sizing_mode="stretch_width",
                    ),
                )
            )

        return _build_card("Quality Flags", items)

    quality_flags_panel = pn.bind(_quality_flags_panel_fn, selected_idx)

    # ------------------------------------------------------------------
    # High-severity quality flag banner
    # ------------------------------------------------------------------
    def _dismiss_banner(_event: t.Any = None) -> None:
        """Dismiss the high-severity banner for the current patient."""
        _banner_visible.rx.value = False  # type: ignore[attr-defined]

    def _build_high_flags_banner_fn(
        idx: int, visible: bool
    ) -> pn.Column:
        """Build a red banner listing high-severity quality concerns."""
        if not visible or idx < 0 or idx >= len(patients):
            return pn.Column()

        pat = patients[idx]
        high_flags = [f for f in pat.quality_flags if f.get("severity") == "high"]
        if not high_flags:
            return pn.Column()

        concerns = ", ".join(f.get("concern", "") for f in high_flags)

        dismiss_btn = pn.widgets.Button(  # type: ignore[call-overload]
            name="✕",
            button_type="light",
            width=30,
            height=30,
            styles={
                "background": "transparent",
                "color": "#fff",
                "border": "none",
                "font-size": "16px",
                "cursor": "pointer",
            },
        )
        dismiss_btn.on_click(_dismiss_banner)

        return pn.Column(
            pn.Row(
                pn.pane.Markdown(
                    f"**High-Severity Quality Concerns:** {concerns}",
                    styles={
                        "color": "#fff",
                        "margin": "0",
                        "flex": "1",
                    },
                ),
                dismiss_btn,
                styles={
                    "display": "flex",
                    "align-items": "center",
                },
            ),
            styles={
                "background": "#e74c3c",
                "border-radius": "8px",
                "padding": "10px 16px",
                "margin": "0 0 12px 0",
            },
            sizing_mode="stretch_width",
        )

    high_flags_banner = pn.bind(
        _build_high_flags_banner_fn, selected_idx, _banner_visible
    )

    # ------------------------------------------------------------------
    # Main area: banner + three viewport cards stacked vertically
    # ------------------------------------------------------------------
    def _step2_viewport_fn(idx: int) -> pn.Column:
        """Build Step 2 Anchors viewport for the selected patient."""
        if idx < 0 or idx >= len(patients):
            return _empty_card(
                "Anchors + Pericardium", "Select a patient to view the 3D model."
            )
        patient_dir = os.path.join(output_dir, patients[idx].patient_id)
        return _build_step2_viewport(patient_dir)

    def _step5_viewport_fn(idx: int) -> pn.Column:
        """Build Step 5 Fat Partition viewport for the selected patient."""
        if idx < 0 or idx >= len(patients):
            return _empty_card(
                "Fat Partition", "Select a patient to view the 3D model."
            )
        patient_dir = os.path.join(output_dir, patients[idx].patient_id)
        return _build_step5_viewport(patient_dir)

    def _step7_viewport_fn(idx: int) -> pn.Column:
        """Build Step 7 Final LA Fat viewport for the selected patient."""
        if idx < 0 or idx >= len(patients):
            return _empty_card(
                "Final LA Fat", "Select a patient to view the 3D model."
            )
        patient_dir = os.path.join(output_dir, patients[idx].patient_id)
        return _build_step7_viewport(patient_dir)

    step2_viewport = pn.bind(_step2_viewport_fn, selected_idx)
    step5_viewport = pn.bind(_step5_viewport_fn, selected_idx)
    step7_viewport = pn.bind(_step7_viewport_fn, selected_idx)

    viewport_cards = pn.Column(
        high_flags_banner,
        step2_viewport,
        step5_viewport,
        step7_viewport,
        sizing_mode="stretch_width",
        styles={"padding": "16px"},
    )
    # ------------------------------------------------------------------
    # Sidebar assembly with collapse
    # ------------------------------------------------------------------
    def _sidebar_content_fn(collapsed: bool) -> pn.Column:
        if collapsed:
            return pn.Column(
                collapse_btn,
                styles={"background": "#1a1a2e", "padding": "8px"},
                width=40,
            )

        return pn.Column(
            # Header row
            pn.Row(
                collapse_btn,
                pn.pane.Markdown(
                    "### LA Fat",
                    styles={"color": "#e0e0e0", "margin": "8px 0"},
                ),
                sizing_mode="stretch_width",
            ),
            # Camera sync controls
            pn.pane.Markdown(
                "#### Camera",
                styles={"margin": "8px 0 4px 0"},
            ),
            pn.Column(
                sync_toggle,
                reset_cam_btn,
                styles={"padding": "4px 0"},
                sizing_mode="stretch_width",
            ),
            pn.layout.Divider(styles={"background": "#3a3a55", "margin": "8px 0"}),
            # Patient list
            pn.pane.Markdown("#### Patients", styles={"margin": "8px 0 4px 0"}),
            _build_patient_list(),
            pn.layout.Divider(styles={"background": "#3a3a55", "margin": "8px 0"}),
            # Key numbers
            key_numbers_card,
            pn.layout.Divider(styles={"background": "#3a3a55", "margin": "8px 0"}),
            # Quality flags
            quality_flags_panel,
            sizing_mode="stretch_width",
            styles={
                "background": "#1a1a2e",
                "padding": "12px",
                "overflow-y": "auto",
                "height": "100vh",
            },
            width=320,
        )

    sidebar_content = pn.bind(_sidebar_content_fn, sidebar_collapsed)

    # ------------------------------------------------------------------
    # Inject custom CSS
    # ------------------------------------------------------------------
    pn.config.raw_css.append(_CSS)

    # ------------------------------------------------------------------
    # Template
    # ------------------------------------------------------------------
    template = pn.template.FastListTemplate(
        title="LA Fat Dashboard",
        theme="dark",
        sidebar=[sidebar_content],
        main=[viewport_cards],
        header_background="#1a1a2e",
        accent_base_color="#4a90d9",
        neutral_color="#2d2d44",
        sidebar_width=320,
    )

    # Store the periodic callback handle so it can be cleaned up.
    template._sync_cb_handle = _sync_cb_handle  # type: ignore[attr-defined]

    # Return as Column-compatible Panel object
    return template  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------


def _empty_card(title: str, message: str) -> pn.Column:  # type: ignore[name-defined]
    """Return a styled card with a placeholder message."""
    return pn.Column(
        pn.pane.Markdown(f"### {title}", styles={"margin": "0 0 4px 0"}),
        pn.pane.Markdown(
            f"_{message}_",
            styles={"color": "#999", "font-style": "italic"},
        ),
        styles={
            "background": "#2d2d44",
            "border-radius": "8px",
            "padding": "12px",
            "border": "1px solid #3a3a55",
        },
    )


def _build_card(  # type: ignore[no-untyped-def]
    title: str, items: list,
) -> pn.Column:
    """Return a styled card with a title and list of content items."""
    return pn.Column(
        pn.pane.Markdown(f"### {title}", styles={"margin": "0 0 4px 0"}),
        *items,
        styles={
            "background": "#2d2d44",
            "border-radius": "8px",
            "padding": "12px",
            "border": "1px solid #3a3a55",
        },
    )
