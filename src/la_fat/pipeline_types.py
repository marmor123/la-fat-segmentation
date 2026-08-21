"""Typed data structures for the LA Fat pipeline.

Provides ``PipelineArtifacts``, ``SurfaceSpec``, and ``ViewportPreset``
dataclasses to replace untyped dict parameters throughout the codebase.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
import typing as t

import numpy as np


class QualitySeverity(str, Enum):
    """Severity tier for quality audit flags."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_str(cls, value: str | QualitySeverity) -> QualitySeverity:
        if isinstance(value, cls):
            return value
        val = str(value).strip().lower()
        if val in ("high", "h"):
            return cls.HIGH
        if val in ("medium", "med", "m"):
            return cls.MEDIUM
        if val in ("low", "l"):
            return cls.LOW
        raise ValueError(f"Unknown QualitySeverity: {value}")


@dataclasses.dataclass(frozen=True)
class QualityFlag:
    """An auditable quality flag with severity and clinical description.

    Supports both legacy pipeline format (concern/detail) and prototype format
    (flag_id/message) with seamless property aliases.
    """

    severity: QualitySeverity | str
    concern: str = ""
    detail: str = ""
    threshold_value: float | None = None
    actual_value: float | None = None
    flag_id: str | None = None
    message: str | None = None
    metric_value: float | None = None

    def __post_init__(self) -> None:
        sev_str = str(
            self.severity.value if isinstance(self.severity, Enum) else self.severity
        ).lower()
        object.__setattr__(self, "severity", sev_str)

        eff_concern = self.concern or self.flag_id or ""
        eff_id = self.flag_id or self.concern or ""
        object.__setattr__(self, "concern", eff_concern)
        object.__setattr__(self, "flag_id", eff_id)

        eff_detail = self.detail or self.message or ""
        eff_msg = self.message or self.detail or ""
        object.__setattr__(self, "detail", eff_detail)
        object.__setattr__(self, "message", eff_msg)

        eff_val = self.actual_value if self.actual_value is not None else self.metric_value
        object.__setattr__(self, "actual_value", eff_val)
        object.__setattr__(self, "metric_value", eff_val)



@dataclasses.dataclass(frozen=True)
class PipelineArtifacts:
    """Typed container for pipeline state passed to mesh extraction.

    Replaces the untyped ``dict[str, Any]`` previously used as the
    ``pipeline_state`` parameter of :func:`~la_fat.mesh_extractor.extract_interactive_meshes`.

    Attributes
    ----------
    anchor_masks:
        Mapping of anchor names (``"LA"``, ``"LV"``, ``"RA"``, ``"RV"``,
        ``"Aorta"``, ``"Pulmonary_Artery"``) to binary mask arrays.
    pericardium_mask:
        Binary mask of the pericardium.
    partition_result:
        Result from the partition engine.  Expected to have
        ``anchor_assignments`` (int array) and ``all_fat_mask`` (bool array)
        attributes.
    cleanup_result:
        Result from the cleanup module.  Expected to have a ``cleaned_mask``
        (bool array) attribute.
    spacing:
        Voxel spacing ``(sx, sy, sz)`` in mm.
    """

    anchor_masks: dict[str, np.ndarray]
    pericardium_mask: np.ndarray
    partition_result: t.Any  # PartitionResult
    cleanup_result: t.Any  # CleanupResult
    spacing: tuple[float, float, float]


@dataclasses.dataclass(frozen=True)
class SurfaceSpec:
    """Visual specification for a 3D surface in a viewport.

    Replaces the untyped inner dict ``dict[str, Any]`` previously used as
    values in the ``surface_specs`` parameter of
    :func:`~la_fat.interactive_dashboard._build_step_viewport`.

    Attributes
    ----------
    color:
        RGB colour tuple with components in ``[0, 1]``.
    opacity:
        Opacity in ``[0, 1]``.
    label:
        Human-readable label shown in the checkbox control.
    show_edges:
        Whether to render mesh edges (default ``False``).
    style:
        PyVista mesh style — ``"surface"`` or ``"wireframe"`` (default
        ``"surface"``).
    """

    color: tuple[float, float, float]
    opacity: float
    label: str
    show_edges: bool = False
    style: str = "surface"


@dataclasses.dataclass(frozen=True)
class ViewportPreset:
    """Preset button configuration for a 3D viewport.

    Replaces the untyped dict ``dict[str, Any]`` previously used as items
    in the ``presets`` parameter of
    :func:`~la_fat.interactive_dashboard._build_step_viewport`.

    Attributes
    ----------
    name:
        Internal name used as the handler key (e.g. ``"Show All"``).
    label:
        Button label shown to the user.
    button_type:
        Panel button type (e.g. ``"primary"``, ``"warning"``, ``"default"``).
    hide:
        List of surface names to hide when the preset is activated.
        Mutually exclusive with *show_only*.
    show_only:
        List of surface names to show when the preset is activated.
        All other surfaces are hidden.  Mutually exclusive with *hide*.
    """

    name: str
    label: str
    button_type: str = "default"
    hide: list[str] | None = None
    show_only: list[str] | None = None
