"""Typed data structures for the LA Fat pipeline.

Provides ``PipelineArtifacts``, ``SurfaceSpec``, and ``ViewportPreset``
dataclasses to replace untyped dict parameters throughout the codebase.
"""

from __future__ import annotations

import dataclasses
import typing as t

import numpy as np


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
