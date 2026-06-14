"""Pipeline result data layer — typed serialization for dashboard consumption.

Defines ``PipelineResultData``, a frozen dataclass that captures all pipeline
outputs needed by the interactive and QA dashboards.  The pipeline constructs
it from in-memory results (NOT by re-parsing files), then saves it to a single
JSON file.  Dashboards load it back with full type fidelity.

This replaces the scattered CSV, JSON, and text-file re-parsing that previously
happened independently in each dashboard.
"""

from __future__ import annotations

import dataclasses
import json
import os
import typing as t


# ---------------------------------------------------------------------------
# PipelineResultData — serializable pipeline output
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PipelineResultData:
    """Serializable pipeline result for dashboard consumption.

    All fields are JSON-compatible Python primitives or collections thereof.
    This dataclass is designed for one-shot serialization/deserialization via
    :func:`save_pipeline_result` / :func:`load_pipeline_result`.

    Attributes
    ----------
    patient_id:
        Patient identifier string.
    la_fat_volume_ml:
        Volume of LA-associated epicardial fat in ml.
    total_fat_volume_ml:
        Total epicardial fat volume in ml (assigned + unassigned).
    pericardium_volume_ml:
        Volume of the pericardium mask in ml.
    unassigned_volume_ml:
        Volume of epicardial fat that could not be assigned to any anchor.
    unassigned_fat_pct:
        Percentage of total epicardial fat that is unassigned.
    anchor_volumes_ml:
        Volume in ml per Partition Anchor.
    quality_flags:
        Serialized quality flags as a list of dicts (JSON-compatible).
    fat_hu_range:
        Fat HU threshold range ``(low, high)``.
    voxel_volume_ml:
        Volume of a single voxel in ml.
    excluded_anchors:
        Anchors excluded from the partition (e.g. below minimum volume).
    islands_removed:
        Number of small connected components removed during cleanup.
    total_removed_volume_mm3:
        Total volume of removed islands in cubic mm.
    warnings:
        Non-fatal warnings accumulated during pipeline execution.
    errors:
        Fatal error messages accumulated during pipeline execution.
    """

    patient_id: str
    la_fat_volume_ml: float
    total_fat_volume_ml: float
    pericardium_volume_ml: float
    unassigned_volume_ml: float
    unassigned_fat_pct: float
    anchor_volumes_ml: dict[str, float]
    quality_flags: list[dict[str, t.Any]]
    fat_hu_range: tuple[float, float]
    voxel_volume_ml: float
    excluded_anchors: list[str]
    islands_removed: int
    total_removed_volume_mm3: float
    warnings: list[str]
    errors: list[str]


# ---------------------------------------------------------------------------
# JSON encoding / decoding helpers
# ---------------------------------------------------------------------------

_RESULT_FILENAME = "pipeline_result.json"


def _asdict_preserve_tuples(obj: t.Any) -> t.Any:
    """Recursively convert a dataclass to a dict, preserving tuples via markers.

    ``dataclasses.asdict`` recursively converts tuples to lists, losing type
    information.  This replacement emits ``{"__tuple__": True, "items": [...]}``
    for tuple fields so the JSON decoder can reconstruct them.
    """
    if dataclasses.is_dataclass(obj):
        return {
            f.name: _asdict_preserve_tuples(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    if isinstance(obj, tuple):
        return {"__tuple__": True, "items": [_asdict_preserve_tuples(item) for item in obj]}
    if isinstance(obj, list):
        return [_asdict_preserve_tuples(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _asdict_preserve_tuples(v) for k, v in obj.items()}
    return obj


def _decode_result(obj: t.Any) -> t.Any:
    """JSON object hook that deserialises ``__tuple__`` markers back to tuples."""
    if isinstance(obj, dict) and obj.get("__tuple__") is True:
        return tuple(obj["items"])
    return obj


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_pipeline_result(result: PipelineResultData, output_dir: str) -> str:
    """Serialize *result* to ``pipeline_result.json`` inside *output_dir*.

    Parameters
    ----------
    result:
        The pipeline result to persist.
    output_dir:
        Directory where ``pipeline_result.json`` will be written.

    Returns
    -------
    str
        Absolute path to the created JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert the frozen dataclass to a dict for serialization, preserving
    # tuples via markers.
    data = _asdict_preserve_tuples(result)

    json_path = os.path.join(output_dir, _RESULT_FILENAME)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return json_path


def load_pipeline_result(output_dir: str) -> PipelineResultData:
    """Deserialize ``pipeline_result.json`` from *output_dir*.

    Parameters
    ----------
    output_dir:
        Directory containing ``pipeline_result.json``.

    Returns
    -------
    PipelineResultData
        The deserialized pipeline result with full type fidelity.

    Raises
    ------
    FileNotFoundError
        If *output_dir* does not contain ``pipeline_result.json``.
    ValueError
        If the JSON file exists but cannot be parsed or is missing
        required fields.
    """
    json_path = os.path.join(output_dir, _RESULT_FILENAME)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"Pipeline result not found: {json_path}"
        )

    try:
        with open(json_path, encoding="utf-8") as f:
            raw: dict[str, t.Any] = json.load(f, object_hook=_decode_result)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse pipeline result JSON: {exc}"
        ) from exc

    # Reconstruct the frozen dataclass.
    try:
        return PipelineResultData(**raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Failed to reconstruct PipelineResultData from JSON: {exc}"
        ) from exc
