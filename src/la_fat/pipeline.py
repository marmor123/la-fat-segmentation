"""Pipeline orchestrator for LA Fat Segmentation.

Wires all processing modules together into a single end-to-end
``run_fat_extraction_pipeline`` function.  See
:doc:`/docs/pipeline` for the detailed architecture.
"""

import dataclasses
import json
import logging
import os
import time
import typing as t

import numpy as np
import SimpleITK as sitk

from la_fat.anatomy import CANONICAL_ANCHORS, voxel_volume_ml
from la_fat import nifti_io
from la_fat.cleanup import CleanupResult, cleanup_la_fat_mask
from la_fat.config import PipelineConfig
from la_fat.fat_thresholder import FatThresholdResult, compute_fat_threshold
from la_fat.mesh_extractor import extract_interactive_meshes
from la_fat.pipeline_types import PipelineArtifacts
from la_fat.partition_engine import PartitionResult, partition_fat
from la_fat.pericardium_resolver import PericardiumResult, resolve_pericardium
from la_fat.preprocessor import ResampleResult, resample_to_isotropic
from la_fat.qa_dashboard import DashboardOutput, generate_dashboard
from la_fat.pipeline_result import PipelineResultData, save_pipeline_result
from la_fat.quality_flagger import QualityFlag, generate_quality_flags
from la_fat.ts_runner import resolve_ts_mask_path

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Mapping from internal structure names (used by pipeline modules) to the
#: filename stems produced by the TS Pre-Compute runner.
#: The TS runner saves::
#:     <intermediate_dir>/<patient_id>/<patient_id>_<stem>.nii.gz
_STRUCTURE_FILENAMES: dict[str, str] = {
    name: name.replace("_", " ") if "_" in name else name
    for name in CANONICAL_ANCHORS
}
_STRUCTURE_FILENAMES.update({
    "Pericardium": "Pericardium",
    "Pulmonary_Veins": "Pulmonary Veins",
})

#: Chamber keys expected by the pericardium resolver (all 6 Partition Anchors).
_CHAMBER_KEYS: list[str] = list(CANONICAL_ANCHORS)

#: Six canonical Partition Anchors expected by the partition engine.
_ANCHOR_KEYS: list[str] = list(CANONICAL_ANCHORS)

#: Name of the pre-resampled CT cache file (relative to patient dir).
_RESAMPLED_CT_FILENAME = "{patient_id}_ct_resampled.nii.gz"


# ---------------------------------------------------------------------------
# Pipeline state (mutable, for intra-pipeline coordination)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PipelineState:
    """Mutable state container for pipeline steps.

    Each step function receives this and mutates it.  Non-frozen so
    steps can freely assign attributes.
    """

    patient_id: str
    cfg: PipelineConfig | None = None
    data_dir: str = ""
    output_dir: str = ""
    intermediate_dir: str = ""
    raw_ct_dir: str = ""
    patient_output_dir: str = ""
    spacing: tuple[float, float, float] = (1.5, 1.5, 1.5)
    ct_array: np.ndarray | None = None
    ct_spacing: tuple[float, float, float] = (1.5, 1.5, 1.5)
    ct_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ct_direction: np.ndarray = dataclasses.field(default_factory=lambda: np.eye(3))
    loaded_masks: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    pericardium_result: PericardiumResult | None = None
    fat_threshold_result: FatThresholdResult | None = None
    partition_result: PartitionResult | None = None
    anchor_masks: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    cleanup_result: CleanupResult | None = None
    mesh_paths: dict[str, list[str]] | None = None
    quality_flags: list[QualityFlag] = dataclasses.field(default_factory=list)
    dashboard_output: DashboardOutput | None = None
    errors: list[str] = dataclasses.field(default_factory=list)
    warnings: list[str] = dataclasses.field(default_factory=list)
    start_time: float = 0.0
    step: int = 0
    step_total: int = 0


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PipelineResult:
    """Result of running the full fat extraction pipeline.

    Attributes
    ----------
    patient_id:
        Patient identifier string.
    success:
        ``True`` if the pipeline completed without errors.
    partition_result:
        Result from the partition engine, or ``None`` if partition failed.
    fat_threshold_result:
        Result from the fat thresholder, or ``None`` if thresholding failed.
    pericardium_result:
        Result from the pericardium resolver, or ``None`` if resolution
        failed.
    cleanup_result:
        Result from the cleanup module, or ``None`` if cleanup failed.
    quality_flags:
        List of quality flags (empty if pipeline failed early).
    dashboard_output:
        Output paths for the QA dashboard, or ``None`` if dashboard
        generation did not run.
    errors:
        Human-readable error messages for any steps that failed.
    warnings:
        Non-fatal warnings (e.g. fallback triggered, anchors excluded).
    total_runtime_seconds:
        Wall-clock duration of the full pipeline in seconds.
    mesh_paths:
        Mapping from step name (e.g. ``"step2_anchors"``) to list of
        ``.ply`` file paths, or ``None`` if mesh extraction failed or
        was skipped.
    """

    patient_id: str
    success: bool
    partition_result: PartitionResult | None
    fat_threshold_result: FatThresholdResult | None
    pericardium_result: PericardiumResult | None
    cleanup_result: CleanupResult | None
    quality_flags: list[QualityFlag]
    dashboard_output: DashboardOutput | None
    errors: list[str]
    warnings: list[str]
    total_runtime_seconds: float
    mesh_paths: dict[str, list[str]] | None = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_fat_extraction_pipeline(
    patient_id: str,
    config: PipelineConfig | None = None,
    config_path: str | None = None,
) -> PipelineResult:
    """Run the complete fat extraction pipeline for a single patient.

    The pipeline follows these steps:

    1. Load or create configuration.
    2. Construct file-system paths.
    3. Resample the raw CT to isotropic spacing (or load cached copy).
    4. Load TotalSegmentator masks from disk.
    5. Resolve pericardium (direct or fallback).
    6. Compute fat HU threshold (Gaussian fit or fallback).
    7. Partition epicardial fat to nearest anchor surface.
    8. Clean the LA fat mask (island removal).
    9. Extract meshes for interactive visualization.
    10. Generate quality flags.
    11. Generate QA dashboard.
    12. Save LA fat mask and quality flags to output directory.

    Parameters
    ----------
    patient_id:
        Patient identifier.  Used to locate input files and name outputs.
    config:
        Optional pre-built ``PipelineConfig``.  If not provided, a default
        config is used (unless *config_path* is given).
    config_path:
        Path to a YAML configuration file to load.  Ignored if *config* is
        provided directly.

    Returns
    -------
    PipelineResult
        Always returned — the pipeline never raises.
    """
    start_time = time.perf_counter()

    # ── Step 0: Configuration ───────────────────────────────────────────────
    if config is not None:
        cfg = config
    elif config_path is not None:
        cfg = PipelineConfig.from_yaml(config_path)
        logger.info(
            "[%s] Step 0/12: Loaded config from %s",
            _ts(), config_path,
        )
    else:
        cfg = PipelineConfig()
        logger.info(
            "[%s] Step 0/12: Using default configuration",
            _ts(),
        )

    data_dir: str = cfg.data_dir
    output_dir: str = cfg.output_dir
    intermediate_dir = os.path.join(data_dir, cfg.intermediate_subdir, patient_id)
    raw_ct_dir = os.path.join(data_dir, cfg.raw_subdir)
    patient_output_dir = os.path.join(output_dir, patient_id)
    spacing: tuple[float, float, float] = (
        cfg.spacing_mm,
        cfg.spacing_mm,
        cfg.spacing_mm,
    )

    # ── Build state ─────────────────────────────────────────────────────────
    state = PipelineState(
        patient_id=patient_id,
        cfg=cfg,
        data_dir=data_dir,
        output_dir=output_dir,
        intermediate_dir=intermediate_dir,
        raw_ct_dir=raw_ct_dir,
        patient_output_dir=patient_output_dir,
        spacing=spacing,
        start_time=start_time,
    )

    # ── Pipeline steps ──────────────────────────────────────────────────────
    steps: list[tuple[str, t.Callable, str, bool]] = [
        ("Load/resample CT", _step_load_ct, "CT loading", True),
        ("Load TS masks", _step_load_masks, "TS mask loading", True),
        ("Resolve pericardium", _step_resolve_pericardium,
         "Pericardium resolution", True),
        ("Compute fat threshold", _step_compute_fat_threshold,
         "Fat threshold", True),
        ("Partition fat", _step_partition_fat, "Fat partition", True),
        ("Cleanup LA fat mask", _step_cleanup, "Cleanup", False),
        ("Extract meshes", _step_extract_meshes, "Mesh extraction", False),
        ("Generate quality flags", _step_generate_quality_flags,
         "Quality flag generation", False),
        ("Generate QA dashboard", _step_generate_dashboard,
         "QA dashboard generation", False),
        ("Save LA fat mask", _step_save_la_fat_mask,
         "Saving LA fat mask", False),
        ("Save quality flags", _step_save_quality_flags,
         "Saving quality flags", False),
        ("Save pipeline result", _step_save_pipeline_result,
         "Saving pipeline result data", False),
    ]

    early_result = _run_steps(state, steps)
    if early_result is not None:
        return early_result

    # ── Build final result ──────────────────────────────────────────────────
    success = len(state.errors) == 0
    total_runtime = time.perf_counter() - start_time

    logger.info(
        "[%s] Pipeline %s for %s (%.1f s, %d error(s), %d warning(s))",
        _ts(),
        "succeeded" if success else "failed",
        patient_id,
        total_runtime,
        len(state.errors),
        len(state.warnings),
    )

    return PipelineResult(
        patient_id=patient_id,
        success=success,
        partition_result=state.partition_result,
        fat_threshold_result=state.fat_threshold_result,
        pericardium_result=state.pericardium_result,
        cleanup_result=state.cleanup_result,
        quality_flags=state.quality_flags,
        dashboard_output=state.dashboard_output,
        mesh_paths=state.mesh_paths,
        errors=state.errors,
        warnings=state.warnings,
        total_runtime_seconds=total_runtime,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_masks(
    intermediate_dir: str,
    patient_id: str,
) -> dict[str, np.ndarray]:
    """Load TS masks from disk into a ``{name: array}`` dictionary.

    Only structures whose mask file exists on disk are included.
    Resolution of v2 and v1 native filenames is delegated to
    :func:`la_fat.ts_runner.resolve_ts_mask_path`.
    """
    masks: dict[str, np.ndarray] = {}
    for internal_key in _STRUCTURE_FILENAMES:
        mask_path = resolve_ts_mask_path(
            intermediate_dir, patient_id, internal_key,
        )
        if mask_path:
            try:
                array, _ = nifti_io.load_nifti(mask_path)
                masks[internal_key] = array
                logger.debug("Loaded mask %s (%s)", internal_key, mask_path)
            except Exception as exc:
                logger.warning(
                    "Failed to load mask %s from %s: %s",
                    internal_key, mask_path, exc,
                )
    return masks


def _save_quality_flags_json(
    flags: list[QualityFlag],
    path: str,
) -> None:
    """Write quality flags as a JSON array."""
    data = [
        {
            "severity": f.severity,
            "concern": f.concern,
            "detail": f.detail,
            "threshold_value": f.threshold_value,
            "actual_value": f.actual_value,
        }
        for f in flags
    ]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _empty_cleanup_result() -> CleanupResult:
    """Return an empty ``CleanupResult`` for flagging/dashboard when cleanup
    was skipped."""
    return CleanupResult(
        cleaned_mask=np.empty(0, dtype=bool),
        islands_removed=0,
        island_volumes_mm3=[],
        total_removed_volume_mm3=0.0,
        morphological_opening_applied=False,
        vessel_filling_applied=False,
    )


def _ts() -> str:
    """Return the current time formatted as ``HH:MM:SS`` for log messages."""
    return time.strftime("%H:%M:%S")


def _run_steps(
    state: PipelineState,
    steps: list[tuple[str, t.Callable, str, bool]],
) -> PipelineResult | None:
    """Run a sequence of pipeline steps, accumulating errors.

    Each step is a tuple of ``(name, callable, error_context, is_fatal)``.

    *name* is the human-readable label for logging.  *callable* receives
    the mutable *state* and is expected to mutate it.  *error_context* is
    the prefix used in error messages.  *is_fatal* controls whether a
    failure should halt the pipeline (``True``) or just be recorded and
    continue (``False``).

    Returns ``None`` when all steps run (even with non-fatal errors).
    Returns a partial :class:`PipelineResult` when a fatal step fails
    (the caller should return this immediately).
    """
    state.step_total = len(steps)
    for step_idx, (step_name, step_fn, error_context, is_fatal) in enumerate(
        steps, start=1,
    ):
        state.step = step_idx
        logger.info(
            "[%s] Step %d/%d: %s",
            _ts(), state.step, state.step_total, step_name,
        )
        try:
            step_fn(state)
        except Exception as exc:
            state.errors.append(f"{error_context} failed: {exc}")
            logger.error(
                "[%s] Step %d/%d: %s",
                _ts(), state.step, state.step_total, exc,
            )
            if is_fatal:
                return _build_result(
                    patient_id=state.patient_id,
                    start_time=state.start_time,
                    errors=state.errors,
                    warnings=state.warnings,
                    pericardium_result=state.pericardium_result,
                    fat_threshold_result=state.fat_threshold_result,
                    partition_result=state.partition_result,
                    cleanup_result=state.cleanup_result,
                    quality_flags=state.quality_flags,
                    dashboard_output=state.dashboard_output,
                    mesh_paths=state.mesh_paths,
                )
    return None


# ---------------------------------------------------------------------------
# Step functions (each mutates PipelineState)
# ---------------------------------------------------------------------------


def _step_load_ct(state: PipelineState) -> None:
    """Load / resample CT to isotropic spacing (pipeline step 1)."""
    patient_id = state.patient_id
    intermediate_dir = state.intermediate_dir
    raw_ct_dir = state.raw_ct_dir
    spacing = state.spacing
    spacing_mm = state.cfg.spacing_mm if state.cfg else 1.5

    ct_array: np.ndarray | None = None
    ct_spacing: tuple[float, float, float] = spacing
    ct_origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ct_direction: np.ndarray = np.eye(3)

    # Check for pre-resampled cache (try new + old naming).
    resampled_path = os.path.join(
        intermediate_dir,
        _RESAMPLED_CT_FILENAME.format(patient_id=patient_id),
    )
    if not os.path.isfile(resampled_path):
        legacy_ct_path = os.path.join(intermediate_dir, "ct_resampled.nii.gz")
        if os.path.isfile(legacy_ct_path):
            resampled_path = legacy_ct_path
    if os.path.isfile(resampled_path):
        logger.info(
            "[%s] Step %d/%d: Loading pre-resampled CT from %s",
            _ts(), state.step, state.step_total, resampled_path,
        )
        img = sitk.ReadImage(resampled_path)
        ct_array = sitk.GetArrayFromImage(img)
        ct_spacing = img.GetSpacing()
        ct_origin = img.GetOrigin()
        ct_direction = np.array(img.GetDirection()).reshape(3, 3)
    else:
        raw_ct_path = os.path.join(raw_ct_dir, f"{patient_id}.nii.gz")
        if not os.path.isfile(raw_ct_path):
            raw_ct_path = os.path.join(raw_ct_dir, f"{patient_id}.nii")
        if not os.path.isfile(raw_ct_path):
            raise FileNotFoundError(
                f"Raw CT volume not found for patient {patient_id} "
                f"(searched for .nii.gz and .nii in {raw_ct_dir})"
            )

        logger.info(
            "[%s] Step %d/%d: Resampling CT %s to isotropic %.1f mm",
            _ts(), state.step, state.step_total, raw_ct_path, spacing_mm,
        )
        resample_result: ResampleResult = resample_to_isotropic(
            raw_ct_path, spacing_mm,
        )
        ct_array = resample_result.ct_array
        ct_spacing = resample_result.spacing
        ct_origin = resample_result.origin
        ct_direction = resample_result.direction

        os.makedirs(os.path.dirname(resampled_path), exist_ok=True)
        nifti_io.save_nifti(
            ct_array,
            resampled_path,
            spacing=ct_spacing,
            origin=ct_origin,
            direction=ct_direction,
        )
        logger.info(
            "[%s] Step %d/%d: Cached resampled CT to %s",
            _ts(), state.step, state.step_total, resampled_path,
        )

    state.ct_array = ct_array
    state.ct_spacing = ct_spacing
    state.ct_origin = ct_origin
    state.ct_direction = ct_direction


def _step_load_masks(state: PipelineState) -> None:
    """Load TS masks from disk (pipeline step 2)."""
    loaded_masks: dict[str, np.ndarray] = _load_masks(
        state.intermediate_dir, state.patient_id,
    )
    if not loaded_masks:
        raise FileNotFoundError(
            f"No TS masks found in {state.intermediate_dir} "
            f"for patient {state.patient_id}"
        )
    logger.info(
        "[%s] Step %d/%d: Loaded %d mask(s): %s",
        _ts(), state.step, state.step_total,
        len(loaded_masks),
        ", ".join(sorted(loaded_masks.keys())),
    )
    state.loaded_masks = loaded_masks

    # Build anchor_masks used by later steps.
    anchor_masks: dict[str, np.ndarray] = {}
    for k in _ANCHOR_KEYS:
        if k in loaded_masks:
            anchor_masks[k] = loaded_masks[k]
    state.anchor_masks = anchor_masks


def _step_resolve_pericardium(state: PipelineState) -> None:
    """Resolve pericardium (pipeline step 3, fatal)."""
    ts_masks_for_resolver: dict[str, np.ndarray] = {}
    for k in _CHAMBER_KEYS:
        if k in state.loaded_masks:
            ts_masks_for_resolver[k] = state.loaded_masks[k]
    if "Pericardium" in state.loaded_masks:
        ts_masks_for_resolver["pericardium"] = state.loaded_masks["Pericardium"]

    pericardium_result = resolve_pericardium(
        ts_masks_for_resolver, state.cfg, state.spacing,
    )
    if pericardium_result.fallback_triggered:
        msg = (
            f"Pericardium fallback triggered: "
            f"{pericardium_result.fallback_reason}"
        )
        state.warnings.append(msg)
        logger.warning(
            "[%s] Step %d/%d: %s",
            _ts(), state.step, state.step_total, msg,
        )
    else:
        logger.info(
            "[%s] Step %d/%d: Pericardium resolved via %s",
            _ts(), state.step, state.step_total, pericardium_result.method,
        )
    state.pericardium_result = pericardium_result


def _step_compute_fat_threshold(state: PipelineState) -> None:
    """Compute fat HU threshold (pipeline step 4)."""
    fat_threshold_result = compute_fat_threshold(
        state.ct_array, state.pericardium_result.mask, state.cfg,
    )
    if fat_threshold_result.fallback_triggered:
        msg = (
            f"Fat threshold fallback triggered: "
            f"{fat_threshold_result.fallback_reason}"
        )
        state.warnings.append(msg)
        logger.warning(
            "[%s] Step %d/%d: %s",
            _ts(), state.step, state.step_total, msg,
        )
    else:
        logger.info(
            "[%s] Step %d/%d: Fat threshold — HU range [%.1f, %.1f] "
            "(mean=%.1f, sigma=%.1f, n=%d)",
            _ts(), state.step, state.step_total,
            fat_threshold_result.hu_low,
            fat_threshold_result.hu_high,
            fat_threshold_result.mean_hu,
            fat_threshold_result.sigma_hu,
            fat_threshold_result.num_voxels_fit,
        )
    state.fat_threshold_result = fat_threshold_result


def _step_partition_fat(state: PipelineState) -> None:
    """Partition epicardial fat (pipeline step 5, fatal)."""
    partition_result = partition_fat(
        ct_array=state.ct_array,
        pericardium_mask=state.pericardium_result.mask,
        fat_hu_range=(
            state.fat_threshold_result.hu_low,
            state.fat_threshold_result.hu_high,
        ),
        anchor_masks=state.anchor_masks,
        config=state.cfg,
        spacing=state.spacing,
    )
    if partition_result.excluded_anchors:
        msg = (
            f"Anchor(s) excluded: "
            f"{', '.join(partition_result.excluded_anchors)}"
        )
        state.warnings.append(msg)
        logger.warning(
            "[%s] Step %d/%d: %s",
            _ts(), state.step, state.step_total, msg,
        )
    logger.info(
        "[%s] Step %d/%d: Partition complete — "
        "LA fat=%.2f ml, total fat=%.2f ml, "
        "unassigned=%.2f ml",
        _ts(), state.step, state.step_total,
        partition_result.anchor_volumes_ml.get("LA", 0.0),
        partition_result.total_fat_volume_ml,
        partition_result.unassigned_volume_ml,
    )
    state.partition_result = partition_result


def _step_cleanup(state: PipelineState) -> None:
    """Clean LA fat mask (pipeline step 6, non-fatal)."""
    cleanup_result = cleanup_la_fat_mask(
        state.partition_result.la_fat_mask, state.cfg, state.spacing,
        apply_opening=False,
        apply_vessel_filling=False,
    )
    if cleanup_result.islands_removed > 0:
        logger.info(
            "[%s] Step %d/%d: Removed %d island(s) "
            "(total %.2f mm³)",
            _ts(), state.step, state.step_total,
            cleanup_result.islands_removed,
            cleanup_result.total_removed_volume_mm3,
        )
    else:
        logger.info(
            "[%s] Step %d/%d: No islands removed",
            _ts(), state.step, state.step_total,
        )
    state.cleanup_result = cleanup_result


def _step_extract_meshes(state: PipelineState) -> None:
    """Extract interactive meshes (pipeline step 7, non-fatal)."""
    artifacts = PipelineArtifacts(
        anchor_masks=state.anchor_masks,
        pericardium_mask=state.pericardium_result.mask,
        partition_result=state.partition_result,
        cleanup_result=state.cleanup_result or _empty_cleanup_result(),
        spacing=state.spacing,
    )
    mesh_results = extract_interactive_meshes(
        artifacts, state.patient_output_dir,
    )
    mesh_paths: dict[str, list[str]] = {}
    for step_name, step_meshes in mesh_results.items():
        ply_files = [
            os.path.join(
                state.patient_output_dir,
                "meshes",
                step_name,
                f"{sn}.ply",
            )
            for sn, md in step_meshes.items()
            if md is not None
        ]
        mesh_paths[step_name] = ply_files
    total_meshes = sum(len(files) for files in mesh_paths.values())
    logger.info(
        "[%s] Step %d/%d: Mesh extraction complete — "
        "%d meshes saved to %s/meshes",
        _ts(), state.step, state.step_total,
        total_meshes, state.patient_output_dir,
    )
    state.mesh_paths = mesh_paths


def _step_generate_quality_flags(state: PipelineState) -> None:
    """Generate quality flags (pipeline step 8, non-fatal)."""
    quality_flags = generate_quality_flags(
        partition_result=state.partition_result,
        fat_threshold_result=state.fat_threshold_result,
        pericardium_result=state.pericardium_result,
        cleanup_result=state.cleanup_result or _empty_cleanup_result(),
        config=state.cfg,
    )
    logger.info(
        "[%s] Step %d/%d: %d quality flag(s) generated",
        _ts(), state.step, state.step_total, len(quality_flags),
    )
    state.quality_flags = quality_flags


def _step_generate_dashboard(state: PipelineState) -> None:
    """Generate QA dashboard (pipeline step 9, non-fatal)."""
    dashboard_output = generate_dashboard(
        ct_array=state.ct_array,
        anchor_masks=state.anchor_masks,
        pericardium_result=state.pericardium_result,
        partition_result=state.partition_result,
        fat_threshold_result=state.fat_threshold_result,
        cleanup_result=state.cleanup_result or _empty_cleanup_result(),
        quality_flags=state.quality_flags,
        config=state.cfg,
        patient_id=state.patient_id,
        output_dir=state.patient_output_dir,
        spacing=state.spacing,
    )
    logger.info(
        "[%s] Step %d/%d: Dashboard saved to %s",
        _ts(), state.step, state.step_total, state.patient_output_dir,
    )
    state.dashboard_output = dashboard_output


def _step_save_la_fat_mask(state: PipelineState) -> None:
    """Save LA fat mask as NIfTI (pipeline step 10, non-fatal)."""
    la_fat_mask_path = os.path.join(state.patient_output_dir, "la_fat_mask.nii.gz")
    if state.cleanup_result is not None:
        os.makedirs(state.patient_output_dir, exist_ok=True)
        nifti_io.save_nifti(
            state.cleanup_result.cleaned_mask.astype(np.uint8),
            la_fat_mask_path,
            spacing=state.ct_spacing,
            origin=state.ct_origin,
            direction=state.ct_direction,
        )
        logger.info(
            "[%s] Step %d/%d: LA fat mask saved to %s",
            _ts(), state.step, state.step_total, la_fat_mask_path,
        )
    else:
        state.warnings.append("LA fat mask not saved (cleanup result unavailable)")
        logger.warning(
            "[%s] Step %d/%d: Cleanup result unavailable, skipping mask save",
            _ts(), state.step, state.step_total,
        )


def _step_save_quality_flags(state: PipelineState) -> None:
    """Save quality flags as JSON (pipeline step 11, non-fatal)."""
    quality_flags_path = os.path.join(
        state.patient_output_dir, "quality_flags.json",
    )
    os.makedirs(state.patient_output_dir, exist_ok=True)
    _save_quality_flags_json(state.quality_flags, quality_flags_path)
    logger.info(
        "[%s] Step %d/%d: Quality flags saved to %s",
        _ts(), state.step, state.step_total, quality_flags_path,
    )


def _step_save_pipeline_result(state: PipelineState) -> None:
    """Save PipelineResultData for dashboard consumption (pipeline step 12, non-fatal)."""
    voxel_vol = voxel_volume_ml(state.spacing)
    total_fat = (
        state.partition_result.total_fat_volume_ml
        if state.partition_result is not None
        else 0.0
    )
    unassigned_vol = (
        state.partition_result.unassigned_volume_ml
        if state.partition_result is not None
        else 0.0
    )
    result_data = PipelineResultData(
        patient_id=state.patient_id,
        la_fat_volume_ml=(
            state.partition_result.anchor_volumes_ml.get("LA", 0.0)
            if state.partition_result is not None
            else 0.0
        ),
        total_fat_volume_ml=total_fat,
        pericardium_volume_ml=(
            state.pericardium_result.volume_ml
            if state.pericardium_result is not None
            else 0.0
        ),
        unassigned_volume_ml=unassigned_vol,
        unassigned_fat_pct=(
            (unassigned_vol / max(total_fat, 0.001)) * 100.0
        ),
        anchor_volumes_ml=(
            state.partition_result.anchor_volumes_ml
            if state.partition_result is not None
            else {}
        ),
        quality_flags=[
            {
                "severity": f.severity,
                "concern": f.concern,
                "detail": f.detail,
                "threshold_value": f.threshold_value,
                "actual_value": f.actual_value,
            }
            for f in state.quality_flags
        ],
        fat_hu_range=(
            (state.fat_threshold_result.hu_low, state.fat_threshold_result.hu_high)
            if state.fat_threshold_result is not None
            else (0.0, 0.0)
        ),
        voxel_volume_ml=voxel_vol,
        excluded_anchors=(
            list(state.partition_result.excluded_anchors)
            if state.partition_result is not None
            else []
        ),
        islands_removed=(
            state.cleanup_result.islands_removed
            if state.cleanup_result is not None
            else 0
        ),
        total_removed_volume_mm3=(
            state.cleanup_result.total_removed_volume_mm3
            if state.cleanup_result is not None
            else 0.0
        ),
        warnings=list(state.warnings),
        errors=list(state.errors),
    )
    save_pipeline_result(result_data, state.patient_output_dir)
    logger.info(
        "[%s] Step %d/%d: PipelineResultData saved to %s",
        _ts(), state.step, state.step_total,
        os.path.join(state.patient_output_dir, "pipeline_result.json"),
    )


def _build_result(
    patient_id: str,
    start_time: float,
    errors: list[str],
    warnings: list[str],
    pericardium_result: PericardiumResult | None = None,
    fat_threshold_result: FatThresholdResult | None = None,
    partition_result: PartitionResult | None = None,
    cleanup_result: CleanupResult | None = None,
    quality_flags: list[QualityFlag] | None = None,
    dashboard_output: DashboardOutput | None = None,
    mesh_paths: dict[str, list[str]] | None = None,
) -> PipelineResult:
    """Build a ``PipelineResult`` with the accumulated state.

    Used for early-exit paths when a fatal error occurs mid-pipeline.
    """
    total_runtime = time.perf_counter() - start_time
    return PipelineResult(
        patient_id=patient_id,
        success=False,
        partition_result=partition_result,
        fat_threshold_result=fat_threshold_result,
        pericardium_result=pericardium_result,
        cleanup_result=cleanup_result,
        quality_flags=quality_flags or [],
        dashboard_output=dashboard_output,
        mesh_paths=mesh_paths,
        errors=errors,
        warnings=warnings,
        total_runtime_seconds=total_runtime,
    )


