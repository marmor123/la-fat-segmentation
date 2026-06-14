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
    errors: list[str] = []
    warnings: list[str] = []

    pericardium_result: PericardiumResult | None = None
    fat_threshold_result: FatThresholdResult | None = None
    partition_result: PartitionResult | None = None
    cleanup_result: CleanupResult | None = None
    quality_flags: list[QualityFlag] = []
    dashboard_output: DashboardOutput | None = None
    mesh_paths: dict[str, list[str]] | None = None

    # ── Step 0: Configuration ───────────────────────────────────────────────
    step_total = 13
    step = 0

    try:
        if config is not None:
            cfg = config
        elif config_path is not None:
            cfg = PipelineConfig.from_yaml(config_path)
            logger.info(
                "[%s] Step 0/%d: Loaded config from %s",
                _ts(), step_total, config_path,
            )
        else:
            cfg = PipelineConfig()
            logger.info(
                "[%s] Step 0/%d: Using default configuration",
                _ts(), step_total,
            )

        data_dir: str = cfg.data_dir
        output_dir: str = cfg.output_dir
        intermediate_subdir: str = cfg.intermediate_subdir
        raw_subdir: str = cfg.raw_subdir

        intermediate_dir = os.path.join(data_dir, intermediate_subdir, patient_id)
        raw_ct_dir = os.path.join(data_dir, raw_subdir)
        patient_output_dir = os.path.join(output_dir, patient_id)
        spacing: tuple[float, float, float] = (
            cfg.spacing_mm,
            cfg.spacing_mm,
            cfg.spacing_mm,
        )

        # ── Step 1: Load / resample CT ────────────────────────────────────
        step += 1
        logger.info(
            "[%s] Step %d/%d: Loading / resampling CT for patient %s",
            _ts(), step, step_total, patient_id,
        )

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
            # Fallback: old naming without patient_id prefix
            legacy_ct_path = os.path.join(intermediate_dir, "ct_resampled.nii.gz")
            if os.path.isfile(legacy_ct_path):
                resampled_path = legacy_ct_path
        if os.path.isfile(resampled_path):
            logger.info(
                "[%s] Step %d/%d: Loading pre-resampled CT from %s",
                _ts(), step, step_total, resampled_path,
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
                _ts(), step, step_total, raw_ct_path, cfg.spacing_mm,
            )
            resample_result: ResampleResult = resample_to_isotropic(
                raw_ct_path, cfg.spacing_mm,
            )
            ct_array = resample_result.ct_array
            ct_spacing = resample_result.spacing
            ct_origin = resample_result.origin
            ct_direction = resample_result.direction

            # Cache resampled CT for future runs.
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
                _ts(), step, step_total, resampled_path,
            )

        # ── Step 2: Load TS masks from disk ───────────────────────────────
        step += 1
        logger.info(
            "[%s] Step %d/%d: Loading TS masks from %s",
            _ts(), step, step_total, intermediate_dir,
        )

        loaded_masks: dict[str, np.ndarray] = _load_masks(
            intermediate_dir, patient_id,
        )

        if not loaded_masks:
            raise FileNotFoundError(
                f"No TS masks found in {intermediate_dir} "
                f"for patient {patient_id}"
            )

        logger.info(
            "[%s] Step %d/%d: Loaded %d mask(s): %s",
            _ts(), step, step_total,
            len(loaded_masks),
            ", ".join(sorted(loaded_masks.keys())),
        )

        # ── Step 3: Resolve pericardium ───────────────────────────────────
        step += 1
        logger.info(
            "[%s] Step %d/%d: Resolving pericardium",
            _ts(), step, step_total,
        )

        ts_masks_for_resolver: dict[str, np.ndarray] = {}
        for k in _CHAMBER_KEYS:
            if k in loaded_masks:
                ts_masks_for_resolver[k] = loaded_masks[k]
        if "Pericardium" in loaded_masks:
            ts_masks_for_resolver["pericardium"] = loaded_masks["Pericardium"]

        try:
            pericardium_result = resolve_pericardium(
                ts_masks_for_resolver, cfg, spacing,
            )
            if pericardium_result.fallback_triggered:
                msg = (
                    f"Pericardium fallback triggered: "
                    f"{pericardium_result.fallback_reason}"
                )
                warnings.append(msg)
                logger.warning(
                    "[%s] Step %d/%d: %s",
                    _ts(), step, step_total, msg,
                )
            else:
                logger.info(
                    "[%s] Step %d/%d: Pericardium resolved via %s",
                    _ts(), step, step_total, pericardium_result.method,
                )
        except ValueError as exc:
            errors.append(f"Pericardium resolution failed: {exc}")
            logger.error(
                "[%s] Step %d/%d: %s",
                _ts(), step, step_total, exc,
            )
            # Cannot continue without pericardium
            return _build_result(
                patient_id=patient_id,
                start_time=start_time,
                errors=errors,
                warnings=warnings,
            )

        # ── Step 4: Compute fat threshold ─────────────────────────────────
        step += 1
        logger.info(
            "[%s] Step %d/%d: Computing fat HU threshold",
            _ts(), step, step_total,
        )

        fat_threshold_result = compute_fat_threshold(
            ct_array, pericardium_result.mask, cfg,
        )
        if fat_threshold_result.fallback_triggered:
            msg = (
                f"Fat threshold fallback triggered: "
                f"{fat_threshold_result.fallback_reason}"
            )
            warnings.append(msg)
            logger.warning(
                "[%s] Step %d/%d: %s",
                _ts(), step, step_total, msg,
            )
        else:
            logger.info(
                "[%s] Step %d/%d: Fat threshold — HU range [%.1f, %.1f] "
                "(mean=%.1f, sigma=%.1f, n=%d)",
                _ts(), step, step_total,
                fat_threshold_result.hu_low,
                fat_threshold_result.hu_high,
                fat_threshold_result.mean_hu,
                fat_threshold_result.sigma_hu,
                fat_threshold_result.num_voxels_fit,
            )

        # ── Step 5: Partition fat ─────────────────────────────────────────
        step += 1
        logger.info(
            "[%s] Step %d/%d: Partitioning epicardial fat",
            _ts(), step, step_total,
        )

        anchor_masks: dict[str, np.ndarray] = {}
        for k in _ANCHOR_KEYS:
            if k in loaded_masks:
                anchor_masks[k] = loaded_masks[k]

        try:
            partition_result = partition_fat(
                ct_array=ct_array,
                pericardium_mask=pericardium_result.mask,
                fat_hu_range=(
                    fat_threshold_result.hu_low,
                    fat_threshold_result.hu_high,
                ),
                anchor_masks=anchor_masks,
                config=cfg,
                spacing=spacing,
            )
            if partition_result.excluded_anchors:
                msg = (
                    f"Anchor(s) excluded: "
                    f"{', '.join(partition_result.excluded_anchors)}"
                )
                warnings.append(msg)
                logger.warning(
                    "[%s] Step %d/%d: %s",
                    _ts(), step, step_total, msg,
                )
            logger.info(
                "[%s] Step %d/%d: Partition complete — "
                "LA fat=%.2f ml, total fat=%.2f ml, "
                "unassigned=%.2f ml",
                _ts(), step, step_total,
                partition_result.anchor_volumes_ml.get("LA", 0.0),
                partition_result.total_fat_volume_ml,
                partition_result.unassigned_volume_ml,
            )
        except ValueError as exc:
            errors.append(f"Fat partition failed: {exc}")
            logger.error(
                "[%s] Step %d/%d: %s",
                _ts(), step, step_total, exc,
            )
            # Cannot continue without partition
            return _build_result(
                patient_id=patient_id,
                start_time=start_time,
                errors=errors,
                warnings=warnings,
                pericardium_result=pericardium_result,
                fat_threshold_result=fat_threshold_result,
            )

        # ── Step 6: Cleanup LA fat mask ───────────────────────────────────
        step += 1
        logger.info(
            "[%s] Step %d/%d: Cleaning LA fat mask",
            _ts(), step, step_total,
        )

        try:
            cleanup_result = cleanup_la_fat_mask(
                partition_result.la_fat_mask, cfg, spacing,
                apply_opening=False,
                apply_vessel_filling=False,
            )
            if cleanup_result.islands_removed > 0:
                logger.info(
                    "[%s] Step %d/%d: Removed %d island(s) "
                    "(total %.2f mm³)",
                    _ts(), step, step_total,
                    cleanup_result.islands_removed,
                    cleanup_result.total_removed_volume_mm3,
                )
            else:
                logger.info(
                    "[%s] Step %d/%d: No islands removed",
                    _ts(), step, step_total,
                )
        except Exception as exc:
            errors.append(f"Cleanup failed: {exc}")
            logger.error(
                "[%s] Step %d/%d: %s",
                _ts(), step, step_total, exc,
            )
            cleanup_result = None

        # ── Step 7: Extract meshes ────────────────────────────────────────
        step += 1
        logger.info(
            "[%s] Step %d/%d: Extracting meshes for interactive visualization",
            _ts(), step, step_total,
        )

        try:
            pipeline_artifacts = PipelineArtifacts(
                anchor_masks=anchor_masks,
                pericardium_mask=pericardium_result.mask,
                partition_result=partition_result,
                cleanup_result=cleanup_result or _empty_cleanup_result(),
                spacing=spacing,
            )
            mesh_results = extract_interactive_meshes(
                pipeline_artifacts, patient_output_dir,
            )
            mesh_paths = {}
            for step_name, step_meshes in mesh_results.items():
                ply_files = [
                    os.path.join(
                        patient_output_dir,
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
                _ts(), step, step_total,
                total_meshes, patient_output_dir,
            )
        except Exception as exc:
            errors.append(f"Mesh extraction failed: {exc}")
            logger.error(
                "[%s] Step %d/%d: %s",
                _ts(), step, step_total, exc,
            )
            mesh_paths = None

        # ── Step 8: Generate quality flags ────────────────────────────────
        step += 1
        logger.info(
            "[%s] Step %d/%d: Generating quality flags",
            _ts(), step, step_total,
        )

        try:
            quality_flags = generate_quality_flags(
                partition_result=partition_result,
                fat_threshold_result=fat_threshold_result,
                pericardium_result=pericardium_result,
                cleanup_result=cleanup_result or _empty_cleanup_result(),
                config=cfg,
            )
            logger.info(
                "[%s] Step %d/%d: %d quality flag(s) generated",
                _ts(), step, step_total, len(quality_flags),
            )
        except Exception as exc:
            errors.append(f"Quality flag generation failed: {exc}")
            logger.error(
                "[%s] Step %d/%d: %s",
                _ts(), step, step_total, exc,
            )

        # ── Step 9: Generate QA dashboard ─────────────────────────────────
        step += 1
        logger.info(
            "[%s] Step %d/%d: Generating QA dashboard",
            _ts(), step, step_total,
        )

        try:
            dashboard_output = generate_dashboard(
                ct_array=ct_array,
                anchor_masks=anchor_masks,
                pericardium_result=pericardium_result,
                partition_result=partition_result,
                fat_threshold_result=fat_threshold_result,
                cleanup_result=cleanup_result or _empty_cleanup_result(),
                quality_flags=quality_flags,
                config=cfg,
                patient_id=patient_id,
                output_dir=patient_output_dir,
                spacing=spacing,
            )
            logger.info(
                "[%s] Step %d/%d: Dashboard saved to %s",
                _ts(), step, step_total, patient_output_dir,
            )
        except Exception as exc:
            errors.append(f"QA dashboard generation failed: {exc}")
            logger.error(
                "[%s] Step %d/%d: %s",
                _ts(), step, step_total, exc,
            )

        # ── Step 10: Save LA fat mask ─────────────────────────────────────
        step += 1
        logger.info(
            "[%s] Step %d/%d: Saving LA fat mask",
            _ts(), step, step_total,
        )

        la_fat_mask_path = os.path.join(patient_output_dir, "la_fat_mask.nii.gz")
        if cleanup_result is not None:
            try:
                os.makedirs(patient_output_dir, exist_ok=True)
                nifti_io.save_nifti(
                    cleanup_result.cleaned_mask.astype(np.uint8),
                    la_fat_mask_path,
                    spacing=ct_spacing,
                    origin=ct_origin,
                    direction=ct_direction,
                )
                logger.info(
                    "[%s] Step %d/%d: LA fat mask saved to %s",
                    _ts(), step, step_total, la_fat_mask_path,
                )
            except Exception as exc:
                errors.append(f"Saving LA fat mask failed: {exc}")
                logger.error(
                    "[%s] Step %d/%d: %s",
                    _ts(), step, step_total, exc,
                )
        else:
            warnings.append("LA fat mask not saved (cleanup result unavailable)")
            logger.warning(
                "[%s] Step %d/%d: Cleanup result unavailable, skipping mask save",
                _ts(), step, step_total,
            )

        # ── Step 11: Save quality flags as JSON ───────────────────────────
        step += 1
        logger.info(
            "[%s] Step %d/%d: Saving quality flags",
            _ts(), step, step_total,
        )

        quality_flags_path = os.path.join(
            patient_output_dir, "quality_flags.json",
        )
        try:
            os.makedirs(patient_output_dir, exist_ok=True)
            _save_quality_flags_json(quality_flags, quality_flags_path)
            logger.info(
                "[%s] Step %d/%d: Quality flags saved to %s",
                _ts(), step, step_total, quality_flags_path,
            )
        except Exception as exc:
            errors.append(f"Saving quality flags failed: {exc}")
            logger.error(
                "[%s] Step %d/%d: %s",
                _ts(), step, step_total, exc,
            )

    except FileNotFoundError as exc:
        errors.append(str(exc))
        logger.error("[%s] %s", _ts(), exc)
    except Exception as exc:
        errors.append(f"Unexpected pipeline error: {exc}")
        logger.exception("[%s] Unexpected pipeline error: %s", _ts(), exc)

    success = len(errors) == 0
    total_runtime = time.perf_counter() - start_time

    # ── Step 12: Save PipelineResultData (single-source-of-truth for dashboards)
    voxel_vol = voxel_volume_ml(spacing)
    total_fat = (
        partition_result.total_fat_volume_ml if partition_result is not None else 0.0
    )
    unassigned_vol = (
        partition_result.unassigned_volume_ml if partition_result is not None else 0.0
    )
    unassigned_pct = (
        (unassigned_vol / total_fat * 100.0) if total_fat > 0.001 else 0.0
    )

    pipeline_result_data = PipelineResultData(
        patient_id=patient_id,
        la_fat_volume_ml=(
            partition_result.anchor_volumes_ml.get("LA", 0.0)
            if partition_result is not None
            else 0.0
        ),
        total_fat_volume_ml=total_fat,
        pericardium_volume_ml=(
            pericardium_result.volume_ml if pericardium_result is not None else 0.0
        ),
        unassigned_volume_ml=unassigned_vol,
        unassigned_fat_pct=unassigned_pct,
        anchor_volumes_ml=(
            partition_result.anchor_volumes_ml
            if partition_result is not None
            else {}
        ),
        quality_flags=[dataclasses.asdict(f) for f in quality_flags],
        fat_hu_range=(
            (fat_threshold_result.hu_low, fat_threshold_result.hu_high)
            if fat_threshold_result is not None
            else (0.0, 0.0)
        ),
        voxel_volume_ml=voxel_vol,
        excluded_anchors=(
            partition_result.excluded_anchors
            if partition_result is not None
            else []
        ),
        islands_removed=(
            cleanup_result.islands_removed if cleanup_result is not None else 0
        ),
        total_removed_volume_mm3=(
            cleanup_result.total_removed_volume_mm3
            if cleanup_result is not None
            else 0.0
        ),
        warnings=list(warnings),
        errors=list(errors),
    )
    try:
        save_pipeline_result(pipeline_result_data, patient_output_dir)
        logger.info(
            "[%s] Step %d/%d: Pipeline result data saved to %s",
            _ts(), step + 1, step_total, patient_output_dir,
        )
    except Exception as exc:
        errors.append(f"Saving pipeline result data failed: {exc}")
        logger.error(
            "[%s] Step %d/%d: %s",
            _ts(), step + 1, step_total, exc,
        )

    logger.info(
        "[%s] Pipeline %s for %s (%.1f s, %d error(s), %d warning(s))",
        _ts(),
        "succeeded" if success else "failed",
        patient_id,
        total_runtime,
        len(errors),
        len(warnings),
    )

    return PipelineResult(
        patient_id=patient_id,
        success=success,
        partition_result=partition_result,
        fat_threshold_result=fat_threshold_result,
        pericardium_result=pericardium_result,
        cleanup_result=cleanup_result,
        quality_flags=quality_flags,
        dashboard_output=dashboard_output,
        mesh_paths=mesh_paths,
        errors=errors,
        warnings=warnings,
        total_runtime_seconds=total_runtime,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main_cli() -> None:
    """Console-script entry point for ``la-fat``.

    Parses command-line arguments and runs
    :func:`run_fat_extraction_pipeline`.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="LA Fat Segmentation Pipeline — "
        "extract epicardial adipose tissue from CT scans.",
    )
    parser.add_argument(
        "--patient",
        required=True,
        help="Patient identifier (e.g. '0674')",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data directory (default: value in config)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: value in config)",
    )
    args = parser.parse_args()

    # Configure logging to stderr so stdout stays clean for the summary.
    _configure_cli_logging()

    # Build config, optionally overriding paths.
    config: PipelineConfig | None = None
    config_path: str | None = args.config

    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    if args.data_dir is not None or args.output_dir is not None:
        if config is None:
            config = PipelineConfig()
        kwargs: dict[str, str] = {}
        if args.data_dir is not None:
            kwargs["data_dir"] = args.data_dir
        if args.output_dir is not None:
            kwargs["output_dir"] = args.output_dir
        if kwargs:
            # Frozen dataclass — replace with a new instance.
            config = dataclasses.replace(config, **kwargs)

    logger.info("Starting LA Fat extraction pipeline for patient %s", args.patient)
    result = run_fat_extraction_pipeline(
        patient_id=args.patient,
        config=config,
        config_path=None if config is not None else config_path,
    )

    # ── Print summary to stdout ──────────────────────────────────────────
    _print_cli_summary(result)

    import sys
    sys.exit(0 if result.success else 1)


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


def _configure_cli_logging() -> None:
    """Configure logging for the CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=__import__("sys").stderr,
    )


def _print_cli_summary(result: PipelineResult) -> None:
    """Print a human-readable summary of pipeline results to stdout."""
    lines: list[str] = []
    add = lines.append

    add("=" * 56)
    add(f"  LA FAT SEGMENTATION PIPELINE — {result.patient_id}")
    add("=" * 56)
    add("")

    if result.success:
        add("  Status:   SUCCESS")
    else:
        add("  Status:   FAILED  ({0} error(s))".format(len(result.errors)))
    add("  Runtime:  {0:.1f} s".format(result.total_runtime_seconds))
    add("")

    if result.pericardium_result is not None:
        add("  Pericardium:  {0}".format(result.pericardium_result.method))
        if result.pericardium_result.fallback_triggered:
            add("    (fallback: {0})".format(
                result.pericardium_result.fallback_reason or "",
            ))

    if result.fat_threshold_result is not None:
        add("  Fat range:    [{0:.1f}, {1:.1f}] HU  ({2})".format(
            result.fat_threshold_result.hu_low,
            result.fat_threshold_result.hu_high,
            result.fat_threshold_result.method,
        ))

    if result.partition_result is not None:
        la_vol = result.partition_result.anchor_volumes_ml.get("LA", 0.0)
        total = result.partition_result.total_fat_volume_ml
        unassigned = result.partition_result.unassigned_volume_ml
        add("  LA Fat:       {0:.2f} ml  (total epicardial: {1:.2f} ml)".format(
            la_vol, total,
        ))
        add("  Unassigned:   {0:.2f} ml".format(unassigned))
        if result.partition_result.excluded_anchors:
            add("  Excluded:     {0}".format(
                ", ".join(result.partition_result.excluded_anchors),
            ))

    if result.cleanup_result is not None:
        add("  Cleanup:      {0} island(s) removed ({1:.1f} mm³)".format(
            result.cleanup_result.islands_removed,
            result.cleanup_result.total_removed_volume_mm3,
        ))

    if result.mesh_paths is not None:
        total_meshes = sum(len(files) for files in result.mesh_paths.values())
        add("  Meshes:       {0} saved ({1} step(s))".format(
            total_meshes, len(result.mesh_paths),
        ))
    else:
        add("  Meshes:       not extracted")

    if result.dashboard_output is not None:
        add("  Dashboard:    {0}".format(result.dashboard_output.output_dir))

    if result.quality_flags:
        add("")
        add("  Quality Flags:")
        for flag in result.quality_flags:
            add("    [{0}] {1}: {2}".format(
                flag.severity, flag.concern, flag.detail,
            ))

    if result.errors:
        add("")
        add("  Errors:")
        for err in result.errors:
            add("    - {0}".format(err))

    if result.warnings:
        add("")
        add("  Warnings:")
        for warn in result.warnings:
            add("    - {0}".format(warn))

    add("")
    add("=" * 56)

    __import__("sys").stdout.write("\n".join(lines) + "\n")
