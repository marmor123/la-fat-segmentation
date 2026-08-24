"""Deep Pipeline Orchestrator for LA Fat Segmentation.

Wires all processing stages into a cohesive, deep pure-function entry point:
``run_fat_extraction`` (and backward-compatible alias ``run_fat_extraction_pipeline``).

Execution sequence:
1. Ingest raw CT scan into native 3D GridGeometry (512x512xZ).
2. Ingest & align canonical anatomical masks to native CT geometry.
3. Resolve pericardial envelope via ``pericardium_resolver``.
4. Perform adaptive trimmed-Gaussian fat thresholding via ``thresholding``.
5. Compute 3D multi-anchor solid Euclidean Distance Transform partition via ``partition_engine``.
6. Filter disconnected fat islands via ``cleanup``.
7. Export native radiomics masks (Adaptive, Conservative, and GMM Bayes).
8. Audit quality concerns via ``quality_flagger``.
9. Generate zero-footprint standalone HTML5 QA Studio via ``cohort_qa_generator``.
10. Return immutable, typed ``SegmentationResult``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk

from la_fat.anatomy import CANONICAL_ANCHORS, voxel_volume_ml
from la_fat.cleanup import CleanupResult, cleanup_la_fat_mask
from la_fat.cohort_qa_generator import extract_patient_qa_record, generate_cohort_qa_html
from la_fat.config import PipelineConfig
from la_fat.image_ops import (
    GridGeometry,
    ResampleResult,
    resample_to_reference,
)
from la_fat import nifti_io
from la_fat.partition_engine import PartitionResult, partition_fat
from la_fat.pericardium_resolver import PericardiumResult, resolve_pericardium
from la_fat.quality_flagger import QualityFlag, generate_quality_flags
from la_fat.thresholding import ThresholdConfig, ThresholdResult, compute_fat_threshold
from la_fat.ts_runner import resolve_ts_mask_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Consolidated Immutable Result Model
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SegmentationResult:
    """Consolidated immutable result of the LA fat extraction pipeline.

    Contains all computed volumetric metrics (both adaptive Gaussian and
    conservative clinical windows), spatial NIfTI paths, quality flags, and
    QA visualization artifacts.
    """

    patient_id: str
    success: bool = True
    la_fat_volume_adaptive_ml: float = 0.0
    la_fat_volume_conservative_ml: float = 0.0
    la_fat_volume_gmm_bayes_ml: float = 0.0
    total_eat_volume_adaptive_ml: float = 0.0
    total_eat_volume_conservative_ml: float = 0.0
    total_eat_volume_gmm_bayes_ml: float = 0.0
    pericardium_volume_ml: float = 0.0
    unassigned_volume_ml: float = 0.0
    unassigned_fat_pct: float = 0.0
    anchor_volumes_ml: Dict[str, float] = dataclasses.field(default_factory=dict)
    fat_hu_range_adaptive: Tuple[float, float] = (-190.0, -30.0)
    fat_hu_range_conservative: Tuple[float, float] = (-190.0, -30.0)
    fat_hu_range_gmm_bayes: Tuple[float, float] = (-190.0, -30.0)
    gaussian_fit_mu: Optional[float] = None
    gaussian_fit_sigma: Optional[float] = None
    gaussian_fit_success: bool = False
    gmm_bayes_mu_fat: Optional[float] = None
    gmm_bayes_sigma_fat: Optional[float] = None
    gmm_bayes_weight_fat: Optional[float] = None
    quality_flags: List[QualityFlag] = dataclasses.field(default_factory=list)
    quality_flags_count_by_tier: Dict[str, int] = dataclasses.field(
        default_factory=lambda: {"high": 0, "medium": 0, "low": 0}
    )
    islands_removed: int = 0
    total_removed_volume_mm3: float = 0.0
    mask_1_5mm_path: Optional[str] = None
    mask_native_path: Optional[str] = None
    mask_final_native_path: Optional[str] = None
    mask_conservative_native_path: Optional[str] = None
    mask_gmm_bayes_native_path: Optional[str] = None
    qa_report_path: Optional[str] = None
    qa_record: Optional[Dict[str, Any]] = None
    errors: List[str] = dataclasses.field(default_factory=list)
    warnings: List[str] = dataclasses.field(default_factory=list)
    total_runtime_seconds: float = 0.0
    partition_result: Optional[Any] = None
    pericardium_result: Optional[Any] = None
    cleanup_result: Optional[Any] = None
    dashboard_output: Optional[Any] = None
    mesh_paths: Optional[Any] = None

    # ── Backward Compatibility Properties ───────────────────────────────────

    @property
    def la_fat_volume_ml(self) -> float:
        """Primary LA fat volume in mL (alias to adaptive volume)."""
        return self.la_fat_volume_adaptive_ml

    @property
    def total_fat_volume_ml(self) -> float:
        """Total epicardial fat volume in mL (alias to adaptive volume)."""
        return self.total_eat_volume_adaptive_ml

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics and metadata to a Python dictionary."""
        return {
            "patient_id": self.patient_id,
            "success": self.success,
            "la_fat_volume_adaptive_ml": self.la_fat_volume_adaptive_ml,
            "la_fat_volume_conservative_ml": self.la_fat_volume_conservative_ml,
            "la_fat_volume_gmm_bayes_ml": self.la_fat_volume_gmm_bayes_ml,
            "total_eat_volume_adaptive_ml": self.total_eat_volume_adaptive_ml,
            "total_eat_volume_conservative_ml": self.total_eat_volume_conservative_ml,
            "total_eat_volume_gmm_bayes_ml": self.total_eat_volume_gmm_bayes_ml,
            "pericardium_volume_ml": self.pericardium_volume_ml,
            "unassigned_volume_ml": self.unassigned_volume_ml,
            "unassigned_fat_pct": self.unassigned_fat_pct,
            "anchor_volumes_ml": self.anchor_volumes_ml,
            "fat_hu_range_adaptive": list(self.fat_hu_range_adaptive),
            "fat_hu_range_conservative": list(self.fat_hu_range_conservative),
            "fat_hu_range_gmm_bayes": list(self.fat_hu_range_gmm_bayes),
            "gaussian_fit_mu": self.gaussian_fit_mu,
            "gaussian_fit_sigma": self.gaussian_fit_sigma,
            "gaussian_fit_success": self.gaussian_fit_success,
            "gmm_bayes_mu_fat": self.gmm_bayes_mu_fat,
            "gmm_bayes_sigma_fat": self.gmm_bayes_sigma_fat,
            "gmm_bayes_weight_fat": self.gmm_bayes_weight_fat,
            "quality_flags": [
                {
                    "severity": f.severity,
                    "concern": f.concern or getattr(f, "flag_id", ""),
                    "detail": f.detail or getattr(f, "message", ""),
                    "threshold_value": f.threshold_value,
                    "actual_value": f.actual_value,
                }
                for f in self.quality_flags
            ],
            "quality_flags_count_by_tier": self.quality_flags_count_by_tier,
            "islands_removed": self.islands_removed,
            "mask_native_path": self.mask_native_path,
            "mask_final_native_path": self.mask_final_native_path,
            "mask_conservative_native_path": self.mask_conservative_native_path,
            "mask_gmm_bayes_native_path": self.mask_gmm_bayes_native_path,
            "mask_1_5mm_path": self.mask_1_5mm_path,
            "qa_report_path": self.qa_report_path,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "total_runtime_seconds": self.total_runtime_seconds,
        }

    def save_json(self, output_path: str) -> None:
        """Write serialized result metrics to JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)


# Backwards compatibility alias
PipelineResult = SegmentationResult


# ---------------------------------------------------------------------------
# Seam Functions
# ---------------------------------------------------------------------------


def load_and_resample_masks(
    intermediate_dir: str,
    patient_id: str,
    reference_geometry: GridGeometry,
) -> Dict[str, np.ndarray]:
    """Load canonical anatomical masks and resample them onto reference grid.

    Parameters
    ----------
    intermediate_dir:
        Directory containing TotalSegmentator mask files.
    patient_id:
        Canonical 4-digit patient ID.
    reference_geometry:
        Target 3D grid geometry (1.5mm isotropic CT).

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping from structure name to boolean NumPy array matching reference shape.
    """
    structures_to_load = list(CANONICAL_ANCHORS) + ["Pericardium", "Pulmonary_Veins"]
    loaded_masks: Dict[str, np.ndarray] = {}

    ref_dummy = reference_geometry.to_sitk_image(np.zeros(reference_geometry.shape_zyx, dtype=np.uint8))
    for name in structures_to_load:
        mask_path = resolve_ts_mask_path(intermediate_dir, patient_id, name)
        if mask_path and os.path.isfile(mask_path):
            try:
                res = resample_to_reference(
                    mask_path,
                    reference_or_path=ref_dummy,
                    is_label=True,
                )
                loaded_masks[name] = res.array.astype(bool)
                logger.debug("Ingested & aligned mask %s for %s", name, patient_id)
            except Exception as exc:
                logger.warning(
                    "Failed to resample mask %s for %s from %s: %s",
                    name, patient_id, mask_path, exc,
                )

    return loaded_masks


# ---------------------------------------------------------------------------
# Public Entry Point
# ---------------------------------------------------------------------------


def run_fat_extraction(
    patient_id: str,
    config: Optional[PipelineConfig] = None,
    config_path: Optional[str] = None,
    raw_ct_path: Optional[str] = None,
    mask_dir: Optional[str] = None,
    generate_qa: bool = True,
) -> SegmentationResult:
    """Execute complete end-to-end fat extraction pipeline for a single patient.

    Parameters
    ----------
    patient_id:
        Canonical 4-digit patient identifier.
    config:
        Optional pre-built PipelineConfig.
    config_path:
        Optional path to YAML configuration file.
    raw_ct_path:
        Optional direct path to raw CT NIfTI file.
    mask_dir:
        Optional direct directory containing pre-computed TS masks.
    generate_qa:
        Whether to generate the zero-footprint HTML5 QA Studio report.

    Returns
    -------
    SegmentationResult
        Immutable result containing all volumetric metrics, mask paths, and QA data.
    """
    start_time = time.perf_counter()
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Configuration Resolution
    if config is not None:
        cfg = config
    elif config_path is not None:
        cfg = PipelineConfig.from_yaml(config_path)
    else:
        cfg = PipelineConfig()

    spacing_mm = cfg.spacing_mm
    data_dir = cfg.data_dir
    output_dir = cfg.output_dir
    patient_output_dir = os.path.join(output_dir, patient_id)
    target_mask_dir = mask_dir or os.path.join(data_dir, cfg.intermediate_subdir, patient_id)
    raw_ct_dir = os.path.join(data_dir, cfg.raw_subdir)

    # 2. Raw CT Path Resolution
    if raw_ct_path is None or not os.path.isfile(raw_ct_path):
        candidate_paths = [
            os.path.join(raw_ct_dir, f"{patient_id}.nii.gz"),
            os.path.join(raw_ct_dir, f"{patient_id}.nii"),
            os.path.join(data_dir, f"{patient_id}.nii.gz"),
            os.path.join(data_dir, f"{patient_id}.nii"),
        ]
        resolved_ct_path = next((p for p in candidate_paths if os.path.isfile(p)), None)
        if resolved_ct_path is None:
            err = f"Raw CT volume not found for patient {patient_id} in {raw_ct_dir}"
            logger.error(err)
            errors.append(err)
            return _build_empty_result(patient_id, start_time, errors, warnings)
        raw_ct_path = resolved_ct_path

    # 3. Ingest Raw CT (Native Resolution Master Grid)
    try:
        raw_img = sitk.ReadImage(raw_ct_path)
        ref_geometry = GridGeometry.from_sitk_image(raw_img)
        raw_geometry = ref_geometry
        ct_array = sitk.GetArrayFromImage(raw_img).astype(np.float32)
        spacing = ref_geometry.spacing
        voxel_vol_ml = ref_geometry.voxel_volume_ml
        logger.info(
            "Loaded native CT %s (shape: %s, spacing: %s mm)",
            patient_id, ct_array.shape, spacing,
        )
    except Exception as exc:
        err = f"CT native ingestion failed: {exc}"
        logger.error(err)
        errors.append(err)
        return _build_empty_result(patient_id, start_time, errors, warnings)

    # 4. Ingest & Align TS Masks to Native Grid
    loaded_masks = load_and_resample_masks(target_mask_dir, patient_id, ref_geometry)
    if not loaded_masks:
        err = f"No anatomical masks found for {patient_id} in {target_mask_dir}"
        logger.error(err)
        errors.append(err)
        return _build_empty_result(patient_id, start_time, errors, warnings)

    anchor_masks = {
        name: loaded_masks[name]
        for name in CANONICAL_ANCHORS
        if name in loaded_masks
    }

    # 5. Resolve Pericardium
    ts_masks_for_resolver: Dict[str, np.ndarray] = dict(anchor_masks)
    if "Pericardium" in loaded_masks:
        ts_masks_for_resolver["pericardium"] = loaded_masks["Pericardium"]

    try:
        pericardium_result: PericardiumResult = resolve_pericardium(
            ts_masks_for_resolver, cfg, spacing,
        )
        pericardium_mask = pericardium_result.mask
        if pericardium_result.fallback_triggered:
            msg = f"Pericardium fallback triggered: {pericardium_result.fallback_reason}"
            warnings.append(msg)
            logger.warning(msg)
    except Exception as exc:
        err = f"Pericardium resolution failed: {exc}"
        logger.error(err)
        errors.append(err)
        return _build_empty_result(patient_id, start_time, errors, warnings)

    # 6. Adaptive Trimmed-Gaussian Fat Thresholding
    thresh_cfg = ThresholdConfig.from_pipeline_config(cfg)
    threshold_result: ThresholdResult = compute_fat_threshold(
        ct_volume=ct_array,
        pericardium_mask=pericardium_mask,
        geometry=ref_geometry,
        config=thresh_cfg,
    )
    if threshold_result.is_fallback:
        warnings.append("Gaussian fat threshold fit failed; used consensus fallback window")

    # 7. 3D Multi-Anchor Solid EDT Partition
    try:
        partition_result: PartitionResult = partition_fat(
            ct_array=ct_array,
            pericardium_mask=pericardium_mask,
            fat_hu_range=(threshold_result.hu_low, threshold_result.hu_high),
            anchor_masks=anchor_masks,
            config=cfg,
            geometry=ref_geometry,
        )
        if partition_result.excluded_anchors:
            warnings.append(f"Anchor(s) excluded: {', '.join(partition_result.excluded_anchors)}")
    except Exception as exc:
        err = f"Fat partition failed: {exc}"
        logger.error(err)
        errors.append(err)
        return _build_empty_result(patient_id, start_time, errors, warnings)

    # 8. Connected-Component Island Cleanup
    try:
        cleanup_result: CleanupResult = cleanup_la_fat_mask(
            partition_result.la_fat_mask, cfg, spacing,
            apply_opening=False,
            apply_vessel_filling=False,
        )
        cleaned_la_fat_mask = cleanup_result.cleaned_mask
    except Exception as exc:
        warnings.append(f"Cleanup failed ({exc}); using raw partitioned LA fat mask")
        cleanup_result = CleanupResult(
            cleaned_mask=partition_result.la_fat_mask,
            islands_removed=0,
            island_volumes_mm3=[],
            total_removed_volume_mm3=0.0,
            morphological_opening_applied=False,
            vessel_filling_applied=False,
        )
        cleaned_la_fat_mask = partition_result.la_fat_mask

    # 9. Compute Volumetric Metrics (Adaptive, Conservative, and GMM Bayes)
    la_fat_volume_adaptive_ml = float(np.sum(cleaned_la_fat_mask) * voxel_vol_ml)
    total_eat_volume_adaptive_ml = float(partition_result.total_fat_volume_ml)
    pericardium_volume_ml = float(pericardium_result.volume_ml)
    unassigned_vol = float(partition_result.unassigned_volume_ml)
    unassigned_pct = float((unassigned_vol / max(total_eat_volume_adaptive_ml, 0.001)) * 100.0)

    # Conservative standard window [-190, -30] HU
    conservative_fat_mask = (
        pericardium_mask & (ct_array >= -190.0) & (ct_array <= -30.0)
    )
    la_conservative_mask = conservative_fat_mask & (partition_result.anchor_assignments == 1)
    la_fat_volume_conservative_ml = float(np.sum(la_conservative_mask) * voxel_vol_ml)
    total_eat_volume_conservative_ml = float(np.sum(conservative_fat_mask) * voxel_vol_ml)

    # GMM Bayes decision boundary window
    gmm_low, gmm_high = threshold_result.gmm_bayes_window
    gmm_fat_mask = (
        pericardium_mask & (ct_array >= gmm_low) & (ct_array <= gmm_high)
    )
    la_gmm_mask = gmm_fat_mask & (partition_result.anchor_assignments == 1)
    la_fat_volume_gmm_bayes_ml = float(np.sum(la_gmm_mask) * voxel_vol_ml)
    total_eat_volume_gmm_bayes_ml = float(np.sum(gmm_fat_mask) * voxel_vol_ml)

    # 10. Tri-Track Radiomics Export
    os.makedirs(patient_output_dir, exist_ok=True)
    mask_native_path = os.path.join(patient_output_dir, f"{patient_id}_la_fat_native.nii.gz")
    mask_final_native_path = os.path.join(patient_output_dir, f"{patient_id}_la_fat_final_native.nii.gz")
    mask_conservative_native_path = os.path.join(patient_output_dir, f"{patient_id}_la_fat_conservative_native.nii.gz")
    mask_gmm_bayes_native_path = os.path.join(patient_output_dir, f"{patient_id}_la_fat_gmm_bayes_native.nii.gz")
    legacy_mask_path = os.path.join(patient_output_dir, "la_fat_mask.nii.gz")
    mask_1_5mm_path = mask_final_native_path

    try:
        # Save Native adaptive mask directly
        nifti_io.save_nifti(
            cleaned_la_fat_mask.astype(np.uint8),
            mask_native_path,
            spacing=ref_geometry.spacing,
            origin=ref_geometry.origin,
            direction=ref_geometry.direction,
        )
        nifti_io.save_nifti(
            cleaned_la_fat_mask.astype(np.uint8),
            mask_final_native_path,
            spacing=ref_geometry.spacing,
            origin=ref_geometry.origin,
            direction=ref_geometry.direction,
        )
        nifti_io.save_nifti(
            cleaned_la_fat_mask.astype(np.uint8),
            legacy_mask_path,
            spacing=ref_geometry.spacing,
            origin=ref_geometry.origin,
            direction=ref_geometry.direction,
        )

        # Save Native Conservative mask directly
        nifti_io.save_nifti(
            la_conservative_mask.astype(np.uint8),
            mask_conservative_native_path,
            spacing=ref_geometry.spacing,
            origin=ref_geometry.origin,
            direction=ref_geometry.direction,
        )

        # Save Native GMM Bayes mask directly
        nifti_io.save_nifti(
            la_gmm_mask.astype(np.uint8),
            mask_gmm_bayes_native_path,
            spacing=ref_geometry.spacing,
            origin=ref_geometry.origin,
            direction=ref_geometry.direction,
        )

        logger.info("Saved tri-track native masks: Adaptive, Conservative, GMM Bayes for %s", patient_id)
    except Exception as exc:
        msg = f"Failed to export NIfTI masks: {exc}"
        warnings.append(msg)
        logger.warning(msg)

    # 11. Quality Flagging
    quality_flags = generate_quality_flags(
        partition_result=partition_result,
        pericardium_result=pericardium_result,
        cleanup_result=cleanup_result,
        config=cfg,
    )
    if threshold_result.flags:
        quality_flags.extend(threshold_result.flags)

    tier_counts = {"high": 0, "medium": 0, "low": 0}
    for qf in quality_flags:
        sev = str(qf.severity).lower()
        if sev in tier_counts:
            tier_counts[sev] += 1

    # 12. Single-Patient QA HTML Report Generation
    qa_report_path: Optional[str] = None
    qa_record: Optional[Dict[str, Any]] = None

    if generate_qa:
        try:
            high_flags = [f for f in quality_flags if str(f.severity).lower() == "high"]
            med_flags = [f for f in quality_flags if str(f.severity).lower() in ("med", "medium")]
            low_flags = [f for f in quality_flags if str(f.severity).lower() == "low"]

            qa_metrics = {
                "patient_id": patient_id,
                "la_fat_volume_ml": la_fat_volume_adaptive_ml,
                "total_eat_volume_ml": total_eat_volume_adaptive_ml,
                "pericardium_volume_ml": pericardium_volume_ml,
                "la_conservative_volume_ml": la_fat_volume_conservative_ml,
                "eat_conservative_volume_ml": total_eat_volume_conservative_ml,
                "la_gmm_bayes_volume_ml": la_fat_volume_gmm_bayes_ml,
                "eat_gmm_bayes_volume_ml": total_eat_volume_gmm_bayes_ml,
                # UI and Cohort QA Viewer Aliases
                "la_vol_adaptive": la_fat_volume_adaptive_ml,
                "la_vol_std": la_fat_volume_conservative_ml,
                "la_vol_gmm_bayes": la_fat_volume_gmm_bayes_ml,
                "total_eat_vol": total_eat_volume_adaptive_ml,
                "total_eat_std": total_eat_volume_conservative_ml,
                "total_eat_gmm_bayes": total_eat_volume_gmm_bayes_ml,
                "fitted_mu_hu": threshold_result.fitted_mu,
                "fitted_sigma_hu": threshold_result.fitted_sigma,
                "high_flags": len(high_flags),
                "med_flags": len(med_flags),
                "low_flags": len(low_flags),
                "primary_component_purity": partition_result.metrics.primary_component_fraction if partition_result and partition_result.metrics else 1.0,
                "gaussian_fit": {
                    "success": not threshold_result.is_fallback,
                    "mu": threshold_result.fitted_mu,
                    "sigma": threshold_result.fitted_sigma,
                    "low_hu": threshold_result.hu_low,
                    "high_hu": threshold_result.hu_high,
                },
                "quality_flags": [
                    {
                        "severity": f.severity,
                        "concern": f.concern or getattr(f, "flag_id", ""),
                        "detail": f.detail or getattr(f, "message", ""),
                    }
                    for f in quality_flags
                ],
            }
            qa_record = extract_patient_qa_record(
                patient_id=patient_id,
                ct_volume=ct_array,
                pericardium_mask=pericardium_mask,
                anchor_masks=anchor_masks,
                partition_assignments=partition_result.anchor_assignments,
                la_fat_mask=cleaned_la_fat_mask,
                metrics=qa_metrics,
                spacing=ref_geometry.spacing,
            )
            qa_report_path = os.path.join(patient_output_dir, "qa_report.html")
            generate_cohort_qa_html({patient_id: qa_record}, qa_report_path)
            logger.info("Generated standalone QA Studio report at: %s", qa_report_path)
        except Exception as exc:
            msg = f"Failed to generate QA report: {exc}"
            warnings.append(msg)
            logger.warning(msg)

    # 13. Construct and Save SegmentationResult
    gmm_mu = threshold_result.gmm_bayes_result.fitted_mu_fat if threshold_result.gmm_bayes_result else None
    gmm_sigma = threshold_result.gmm_bayes_result.fitted_sigma_fat if threshold_result.gmm_bayes_result else None
    gmm_wt = threshold_result.gmm_bayes_result.weight_fat if threshold_result.gmm_bayes_result else None

    total_runtime = time.perf_counter() - start_time
    result = SegmentationResult(
        patient_id=patient_id,
        success=len(errors) == 0,
        la_fat_volume_adaptive_ml=la_fat_volume_adaptive_ml,
        la_fat_volume_conservative_ml=la_fat_volume_conservative_ml,
        la_fat_volume_gmm_bayes_ml=la_fat_volume_gmm_bayes_ml,
        total_eat_volume_adaptive_ml=total_eat_volume_adaptive_ml,
        total_eat_volume_conservative_ml=total_eat_volume_conservative_ml,
        total_eat_volume_gmm_bayes_ml=total_eat_volume_gmm_bayes_ml,
        pericardium_volume_ml=pericardium_volume_ml,
        unassigned_volume_ml=unassigned_vol,
        unassigned_fat_pct=unassigned_pct,
        anchor_volumes_ml=partition_result.anchor_volumes_ml,
        fat_hu_range_adaptive=(threshold_result.hu_low, threshold_result.hu_high),
        fat_hu_range_conservative=(threshold_result.conservative_hu_low, threshold_result.conservative_hu_high),
        fat_hu_range_gmm_bayes=threshold_result.gmm_bayes_window,
        gaussian_fit_mu=threshold_result.fitted_mu,
        gaussian_fit_sigma=threshold_result.fitted_sigma,
        gaussian_fit_success=not threshold_result.is_fallback,
        gmm_bayes_mu_fat=gmm_mu,
        gmm_bayes_sigma_fat=gmm_sigma,
        gmm_bayes_weight_fat=gmm_wt,
        quality_flags=quality_flags,
        quality_flags_count_by_tier=tier_counts,
        islands_removed=cleanup_result.islands_removed,
        total_removed_volume_mm3=cleanup_result.total_removed_volume_mm3,
        mask_1_5mm_path=mask_1_5mm_path,
        mask_native_path=mask_native_path,
        mask_final_native_path=mask_final_native_path,
        mask_conservative_native_path=mask_conservative_native_path,
        mask_gmm_bayes_native_path=mask_gmm_bayes_native_path,
        qa_report_path=qa_report_path,
        qa_record=qa_record,
        errors=errors,
        warnings=warnings,
        total_runtime_seconds=total_runtime,
        partition_result=partition_result,
        pericardium_result=pericardium_result,
        cleanup_result=cleanup_result,
    )

    result_json_path = os.path.join(patient_output_dir, "pipeline_result.json")
    flags_json_path = os.path.join(patient_output_dir, "quality_flags.json")
    try:
        result.save_json(result_json_path)
        flags_data = [
            {
                "severity": f.severity,
                "concern": f.concern or getattr(f, "flag_id", ""),
                "detail": f.detail or getattr(f, "message", ""),
                "threshold_value": f.threshold_value,
                "actual_value": f.actual_value,
            }
            for f in quality_flags
        ]
        with open(flags_json_path, "w", encoding="utf-8") as fh:
            json.dump(flags_data, fh, indent=2)
    except Exception as exc:
        logger.warning("Failed to save result/flag JSON files: %s", exc)

    logger.info(
        "Extraction finished for %s (LA fat: %.2f mL, EAT: %.2f mL, %.1fs)",
        patient_id, la_fat_volume_adaptive_ml, total_eat_volume_adaptive_ml, total_runtime,
    )
    return result


# ---------------------------------------------------------------------------
# Backward Compatibility Alias
# ---------------------------------------------------------------------------

run_fat_extraction_pipeline = run_fat_extraction


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _build_empty_result(
    patient_id: str,
    start_time: float,
    errors: List[str],
    warnings: List[str],
) -> SegmentationResult:
    """Construct a failed SegmentationResult when fatal error occurs early."""
    return SegmentationResult(
        patient_id=patient_id,
        success=False,
        la_fat_volume_adaptive_ml=0.0,
        la_fat_volume_conservative_ml=0.0,
        total_eat_volume_adaptive_ml=0.0,
        total_eat_volume_conservative_ml=0.0,
        pericardium_volume_ml=0.0,
        unassigned_volume_ml=0.0,
        unassigned_fat_pct=0.0,
        anchor_volumes_ml={},
        fat_hu_range_adaptive=(-190.0, -30.0),
        fat_hu_range_conservative=(-190.0, -30.0),
        gaussian_fit_mu=None,
        gaussian_fit_sigma=None,
        gaussian_fit_success=False,
        quality_flags=[],
        quality_flags_count_by_tier={"high": 0, "medium": 0, "low": 0},
        islands_removed=0,
        total_removed_volume_mm3=0.0,
        mask_native_path=None,
        mask_final_native_path=None,
        mask_conservative_native_path=None,
        mask_gmm_bayes_native_path=None,
        mask_1_5mm_path=None,
        qa_report_path=None,
        qa_record=None,
        errors=errors,
        warnings=warnings,
        total_runtime_seconds=time.perf_counter() - start_time,
    )
