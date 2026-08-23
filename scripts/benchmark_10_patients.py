"""10 Real Patient Cohort Benchmark & QA Generation Script.

Part of Wayfinder Ticket 9 (Issue #39).
Executes the rebuilt LA Fat segmentation pipeline across real patient scans,
extracts dual-window volumes, computes correlation against clinical scanner
software measurements, and generates the standalone HTML5 cohort QA dashboard.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import stats

from la_fat import nifti_io
from la_fat.anatomy import CANONICAL_ANCHORS, voxel_volume_ml
from la_fat.cleanup import cleanup_la_fat_mask
from la_fat.cohort_qa_generator import extract_patient_qa_record, generate_cohort_qa_html
from la_fat.config import PipelineConfig
from la_fat.image_ops import GridGeometry, resample_to_isotropic, resample_to_reference
from la_fat.partition_engine import partition_fat, PartitionConfig
from la_fat.pericardium_resolver import resolve_pericardium
from la_fat.quality_flagger import generate_quality_flags
from la_fat.thresholding import compute_fat_threshold, ThresholdConfig
from la_fat.ts_runner import resolve_ts_mask_path


# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------

DATA_DIR = r"C:\Users\marmo\Downloads\ctscans"
LEGACY_INTERMEDIATE_DIR = r"C:\Users\marmo\Downloads\la_eat_segmentation\data\intermediate"
MANIFEST_PATH = "data/cohort_manifest.json"
OUTPUT_DIR = "data/outputs"


def load_manifest() -> Dict[str, Any]:
    """Load canonical patient metadata and clinical scanner ground truth."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_benchmark_for_patient(
    patient_id: str,
    meta: Dict[str, Any],
    config: PipelineConfig,
) -> Optional[Dict[str, Any]]:
    """Execute complete segmentation pipeline for a single patient."""
    print(f"\n=======================================================")
    print(f"[*] Processing Patient {patient_id} ({meta['age']}y, {meta['sex'].upper()})")
    print(f"=======================================================")

    # 1. Resolve Paths
    raw_ct_path = os.path.join(DATA_DIR, f"{patient_id}.nii.gz")
    if not os.path.isfile(raw_ct_path):
        raw_ct_path = os.path.join(DATA_DIR, f"{patient_id}.nii")
    if not os.path.isfile(raw_ct_path):
        print(f"[-] Raw CT not found for {patient_id} at: {raw_ct_path}")
        return None

    # Check mask cache directory
    target_mask_dir = os.path.join(DATA_DIR, "masks", patient_id)
    if not os.path.isdir(target_mask_dir):
        legacy_dir = os.path.join(LEGACY_INTERMEDIATE_DIR, patient_id)
        if os.path.isdir(legacy_dir):
            target_mask_dir = legacy_dir
        else:
            print(f"[-] Anatomical mask cache not found for {patient_id}. Needs TS inference.")
            return None

    print(f"[+] Using mask cache: {target_mask_dir}")

    # 2. Resample Raw CT to 1.5mm Isotropic Reference Grid
    resample_result = resample_to_isotropic(raw_ct_path, target_spacing_mm=1.5, is_label=False)
    ct_array = resample_result.array # (Z, Y, X)
    geo_1_5mm = resample_result.geometry
    spacing = geo_1_5mm.spacing
    voxel_vol_ml = voxel_volume_ml(spacing)

    # 3. Load & Resample Canonical Masks onto Reference 1.5mm Grid
    loaded_masks: Dict[str, np.ndarray] = {}
    for anchor in CANONICAL_ANCHORS:
        mask_path = resolve_ts_mask_path(target_mask_dir, patient_id, anchor)
        if mask_path and os.path.isfile(mask_path):
            mask_resample = resample_to_reference(
                mask_path,
                reference_or_path=ct_array,
                reference_geometry=geo_1_5mm,
                is_label=True,
            )
            loaded_masks[anchor] = mask_resample.array.astype(bool)
        else:
            print(f"[-] Missing mask for {anchor} in {target_mask_dir}")

    if "LA" not in loaded_masks:
        print(f"[-] Fatal: Left Atrium mask missing for {patient_id}. Skipping.")
        return None

    # 4. Resolve Pericardium
    peri_path = resolve_ts_mask_path(target_mask_dir, patient_id, "Pericardium")
    ts_peri_mask = None
    if peri_path and os.path.isfile(peri_path):
        peri_resample = resample_to_reference(
            peri_path,
            reference_or_path=ct_array,
            reference_geometry=geo_1_5mm,
            is_label=True,
        )
        ts_peri_mask = peri_resample.array.astype(bool)

    ts_resolver_dict = dict(loaded_masks)
    if ts_peri_mask is not None:
        ts_resolver_dict["pericardium"] = ts_peri_mask

    peri_result = resolve_pericardium(ts_resolver_dict, config, spacing)
    print(f"[+] Pericardium resolved: {peri_result.volume_ml:.2f} mL (Method: {peri_result.method})")

    # 5. Trimmed-Gaussian Fat Thresholding
    thresh_cfg = ThresholdConfig.from_pipeline_config(config)
    thresh_result = compute_fat_threshold(
        ct_volume=ct_array,
        pericardium_mask=peri_result.mask,
        geometry=geo_1_5mm,
        config=thresh_cfg,
    )

    if thresh_result.fitted_mu is not None:
        print(f"[+] Gaussian Fit: mu={thresh_result.fitted_mu:.1f} HU, sigma={thresh_result.fitted_sigma:.1f} HU, Window=[{thresh_result.hu_low:.1f}, {thresh_result.hu_high:.1f}] HU")
    else:
        print(f"[+] Gaussian Fit: mu=Fallback, sigma=Fallback, Window=[{thresh_result.hu_low:.1f}, {thresh_result.hu_high:.1f}] HU")

    # 6. EAT Extraction
    print(f"[+] Adaptive EAT Volume: {thresh_result.fat_volume_adaptive_ml:.2f} mL (Conservative: {thresh_result.fat_volume_conservative_ml:.2f} mL)")

    # 7. Multi-Anchor Solid EDT Partition
    part_cfg = PartitionConfig.from_pipeline_config(config)
    anchor_masks = {k: loaded_masks[k] for k in CANONICAL_ANCHORS if k in loaded_masks}
    part_result = partition_fat(
        ct_array=ct_array,
        pericardium_mask=peri_result.mask,
        fat_hu_range=(thresh_result.hu_low, thresh_result.hu_high),
        anchor_masks=anchor_masks,
        config=part_cfg,
        geometry=geo_1_5mm,
    )

    # 8. Cleanup & Topology
    cleanup_result = cleanup_la_fat_mask(part_result.la_fat_mask, config, spacing)
    cleaned_la_fat = cleanup_result.cleaned_mask

    # Dual-window volume quantification for LA
    la_fat_voxels_adaptive = int(np.sum(cleaned_la_fat))
    la_fat_vol_adaptive_ml = la_fat_voxels_adaptive * voxel_vol_ml

    # Conservative LA Fat (within standard [-190, -30] HU)
    cons_fat_mask = (ct_array >= -190.0) & (ct_array <= -30.0) & peri_result.mask
    cons_part_result = partition_fat(
        pericardium_mask=peri_result.mask,
        fat_mask=cons_fat_mask,
        anchor_masks=anchor_masks,
        config=part_cfg,
        geometry=geo_1_5mm,
    )
    cons_cleanup = cleanup_la_fat_mask(cons_part_result.la_fat_mask, config, spacing)
    la_fat_vol_conservative_ml = int(np.sum(cons_cleanup.cleaned_mask)) * voxel_vol_ml

    # GMM Bayes LA Fat (P(Fat|x) >= 0.5 decision boundary)
    gmm_low, gmm_high = thresh_result.gmm_bayes_window
    gmm_fat_mask = (ct_array >= gmm_low) & (ct_array <= gmm_high) & peri_result.mask
    gmm_part_result = partition_fat(
        pericardium_mask=peri_result.mask,
        fat_mask=gmm_fat_mask,
        anchor_masks=anchor_masks,
        config=part_cfg,
        geometry=geo_1_5mm,
    )
    gmm_cleanup = cleanup_la_fat_mask(gmm_part_result.la_fat_mask, config, spacing)
    la_fat_vol_gmm_bayes_ml = int(np.sum(gmm_cleanup.cleaned_mask)) * voxel_vol_ml

    # 9. Quality Flags
    flags = generate_quality_flags(
        partition_result=part_result,
        pericardium_result=peri_result,
        cleanup_result=cleanup_result,
        config=config,
    )
    if thresh_result.flags:
        flags.extend(thresh_result.flags)
    high_flags = [f for f in flags if f.severity == "HIGH" or f.severity == "high"]
    med_flags = [f for f in flags if f.severity == "MEDIUM" or f.severity == "medium"]
    low_flags = [f for f in flags if f.severity == "LOW" or f.severity == "low"]

    print(f"[+] LA Fat (Adaptive): {la_fat_vol_adaptive_ml:.2f} mL vs Scanner Baseline: {meta['scanner_la_eat_ml']:.2f} mL")
    print(f"[+] LA Fat (GMM Bayes): {la_fat_vol_gmm_bayes_ml:.2f} mL (Window: [{gmm_low:.1f}, {gmm_high:.1f}] HU)")
    print(f"[+] LA Fat (Conservative): {la_fat_vol_conservative_ml:.2f} mL")
    print(f"[+] Topological Purity: {part_result.metrics.primary_component_fraction * 100:.1f}%")

    # 10. Save Output NIfTI Masks
    patient_out_dir = os.path.join(OUTPUT_DIR, patient_id)
    os.makedirs(patient_out_dir, exist_ok=True)

    # 1.5mm mask
    nifti_io.save_nifti(
        cleaned_la_fat.astype(np.uint8),
        os.path.join(patient_out_dir, "la_fat_mask.nii.gz"),
        spacing=geo_1_5mm.spacing,
        origin=geo_1_5mm.origin,
        direction=geo_1_5mm.direction,
    )

    # Native resolution projection & save
    raw_img = sitk.ReadImage(raw_ct_path)
    raw_geo = GridGeometry.from_sitk_image(raw_img)

    # Native Adaptive Mask
    native_resample = resample_to_reference(
        cleaned_la_fat.astype(np.uint8),
        raw_img,
        is_label=True,
        moving_geometry=geo_1_5mm,
        reference_geometry=raw_geo,
    )
    nifti_io.save_nifti(
        native_resample.array.astype(np.uint8),
        os.path.join(patient_out_dir, "la_fat_final_native.nii.gz"),
        spacing=raw_geo.spacing,
        origin=raw_geo.origin,
        direction=raw_geo.direction,
    )

    # Native Conservative Mask
    native_cons_resample = resample_to_reference(
        cons_cleanup.cleaned_mask.astype(np.uint8),
        raw_img,
        is_label=True,
        moving_geometry=geo_1_5mm,
        reference_geometry=raw_geo,
    )
    nifti_io.save_nifti(
        native_cons_resample.array.astype(np.uint8),
        os.path.join(patient_out_dir, "la_fat_conservative_native.nii.gz"),
        spacing=raw_geo.spacing,
        origin=raw_geo.origin,
        direction=raw_geo.direction,
    )

    # Native GMM Bayes Mask
    native_gmm_resample = resample_to_reference(
        gmm_cleanup.cleaned_mask.astype(np.uint8),
        raw_img,
        is_label=True,
        moving_geometry=geo_1_5mm,
        reference_geometry=raw_geo,
    )
    nifti_io.save_nifti(
        native_gmm_resample.array.astype(np.uint8),
        os.path.join(patient_out_dir, "la_fat_gmm_bayes_native.nii.gz"),
        spacing=raw_geo.spacing,
        origin=raw_geo.origin,
        direction=raw_geo.direction,
    )

    print(f"[+] Saved tri-track native radiomics masks for {patient_id}")

    # 11. Extract Slices and Metrics for QA Dashboard
    delta_la = la_fat_vol_adaptive_ml - meta["scanner_la_eat_ml"]
    delta_pct = (delta_la / meta["scanner_la_eat_ml"]) * 100.0

    metrics_dict = {
        "age": meta["age"],
        "sex": meta["sex"],
        "smoker": meta["smoker"],
        "diabetes": meta["diabetes"],
        "hypertension": meta["hypertension"],
        "dyslipidemia": meta["dyslipidemia"],
        "chads2": meta["chads2"],
        "control": meta["control"],
        "scanner_la_eat_ml": meta["scanner_la_eat_ml"],
        "scanner_total_eat_ml": meta["scanner_total_eat_ml"],
        "la_vol_adaptive": la_fat_vol_adaptive_ml,
        "la_vol_gmm_bayes": la_fat_vol_gmm_bayes_ml,
        "la_vol_std": la_fat_vol_conservative_ml,
        "total_eat_vol": thresh_result.fat_volume_adaptive_ml,
        "total_eat_gmm_bayes": thresh_result.fat_volume_gmm_bayes_ml,
        "total_eat_std": thresh_result.fat_volume_conservative_ml,
        "delta_la_adaptive_ml": delta_la,
        "delta_la_adaptive_pct": delta_pct,
        "fitted_mu_hu": thresh_result.fitted_mu,
        "fitted_sigma_hu": thresh_result.fitted_sigma,
        "gmm_bayes_low": gmm_low,
        "gmm_bayes_high": gmm_high,
        "primary_component_purity": part_result.metrics.primary_component_fraction if part_result.metrics else 1.0,
        "high_flags": len(high_flags),
        "med_flags": len(med_flags),
        "low_flags": len(low_flags),
    }

    qa_record = extract_patient_qa_record(
        patient_id=patient_id,
        ct_volume=ct_array,
        la_fat_mask=cleaned_la_fat,
        pericardium_mask=peri_result.mask,
        anchor_masks=anchor_masks,
        partition_assignments=part_result.anchor_assignments,
        metrics=metrics_dict,
    )

    return qa_record


def main() -> None:
    """Run cohort benchmark on available scans and compile results."""
    manifest = load_manifest()
    config = PipelineConfig()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results: List[Dict[str, Any]] = []

    print(f"Starting Cohort Benchmark for {len(manifest)} Patients...")
    for pid, meta in manifest.items():
        res = run_benchmark_for_patient(pid, meta, config)
        if res is not None:
            results.append(res)

    if not results:
        print("[-] No patient scans could be processed. Exiting.")
        return

    print(f"\n[+] Successfully processed {len(results)}/{len(manifest)} patient scans!")

    # 1. Compile Summary DataFrame
    df_rows = []
    cohort_payload = {}
    for r in results:
        m = r["metrics"]
        cohort_payload[r["id"]] = r
        df_rows.append({
            "patient_id": r["id"],
            "age": m["age"],
            "sex": m["sex"],
            "scanner_la_eat_ml": m["scanner_la_eat_ml"],
            "pipeline_la_adaptive_ml": m["la_vol_adaptive"],
            "pipeline_la_gmm_bayes_ml": m["la_vol_gmm_bayes"],
            "pipeline_la_conservative_ml": m["la_vol_std"],
            "delta_la_adaptive_ml": m["delta_la_adaptive_ml"],
            "delta_la_adaptive_pct": m["delta_la_adaptive_pct"],
            "scanner_total_eat_ml": m["scanner_total_eat_ml"],
            "pipeline_total_adaptive_ml": m["total_eat_vol"],
            "pipeline_total_gmm_bayes_ml": m["total_eat_gmm_bayes"],
            "pipeline_total_conservative_ml": m["total_eat_std"],
            "primary_component_purity": m["primary_component_purity"],
            "fitted_mu_hu": m["fitted_mu_hu"],
            "fitted_sigma_hu": m["fitted_sigma_hu"],
            "gmm_bayes_low": m["gmm_bayes_low"],
            "gmm_bayes_high": m["gmm_bayes_high"],
            "high_flags": m["high_flags"],
            "med_flags": m["med_flags"],
            "low_flags": m["low_flags"],
        })

    summary_df = pd.DataFrame(df_rows)
    csv_path = os.path.join(OUTPUT_DIR, "cohort_benchmark_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"[+] Saved summary CSV to: {csv_path}")

    # 2. Compute Correlation Metrics across all 3 tracks
    x = summary_df["scanner_la_eat_ml"].values
    y_adapt = summary_df["pipeline_la_adaptive_ml"].values
    y_gmm = summary_df["pipeline_la_gmm_bayes_ml"].values
    y_cons = summary_df["pipeline_la_conservative_ml"].values

    if len(x) > 1:
        r_adapt, p_adapt = stats.pearsonr(x, y_adapt)
        rho_adapt, p_rho_adapt = stats.spearmanr(x, y_adapt)
        mape_adapt = np.mean(np.abs((y_adapt - x) / x)) * 100.0

        r_gmm, p_gmm = stats.pearsonr(x, y_gmm)
        rho_gmm, p_rho_gmm = stats.spearmanr(x, y_gmm)
        mape_gmm = np.mean(np.abs((y_gmm - x) / x)) * 100.0

        r_cons, p_cons = stats.pearsonr(x, y_cons)
        rho_cons, p_rho_cons = stats.spearmanr(x, y_cons)
        mape_cons = np.mean(np.abs((y_cons - x) / x)) * 100.0

        print(f"\n=======================================================")
        print(f"COHORT CORRELATION RESULTS (N = {len(x)})")
        print(f"=======================================================")
        print(f"1. Adaptive Trimmed Gaussian: r = {r_adapt:.4f} (p = {p_adapt:.2e}), rho = {rho_adapt:.4f}, MAPE = {mape_adapt:.2f}%")
        print(f"2. GMM Bayes (P >= 0.5):      r = {r_gmm:.4f} (p = {p_gmm:.2e}), rho = {rho_gmm:.4f}, MAPE = {mape_gmm:.2f}%")
        print(f"3. Conservative [-190,-30]:  r = {r_cons:.4f} (p = {p_cons:.2e}), rho = {rho_cons:.4f}, MAPE = {mape_cons:.2f}%")
        print(f"=======================================================")

    # 3. Generate HTML5 Cohort Viewer
    html_path = os.path.join(OUTPUT_DIR, "cohort_qa_viewer.html")
    generate_cohort_qa_html(cohort_payload, html_path)
    print(f"[+] Saved interactive HTML5 Cohort QA Viewer to: {html_path}")


if __name__ == "__main__":
    main()
