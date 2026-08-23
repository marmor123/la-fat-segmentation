"""Native-Grid Density Modeling & Bayesian Prior Thresholding Benchmark.

Wayfinder Ticket 12 (Issue #42).
Investigates density distribution modeling techniques directly on native-resolution
(512x512xN, ~0.35mm) non-contrast cardiac CT scans:
1. Baseline: Trimmed Gaussian Mode Fit (1.5mm vs Native 512x512).
2. Method 2: 2-Component EM Gaussian Mixture Model (Bayes Boundary & Gaussian Tail).
3. Method 3: Non-Parametric Kernel Density Estimation (KDE Anti-Mode Valley).
4. Method 4: Bayesian MAP Prior Regularization (Empirical Bayes).

Computes volume correlation against clinical scanner baseline and generates
comparative diagnostic plots and summary tables across all 10 cohort scans.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage
import scipy.optimize
import scipy.signal
import scipy.stats
import SimpleITK as sitk
from sklearn.mixture import GaussianMixture

from la_fat import nifti_io
from la_fat.anatomy import CANONICAL_ANCHORS, voxel_volume_ml
from la_fat.cleanup import cleanup_la_fat_mask
from la_fat.config import PipelineConfig
from la_fat.image_ops import GridGeometry, resample_to_isotropic, resample_to_reference
from la_fat.partition_engine import partition_fat, PartitionConfig
from la_fat.pericardium_resolver import resolve_pericardium
from la_fat.thresholding import fit_trimmed_gaussian, ThresholdConfig, ThresholdResult
from la_fat.ts_runner import resolve_ts_mask_path

DATA_DIR = r"C:\Users\marmo\Downloads\ctscans"
LEGACY_INTERMEDIATE_DIR = r"C:\Users\marmo\Downloads\la_eat_segmentation\data\intermediate"
MANIFEST_PATH = "data/cohort_manifest.json"
RESEARCH_OUTPUT_DIR = "docs/research"


@dataclass
class MethodResult:
    name: str
    hu_low: float
    hu_high: float
    mu: Optional[float]
    sigma: Optional[float]
    converged: bool
    fallback: bool
    eat_vol_ml: float
    la_eat_vol_ml: float
    details: Dict[str, Any]


# ---------------------------------------------------------------------------
# Modeling Algorithms
# ---------------------------------------------------------------------------

def fit_gmm_2component(
    voxels: np.ndarray,
    clamping_max_hu: float = 0.0,
    fallback_low: float = -190.0,
    fallback_high: float = -30.0,
) -> Tuple[MethodResult, MethodResult]:
    """Fit 2-component Gaussian Mixture Model (Adipose + Soft-Tissue Shoulder)."""
    v = voxels[(voxels >= -250.0) & (voxels <= 0.0)]
    if len(v) < 500:
        res_bayes = MethodResult(
            name="GMM_Bayes_P0.5",
            hu_low=fallback_low,
            hu_high=fallback_high,
            mu=None,
            sigma=None,
            converged=False,
            fallback=True,
            eat_vol_ml=0.0,
            la_eat_vol_ml=0.0,
            details={"reason": "Insufficient voxels"},
        )
        res_tail = MethodResult(
            name="GMM_Gaussian_Tail",
            hu_low=fallback_low,
            hu_high=fallback_high,
            mu=None,
            sigma=None,
            converged=False,
            fallback=True,
            eat_vol_ml=0.0,
            la_eat_vol_ml=0.0,
            details={"reason": "Insufficient voxels"},
        )
        return res_bayes, res_tail

    # Subsample if too huge for fast EM, or use full dataset
    sample = v if len(v) <= 100000 else np.random.choice(v, size=100000, replace=False)
    X = sample.reshape(-1, 1)

    # Initial means: -95 HU (fat) and -25 HU (partial volume / soft tissue)
    init_means = np.array([[-95.0], [-25.0]])
    gmm = GaussianMixture(
        n_components=2,
        means_init=init_means,
        covariance_type="full",
        max_iter=200,
        random_state=42,
    )
    gmm.fit(X)

    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    weights = gmm.weights_.flatten()

    # Identify fat component (lower mean)
    fat_idx = int(np.argmin(means))
    soft_idx = int(np.argmax(means))

    mu_fat, sigma_fat = float(means[fat_idx]), float(stds[fat_idx])
    mu_soft, sigma_soft = float(means[soft_idx]), float(stds[soft_idx])
    w_fat, w_soft = float(weights[fat_idx]), float(weights[soft_idx])

    # 1. Bayes decision boundary P(Fat|x) == 0.5
    grid_hu = np.linspace(-200.0, 0.0, 2001)
    probs = gmm.predict_proba(grid_hu.reshape(-1, 1))
    fat_probs = probs[:, fat_idx]

    # Find crossing where fat_prob drops below 0.5 between mu_fat and 0
    valid_crossings = grid_hu[(grid_hu > mu_fat) & (fat_probs < 0.5)]
    if len(valid_crossings) > 0:
        bayes_high = float(valid_crossings[0])
    else:
        bayes_high = min(mu_fat + 2.0 * sigma_fat, clamping_max_hu)

    bayes_low = max(mu_fat - 2.0 * sigma_fat, fallback_low)
    bayes_high = min(bayes_high, clamping_max_hu)

    # 2. Gaussian Tail Formulation: mu_fat + 2*sigma_fat clamped at clamping_max_hu
    tail_low = max(mu_fat - 2.0 * sigma_fat, fallback_low)
    tail_high = min(mu_fat + 2.0 * sigma_fat, clamping_max_hu)

    details = {
        "mu_fat": mu_fat,
        "sigma_fat": sigma_fat,
        "w_fat": w_fat,
        "mu_soft": mu_soft,
        "sigma_soft": sigma_soft,
        "w_soft": w_soft,
    }

    res_bayes = MethodResult(
        name="GMM_Bayes_P0.5",
        hu_low=bayes_low,
        hu_high=bayes_high,
        mu=mu_fat,
        sigma=sigma_fat,
        converged=True,
        fallback=False,
        eat_vol_ml=0.0,
        la_eat_vol_ml=0.0,
        details=details,
    )
    res_tail = MethodResult(
        name="GMM_Gaussian_Tail",
        hu_low=tail_low,
        hu_high=tail_high,
        mu=mu_fat,
        sigma=sigma_fat,
        converged=True,
        fallback=False,
        eat_vol_ml=0.0,
        la_eat_vol_ml=0.0,
        details=details,
    )
    return res_bayes, res_tail


def fit_kde_valley(
    voxels: np.ndarray,
    clamping_max_hu: float = 0.0,
    fallback_low: float = -190.0,
    fallback_high: float = -30.0,
) -> MethodResult:
    """Non-parametric Kernel Density Estimation with Anti-Mode (Valley) Detection."""
    v = voxels[(voxels >= -250.0) & (voxels <= 0.0)]
    if len(v) < 500:
        return MethodResult(
            name="KDE_Valley",
            hu_low=fallback_low,
            hu_high=fallback_high,
            mu=None,
            sigma=None,
            converged=False,
            fallback=True,
            eat_vol_ml=0.0,
            la_eat_vol_ml=0.0,
            details={"reason": "Insufficient voxels"},
        )

    sample = v if len(v) <= 50000 else np.random.choice(v, size=50000, replace=False)
    kde = scipy.stats.gaussian_kde(sample, bw_method="silverman")

    x_grid = np.linspace(-220.0, 0.0, 440)
    density = kde.evaluate(x_grid)

    # Detect peaks (modes) and valleys (anti-modes)
    peaks, _ = scipy.signal.find_peaks(density, distance=15)
    valleys, _ = scipy.signal.find_peaks(-density, distance=15)

    fat_peaks = [p for p in peaks if -140.0 <= x_grid[p] <= -60.0]

    if not fat_peaks:
        # Fallback
        return MethodResult(
            name="KDE_Valley",
            hu_low=fallback_low,
            hu_high=fallback_high,
            mu=None,
            sigma=None,
            converged=False,
            fallback=True,
            eat_vol_ml=0.0,
            la_eat_vol_ml=0.0,
            details={"reason": "No prominent fat peak detected in KDE"},
        )

    best_fat_peak = fat_peaks[np.argmax(density[fat_peaks])]
    fat_mode_hu = float(x_grid[best_fat_peak])

    # Find valley between fat mode and soft-tissue region (> fat_mode_hu)
    right_valleys = [v for v in valleys if x_grid[v] > fat_mode_hu]

    if right_valleys:
        valley_hu = float(x_grid[right_valleys[0]])
        hu_high = min(valley_hu, clamping_max_hu)
    else:
        hu_high = min(fat_mode_hu + 50.0, clamping_max_hu)

    hu_low = max(fat_mode_hu - 60.0, fallback_low)

    return MethodResult(
        name="KDE_Valley",
        hu_low=hu_low,
        hu_high=hu_high,
        mu=fat_mode_hu,
        sigma=None,
        converged=True,
        fallback=False,
        eat_vol_ml=0.0,
        la_eat_vol_ml=0.0,
        details={"fat_mode_hu": fat_mode_hu, "valleys": [float(x_grid[v]) for v in valleys]},
    )


def fit_bayesian_map_regularization(
    voxels: np.ndarray,
    prior_mu: float = -85.3,
    prior_sigma: float = 32.6,
    prior_weight: float = 1500.0,
    clamping_max_hu: float = 0.0,
    fallback_low: float = -190.0,
    fallback_high: float = -30.0,
) -> MethodResult:
    """Bayesian Maximum A Posteriori (MAP) with Cohort Normal-Inverse-Gamma Prior."""
    v = voxels[(voxels >= -250.0) & (voxels <= 0.0)]
    n = len(v)
    if n == 0:
        return MethodResult(
            name="Bayesian_MAP",
            hu_low=fallback_low,
            hu_high=fallback_high,
            mu=prior_mu,
            sigma=prior_sigma,
            converged=False,
            fallback=True,
            eat_vol_ml=0.0,
            la_eat_vol_ml=0.0,
            details={"reason": "Empty voxels"},
        )

    # Focus on sub-0 candidate fat range for sample stats
    sample_v = v[(v >= -180.0) & (v <= -10.0)]
    if len(sample_v) > 20:
        sample_mean = float(np.mean(sample_v))
        sample_var = float(np.var(sample_v))
        n_eff = len(sample_v)
    else:
        sample_mean = prior_mu
        sample_var = prior_sigma ** 2
        n_eff = 0

    # Conjugate Normal-Inverse-Gamma MAP update
    # mu_map = (k0 * mu0 + n * x_bar) / (k0 + n)
    k0 = prior_weight
    mu_map = (k0 * prior_mu + n_eff * sample_mean) / (k0 + n_eff)

    # Variance MAP shrinkage
    alpha0 = 10.0
    beta0 = alpha0 * (prior_sigma ** 2)
    alpha_post = alpha0 + n_eff / 2.0
    beta_post = beta0 + 0.5 * np.sum((sample_v - sample_mean) ** 2) + (
        (k0 * n_eff * (sample_mean - prior_mu) ** 2) / (2.0 * (k0 + n_eff))
    )
    sigma_map = float(np.sqrt(beta_post / (alpha_post + 1.5)))

    hu_low = max(mu_map - 2.0 * sigma_map, fallback_low)
    hu_high = min(mu_map + 2.0 * sigma_map, clamping_max_hu)

    data_weight_pct = (n_eff / (k0 + n_eff)) * 100.0

    return MethodResult(
        name="Bayesian_MAP",
        hu_low=hu_low,
        hu_high=hu_high,
        mu=mu_map,
        sigma=sigma_map,
        converged=True,
        fallback=False,
        eat_vol_ml=0.0,
        la_eat_vol_ml=0.0,
        details={"data_weight_pct": data_weight_pct, "n_eff": n_eff},
    )


# ---------------------------------------------------------------------------
# Cohort Benchmark Runner
# ---------------------------------------------------------------------------

def run_cohort_density_research() -> None:
    """Execute density modeling research across all 10 patient scans."""
    print("===================================================================")
    print("Starting Ticket 12 Native-Grid Density Modeling & Bayesian Prior Research")
    print("===================================================================")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    config = PipelineConfig()
    thresh_cfg = ThresholdConfig.from_pipeline_config(config)

    records: List[Dict[str, Any]] = []
    patient_plot_data: Dict[str, Any] = {}

    for patient_id, meta in manifest.items():
        print(f"\n---> Analyzing Patient {patient_id} (Age: {meta['age']}, Sex: {meta['sex'].upper()})")

        # 1. Load Raw CT
        raw_ct_path = os.path.join(DATA_DIR, f"{patient_id}.nii.gz")
        if not os.path.isfile(raw_ct_path):
            raw_ct_path = os.path.join(DATA_DIR, f"{patient_id}.nii")
        if not os.path.isfile(raw_ct_path):
            print(f"[-] Raw CT not found for {patient_id}")
            continue

        target_mask_dir = os.path.join(DATA_DIR, "masks", patient_id)
        if not os.path.isdir(target_mask_dir):
            legacy_dir = os.path.join(LEGACY_INTERMEDIATE_DIR, patient_id)
            if os.path.isdir(legacy_dir):
                target_mask_dir = legacy_dir

        raw_img = sitk.ReadImage(raw_ct_path)
        raw_geo = GridGeometry.from_sitk_image(raw_img)
        raw_ct_arr = sitk.GetArrayFromImage(raw_img)
        raw_voxel_vol_ml = voxel_volume_ml(raw_geo.spacing)

        # 2. Load 1.5mm Isotropic CT & Anchors
        res_1_5mm = resample_to_isotropic(raw_ct_path, target_spacing_mm=1.5, is_label=False)
        ct_1_5mm = res_1_5mm.array
        geo_1_5mm = res_1_5mm.geometry
        voxel_vol_1_5mm = voxel_volume_ml(geo_1_5mm.spacing)

        loaded_masks_1_5mm: Dict[str, np.ndarray] = {}
        for anchor in CANONICAL_ANCHORS:
            mask_path = resolve_ts_mask_path(target_mask_dir, patient_id, anchor)
            if mask_path and os.path.isfile(mask_path):
                m_res = resample_to_reference(
                    mask_path,
                    reference_or_path=ct_1_5mm,
                    reference_geometry=geo_1_5mm,
                    is_label=True,
                )
                loaded_masks_1_5mm[anchor] = m_res.array.astype(bool)

        peri_path = resolve_ts_mask_path(target_mask_dir, patient_id, "Pericardium")
        ts_peri_mask = None
        if peri_path and os.path.isfile(peri_path):
            peri_res = resample_to_reference(
                peri_path,
                reference_or_path=ct_1_5mm,
                reference_geometry=geo_1_5mm,
                is_label=True,
            )
            ts_peri_mask = peri_res.array.astype(bool)

        resolver_dict = dict(loaded_masks_1_5mm)
        if ts_peri_mask is not None:
            resolver_dict["pericardium"] = ts_peri_mask

        peri_res_1_5mm = resolve_pericardium(resolver_dict, config, geo_1_5mm.spacing)

        # Project Pericardium to Native Grid
        native_peri_res = resample_to_reference(
            peri_res_1_5mm.mask.astype(np.uint8),
            raw_img,
            is_label=True,
            moving_geometry=geo_1_5mm,
            reference_geometry=raw_geo,
        )
        native_peri_mask = native_peri_res.array.astype(bool)

        # Project Anchors to Native Grid for precise LA Partitioning at native resolution
        native_anchors: Dict[str, np.ndarray] = {}
        for anchor in CANONICAL_ANCHORS:
            if anchor in loaded_masks_1_5mm:
                a_res = resample_to_reference(
                    loaded_masks_1_5mm[anchor].astype(np.uint8),
                    raw_img,
                    is_label=True,
                    moving_geometry=geo_1_5mm,
                    reference_geometry=raw_geo,
                )
                native_anchors[anchor] = a_res.array.astype(bool)

        # Extract Voxel Arrays
        vox_1_5mm = ct_1_5mm[peri_res_1_5mm.mask]
        vox_native = raw_ct_arr[native_peri_mask]

        sub0_1_5mm = vox_1_5mm[(vox_1_5mm >= -250.0) & (vox_1_5mm <= 0.0)]
        sub0_native = vox_native[(vox_native >= -250.0) & (vox_native <= 0.0)]

        print(f"  Voxel Counts in Pericardium: 1.5mm = {len(sub0_1_5mm):,}, Native = {len(sub0_native):,}")

        # -------------------------------------------------------------------
        # Execute Candidate Density Modeling Methods
        # -------------------------------------------------------------------

        # Method 1A: Baseline Trimmed Gaussian (1.5mm)
        res_1a = fit_trimmed_gaussian(sub0_1_5mm, thresh_cfg, voxel_vol_1_5mm)

        # Method 1B: Trimmed Gaussian on Native Grid
        res_1b = fit_trimmed_gaussian(sub0_native, thresh_cfg, raw_voxel_vol_ml)

        # Method 2A & 2B: 2-Component EM Gaussian Mixture Model (Native)
        res_2a_bayes, res_2b_tail = fit_gmm_2component(sub0_native)

        # Method 3: Non-Parametric Kernel Density Estimation (Native)
        res_3_kde = fit_kde_valley(sub0_native)

        # Method 4: Bayesian MAP Regularization (Native)
        res_4_bayes_map = fit_bayesian_map_regularization(sub0_native)

        # -------------------------------------------------------------------
        # Compute EAT and LA EAT Volumes for Each Method on Native Grid
        # -------------------------------------------------------------------
        methods_to_eval = [
            ("1.5mm_Gaussian_Mode", res_1a.hu_low, res_1a.hu_high, res_1a.fitted_mu, res_1a.fitted_sigma, res_1a.is_fallback),
            ("Native_Gaussian_Mode", res_1b.hu_low, res_1b.hu_high, res_1b.fitted_mu, res_1b.fitted_sigma, res_1b.is_fallback),
            ("Native_GMM_Bayes", res_2a_bayes.hu_low, res_2a_bayes.hu_high, res_2a_bayes.mu, res_2a_bayes.sigma, res_2a_bayes.fallback),
            ("Native_GMM_Tail", res_2b_tail.hu_low, res_2b_tail.hu_high, res_2b_tail.mu, res_2b_tail.sigma, res_2b_tail.fallback),
            ("Native_KDE_Valley", res_3_kde.hu_low, res_3_kde.hu_high, res_3_kde.mu, res_3_kde.sigma, res_3_kde.fallback),
            ("Native_Bayesian_MAP", res_4_bayes_map.hu_low, res_4_bayes_map.hu_high, res_4_bayes_map.mu, res_4_bayes_map.sigma, res_4_bayes_map.fallback),
        ]

        patient_record = {
            "patient_id": patient_id,
            "age": meta["age"],
            "sex": meta["sex"],
            "scanner_la_eat_ml": meta["scanner_la_eat_ml"],
            "scanner_total_eat_ml": meta["scanner_total_eat_ml"],
        }

        # Calculate partition and volumes
        part_cfg = PartitionConfig.from_pipeline_config(config)

        for m_name, low_hu, high_hu, mu_val, sig_val, is_fb in methods_to_eval:
            fat_mask_native = (raw_ct_arr >= low_hu) & (raw_ct_arr <= high_hu) & native_peri_mask
            total_eat_ml = float(np.sum(fat_mask_native)) * raw_voxel_vol_ml

            # Partition on 1.5mm grid with corresponding thresholds for fast, topologically equivalent LA isolation
            fat_mask_1_5mm = (ct_1_5mm >= low_hu) & (ct_1_5mm <= high_hu) & peri_res_1_5mm.mask
            part_res = partition_fat(
                fat_mask=fat_mask_1_5mm,
                pericardium_mask=peri_res_1_5mm.mask,
                anchor_masks=loaded_masks_1_5mm,
                geometry=geo_1_5mm,
                config=part_cfg,
            )
            la_clean = cleanup_la_fat_mask(part_res.la_fat_mask, config, geo_1_5mm.spacing)
            la_eat_ml = float(np.sum(la_clean.cleaned_mask)) * voxel_vol_1_5mm

            patient_record[f"{m_name}_low"] = low_hu
            patient_record[f"{m_name}_high"] = high_hu
            patient_record[f"{m_name}_mu"] = mu_val
            patient_record[f"{m_name}_sigma"] = sig_val
            patient_record[f"{m_name}_fallback"] = is_fb
            patient_record[f"{m_name}_total_eat_ml"] = total_eat_ml
            patient_record[f"{m_name}_la_eat_ml"] = la_eat_ml

            print(f"    [{m_name:<20}] Window: [{low_hu:6.1f}, {high_hu:5.1f}] HU | mu={str(f'{mu_val:.1f}' if mu_val else 'None'):>5} | EAT: {total_eat_ml:6.2f} mL | LA: {la_eat_ml:5.2f} mL | Fallback: {is_fb}")

        records.append(patient_record)

        # Store data for plotting
        patient_plot_data[patient_id] = {
            "sub0_1_5mm": sub0_1_5mm,
            "sub0_native": sub0_native,
            "res_1a": res_1a,
            "res_1b": res_1b,
            "res_2a": res_2a_bayes,
            "res_2b": res_2b_tail,
            "res_3": res_3_kde,
            "res_4": res_4_bayes_map,
            "meta": meta,
        }

    # -----------------------------------------------------------------------
    # Comparative Statistics & Correlation Summary
    # -----------------------------------------------------------------------
    df = pd.DataFrame(records)
    csv_out_path = os.path.join(RESEARCH_OUTPUT_DIR, "density_modeling_benchmark_results.csv")
    df.to_csv(csv_out_path, index=False)
    print(f"\n[+] Saved detailed benchmark CSV: {csv_out_path}")

    # Compute correlation metrics across methods
    summary_metrics = []
    method_keys = [
        "1.5mm_Gaussian_Mode",
        "Native_Gaussian_Mode",
        "Native_GMM_Bayes",
        "Native_GMM_Tail",
        "Native_KDE_Valley",
        "Native_Bayesian_MAP",
    ]

    print("\n" + "=" * 80)
    print(f"{'Method':<22} | {'r (LA)':<8} | {'p (LA)':<9} | {'r (Total)':<9} | {'MAE LA (mL)':<11} | {'Fallbacks':<9}")
    print("-" * 80)

    for m in method_keys:
        la_vals = df[f"{m}_la_eat_ml"].values
        total_vals = df[f"{m}_total_eat_ml"].values
        scanner_la = df["scanner_la_eat_ml"].values
        scanner_total = df["scanner_total_eat_ml"].values
        fallbacks = int(df[f"{m}_fallback"].sum())

        r_la, p_la = scipy.stats.pearsonr(la_vals, scanner_la)
        r_tot, p_tot = scipy.stats.pearsonr(total_vals, scanner_total)
        mae_la = float(np.mean(np.abs(la_vals - scanner_la)))

        summary_metrics.append({
            "Method": m,
            "r_la": r_la,
            "p_la": p_la,
            "r_total": r_tot,
            "p_total": p_tot,
            "mae_la": mae_la,
            "fallbacks": fallbacks,
        })
        print(f"{m:<22} | {r_la:8.4f} | {p_la:9.2e} | {r_tot:9.4f} | {mae_la:11.2f} | {fallbacks:>2} / 10")

    print("=" * 80)

    # -----------------------------------------------------------------------
    # Generate Multi-Panel Diagnostic Plot
    # -----------------------------------------------------------------------
    print("\n[+] Generating cohort multi-panel diagnostic figure...")
    fig, axes = plt.subplots(5, 2, figsize=(18, 24))
    axes = axes.flatten()

    for idx, (patient_id, pdata) in enumerate(patient_plot_data.items()):
        ax = axes[idx]
        sub0_nat = pdata["sub0_native"]
        sub0_15 = pdata["sub0_1_5mm"]
        meta = pdata["meta"]

        # Histograms
        counts_nat, bin_edges = np.histogram(sub0_nat, bins=125, range=(-250, 0), density=True)
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        counts_15, _ = np.histogram(sub0_15, bins=125, range=(-250, 0), density=True)

        ax.plot(centers, counts_15, color="gray", linestyle="--", alpha=0.7, label="1.5mm Grid Density")
        ax.plot(centers, counts_nat, color="black", lw=1.8, label="Native 512x512 Density")

        # Overlay fitted thresholds
        r1b = pdata["res_1b"]
        r2b = pdata["res_2b"]
        r3 = pdata["res_3"]
        r4 = pdata["res_4"]

        # Color lines for thresholds
        ax.axvline(r1b.hu_high, color="tab:blue", linestyle="-", lw=1.8, label=f"Native Gaussian: {r1b.hu_high:.1f} HU")
        ax.axvline(r2b.hu_high, color="tab:red", linestyle="-.", lw=1.5, label=f"Native GMM Tail: {r2b.hu_high:.1f} HU")
        ax.axvline(r3.hu_high, color="tab:green", linestyle=":", lw=1.5, label=f"Native KDE Valley: {r3.hu_high:.1f} HU")
        ax.axvline(r4.hu_high, color="tab:purple", linestyle="--", lw=1.5, label=f"Bayesian MAP: {r4.hu_high:.1f} HU")

        status_str = "FALLBACK in 1.5mm" if pdata["res_1a"].is_fallback else "NORMAL 1.5mm"
        ax.set_title(f"Patient {patient_id} ({meta['age']}y {meta['sex'].upper()}) - [{status_str}]", fontsize=11, fontweight="bold")
        ax.set_xlim(-220, 10)
        ax.set_xlabel("Attenuation (HU)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plot_out_path = os.path.join(RESEARCH_OUTPUT_DIR, "density_modeling_native_grid_comparison.png")
    plt.savefig(plot_out_path, dpi=200)
    plt.close()
    print(f"[+] Saved diagnostic plot: {plot_out_path}")


if __name__ == "__main__":
    run_cohort_density_research()
