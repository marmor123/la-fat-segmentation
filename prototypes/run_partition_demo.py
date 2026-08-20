"""Demo Runner & Multi-Scenario Evaluation for Ticket 4.

Evaluates 3D Surface Distance Partition across 6 challenging scenarios:
1. Ideal 6-Chamber Phantom with Non-Planar AV Saddle
2. Acute AV Groove Concavity Stress-Test (Deep Cleft)
3. Ultra-Thin Interatrial Septum (1-voxel thickness)
4. Distance Radius Sensitivity Sweep (20mm, 30mm, 40mm, inf)
5. Algorithmic Comparison: Solid-Mask EDT vs Surface-Erosion EDT vs Domain-Constrained EDT
6. Real Clinical Cardiac Scan (Patient 0674 / 3664 TotalSegmentator Masks)

Generates a multi-panel orthogonal visualization report saved to
docs/prototypes/ticket_4_partition_phantom_demo.png and prints quantitative metrics.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from prototype_partition_phantom import (
    ANCHOR_LABELS,
    CANONICAL_ANCHORS,
    PartitionMetrics,
    create_synthetic_cardiac_phantom,
    evaluate_partition_metrics,
    partition_domain_constrained_edt,
    partition_solid_edt,
    partition_surface_edt,
)


def load_real_patient_data(patient_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray, Tuple[float, float, float]]:
    """Load real CT and TotalSegmentator masks from intermediate directory."""
    ct_img = nib.load(os.path.join(patient_dir, "ct_resampled.nii.gz"))
    ct_array = np.asarray(ct_img.dataobj, dtype=np.float32)
    spacing = tuple(float(s) for s in ct_img.header.get_zooms()[:3])

    peri_img = nib.load(os.path.join(patient_dir, "pericardium.nii.gz"))
    pericardium_mask = np.asarray(peri_img.dataobj, dtype=np.uint8)

    # Standard TS v2 filenames mapped to canonical anchor names
    name_map = {
        "LA": "heart_atrium_left.nii.gz",
        "LV": "heart_ventricle_left.nii.gz",
        "RA": "heart_atrium_right.nii.gz",
        "RV": "heart_ventricle_right.nii.gz",
        "Aorta": "aorta.nii.gz",
        "Pulmonary_Artery": "pulmonary_artery.nii.gz",
    }

    anchor_masks: Dict[str, np.ndarray] = {}
    for anchor_name, fname in name_map.items():
        fpath = os.path.join(patient_dir, fname)
        if os.path.exists(fpath):
            img = nib.load(fpath)
            anchor_masks[anchor_name] = np.asarray(img.dataobj, dtype=np.uint8)
        else:
            anchor_masks[anchor_name] = np.zeros_like(pericardium_mask)

    # Combined chamber mask
    chambers = np.zeros_like(pericardium_mask, dtype=bool)
    for m in anchor_masks.values():
        chambers |= (m > 0)

    # Fat in HU range [-190, -30] within pericardium excluding chambers
    fat_hu = (ct_array >= -190.0) & (ct_array <= -30.0)
    fat_mask = ((pericardium_mask > 0) & fat_hu & ~chambers).astype(np.uint8)

    return ct_array, pericardium_mask, anchor_masks, fat_mask, spacing


def run_all_evaluations() -> None:
    """Execute evaluation across all scenarios and produce visualization report."""
    os.makedirs(r"docs\prototypes", exist_ok=True)

    print("\n" + "=" * 115)
    print(f"{'Scenario':<42} | {'Algorithm':<14} | {'LA Fat (mL)':<11} | {'LA Share':<9} | {'CCs':<4} | {'Primary %':<10} | {'Leak Vox':<8} | {'Time (ms)'}")
    print("=" * 115)

    results_for_plot = []

    # -----------------------------------------------------------------------
    # Scenario 1: Baseline Ideal Phantom (Sharpness=1.0)
    # -----------------------------------------------------------------------
    ct_1, peri_1, anchors_1, fat_1 = create_synthetic_cardiac_phantom(shape=(128, 128, 128), av_groove_sharpness=1.0)
    la_fat_1, assign_1, t_1 = partition_solid_edt(anchors_1, peri_1, fat_1, spacing=(1.5, 1.5, 1.5), max_assign_distance_mm=35.0)
    m_1 = evaluate_partition_metrics(la_fat_1, fat_1, assign_1, (1.5, 1.5, 1.5), t_1, septal_plane_x=48)
    print(f"{'1. Baseline 6-Chamber Phantom':<42} | {'Solid EDT':<14} | {m_1.la_fat_volume_ml:<11.2f} | {m_1.la_fat_share_pct:<8.1f}% | {m_1.num_connected_components:<4} | {m_1.primary_component_fraction*100:<9.1f}% | {m_1.septal_leakage_voxels:<8} | {m_1.execution_time_ms:.1f}ms")
    results_for_plot.append(("1. Baseline Phantom", ct_1, peri_1, anchors_1, fat_1, la_fat_1, assign_1, m_1))

    # -----------------------------------------------------------------------
    # Scenario 2: Acute AV Groove Concavity Stress (Sharpness=2.5)
    # -----------------------------------------------------------------------
    ct_2, peri_2, anchors_2, fat_2 = create_synthetic_cardiac_phantom(shape=(128, 128, 128), av_groove_sharpness=2.5)
    la_fat_2, assign_2, t_2 = partition_solid_edt(anchors_2, peri_2, fat_2, spacing=(1.5, 1.5, 1.5), max_assign_distance_mm=35.0)
    m_2 = evaluate_partition_metrics(la_fat_2, fat_2, assign_2, (1.5, 1.5, 1.5), t_2, septal_plane_x=48)
    print(f"{'2. Acute AV Groove Stress (2.5x)':<42} | {'Solid EDT':<14} | {m_2.la_fat_volume_ml:<11.2f} | {m_2.la_fat_share_pct:<8.1f}% | {m_2.num_connected_components:<4} | {m_2.primary_component_fraction*100:<9.1f}% | {m_2.septal_leakage_voxels:<8} | {m_2.execution_time_ms:.1f}ms")
    results_for_plot.append(("2. Acute AV Groove", ct_2, peri_2, anchors_2, fat_2, la_fat_2, assign_2, m_2))

    # -----------------------------------------------------------------------
    # Scenario 3: Ultra-Thin Interatrial Septum (1-Voxel Thickness)
    # -----------------------------------------------------------------------
    ct_3, peri_3, anchors_3, fat_3 = create_synthetic_cardiac_phantom(shape=(128, 128, 128), ias_thickness_voxels=1)
    la_fat_3, assign_3, t_3 = partition_solid_edt(anchors_3, peri_3, fat_3, spacing=(1.5, 1.5, 1.5), max_assign_distance_mm=35.0)
    m_3 = evaluate_partition_metrics(la_fat_3, fat_3, assign_3, (1.5, 1.5, 1.5), t_3, septal_plane_x=48)
    print(f"{'3. Thin Septum (1-voxel thickness)':<42} | {'Solid EDT':<14} | {m_3.la_fat_volume_ml:<11.2f} | {m_3.la_fat_share_pct:<8.1f}% | {m_3.num_connected_components:<4} | {m_3.primary_component_fraction*100:<9.1f}% | {m_3.septal_leakage_voxels:<8} | {m_3.execution_time_ms:.1f}ms")
    results_for_plot.append(("3. Thin Septum", ct_3, peri_3, anchors_3, fat_3, la_fat_3, assign_3, m_3))

    # -----------------------------------------------------------------------
    # Scenario 4: Algorithmic Comparison (Solid vs Surface vs Constrained)
    # -----------------------------------------------------------------------
    la_fat_surf, assign_surf, t_surf = partition_surface_edt(anchors_1, peri_1, fat_1, spacing=(1.5, 1.5, 1.5), max_assign_distance_mm=35.0)
    m_surf = evaluate_partition_metrics(la_fat_surf, fat_1, assign_surf, (1.5, 1.5, 1.5), t_surf, septal_plane_x=48)
    print(f"{'4a. Algorithm: Surface-Erosion EDT':<42} | {'Surface EDT':<14} | {m_surf.la_fat_volume_ml:<11.2f} | {m_surf.la_fat_share_pct:<8.1f}% | {m_surf.num_connected_components:<4} | {m_surf.primary_component_fraction*100:<9.1f}% | {m_surf.septal_leakage_voxels:<8} | {m_surf.execution_time_ms:.1f}ms")

    la_fat_geo, assign_geo, t_geo = partition_domain_constrained_edt(anchors_1, peri_1, fat_1, spacing=(1.5, 1.5, 1.5), max_assign_distance_mm=35.0)
    m_geo = evaluate_partition_metrics(la_fat_geo, fat_1, assign_geo, (1.5, 1.5, 1.5), t_geo, septal_plane_x=48)
    print(f"{'4b. Algorithm: Domain-Constrained EDT':<42} | {'Constrained EDT':<14} | {m_geo.la_fat_volume_ml:<11.2f} | {m_geo.la_fat_share_pct:<8.1f}% | {m_geo.num_connected_components:<4} | {m_geo.primary_component_fraction*100:<9.1f}% | {m_geo.septal_leakage_voxels:<8} | {m_geo.execution_time_ms:.1f}ms")

    # -----------------------------------------------------------------------
    # Scenario 5: Distance Radius Sensitivity Sweep (20mm vs 35mm vs inf)
    # -----------------------------------------------------------------------
    for dist_cutoff in [20.0, 35.0, 50.0, np.inf]:
        la_fat_r, assign_r, t_r = partition_solid_edt(anchors_1, peri_1, fat_1, spacing=(1.5, 1.5, 1.5), max_assign_distance_mm=dist_cutoff)
        m_r = evaluate_partition_metrics(la_fat_r, fat_1, assign_r, (1.5, 1.5, 1.5), t_r, septal_plane_x=48)
        dist_str = f"Cutoff R={dist_cutoff:.0f}mm" if dist_cutoff < np.inf else "Cutoff R=Inf"
        print(f"{f'5. Radius Sweep: {dist_str}':<42} | {'Solid EDT':<14} | {m_r.la_fat_volume_ml:<11.2f} | {m_r.la_fat_share_pct:<8.1f}% | {m_r.num_connected_components:<4} | {m_r.primary_component_fraction*100:<9.1f}% | {m_r.septal_leakage_voxels:<8} | {m_r.execution_time_ms:.1f}ms")

    # -----------------------------------------------------------------------
    # Scenario 6: Real Clinical Cardiac Scans (Patient 0674 & 3664)
    # -----------------------------------------------------------------------
    real_paths = [
        (r"C:\Users\marmo\Downloads\la_eat_segmentation\data\intermediate\0674", "6a. Clinical CT 0674"),
        (r"C:\Users\marmo\Downloads\la_eat_segmentation\data\intermediate\3664", "6b. Clinical CT 3664"),
    ]

    for p_dir, sc_name in real_paths:
        if os.path.exists(p_dir):
            ct_real, peri_real, anchors_real, fat_real, sp_real = load_real_patient_data(p_dir)
            la_fat_real, assign_real, t_real = partition_solid_edt(anchors_real, peri_real, fat_real, spacing=sp_real, max_assign_distance_mm=35.0)
            m_real = evaluate_partition_metrics(la_fat_real, fat_real, assign_real, sp_real, t_real)
            print(f"{sc_name:<42} | {'Solid EDT':<14} | {m_real.la_fat_volume_ml:<11.2f} | {m_real.la_fat_share_pct:<8.1f}% | {m_real.num_connected_components:<4} | {m_real.primary_component_fraction*100:<9.1f}% | {m_real.septal_leakage_voxels:<8} | {m_real.execution_time_ms:.1f}ms")
            results_for_plot.append((sc_name, ct_real, peri_real, anchors_real, fat_real, la_fat_real, assign_real, m_real))

    print("=" * 115)

    # -----------------------------------------------------------------------
    # Generate Multi-Panel Visual Report
    # -----------------------------------------------------------------------
    print("\nGenerating multi-panel visual report: docs/prototypes/ticket_4_partition_phantom_demo.png ...")
    num_scenarios = len(results_for_plot)
    fig, axes = plt.subplots(num_scenarios, 4, figsize=(18, 4.2 * num_scenarios), dpi=140)
    if num_scenarios == 1:
        axes = [axes]

    # Color palette for 6 anchors + background
    anchor_colors = {
        1: (1.0, 0.2, 0.2),  # LA = Red
        2: (0.2, 0.4, 1.0),  # LV = Blue
        3: (0.2, 0.9, 0.2),  # RA = Green
        4: (1.0, 0.8, 0.1),  # RV = Yellow
        5: (0.9, 0.1, 0.9),  # Aorta = Magenta
        6: (0.1, 0.9, 0.9),  # PA = Cyan
    }

    for row_idx, (title, ct, peri, anchors, fat, la_fat, assign, metrics) in enumerate(results_for_plot):
        shape = ct.shape
        # Mid-cardiac slice (Z for axial, Y for coronal, X for sagittal)
        z_slice = shape[0] // 2
        y_slice = shape[1] // 2
        x_slice = shape[2] // 2

        # 1. Panel 1: CT with Chamber Overlays (Axial Cut)
        ax1 = axes[row_idx][0]
        ct_slice = ct[z_slice, :, :]
        ax1.imshow(ct_slice, cmap="gray", vmin=-250, vmax=200)
        # Overlay chamber contours
        for aname, amask in anchors.items():
            if np.count_nonzero(amask[z_slice, :, :]) > 0:
                lbl = ANCHOR_LABELS[aname]
                ax1.contour(amask[z_slice, :, :], levels=[0.5], colors=[anchor_colors[lbl]], linewidths=1.5)
        if np.count_nonzero(peri[z_slice, :, :]) > 0:
            ax1.contour(peri[z_slice, :, :], levels=[0.5], colors=["white"], linewidths=1.2, linestyles="dashed")
        ax1.set_title(f"{title}\nAxial CT & Chamber Contours", fontsize=11, fontweight="bold")
        ax1.axis("off")

        # 2. Panel 2: Multi-Anchor Distance Fields (Coronal Cut)
        ax2 = axes[row_idx][1]
        # Build 3-channel RGB overlay of anchor assignments on Coronal slice
        coronal_assign = assign[:, y_slice, :]
        rgb_coronal = np.zeros((coronal_assign.shape[0], coronal_assign.shape[1], 3), dtype=np.float32)
        # Background CT
        ct_coronal = ct[:, y_slice, :]
        norm_ct = np.clip((ct_coronal - (-200)) / 400.0, 0.0, 1.0)
        rgb_coronal[:, :, 0] = norm_ct * 0.4
        rgb_coronal[:, :, 1] = norm_ct * 0.4
        rgb_coronal[:, :, 2] = norm_ct * 0.4

        for lbl, col in anchor_colors.items():
            mask_lbl = coronal_assign == lbl
            rgb_coronal[mask_lbl] = np.array(col) * 0.85

        ax2.imshow(rgb_coronal)
        ax2.set_title(f"Coronal Cut: Multi-Anchor Partition Map\nLA=Red, LV=Blue, RA=Green, RV=Yellow", fontsize=10)
        ax2.axis("off")

        # 3. Panel 3: Sagittal Cut with LA Fat Overlay
        ax3 = axes[row_idx][2]
        sag_ct = ct[:, :, x_slice]
        ax3.imshow(sag_ct, cmap="gray", vmin=-250, vmax=200)
        if np.count_nonzero(la_fat[:, :, x_slice]) > 0:
            ax3.imshow(np.ma.masked_where(la_fat[:, :, x_slice] == 0, la_fat[:, :, x_slice]), cmap="autumn", alpha=0.75)
        ax3.set_title(f"Sagittal Cut: Segmented LA Fat\nVolume: {metrics.la_fat_volume_ml:.1f} mL ({metrics.la_fat_share_pct:.1f}%)", fontsize=10)
        ax3.axis("off")

        # 4. Panel 4: Quality & Topology Scorecard
        ax4 = axes[row_idx][3]
        ax4.axis("off")
        status_color = "green" if metrics.primary_component_fraction >= 0.98 and metrics.septal_leakage_voxels == 0 else "red"
        status_text = "PASS (Topologically Sound)" if status_color == "green" else "WARN (Minor Fragmentation)"

        card_text = (
            f"--- QA TOPOLOGY SCORECARD ---\n\n"
            f"Status: {status_text}\n"
            f"• LA Fat Volume: {metrics.la_fat_volume_ml:.2f} mL\n"
            f"• Total Epicardial Fat: {metrics.total_fat_volume_ml:.2f} mL\n"
            f"• LA Fat Share: {metrics.la_fat_share_pct:.1f}%\n"
            f"• 3D Connected Components: {metrics.num_connected_components}\n"
            f"• Primary Component: {metrics.primary_component_fraction*100:.1f}%\n"
            f"• Secondary Island Max: {metrics.secondary_component_max_ml:.3f} mL\n"
            f"• Septal Leakage: {metrics.septal_leakage_voxels} voxels\n"
            f"• Runtime (CPU): {metrics.execution_time_ms:.1f} ms\n"
        )
        ax4.text(
            0.05, 0.5, card_text,
            fontsize=10, fontfamily="monospace",
            verticalalignment="center",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor=status_color, linewidth=2.0)
        )

    plt.tight_layout()
    out_path = os.path.join(r"docs\prototypes", "ticket_4_partition_phantom_demo.png")
    plt.savefig(out_path, dpi=140)
    plt.close()
    print(f"Report successfully saved to {out_path}!\n")


if __name__ == "__main__":
    run_all_evaluations()
