"""Forensic Investigation: Spatial Adjacency and HU Profiles of Partial Volume Fat Voxels.

Analyzes the [-30, 0] HU voxels inside the pericardial space in Patient 0674 and 3664
to determine if they represent genuine partial-volume adipose tissue at the myocardial/pericardial
boundaries, or background soft-tissue noise.
"""

from __future__ import annotations

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

sys.path.insert(0, os.path.dirname(__file__))
from run_partition_demo import load_real_patient_data


def analyze_patient(patient_dir: str, patient_id: str) -> dict:
    """Analyze spatial contiguity and HU distribution of transition voxels."""
    ct, peri, anchors, fat_30, sp = load_real_patient_data(patient_dir)
    voxel_vol_ml = sp[0] * sp[1] * sp[2] / 1000.0

    chambers = np.zeros_like(peri, dtype=bool)
    for m in anchors.values():
        chambers |= (m > 0)

    # Zone 1: Pure Fat Core [-190, -30] HU
    vox_core = (peri > 0) & (ct >= -190.0) & (ct <= -30.0) & ~chambers
    # Zone 2: Transition Layer [-30, 0] HU
    vox_trans = (peri > 0) & (ct > -30.0) & (ct <= 0.0) & ~chambers
    # Zone 3: Dense Myocardium / Blood > +20 HU
    vox_myo = chambers & (ct > 20.0)

    # 1-voxel 26-connectivity dilation of core fat
    core_dilated = binary_dilation(vox_core, structure=np.ones((3, 3, 3)))
    # Transition voxels directly touching the core fat mantle
    trans_touching_core = vox_trans & core_dilated
    pct_touching = (np.count_nonzero(trans_touching_core) / np.count_nonzero(vox_trans) * 100.0) if np.count_nonzero(vox_trans) > 0 else 0.0

    # 2-voxel dilation of core fat
    core_dilated_2 = binary_dilation(core_dilated, structure=np.ones((3, 3, 3)))
    trans_within_2vox = vox_trans & core_dilated_2
    pct_within_2vox = (np.count_nonzero(trans_within_2vox) / np.count_nonzero(vox_trans) * 100.0) if np.count_nonzero(vox_trans) > 0 else 0.0

    trans_hu = ct[vox_trans]
    core_hu = ct[vox_core]

    return {
        "patient_id": patient_id,
        "ct": ct,
        "peri": peri,
        "anchors": anchors,
        "vox_core": vox_core,
        "vox_trans": vox_trans,
        "trans_touching_core": trans_touching_core,
        "core_vol_ml": np.count_nonzero(vox_core) * voxel_vol_ml,
        "trans_vol_ml": np.count_nonzero(vox_trans) * voxel_vol_ml,
        "pct_touching_1vox": pct_touching,
        "pct_within_2vox": pct_within_2vox,
        "trans_hu_mean": float(np.mean(trans_hu)),
        "trans_hu_median": float(np.median(trans_hu)),
        "trans_hu_std": float(np.std(trans_hu)),
        "core_hu_mean": float(np.mean(core_hu)),
        "core_hu_std": float(np.std(core_hu)),
    }


def run_forensic_analysis() -> None:
    """Run spatial adjacency analysis and generate visual inspection report."""
    os.makedirs(r"docs\prototypes", exist_ok=True)

    patients = [
        (r"C:\Users\marmo\Downloads\la_eat_segmentation\data\intermediate\0674", "0674"),
        (r"C:\Users\marmo\Downloads\la_eat_segmentation\data\intermediate\3664", "3664"),
    ]

    results = []
    print("\n" + "=" * 100)
    print(f"{'Patient':<10} | {'Core [-190,-30] mL':<19} | {'Trans [-30,0] mL':<16} | {'Touching Core (1-vox)':<22} | {'Within 2-vox of Core'}")
    print("=" * 100)

    for p_dir, p_id in patients:
        if os.path.exists(p_dir):
            res = analyze_patient(p_dir, p_id)
            results.append(res)
            print(f"{p_id:<10} | {res['core_vol_ml']:<19.2f} | {res['trans_vol_ml']:<16.2f} | {res['pct_touching_1vox']:<21.1f}% | {res['pct_within_2vox']:.1f}%")

    print("=" * 100)

    # -----------------------------------------------------------------------
    # Generate Multi-Panel Visual Figure
    # -----------------------------------------------------------------------
    fig, axes = plt.subplots(len(results), 3, figsize=(18, 5.8 * len(results)), dpi=150)
    if len(results) == 1:
        axes = [axes]

    for row_idx, res in enumerate(results):
        ct = res["ct"]
        peri = res["peri"]
        anchors = res["anchors"]
        vox_core = res["vox_core"]
        vox_trans = res["vox_trans"]
        p_id = res["patient_id"]

        # Find axial slice with maximum LA fat
        la_mask = anchors.get("LA", np.zeros_like(peri))
        la_fat_slice_counts = [np.count_nonzero(vox_core[z] | vox_trans[z]) for z in range(ct.shape[0])]
        best_z = int(np.argmax(la_fat_slice_counts))

        # Crop around heart for crisp zoom
        peri_indices = np.where(peri[best_z])
        if len(peri_indices[0]) > 0:
            ymin, ymax = max(0, peri_indices[0].min() - 10), min(ct.shape[1], peri_indices[0].max() + 10)
            xmin, xmax = max(0, peri_indices[1].min() - 10), min(ct.shape[2], peri_indices[1].max() + 10)
        else:
            ymin, ymax = 0, ct.shape[1]
            xmin, xmax = 0, ct.shape[2]

        ct_crop = ct[best_z, ymin:ymax, xmin:xmax]
        core_crop = vox_core[best_z, ymin:ymax, xmin:xmax]
        trans_crop = vox_trans[best_z, ymin:ymax, xmin:xmax]
        peri_crop = peri[best_z, ymin:ymax, xmin:xmax]
        la_crop = la_mask[best_z, ymin:ymax, xmin:xmax]

        # 1. Panel 1: CT Attenuation Map with HU window [-160, 80]
        ax1 = axes[row_idx][0]
        im1 = ax1.imshow(ct_crop, cmap="bone", vmin=-160, vmax=80)
        ax1.contour(peri_crop, levels=[0.5], colors=["cyan"], linewidths=1.2, linestyles="dashed")
        if np.count_nonzero(la_crop) > 0:
            ax1.contour(la_crop, levels=[0.5], colors=["red"], linewidths=1.5)
        ax1.set_title(f"Patient {p_id} (Axial Slice {best_z})\nRaw CT Attenuation (Red=LA, Cyan=Pericardium)", fontsize=11, fontweight="bold")
        ax1.axis("off")
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label("HU", fontsize=9)

        # 2. Panel 2: Spatial Adjacency Overlay (Core vs Transition)
        ax2 = axes[row_idx][1]
        # Base grayscale
        norm_ct = np.clip((ct_crop - (-200)) / 350.0, 0.0, 1.0)
        rgb_overlay = np.stack([norm_ct, norm_ct, norm_ct], axis=-1)

        # Color Core Fat [-190, -30] HU as Bright Yellow (1.0, 0.85, 0.0)
        rgb_overlay[core_crop] = np.array([1.0, 0.85, 0.0])
        # Color Transition Fat [-30, 0] HU as Neon Green (0.2, 0.95, 0.2)
        rgb_overlay[trans_crop] = np.array([0.2, 0.95, 0.2])

        ax2.imshow(rgb_overlay)
        ax2.set_title(f"Spatial Adjacency Breakdown\nYellow: Core Fat [-190,-30] | Green: Transition [-30,0] HU", fontsize=11, fontweight="bold")
        ax2.axis("off")

        # 3. Panel 3: 1D Profile Transect across Myocardium -> Fat -> Pericardium
        ax3 = axes[row_idx][2]
        # Sample horizontal line transect through the posterior LA fat pocket
        if np.count_nonzero(core_crop) > 0:
            y_trans = int(np.where(core_crop)[0].mean())
            x_profile = ct_crop[y_trans, :]
            ax3.plot(x_profile, color="#1f77b4", linewidth=2.0, label="CT HU Profile")
            ax3.axhline(-30.0, color="orange", linestyle="--", linewidth=1.5, label="Old Cutoff (-30 HU)")
            ax3.axhline(0.0, color="green", linestyle="--", linewidth=1.5, label="Physical Bound (0 HU)")
            ax3.axhspan(-190.0, -30.0, color="yellow", alpha=0.2, label="Core Fat Window")
            ax3.axhspan(-30.0, 0.0, color="green", alpha=0.15, label="Partial Volume Zone")
            ax3.set_ylim(-180, 100)
            ax3.set_xlabel("Voxel Coordinate X across AV Groove / Periatrial Space", fontsize=9)
            ax3.set_ylabel("CT Attenuation (HU)", fontsize=9)
            ax3.set_title(f"1D HU Profile across Fat Pocket (Y={y_trans})\nContinuous Gaussian Transition from Fat to Wall", fontsize=11, fontweight="bold")
            ax3.legend(loc="upper right", fontsize=8)
            ax3.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_img = r"docs\prototypes\fat_partial_volume_forensics.png"
    plt.savefig(out_img, dpi=150)
    plt.close()
    print(f"\nVisual forensics report saved to {out_img}!")


if __name__ == "__main__":
    run_forensic_analysis()
