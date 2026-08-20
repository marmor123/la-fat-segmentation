"""Demo Runner & Multi-Scenario Evaluation for Ticket 3.

Evaluates the Trimmed-Gaussian Peak Fitting algorithm across 7 clinical and synthetic scenarios:
1. Ideal Pure Gaussian Fat Peak
2. Asymmetric Fat Peak with Heavy Muscle Shoulder
3. Low-Dose CT with High Noise
4. Sparse Fat (N < 500 voxels)
5. Monotonic Soft-Tissue Slope (No Fat Peak)
6. Metal / Outlier Artifact
7. Real Clinical Cardiac CT Voxels (from patient 1512 / 0674)

Computes the thresholded fat volume in mL under both Adaptive Gaussian and Fixed [-190, -30] HU,
and generates a multi-panel visualization report saved to docs/prototypes/ticket_3_gaussian_fit_demo.png.
"""

from __future__ import annotations

import os
from typing import Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from prototype_gaussian_fit import (
    ThresholdConfig,
    ThresholdResult,
    fit_trimmed_gaussian_threshold,
    _gaussian_func,
)


def generate_scenarios() -> list[dict]:
    """Generate synthetic and real CT voxel distributions."""
    np.random.seed(42)
    scenarios = []

    # 1. Ideal Pure Gaussian
    fat_1 = np.random.normal(loc=-105.0, scale=12.0, size=50000)
    scenarios.append({
        "name": "1. Ideal Gaussian Fat Peak",
        "voxels": fat_1,
        "true_mu": -105.0,
        "true_sigma": 12.0,
        "spacing": (1.0, 1.0, 1.0),
    })

    # 2. Asymmetric with Heavy Soft-Tissue Shoulder
    fat_2 = np.random.normal(loc=-95.0, scale=14.0, size=40000)
    # Exponential shoulder rising toward 0 HU
    shoulder = -np.random.exponential(scale=20.0, size=80000)
    shoulder = shoulder[shoulder >= -250.0]
    scenarios.append({
        "name": "2. Asymmetric (Muscle/Blood Shoulder)",
        "voxels": np.concatenate([fat_2, shoulder]),
        "true_mu": -95.0,
        "true_sigma": 14.0,
        "spacing": (1.0, 1.0, 1.0),
    })

    # 3. Low-Dose CT (High Noise & Wide Spread)
    fat_3 = np.random.normal(loc=-100.0, scale=22.0, size=35000)
    noise = np.random.uniform(-250.0, 0.0, size=20000)
    scenarios.append({
        "name": "3. Low-Dose CT (High Noise)",
        "voxels": np.concatenate([fat_3, noise]),
        "true_mu": -100.0,
        "true_sigma": 22.0,
        "spacing": (1.0, 1.0, 1.0),
    })

    # 4. Sparse Fat (N < 500) -> Tests High Concern Fallback
    fat_4 = np.random.normal(loc=-100.0, scale=15.0, size=250)
    scenarios.append({
        "name": "4. Sparse Fat (N=250 < 500)",
        "voxels": fat_4,
        "true_mu": None,
        "true_sigma": None,
        "spacing": (1.0, 1.0, 1.0),
    })

    # 5. Monotonic Soft-Tissue Slope (No Fat Peak) -> Tests No-Peak Fallback
    # Pure soft-tissue tail with zero fat mode
    slope = -np.random.exponential(scale=15.0, size=60000)
    slope = slope[slope >= -250.0]
    scenarios.append({
        "name": "5. Monotonic Slope (No Fat Peak)",
        "voxels": slope,
        "true_mu": None,
        "true_sigma": None,
        "spacing": (1.0, 1.0, 1.0),
    })

    # 6. Metal / Extreme Outlier Artifacts
    fat_6 = np.random.normal(loc=-90.0, scale=12.0, size=30000)
    outliers = np.array([-1024.0] * 5000 + [2000.0] * 5000)
    scenarios.append({
        "name": "6. Outlier / Metal Spikes",
        "voxels": np.concatenate([fat_6, outliers]),
        "true_mu": -90.0,
        "true_sigma": 12.0,
        "spacing": (1.0, 1.0, 1.0),
    })

    # 7. Real Clinical CT Extract (Patient 1512, cardiac slices 30:85)
    real_path = r"C:\Users\marmo\Downloads\ctscans\1512.nii.gz"
    if os.path.exists(real_path):
        img = nib.load(real_path)
        data = img.get_fdata()
        zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
        # Cardiac slices in Z
        crop = data[:, :, 30:85]
        scenarios.append({
            "name": "7. Real Scan Extract (Patient 1512)",
            "voxels": crop.flatten(),
            "true_mu": None,
            "true_sigma": None,
            "spacing": zooms,
        })

    return scenarios


def run_evaluation() -> None:
    """Run all scenarios, print quantitative volume comparison table, and generate plot."""
    scenarios = generate_scenarios()
    config = ThresholdConfig()

    os.makedirs(r"docs\prototypes", exist_ok=True)
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(11, 3.4 * len(scenarios)), dpi=150)
    if len(scenarios) == 1:
        axes = [axes]

    print("\n" + "=" * 90)
    print(f"{'Scenario':<38} | {'Status':<10} | {'Window (HU)':<15} | {'Adaptive mL':<12} | {'Fixed mL':<10}")
    print("=" * 90)

    for idx, (sc, ax) in enumerate(zip(scenarios, axes)):
        vox = sc["voxels"]
        spacing = sc["spacing"]
        voxel_vol_ml = (spacing[0] * spacing[1] * spacing[2]) / 1000.0

        res: ThresholdResult = fit_trimmed_gaussian_threshold(vox, config=config, voxel_spacing_mm=spacing)

        # Compute fixed window volume for comparison
        fixed_count = int(np.sum((vox >= -190.0) & (vox <= -30.0)))
        fixed_vol_ml = fixed_count * voxel_vol_ml

        status_str = "FALLBACK" if res.is_fallback else "CONVERGED"
        window_str = f"[{res.hu_low:.1f}, {res.hu_high:.1f}]"

        print(f"{sc['name']:<38} | {status_str:<10} | {window_str:<15} | {res.fat_volume_ml:<12.2f} | {fixed_vol_ml:<10.2f}")

        # Plotting histogram
        sub0 = vox[(vox >= -250.0) & (vox <= 0.0)]
        hist, bin_edges = np.histogram(sub0, bins=250, range=(-250.0, 0.0))
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        ax.bar(centers, hist, width=1.0, color="#cbd5e1", edgecolor="none", label="Sub-0 HU Histogram", alpha=0.7)

        # Highlight thresholded regions
        # Fixed window
        ax.axvspan(-190.0, -30.0, color="#94a3b8", alpha=0.15, label="Fixed Window [-190, -30]")

        # Adaptive window
        ax.axvspan(res.hu_low, res.hu_high, color="#3b82f6", alpha=0.25, label=f"Adaptive Window [{res.hu_low:.1f}, {res.hu_high:.1f}]")

        if not res.is_fallback and res.fitted_mu is not None and res.fitted_sigma is not None:
            # Plot fitted curve
            x_curve = np.linspace(-250.0, 0.0, 500)
            y_curve = _gaussian_func(x_curve, res.fitted_amplitude, res.fitted_mu, res.fitted_sigma)
            ax.plot(x_curve, y_curve, color="#dc2626", linewidth=2.0, label=rf"Fitted Gaussian ($\mu={res.fitted_mu:.1f}$, $\sigma={res.fitted_sigma:.1f}$)")
            ax.axvline(res.fitted_mu, color="#b91c1c", linestyle="--", alpha=0.8, label=f"Fat Peak Mode ({res.fitted_mu:.1f} HU)")

        ax.set_title(
            f"{sc['name']}  —  Status: {status_str} | Window: {window_str} HU | Fat Vol: {res.fat_volume_ml:.1f} mL (Fixed: {fixed_vol_ml:.1f} mL)",
            fontsize=10,
            fontweight="bold",
            color="#1e293b"
        )
        ax.set_xlabel("Hounsfield Units (HU)", fontsize=9)
        ax.set_ylabel("Voxel Count", fontsize=9)
        ax.set_xlim(-250, 10)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    plt.tight_layout()
    output_png = r"docs\prototypes\ticket_3_gaussian_fit_demo.png"
    plt.savefig(output_png)
    plt.close()
    print("=" * 90)
    print(f"Visual report saved to {output_png}\n")


if __name__ == "__main__":
    run_evaluation()
