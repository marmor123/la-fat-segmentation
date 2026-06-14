"""QA Dashboard module for LA Fat Segmentation.

Generates per-scan visual QA output including slice gallery, fat overlay,
numeric summary, and a combined self-contained HTML dashboard.
"""

from __future__ import annotations

import csv
import dataclasses
import os
import typing as t

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass

from la_fat.config import PipelineConfig
from la_fat.cleanup import CleanupResult
from la_fat.fat_thresholder import FatThresholdResult
from la_fat.partition_engine import PartitionResult
from la_fat.pericardium_resolver import PericardiumResult
from la_fat.quality_flagger import QualityFlag

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Consistent colors for the 6 Partition Anchors.
# Used across slice gallery and fat overlay.
ANCHOR_COLORS: dict[str, tuple[float, float, float]] = {
    "LA": (1.0, 0.0, 0.0),  # red
    "LV": (0.0, 0.0, 1.0),  # blue
    "RA": (0.0, 0.8, 0.0),  # green
    "RV": (1.0, 0.65, 0.0),  # orange
    "Aorta": (1.0, 1.0, 0.0),  # yellow
    "Pulmonary_Artery": (0.6, 0.0, 0.6),  # purple
}
PERICARDIUM_COLOR: tuple[float, float, float] = (0.0, 0.75, 0.75)  # cyan
LA_FAT_COLOR_3D: tuple[float, float, float] = (1.0, 0.84, 0.0)  # gold
UNASSIGNED_FAT_COLOR: tuple[float, float, float] = (0.7, 0.7, 0.7)  # gray

# Canonical order of Partition Anchors (1-indexed labels).
_CANONICAL_ANCHORS: list[str] = [
    "LA",
    "LV",
    "RA",
    "RV",
    "Aorta",
    "Pulmonary_Artery",
]

# ---------------------------------------------------------------------------
# Dashboard output dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DashboardOutput:
    """Path references for all QA dashboard artifacts.

    Attributes
    ----------
    output_dir:
        Directory where dashboard files were saved.
    slice_gallery_path:
        Path to the slice gallery PNG.
    fat_overlay_path:
        Path to the fat overlay PNG.
    summary_table_path:
        Path to the summary text file (TXT).
    summary_html_path:
        Path to the combined dashboard HTML file.
    """

    output_dir: str
    slice_gallery_path: str
    fat_overlay_path: str
    summary_table_path: str
    summary_html_path: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_dashboard(
    ct_array: np.ndarray,
    anchor_masks: dict[str, np.ndarray],
    pericardium_result: PericardiumResult,
    partition_result: PartitionResult,
    fat_threshold_result: FatThresholdResult,
    cleanup_result: CleanupResult,
    quality_flags: list[QualityFlag],
    config: PipelineConfig,
    patient_id: str,
    output_dir: str,
    spacing: tuple[float, float, float] = (1.5, 1.5, 1.5),
) -> DashboardOutput:
    """Generate the complete QA dashboard.

    Produces 4 output artifacts in *output_dir*:
      1. ``slice_gallery.png`` — 3-plane multi-anchor overlay
      2. ``fat_overlay.png`` — 3-plane color-coded fat overlay
      3. ``summary.txt`` + ``summary.csv`` — numeric tables
      4. ``dashboard.html`` — combined self-contained HTML page

    Returns
    -------
    DashboardOutput
        Paths to all generated artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Heart centre from pericardium centroid.
    centre = _compute_heart_center(pericardium_result.mask)

    # Set of excluded anchors (not shown in overlays).
    excluded: set[str] = set(partition_result.excluded_anchors)

    # ---- Component 1: Slice Gallery ------------------------------------------
    gallery_path = os.path.join(output_dir, "slice_gallery.png")
    _build_slice_gallery(
        ct_array=ct_array,
        anchor_masks=anchor_masks,
        pericardium_mask=pericardium_result.mask,
        centre=centre,
        excluded=excluded,
        save_path=gallery_path,
    )

    # ---- Component 2: Fat Overlay --------------------------------------------
    fat_overlay_path = os.path.join(output_dir, "fat_overlay.png")
    _build_fat_overlay(
        ct_array=ct_array,
        pericardium_mask=pericardium_result.mask,
        anchor_assignments=partition_result.anchor_assignments,
        all_fat_mask=partition_result.all_fat_mask,
        centre=centre,
        excluded=excluded,
        save_path=fat_overlay_path,
    )

    # ---- Component 3: Numeric Summary ----------------------------------------
    summary_path = os.path.join(output_dir, "summary.txt")
    csv_path = os.path.join(output_dir, "summary.csv")
    _build_numeric_summary(
        patient_id=patient_id,
        fat_threshold_result=fat_threshold_result,
        pericardium_result=pericardium_result,
        partition_result=partition_result,
        cleanup_result=cleanup_result,
        quality_flags=quality_flags,
        summary_path=summary_path,
        csv_path=csv_path,
        spacing=spacing,
    )

    # ---- Component 4: Combined HTML ------------------------------------------
    combined_path = os.path.join(output_dir, "dashboard.html")
    _build_combined_html(
        patient_id=patient_id,
        gallery_rel=os.path.basename(gallery_path),
        fat_overlay_rel=os.path.basename(fat_overlay_path),
        summary_path=summary_path,
        save_path=combined_path,
    )

    return DashboardOutput(
        output_dir=output_dir,
        slice_gallery_path=gallery_path,
        fat_overlay_path=fat_overlay_path,
        summary_table_path=summary_path,
        summary_html_path=combined_path,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_heart_center(pericardium_mask: np.ndarray) -> tuple[int, int, int]:
    """Return the centroid of the pericardium mask as integer voxel indices."""
    if np.any(pericardium_mask):
        com = center_of_mass(pericardium_mask)
        return (int(round(com[0])), int(round(com[1])), int(round(com[2])))
    shape = pericardium_mask.shape
    return (shape[0] // 2, shape[1] // 2, shape[2] // 2)


def _extract_slice(
    volume: np.ndarray,
    centre: tuple[int, int, int],
    dim: int,
) -> np.ndarray:
    """Extract a 2D slice from a 3D volume at *centre* along axis *dim*.

    *dim*: 0=axial (z), 1=coronal (y), 2=sagittal (x).
    """
    if dim == 0:
        return volume[centre[0], :, :]
    if dim == 1:
        return volume[:, centre[1], :]
    return volume[:, :, centre[2]]


# ---- Component 1: Slice Gallery ----------------------------------------


def _build_slice_gallery(
    ct_array: np.ndarray,
    anchor_masks: dict[str, np.ndarray],
    pericardium_mask: np.ndarray,
    centre: tuple[int, int, int],
    excluded: set[str],
    save_path: str,
) -> None:
    """Create a 3x2 matplotlib figure: CT alone | CT + overlays."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 10))

    planes = [
        ("Axial", 0),
        ("Coronal", 1),
        ("Sagittal", 2),
    ]

    for row_idx, (plane_name, dim) in enumerate(planes):
        ct_slice = _extract_slice(ct_array, centre, dim)
        peri_slice = _extract_slice(pericardium_mask, centre, dim)

        # Left: CT alone
        ax_left = axes[row_idx, 0]
        ax_left.imshow(ct_slice, cmap="gray", aspect="auto")
        ax_left.set_title(f"{plane_name} -- CT", fontsize=10)
        ax_left.axis("off")

        # Right: CT + overlays
        ax_right = axes[row_idx, 1]
        ax_right.imshow(ct_slice, cmap="gray", aspect="auto")
        ax_right.set_title(f"{plane_name} -- Overlay", fontsize=10)
        ax_right.axis("off")

        # Pericardium outline (dashed cyan)
        if np.any(peri_slice):
            ax_right.contour(
                peri_slice,
                levels=[0.5],
                colors=[PERICARDIUM_COLOR],
                linewidths=1.0,
                alpha=0.6,
                linestyles="dashed",
            )

        # Each anchor outline
        for anchor_name in _CANONICAL_ANCHORS:
            if anchor_name in excluded or anchor_name not in anchor_masks:
                continue
            anchor_slice = _extract_slice(
                anchor_masks[anchor_name], centre, dim
            )
            if not np.any(anchor_slice):
                continue
            color = ANCHOR_COLORS.get(anchor_name, (1.0, 1.0, 1.0))
            ax_right.contour(
                anchor_slice,
                levels=[0.5],
                colors=[color],
                linewidths=1.5,
                alpha=0.85,
            )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- Component 2: Fat Overlay ------------------------------------------


def _build_fat_overlay(
    ct_array: np.ndarray,
    pericardium_mask: np.ndarray,
    anchor_assignments: np.ndarray,
    all_fat_mask: np.ndarray,
    centre: tuple[int, int, int],
    excluded: set[str],
    save_path: str,
) -> None:
    """Create a 3x2 figure: CT alone | CT + fat color-coded by anchor."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 10))

    # Build label->color mapping for integer labels 1..6.
    label_to_color: dict[int, tuple[float, float, float]] = {}
    for idx, name in enumerate(_CANONICAL_ANCHORS, start=1):
        if name in excluded:
            # Excluded anchors shown as dark gray (should not appear if
            # excluded, but guard against unexpected assignments).
            label_to_color[idx] = (0.3, 0.3, 0.3)
        else:
            label_to_color[idx] = ANCHOR_COLORS.get(name, (0.5, 0.5, 0.5))

    planes = [
        ("Axial", 0),
        ("Coronal", 1),
        ("Sagittal", 2),
    ]

    for row_idx, (plane_name, dim) in enumerate(planes):
        ct_slice = _extract_slice(ct_array, centre, dim)
        assign_slice = _extract_slice(anchor_assignments, centre, dim)
        fat_slice = _extract_slice(all_fat_mask, centre, dim)

        # Left: CT alone
        ax_left = axes[row_idx, 0]
        ax_left.imshow(ct_slice, cmap="gray", aspect="auto")
        ax_left.set_title(f"{plane_name} -- CT", fontsize=10)
        ax_left.axis("off")

        # Right: CT + fat overlay
        ax_right = axes[row_idx, 1]
        ax_right.imshow(ct_slice, cmap="gray", aspect="auto")
        ax_right.set_title(f"{plane_name} -- Fat", fontsize=10)
        ax_right.axis("off")

        # Build RGBA overlay for fat voxels
        overlay = np.zeros((*ct_slice.shape, 4), dtype=np.float32)

        # Pericardium outline for spatial reference
        peri_slice = _extract_slice(pericardium_mask, centre, dim)
        if np.any(peri_slice):
            ax_right.contour(
                peri_slice,
                levels=[0.5],
                colors=[PERICARDIUM_COLOR],
                linewidths=0.8,
                alpha=0.4,
                linestyles="dashed",
            )

        # Assigned fat: color per anchor label
        for label_val, color in label_to_color.items():
            mask_2d = assign_slice == label_val
            if np.any(mask_2d):
                overlay[mask_2d, 0] = color[0]
                overlay[mask_2d, 1] = color[1]
                overlay[mask_2d, 2] = color[2]
                overlay[mask_2d, 3] = 0.65

        # Unassigned fat: label == 0 but in all_fat_mask -> gray
        unassigned_mask = (assign_slice == 0) & fat_slice
        if np.any(unassigned_mask):
            overlay[unassigned_mask, 0] = UNASSIGNED_FAT_COLOR[0]
            overlay[unassigned_mask, 1] = UNASSIGNED_FAT_COLOR[1]
            overlay[unassigned_mask, 2] = UNASSIGNED_FAT_COLOR[2]
            overlay[unassigned_mask, 3] = 0.5

        ax_right.imshow(overlay, aspect="auto")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---- Component 3: Numeric Summary --------------------------------------


def _build_numeric_summary(
    patient_id: str,
    fat_threshold_result: FatThresholdResult,
    pericardium_result: PericardiumResult,
    partition_result: PartitionResult,
    cleanup_result: CleanupResult,
    quality_flags: list[QualityFlag],
    summary_path: str,
    csv_path: str,
    spacing: tuple[float, float, float],
) -> None:
    """Write a human-readable TXT summary and a machine-readable CSV."""

    voxel_volume_ml = spacing[0] * spacing[1] * spacing[2] / 1000.0
    pericardium_volume_ml = (
        np.count_nonzero(pericardium_result.mask) * voxel_volume_ml
    )

    total = max(partition_result.total_fat_volume_ml, 0.001)
    unassigned_pct = (
        partition_result.unassigned_volume_ml / total * 100.0
    )

    # -- Build text lines --------------------------------------------------
    lines: list[str] = []
    _add = lines.append

    _add("=" * 50)
    _add("QA DASHBOARD SUMMARY")
    _add(f"Patient: {patient_id}")
    _add("=" * 50)
    _add("")

    # Fat threshold
    _add("--- Fat Threshold ---")
    _add(f"  Method:              {fat_threshold_result.method}")
    _add(f"  Mean HU:             {fat_threshold_result.mean_hu:.2f}")
    _add(f"  Sigma HU:            {fat_threshold_result.sigma_hu:.2f}")
    _add(f"  Range (low):         {fat_threshold_result.hu_low:.2f}")
    _add(f"  Range (high):        {fat_threshold_result.hu_high:.2f}")
    _add(f"  Fallback triggered:  {fat_threshold_result.fallback_triggered}")
    if fat_threshold_result.fallback_reason:
        _add(f"  Fallback reason:     {fat_threshold_result.fallback_reason}")
    _add("")

    # Pericardium
    _add("--- Pericardium ---")
    _add(f"  Method:              {pericardium_result.method}")
    _add(f"  Fallback triggered:  {pericardium_result.fallback_triggered}")
    _add(f"  Volume (ml):         {pericardium_volume_ml:.2f}")
    if pericardium_result.fallback_reason:
        _add(f"  Fallback reason:     {pericardium_result.fallback_reason}")
    _add("")

    # Per-anchor volumes
    _add("--- Per-Anchor Volumes ---")
    _add(f"  {'Anchor':<20} {'Volume (ml)':<15} {'Share (%)':<10}")
    _add(f"  {'-'*20} {'-'*15} {'-'*10}")
    for anchor_name in _CANONICAL_ANCHORS:
        vol = partition_result.anchor_volumes_ml.get(anchor_name, 0.0)
        share = partition_result.anchor_shares.get(anchor_name, 0.0)
        _add(f"  {anchor_name:<20} {vol:<15.2f} {share:<10.1f}")
    _add("")

    # LA Fat summary
    la_vol = partition_result.anchor_volumes_ml.get("LA", 0.0)
    la_share = partition_result.anchor_shares.get("LA", 0.0)
    _add(f"LA Fat Volume:         {la_vol:.2f} ml ({la_share:.1f}% of epicardial fat)")
    _add("")

    # Unassigned fat
    _add(
        f"Unassigned Fat:       "
        f"{partition_result.unassigned_volume_ml:.2f} ml "
        f"({unassigned_pct:.1f}%)"
    )
    _add("")

    # Cleanup
    _add("--- Cleanup ---")
    _add(f"  Islands removed:         {cleanup_result.islands_removed}")
    _add(
        f"  Total volume removed:    "
        f"{cleanup_result.total_removed_volume_mm3:.2f} mm^3"
    )
    _add("")

    # Excluded anchors
    _add("--- Excluded Anchors ---")
    if partition_result.excluded_anchors:
        for anchor in partition_result.excluded_anchors:
            reason = partition_result.exclusion_reasons.get(
                anchor, "no reason"
            )
            _add(f"  {anchor}: {reason}")
    else:
        _add("  None")
    _add("")

    # Quality flags
    _add("--- Quality Flags ---")
    if quality_flags:
        sev_order = {"high": 0, "medium": 1, "low": 2}
        sorted_flags = sorted(quality_flags, key=lambda f: sev_order.get(f.severity, 99))
        for flag in sorted_flags:
            _add(f"  [{flag.severity}] {flag.concern}")
            _add(f"    {flag.detail}")
            if flag.threshold_value is not None:
                _add(f"    Threshold: {flag.threshold_value}")
            if flag.actual_value is not None:
                _add(f"    Actual:    {flag.actual_value}")
            _add("")
    else:
        _add("  None")
    _add("")

    _add("=" * 50)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # -- Write CSV ---------------------------------------------------------
    csv_rows: list[list[str]] = []
    _cr = csv_rows.append
    _cr(["Category", "Key", "Value"])

    _cr(["Patient", "id", patient_id])
    _cr(["Fat Threshold", "method", fat_threshold_result.method])
    _cr(["Fat Threshold", "mean_hu", str(fat_threshold_result.mean_hu)])
    _cr(["Fat Threshold", "sigma_hu", str(fat_threshold_result.sigma_hu)])
    _cr(["Fat Threshold", "hu_low", str(fat_threshold_result.hu_low)])
    _cr(["Fat Threshold", "hu_high", str(fat_threshold_result.hu_high)])
    _cr(
        [
            "Fat Threshold",
            "fallback_triggered",
            str(fat_threshold_result.fallback_triggered),
        ]
    )
    if fat_threshold_result.fallback_reason:
        _cr(
            [
                "Fat Threshold",
                "fallback_reason",
                fat_threshold_result.fallback_reason,
            ]
        )

    _cr(["Pericardium", "method", pericardium_result.method])
    _cr(
        [
            "Pericardium",
            "fallback_triggered",
            str(pericardium_result.fallback_triggered),
        ]
    )
    _cr(
        [
            "Pericardium",
            "volume_ml",
            f"{pericardium_volume_ml:.2f}",
        ]
    )

    for anchor_name in _CANONICAL_ANCHORS:
        vol = partition_result.anchor_volumes_ml.get(anchor_name, 0.0)
        share = partition_result.anchor_shares.get(anchor_name, 0.0)
        _cr([anchor_name, "volume_ml", f"{vol:.2f}"])
        _cr([anchor_name, "share_pct", f"{share:.1f}"])

    _cr(["LA Fat", "volume_ml", f"{la_vol:.2f}"])
    _cr(["LA Fat", "share_pct", f"{la_share:.1f}"])
    _cr(
        ["Unassigned Fat", "volume_ml", f"{partition_result.unassigned_volume_ml:.2f}"]
    )
    _cr(["Unassigned Fat", "share_pct", f"{unassigned_pct:.1f}"])

    _cr(["Cleanup", "islands_removed", str(cleanup_result.islands_removed)])
    _cr(
        [
            "Cleanup",
            "total_removed_volume_mm3",
            f"{cleanup_result.total_removed_volume_mm3:.2f}",
        ]
    )

    for anchor_name in partition_result.excluded_anchors:
        _cr(
            [
                "Excluded Anchor",
                anchor_name,
                partition_result.exclusion_reasons.get(anchor_name, ""),
            ]
        )

    sev_order = {"high": 0, "medium": 1, "low": 2}
    sorted_out = sorted(
        quality_flags, key=lambda f: sev_order.get(f.severity, 99)
    )
    for i, flag in enumerate(sorted_out):
        prefix = f"Quality Flag {i}"
        _cr([prefix, "severity", flag.severity])
        _cr([prefix, "concern", flag.concern])
        _cr([prefix, "detail", flag.detail])
        if flag.threshold_value is not None:
            _cr([prefix, "threshold_value", str(flag.threshold_value)])
        if flag.actual_value is not None:
            _cr([prefix, "actual_value", str(flag.actual_value)])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)


# ---- Component 4: Combined HTML ----------------------------------------


def _build_combined_html(
    patient_id: str,
    gallery_rel: str,
    fat_overlay_rel: str,
    summary_path: str,
    save_path: str,
) -> None:
    """Write a self-contained combined dashboard HTML page.

    All image references are relative paths -- the page is meant to be
    viewed from the dashboard output directory.
    """
    # Read the summary text for embedding into a <pre> block.
    try:
        with open(summary_path, encoding="utf-8") as f:
            summary_text = f.read()
    except Exception:
        summary_text = "(summary not available)"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>QA Dashboard -- {patient_id}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; background: #fafafa; color: #333; }}
  h1 {{ color: #1a1a2e; border-bottom: 3px solid #1a1a2e; padding-bottom: 8px; }}
  h2 {{ color: #16213e; border-bottom: 2px solid #e0e0e0; padding-bottom: 4px; margin-top: 32px; }}
  .section {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 16px; margin: 16px 0; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
  pre {{ background: #f4f4f4; padding: 12px; border-radius: 4px; overflow-x: auto; font-size: 13px; line-height: 1.5; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
  th {{ background: #f0f0f0; }}
  .footer {{ text-align: center; color: #999; font-size: 12px; margin-top: 32px; }}
</style>
</head>
<body>
<h1>QA Dashboard -- {patient_id}</h1>

<div class="section">
  <h2>1. Slice Gallery</h2>
  <p>Axial, coronal, and sagittal views through the pericardium centre.
     Left column: CT alone. Right column: CT with Partition Anchors and Pericardium overlays.</p>
  <img src="{gallery_rel}" alt="Slice Gallery">
</div>

<div class="section">
  <h2>2. Fat Overlay</h2>
  <p>Same three planes with Epicardial Fat color-coded by assigned Partition Anchor.
     Unassigned fat is shown in gray.</p>
  <img src="{fat_overlay_rel}" alt="Fat Overlay">
</div>

<div class="section">
  <h2>3. Numeric Summary</h2>
  <pre>{summary_text}</pre>
</div>

<div class="footer">
  Generated by la-fat-segmentation QA Dashboard
</div>
</body>
</html>"""

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(html)
