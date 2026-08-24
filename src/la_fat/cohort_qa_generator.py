"""Cohort QA Viewer Generator — Full Production Architecture.

Implements the complete 3-variant layout from qa_viewer_prototype.html:
1. 📊 Cohort Scorecard & Focus Inspector (Triage table, biometrics cards, Gaussian fit, landmark filmstrip).
2. 🩻 Multi-Planar PACS (Synchronized 3-plane orthogonal matrix, 6-channel layer toggles, curtain wipe, W/L presets).
3. 🧊 3D Presentation Studio (Interactive true 3D WebGL cardiac stage with real marching-cubes surface meshes, camera presets, and layer toggles).
"""

from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
from skimage.measure import marching_cubes


def hu_to_grayscale(ct_slice: np.ndarray, window_center: float = 40.0, window_width: float = 350.0) -> np.ndarray:
    """Apply medical CT window/level transform to 8-bit grayscale."""
    min_hu = window_center - window_width / 2.0
    max_hu = window_center + window_width / 2.0
    scaled = np.clip((ct_slice - min_hu) / (max_hu - min_hu), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def _to_webp_b64(img: Image.Image, quality: int = 80) -> str:
    """Convert PIL image to base64-encoded WebP string for minimal HTML payload."""
    buf = BytesIO()
    img.save(buf, format="WEBP", quality=quality)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_slice_layers(
    ct_volume: np.ndarray,
    pericardium_mask: Optional[np.ndarray],
    anchor_masks: Optional[Dict[str, np.ndarray]],
    partition_assignments: Optional[np.ndarray],
    la_fat_mask: Optional[np.ndarray],
    slice_idx: int,
    plane: str = "axial",
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, str]:
    """Render the base CT image and transparent overlay masks for a 2D slice."""
    def extract_slice(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if arr is None:
            return None
        if plane == "axial":
            return arr[slice_idx, :, :]
        elif plane == "coronal":
            return arr[:, slice_idx, :]
        else:
            return arr[:, :, slice_idx]

    ct_s = extract_slice(ct_volume)
    if ct_s is None:
        raise ValueError(f"Slice index {slice_idx} out of bounds for plane {plane}")

    h, w = ct_s.shape
    peri_s = extract_slice(pericardium_mask)
    la_fat_s = extract_slice(la_fat_mask)
    partition_s = extract_slice(partition_assignments)

    # 1. Base CT Grayscale
    ct_gray = hu_to_grayscale(ct_s, 40.0, 350.0)
    img_ct = Image.fromarray(ct_gray, mode="L").convert("RGB")

    # 2. Pericardium Outline Mask (Lime Green #22c55e outline)
    peri_contour = np.zeros((h, w, 4), dtype=np.uint8)
    if peri_s is not None and np.any(peri_s):
        try:
            outline = binary_dilation(peri_s) ^ binary_erosion(peri_s)
        except Exception:
            outline = peri_s
        peri_contour[outline] = [34, 197, 94, 240]
    img_peri = Image.fromarray(peri_contour, mode="RGBA")

    # 3. TS 6 Anchors Fill
    anchors_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if anchor_masks is not None:
        la_s = extract_slice(anchor_masks.get("LA"))
        lv_s = extract_slice(anchor_masks.get("LV"))
        ra_s = extract_slice(anchor_masks.get("RA"))
        rv_s = extract_slice(anchor_masks.get("RV"))
        ao_s = extract_slice(anchor_masks.get("Aorta"))
        pa_s = extract_slice(anchor_masks.get("Pulmonary_Artery"))

        if la_s is not None: anchors_rgba[la_s > 0] = [239, 68, 68, 140]
        if lv_s is not None: anchors_rgba[lv_s > 0] = [59, 130, 246, 140]
        if ra_s is not None: anchors_rgba[ra_s > 0] = [6, 182, 212, 140]
        if rv_s is not None: anchors_rgba[rv_s > 0] = [249, 115, 22, 140]
        if ao_s is not None: anchors_rgba[ao_s > 0] = [217, 70, 239, 140]
        if pa_s is not None: anchors_rgba[pa_s > 0] = [234, 179, 8, 140]
    img_anchors = Image.fromarray(anchors_rgba, mode="RGBA")

    # 4. Partitioned EAT (Color-coded by assigned anchor 1..6)
    partition_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if partition_s is not None:
        partition_rgba[partition_s == 1] = [239, 68, 68, 180]   # LA Fat
        partition_rgba[partition_s == 2] = [59, 130, 246, 180]  # LV Fat
        partition_rgba[partition_s == 3] = [6, 182, 212, 180]   # RA Fat
        partition_rgba[partition_s == 4] = [249, 115, 22, 180]  # RV Fat
        partition_rgba[partition_s == 5] = [217, 70, 239, 180]  # Ao Fat
        partition_rgba[partition_s == 6] = [234, 179, 8, 180]   # PA Fat
    img_partition = Image.fromarray(partition_rgba, mode="RGBA")

    # 5. Final LA Fat (Neon Gold #facc15)
    la_fat_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if la_fat_s is not None:
        la_fat_rgba[la_fat_s > 0] = [250, 204, 21, 210]
    img_la_fat = Image.fromarray(la_fat_rgba, mode="RGBA")

    # 6. Partial Volume Tail (0 to -30 HU)
    pv_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if partition_s is not None:
        pv_mask = (partition_s == 1) & (ct_s > -30.0) & (ct_s <= 0.0)
        pv_rgba[pv_mask] = [168, 85, 247, 200]
    img_pv = Image.fromarray(pv_rgba, mode="RGBA")

    # Correct anisotropic slice display aspect ratio if non-square voxels
    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    if plane == "coronal" and abs(sz - sx) > 1e-3:
        target_h = int(round(h * (sz / sx)))
        img_ct = img_ct.resize((w, target_h), Image.Resampling.BILINEAR)
        img_peri = img_peri.resize((w, target_h), Image.Resampling.NEAREST)
        img_anchors = img_anchors.resize((w, target_h), Image.Resampling.NEAREST)
        img_partition = img_partition.resize((w, target_h), Image.Resampling.NEAREST)
        img_la_fat = img_la_fat.resize((w, target_h), Image.Resampling.NEAREST)
        img_pv = img_pv.resize((w, target_h), Image.Resampling.NEAREST)
    elif plane == "sagittal" and abs(sz - sy) > 1e-3:
        target_h = int(round(h * (sz / sy)))
        img_ct = img_ct.resize((w, target_h), Image.Resampling.BILINEAR)
        img_peri = img_peri.resize((w, target_h), Image.Resampling.NEAREST)
        img_anchors = img_anchors.resize((w, target_h), Image.Resampling.NEAREST)
        img_partition = img_partition.resize((w, target_h), Image.Resampling.NEAREST)
        img_la_fat = img_la_fat.resize((w, target_h), Image.Resampling.NEAREST)
        img_pv = img_pv.resize((w, target_h), Image.Resampling.NEAREST)

    return {
        "ct": _to_webp_b64(img_ct),
        "peri": _to_webp_b64(img_peri),
        "anchors": _to_webp_b64(img_anchors),
        "partition": _to_webp_b64(img_partition),
        "la_fat": _to_webp_b64(img_la_fat),
        "pv_zone": _to_webp_b64(img_pv),
    }


def extract_3d_meshes(
    la_fat_mask: Optional[np.ndarray],
    pericardium_mask: Optional[np.ndarray],
    anchor_masks: Optional[Dict[str, np.ndarray]],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, Any]:
    """Extract high-resolution anti-aliased 3D surface meshes for WebGL 3D studio."""
    structures: Dict[str, np.ndarray] = {}
    if la_fat_mask is not None and np.any(la_fat_mask):
        structures["la_fat"] = la_fat_mask
    if pericardium_mask is not None and np.any(pericardium_mask):
        structures["peri"] = pericardium_mask
    if anchor_masks is not None:
        if "LA" in anchor_masks and np.any(anchor_masks["LA"]):
            structures["la"] = anchor_masks["LA"]
        if "LV" in anchor_masks and np.any(anchor_masks["LV"]):
            structures["lv"] = anchor_masks["LV"]
        if "RA" in anchor_masks and np.any(anchor_masks["RA"]):
            structures["ra"] = anchor_masks["RA"]
        if "RV" in anchor_masks and np.any(anchor_masks["RV"]):
            structures["rv"] = anchor_masks["RV"]
        if "Aorta" in anchor_masks and np.any(anchor_masks["Aorta"]):
            structures["ao"] = anchor_masks["Aorta"]
        if "Pulmonary_Artery" in anchor_masks and np.any(anchor_masks["Pulmonary_Artery"]):
            structures["pa"] = anchor_masks["Pulmonary_Artery"]

    # Compute global center of anatomy
    all_centers = []
    for mask in structures.values():
        coords = np.argwhere(mask)
        if len(coords):
            all_centers.append(np.mean(coords, axis=0))
    center = np.mean(all_centers, axis=0) if all_centers else np.array([0.0, 0.0, 0.0])

    sx, sy, sz = float(spacing[0]), float(spacing[1]), float(spacing[2])
    mesh_payload: Dict[str, Any] = {}
    for name, mask in structures.items():
        try:
            # Gentle anti-aliasing Gaussian filter to eliminate voxel stepping
            smooth_arr = gaussian_filter(mask.astype(float), sigma=0.7)
            # Use step_size=2 on large native CT grids (>200 voxels) to optimize WebGL polygon count
            step = 2 if max(mask.shape) > 200 else 1
            verts, faces, _, _ = marching_cubes(smooth_arr, level=0.5, step_size=step)
            
            # verts has shape (N, 3): [Z_idx, Y_idx, X_idx]
            # Center and scale to physical millimeters in (X, Y, Z)
            z_mm = (verts[:, 0] - center[0]) * sz
            y_mm = (verts[:, 1] - center[1]) * sy
            x_mm = (verts[:, 2] - center[2]) * sx
            
            # Three.js Anatomical Coordinates:
            # X: Patient Left (+) / Right (-)
            # Y: Patient Superior (+) / Inferior (-) (Up)
            # Z: Patient Posterior (-) / Anterior (+) (Front)
            reordered_verts = np.column_stack([x_mm, z_mm, -y_mm])

            mesh_payload[name] = {
                "v": np.round(reordered_verts, 1).flatten().tolist(),
                "f": faces.flatten().tolist(),
            }
        except Exception:
            pass

    return mesh_payload


def extract_patient_qa_record(
    patient_id: str,
    ct_volume: np.ndarray,
    la_fat_mask: np.ndarray,
    pericardium_mask: Optional[np.ndarray],
    anchor_masks: Optional[Dict[str, np.ndarray]],
    partition_assignments: Optional[np.ndarray],
    metrics: Dict[str, Any],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, Any]:
    """Extract complete multi-planar slice series, metrics, and 3D surface meshes for a patient."""
    nz, ny, nx = ct_volume.shape

    # Find LA bounding box center
    if la_fat_mask is not None and np.any(la_fat_mask):
        z_indices = np.where(np.any(la_fat_mask, axis=(1, 2)))[0]
        y_indices = np.where(np.any(la_fat_mask, axis=(0, 2)))[0]
        x_indices = np.where(np.any(la_fat_mask, axis=(0, 1)))[0]

        cz = int(np.median(z_indices)) if len(z_indices) else nz // 2
        cy = int(np.median(y_indices)) if len(y_indices) else ny // 2
        cx = int(np.median(x_indices)) if len(x_indices) else nx // 2

        z_min = max(0, int(z_indices[0]) - 5)
        z_max = min(nz - 1, int(z_indices[-1]) + 5)
        y_min = max(0, int(y_indices[0]) - 10)
        y_max = min(ny - 1, int(y_indices[-1]) + 10)
        x_min = max(0, int(x_indices[0]) - 10)
        x_max = min(nx - 1, int(x_indices[-1]) + 10)
    else:
        cz, cy, cx = nz // 2, ny // 2, nx // 2
        z_min, z_max = max(0, cz - 15), min(nz - 1, cz + 15)
        y_min, y_max = max(0, cy - 20), min(ny - 1, cy + 20)
        x_min, x_max = max(0, cx - 20), min(nx - 1, cx + 20)

    # Sample slices evenly across ranges (up to ~25 axial, 15 coronal, 15 sagittal)
    axial_indices = np.linspace(z_min, z_max, min(25, z_max - z_min + 1), dtype=int)
    coronal_indices = np.linspace(y_min, y_max, min(15, y_max - y_min + 1), dtype=int)
    sagittal_indices = np.linspace(x_min, x_max, min(15, x_max - x_min + 1), dtype=int)

    slices_axial = {}
    for z in axial_indices:
        slices_axial[str(z)] = render_slice_layers(
            ct_volume, pericardium_mask, anchor_masks, partition_assignments, la_fat_mask, z, "axial", spacing=spacing
        )

    slices_coronal = {}
    for y in coronal_indices:
        slices_coronal[str(y)] = render_slice_layers(
            ct_volume, pericardium_mask, anchor_masks, partition_assignments, la_fat_mask, y, "coronal", spacing=spacing
        )

    slices_sagittal = {}
    for x in sagittal_indices:
        slices_sagittal[str(x)] = render_slice_layers(
            ct_volume, pericardium_mask, anchor_masks, partition_assignments, la_fat_mask, x, "sagittal", spacing=spacing
        )

    # Landmark slice indices: Apex, Mid-LV, Mid-LA, Mitral Plane, Aorta
    step = max(1, (z_max - z_min) // 5)
    landmark_slices = [
        int(z_min + step // 2),
        int(z_min + step),
        int(cz),
        int(cz + step),
        int(z_max - step // 2),
    ]

    metrics_copy = dict(metrics)
    metrics_copy["landmark_slices"] = landmark_slices

    # Extract true 3D surface meshes with exact physical spacing
    meshes_3d = extract_3d_meshes(la_fat_mask, pericardium_mask, anchor_masks, spacing=spacing)

    return {
        "id": str(patient_id),
        "metrics": metrics_copy,
        "axial_range": [int(axial_indices[0]), int(axial_indices[-1])],
        "coronal_range": [int(coronal_indices[0]), int(coronal_indices[-1])],
        "sagittal_range": [int(sagittal_indices[0]), int(sagittal_indices[-1])],
        "default_axial": int(cz),
        "slices": {
            "axial": slices_axial,
            "coronal": slices_coronal,
            "sagittal": slices_sagittal,
        },
        "meshes": meshes_3d,
    }


def generate_cohort_qa_html(cohort_records: Dict[str, Any], output_html_path: str) -> None:
    """Generate and write the standalone cohort_qa_viewer.html dashboard."""
    patients_json = json.dumps(cohort_records)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LA-FAT QA Viewer — Verified Production Layout</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<style>
:root {{
  --bg-primary: #0a0d14;
  --bg-secondary: #121824;
  --bg-card: #182234;
  --bg-glass: rgba(24, 34, 52, 0.90);
  --border-color: #26354a;
  --border-focus: #3b82f6;
  --text-main: #f1f5f9;
  --text-muted: #94a3b8;
  --text-dim: #64748b;
  --accent-blue: #3b82f6;
  --accent-gold: #facc15;
  --accent-green: #22c55e;
  --accent-red: #ef4444;
  --accent-purple: #a855f7;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  --font-sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}}

* {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  background-color: var(--bg-primary);
  color: var(--text-main);
  font-family: var(--font-sans);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  user-select: none;
}}

/* Top App Header */
header {{
  height: 54px;
  background-color: var(--bg-secondary);
  border-bottom: 1px solid var(--border-color);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px;
  z-index: 50;
}}

.brand {{
  display: flex;
  align-items: center;
  gap: 10px;
  font-weight: 700;
  font-size: 15px;
  letter-spacing: 0.5px;
}}
.brand-badge {{
  background: linear-gradient(135deg, #ef4444, #f59e0b);
  color: white;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
}}

/* Top Navigation Tabs */
.top-nav-tabs {{
  display: flex;
  background: var(--bg-primary);
  padding: 3px;
  border-radius: 8px;
  border: 1px solid var(--border-color);
  gap: 4px;
}}
.top-tab-btn {{
  background: transparent;
  color: var(--text-muted);
  border: none;
  padding: 6px 14px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  transition: all 0.15s ease;
}}
.top-tab-btn:hover {{
  color: var(--text-main);
  background: rgba(255,255,255,0.05);
}}
.top-tab-btn.active {{
  background: var(--accent-blue);
  color: white;
  box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4);
}}

.header-controls {{
  display: flex;
  align-items: center;
  gap: 12px;
}}

.patient-picker {{
  display: flex;
  align-items: center;
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 2px 8px;
  gap: 6px;
}}
.patient-picker select {{
  background: transparent;
  color: var(--text-main);
  border: none;
  font-size: 13px;
  font-weight: 600;
  outline: none;
  cursor: pointer;
}}

.header-stats {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  background: rgba(0,0,0,0.4);
  padding: 4px 10px;
  border-radius: 6px;
  border: 1px solid var(--border-color);
}}
.stat-pill {{
  display: flex;
  align-items: center;
  gap: 4px;
}}
.stat-val {{ font-weight: 700; color: var(--accent-gold); }}

.btn {{
  background: var(--bg-card);
  color: var(--text-main);
  border: 1px solid var(--border-color);
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  transition: all 0.15s ease;
}}
.btn:hover {{
  background: var(--border-color);
  border-color: var(--text-muted);
}}
.btn-primary {{
  background: var(--accent-blue);
  border-color: var(--accent-blue);
  color: white;
}}
.btn-primary:hover {{
  background: #2563eb;
}}

/* Main Workspace Area */
#workspace {{
  flex: 1;
  position: relative;
  overflow: hidden;
}}

.variant-view {{
  position: absolute;
  inset: 0;
  display: none;
  padding: 12px;
  gap: 12px;
}}
.variant-view.active {{
  display: flex;
}}

/* =========================================================================
   VARIANT A: Radiology PACS Layout (3-Panel Synchronized Orthogonal Matrix)
   ========================================================================= */
#variant-a {{
  display: none;
  grid-template-columns: 280px 1fr 340px;
  gap: 12px;
  height: 100%;
}}
#variant-a.active {{
  display: grid;
}}

.sidebar-panel {{
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}}
.panel-header {{
  padding: 10px 14px;
  background: var(--bg-card);
  border-bottom: 1px solid var(--border-color);
  font-size: 13px;
  font-weight: 700;
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
.panel-content {{
  padding: 12px;
  overflow-y: auto;
  flex: 1;
}}

.layer-toggle-group {{
  display: flex;
  flex-direction: column;
  gap: 8px;
}}
.layer-row {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 8px;
  background: rgba(255,255,255,0.03);
  border-radius: 6px;
  border: 1px solid transparent;
  transition: all 0.15s ease;
}}
.layer-row:hover {{
  background: rgba(255,255,255,0.06);
  border-color: var(--border-color);
}}
.layer-label {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  font-weight: 500;
}}
.layer-dot {{
  width: 10px;
  height: 10px;
  border-radius: 50%;
}}

.pacs-viewport-grid {{
  display: grid;
  grid-template-rows: 1fr;
  grid-template-columns: 1fr;
  background: #000;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  position: relative;
  overflow: hidden;
}}

.pacs-main-stage {{
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #000;
}}

.viewport-canvas-stack {{
  position: relative;
  width: 512px;
  height: 512px;
  max-width: 90%;
  max-height: 90%;
  aspect-ratio: 1;
  background: #000;
  box-shadow: 0 0 30px rgba(0,0,0,0.8);
}}
.viewport-layer {{
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: contain;
  image-rendering: pixelated;
  pointer-events: none;
}}
.viewport-layer.base-ct {{
  pointer-events: auto;
  cursor: crosshair;
}}

/* Split Screen Curtain Slider */
.curtain-handle {{
  position: absolute;
  top: 0;
  bottom: 0;
  width: 3px;
  background: var(--accent-gold);
  cursor: ew-resize;
  z-index: 30;
  box-shadow: 0 0 10px rgba(250, 204, 21, 0.6);
  display: none;
}}
.curtain-handle::after {{
  content: "⇹";
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 24px;
  height: 24px;
  background: var(--accent-gold);
  color: #000;
  font-weight: bold;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
}}

.pacs-overlay-hud {{
  position: absolute;
  top: 12px;
  left: 12px;
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--accent-gold);
  background: rgba(0,0,0,0.6);
  padding: 4px 8px;
  border-radius: 4px;
  pointer-events: none;
  line-height: 1.4;
}}
.pacs-overlay-tools {{
  position: absolute;
  bottom: 12px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 8px;
  background: var(--bg-glass);
  padding: 6px 12px;
  border-radius: 30px;
  border: 1px solid var(--border-color);
  backdrop-filter: blur(10px);
  z-index: 40;
}}

.orthogonal-side-stack {{
  display: flex;
  flex-direction: column;
  gap: 12px;
  height: 100%;
}}
.ortho-subview {{
  flex: 1;
  background: #000;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}}
.ortho-subview img {{
  width: 100%;
  height: 100%;
  object-fit: contain;
}}
.ortho-title {{
  position: absolute;
  top: 6px;
  left: 6px;
  font-size: 11px;
  font-weight: 700;
  color: var(--text-muted);
  background: rgba(0,0,0,0.7);
  padding: 2px 6px;
  border-radius: 3px;
  z-index: 10;
}}
.ortho-scrubber {{
  position: absolute;
  bottom: 6px;
  left: 6px;
  right: 6px;
  background: rgba(0,0,0,0.7);
  padding: 3px 8px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  gap: 6px;
  z-index: 10;
}}
.ortho-scrubber input {{ flex: 1; }}

/* =========================================================================
   VARIANT B: Cohort Scorecard & Focus Inspector
   ========================================================================= */
#variant-b {{
  display: none;
  grid-template-rows: 170px 1fr;
  grid-template-columns: 340px 1fr;
  gap: 12px;
  height: 100%;
}}
#variant-b.active {{
  display: grid;
}}

.cohort-table-card {{
  grid-column: 1 / -1;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}}
.table-scroll {{
  flex: 1;
  overflow-y: auto;
}}
table.cohort-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  text-align: left;
}}
table.cohort-table th {{
  background: var(--bg-card);
  padding: 8px 12px;
  color: var(--text-muted);
  font-weight: 600;
  border-bottom: 1px solid var(--border-color);
  position: sticky;
  top: 0;
}}
table.cohort-table td {{
  padding: 8px 12px;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}}
table.cohort-table tr.active-patient {{
  background: rgba(59, 130, 246, 0.15);
  border-left: 3px solid var(--accent-blue);
}}
table.cohort-table tr:hover:not(.active-patient) {{
  background: rgba(255,255,255,0.03);
  cursor: pointer;
}}

.badge {{
  display: inline-block;
  padding: 2px 6px;
  border-radius: 4px;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
}}
.badge-pass {{ background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }}
.badge-warn {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
.badge-fail {{ background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}

.biometrics-sidebar {{
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  overflow-y: auto;
}}

.metric-card {{
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  padding: 10px;
}}
.metric-header {{
  font-size: 11px;
  color: var(--text-muted);
  text-transform: uppercase;
  font-weight: 700;
  margin-bottom: 6px;
}}
.metric-big {{
  font-size: 22px;
  font-weight: 800;
  color: var(--accent-gold);
  display: flex;
  align-items: baseline;
  gap: 4px;
}}
.metric-unit {{ font-size: 12px; color: var(--text-muted); font-weight: 500; }}

.filmstrip-container {{
  display: flex;
  gap: 8px;
  overflow-x: auto;
  padding: 8px 0;
}}
.filmstrip-thumb {{
  width: 90px;
  height: 90px;
  border: 2px solid var(--border-color);
  border-radius: 6px;
  cursor: pointer;
  position: relative;
  background: #000;
  flex-shrink: 0;
  transition: all 0.15s ease;
}}
.filmstrip-thumb.active {{
  border-color: var(--accent-gold);
  box-shadow: 0 0 10px rgba(250, 204, 21, 0.4);
}}
.filmstrip-thumb img {{
  width: 100%;
  height: 100%;
  object-fit: cover;
}}
.filmstrip-label {{
  position: absolute;
  bottom: 2px;
  left: 2px;
  right: 2px;
  font-size: 9px;
  background: rgba(0,0,0,0.7);
  padding: 1px 4px;
  border-radius: 2px;
  text-align: center;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}

/* =========================================================================
   VARIANT C: Colleague 3D Presentation Studio (True 3D WebGL Studio)
   ========================================================================= */
#variant-c {{
  display: none;
  grid-template-columns: 300px 1fr;
  gap: 12px;
  height: 100%;
}}
#variant-c.active {{
  display: grid;
}}

.stage-3d {{
  background: radial-gradient(circle at center, #1a2234 0%, #080c14 100%);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}}

#webgl-container {{
  width: 100%;
  height: 100%;
  position: absolute;
  inset: 0;
}}

.presentation-sidebar {{
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  overflow-y: auto;
}}

.slide-deck-card {{
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-left: 4px solid var(--accent-gold);
  border-radius: 6px;
  padding: 12px;
}}
.slide-deck-title {{
  font-size: 14px;
  font-weight: 700;
  color: var(--text-main);
  margin-bottom: 6px;
}}
.slide-deck-body {{
  font-size: 12px;
  color: var(--text-muted);
  line-height: 1.5;
}}
</style>
</head>
<body>

<!-- Header -->
<header>
  <div class="brand">
    <span>LA-FAT QA STUDIO</span>
    <span class="brand-badge">Production QA</span>
  </div>
  
  <!-- Primary Top Navigation Tabs -->
  <div class="top-nav-tabs">
    <button class="top-tab-btn active" id="tab-btn-b" onclick="setVariantById('B')">
      <span>📊</span> Cohort Scorecard
    </button>
    <button class="top-tab-btn" id="tab-btn-a" onclick="setVariantById('A')">
      <span>🩻</span> Multi-Planar PACS
    </button>
    <button class="top-tab-btn" id="tab-btn-c" onclick="setVariantById('C')">
      <span>🧊</span> 3D Colleague Studio
    </button>
  </div>
  
  <div class="header-controls">
    <div class="patient-picker">
      <span style="font-size:11px; color:var(--text-muted); font-weight:700;">SCAN:</span>
      <select id="patient-select" onchange="selectPatient(this.value)">
      </select>
    </div>
    
    <div class="header-stats">
      <div class="stat-pill">Adaptive LA EAT: <span class="stat-val" id="stat-la-vol">-- mL</span></div>
      <span style="color:var(--border-color);">|</span>
      <div class="stat-pill">Scanner Baseline: <span class="stat-val" id="stat-std-vol">-- mL</span></div>
      <span style="color:var(--border-color);">|</span>
      <div class="stat-pill">Quality: <span id="stat-badge" class="badge badge-pass">PASSED</span></div>
    </div>
    
    <button class="btn" id="btn-export-qa" onclick="window.print()">
      <span>📷</span> Export Figure
    </button>
  </div>
</header>

<!-- Main Workspace -->
<div id="workspace">

  <!-- =========================================================================
       VARIANT A: Radiology PACS Layout
       ========================================================================= -->
  <div id="variant-a" class="variant-view">
    <!-- Left Layer Controls -->
    <div class="sidebar-panel">
      <div class="panel-header">
        <span>Segmentation Layers</span>
        <button class="btn" style="padding:2px 6px; font-size:10px;" id="btn-all-layers" onclick="toggleAllLayers()">Toggle All</button>
      </div>
      <div class="panel-content">
        <div class="layer-toggle-group">
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#fff;"></span>
              <span>Base CT Image</span>
            </div>
            <input type="checkbox" id="layer-ct" checked onchange="updateLayerVisibility()">
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#22c55e;"></span>
              <span>Pericardium Sac (Outline)</span>
            </div>
            <input type="checkbox" id="layer-peri" checked onchange="updateLayerVisibility()">
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#ef4444;"></span>
              <span>TS 6 Anchors (Solid)</span>
            </div>
            <input type="checkbox" id="layer-anchors" onchange="updateLayerVisibility()">
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#3b82f6;"></span>
              <span>Partitioned EAT (6-Color)</span>
            </div>
            <input type="checkbox" id="layer-partition" onchange="updateLayerVisibility()">
          </label>
          
          <label class="layer-row" style="background:rgba(250,204,21,0.1); border-color:rgba(250,204,21,0.3);">
            <div class="layer-label">
              <span class="layer-dot" style="background:#facc15;"></span>
              <span style="font-weight:700; color:#facc15;">Final LA Fat Mask</span>
            </div>
            <input type="checkbox" id="layer-la-fat" checked onchange="updateLayerVisibility()">
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#a855f7;"></span>
              <span>Partial Volume Tail (0 to -30 HU)</span>
            </div>
            <input type="checkbox" id="layer-pv" onchange="updateLayerVisibility()">
          </label>
        </div>
        
        <div style="margin-top:20px;">
          <div style="font-size:11px; font-weight:700; color:var(--text-muted); margin-bottom:8px;">WINDOW / LEVEL PRESET</div>
          <div style="display:flex; flex-direction:column; gap:6px;">
            <button class="btn btn-primary" style="justify-content:flex-start;" id="wl-mediastinal">🫁 Mediastinal (L:40, W:350)</button>
            <button class="btn" style="justify-content:flex-start;" id="wl-fat">🧈 Fat Focused (L:-70, W:200)</button>
            <button class="btn" style="justify-content:flex-start;" id="wl-wide">🔍 Wide Dynamic (L:-100, W:500)</button>
          </div>
        </div>
        
        <div style="margin-top:20px;">
          <div style="font-size:11px; font-weight:700; color:var(--text-muted); margin-bottom:8px;">AXIAL SLICE (Z)</div>
          <div style="display:flex; align-items:center; gap:8px;">
            <input type="range" id="slice-slider-a" style="flex:1;" oninput="onAxialSlider(this.value)">
            <span id="slice-num-a" style="font-family:var(--font-mono); font-size:12px; font-weight:bold;">--</span>
          </div>
          <div style="font-size:10px; color:var(--text-dim); margin-top:4px;">Mouse wheel over viewport to scrub</div>
        </div>
      </div>
    </div>
    
    <!-- Center Primary Viewport (Axial) -->
    <div class="pacs-viewport-grid">
      <div class="pacs-main-stage" id="main-stage-a">
        <div class="viewport-canvas-stack" id="canvas-stack-a">
          <img class="viewport-layer base-ct" id="img-ct-a" src="" alt="CT">
          <img class="viewport-layer" id="img-peri-a" src="" alt="Pericardium">
          <img class="viewport-layer" id="img-anchors-a" src="" alt="Anchors" style="display:none;">
          <img class="viewport-layer" id="img-partition-a" src="" alt="Partition" style="display:none;">
          <img class="viewport-layer" id="img-la-fat-a" src="" alt="LA Fat">
          <img class="viewport-layer" id="img-pv-a" src="" alt="PV" style="display:none;">
          <div class="curtain-handle" id="curtain-handle-a"></div>
        </div>
        
        <div class="pacs-overlay-hud" id="hud-a">
          <div>PATIENT: <span id="hud-patient">--</span> (AXIAL Z: <span id="hud-z">--</span>)</div>
          <div>WINDOW: <span id="hud-wl">L: 40, W: 350 HU</span></div>
          <div>LA-EAT: <span id="hud-la-vol">-- mL</span></div>
        </div>
        
        <div class="pacs-overlay-tools">
          <button class="btn" id="btn-curtain-toggle"><span>⇹</span> Curtain Wipe</button>
          <button class="btn" id="btn-zoom-fit" onclick="resetZoom()"><span>🔍</span> Reset Zoom</button>
        </div>
      </div>
    </div>
    
    <!-- Right Orthogonal Multi-Planar Views (Coronal & Sagittal) -->
    <div class="orthogonal-side-stack">
      <div class="ortho-subview">
        <span class="ortho-title">CORONAL PLANE (Y: <span id="coronal-num">--</span>)</span>
        <img id="img-coronal-a" src="" alt="Coronal">
        <div class="ortho-scrubber">
          <span style="font-size:10px; color:#cbd5e1;">Y:</span>
          <input type="range" id="slider-coronal" oninput="onCoronalSlider(this.value)">
        </div>
      </div>
      <div class="ortho-subview">
        <span class="ortho-title">SAGITTAL PLANE (X: <span id="sagittal-num">--</span>)</span>
        <img id="img-sagittal-a" src="" alt="Sagittal">
        <div class="ortho-scrubber">
          <span style="font-size:10px; color:#cbd5e1;">X:</span>
          <input type="range" id="slider-sagittal" oninput="onSagittalSlider(this.value)">
        </div>
      </div>
    </div>
  </div>

  <!-- =========================================================================
       VARIANT B: Cohort Scorecard & Focus Inspector
       ========================================================================= -->
  <div id="variant-b" class="variant-view active">
    <!-- Top Cohort Table -->
    <div class="cohort-table-card">
      <div class="panel-header">
        <span>Cohort Quality Scorecard & Biometrics (10 Patient Benchmark)</span>
        <span style="font-size:11px; color:var(--text-muted);">Use <kbd style="background:#334155; padding:1px 4px; border-radius:3px;">j</kbd> / <kbd style="background:#334155; padding:1px 4px; border-radius:3px;">k</kbd> to cycle scans</span>
      </div>
      <div class="table-scroll">
        <table class="cohort-table">
          <thead>
            <tr>
              <th>Patient ID</th>
              <th>Demographics</th>
              <th>Scanner Baseline (mL)</th>
              <th>Pipeline Adaptive (mL)</th>
              <th>Std [-190,-30] (mL)</th>
              <th>LA Delta vs Baseline</th>
              <th>Total EAT (mL)</th>
              <th>Gaussian Fit (μ ± 2σ)</th>
              <th>Quality Status</th>
            </tr>
          </thead>
          <tbody id="cohort-tbody">
            <!-- Populated via JS -->
          </tbody>
        </table>
      </div>
    </div>
    
    <!-- Left Forensic Details Sidebar -->
    <div class="biometrics-sidebar">
      <div class="metric-card">
        <div class="metric-header">Adaptive LA-EAT Volume</div>
        <div class="metric-big"><span id="b-la-vol">--</span> <span class="metric-unit">mL</span></div>
        <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">
          Includes 1-2 voxel partial volume boundary layer (0 to -30 HU).
        </div>
      </div>

      <div class="metric-card">
        <div class="metric-header">Scanner Software Baseline</div>
        <div class="metric-big"><span id="b-scanner-vol">--</span> <span class="metric-unit">mL</span></div>
        <div style="font-size:11px; color:var(--text-muted); margin-top:4px;" id="b-scanner-delta">
          Discrepancy vs Pipeline: --
        </div>
      </div>
      
      <div class="metric-card">
        <div class="metric-header">Gaussian HU Threshold Fit</div>
        <div style="font-family:var(--font-mono); font-size:12px; color:var(--accent-gold); margin-bottom:4px;" id="b-gauss-fit">
          μ = -- HU, σ = -- HU
        </div>
        <div style="font-size:11px; color:var(--text-muted);">
          Optimal Window: <strong style="color:#fff;" id="b-gauss-win">[--, --] HU</strong>
        </div>
      </div>
      
      <div class="metric-card">
        <div class="metric-header">Quality Concern Checklist</div>
        <div style="display:flex; flex-direction:column; gap:6px; font-size:11px;" id="b-concern-list">
          <!-- Populated via JS -->
        </div>
      </div>
    </div>
    
    <!-- Center Scroller with Landmark Filmstrip -->
    <div style="display:flex; flex-direction:column; gap:8px; background:var(--bg-secondary); border:1px solid var(--border-color); border-radius:8px; padding:10px;">
      <!-- Filmstrip -->
      <div>
        <div style="font-size:11px; font-weight:700; color:var(--text-muted); margin-bottom:4px;">KEY ANATOMICAL LANDMARKS (CLICK TO JUMP)</div>
        <div class="filmstrip-container" id="filmstrip-b">
          <!-- Populated via JS -->
        </div>
      </div>
      
      <!-- Big Center Slice Viewport -->
      <div style="flex:1; background:#000; border-radius:6px; position:relative; display:flex; align-items:center; justify-content:center; overflow:hidden;">
        <div class="viewport-canvas-stack" style="width:400px; height:400px;">
          <img class="viewport-layer base-ct" id="img-ct-b" src="" alt="CT">
          <img class="viewport-layer" id="img-peri-b" src="" alt="Pericardium">
          <img class="viewport-layer" id="img-la-fat-b" src="" alt="LA Fat">
          <img class="viewport-layer" id="img-pv-b" src="" alt="PV" style="display:none;">
        </div>
        <div class="pacs-overlay-hud">AXIAL SLICE: <span id="hud-z-b">--</span></div>
      </div>
    </div>
  </div>

  <!-- =========================================================================
       VARIANT C: Colleague 3D Presentation Studio (True 3D WebGL Studio)
       ========================================================================= -->
  <div id="variant-c" class="variant-view">
    <!-- Left 3D Controls & Anatomical Legend -->
    <div class="sidebar-panel">
      <div class="panel-header">
        <span>3D Chamber Meshes</span>
      </div>
      <div class="panel-content">
        <div class="layer-toggle-group">
          <label class="layer-row" style="background:rgba(250,204,21,0.1); border-color:rgba(250,204,21,0.3);">
            <div class="layer-label">
              <span class="layer-dot" style="background:#facc15;"></span>
              <span style="font-weight:bold; color:#facc15;">LA Fat Volume</span>
            </div>
            <input type="checkbox" id="m3d-la-fat" checked onchange="update3DMeshVisibility()">
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#22c55e;"></span>
              <span>Pericardium Sac Envelope</span>
            </div>
            <input type="checkbox" id="m3d-peri" checked onchange="update3DMeshVisibility()">
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#ef4444;"></span>
              <span>Left Atrium (LA)</span>
            </div>
            <input type="checkbox" id="m3d-la" checked onchange="update3DMeshVisibility()">
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#3b82f6;"></span>
              <span>Left Ventricle (LV)</span>
            </div>
            <input type="checkbox" id="m3d-lv" checked onchange="update3DMeshVisibility()">
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#06b6d4;"></span>
              <span>Right Atrium (RA)</span>
            </div>
            <input type="checkbox" id="m3d-ra" checked onchange="update3DMeshVisibility()">
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#f97316;"></span>
              <span>Right Ventricle (RV)</span>
            </div>
            <input type="checkbox" id="m3d-rv" checked onchange="update3DMeshVisibility()">
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#d946ef;"></span>
              <span>Aorta / Great Vessels</span>
            </div>
            <input type="checkbox" id="m3d-ao" checked onchange="update3DMeshVisibility()">
          </label>

          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#eab308;"></span>
              <span>Pulmonary Artery</span>
            </div>
            <input type="checkbox" id="m3d-pa" checked onchange="update3DMeshVisibility()">
          </label>
        </div>
        
        <div style="margin-top:20px;">
          <div style="font-size:11px; font-weight:700; color:var(--text-muted); margin-bottom:8px;">SURFACE OPACITY</div>
          <input type="range" id="m3d-opacity" style="width:100%;" min="10" max="100" value="85" oninput="update3DOpacity(this.value)">
        </div>
        
        <div style="margin-top:20px;">
          <div style="font-size:11px; font-weight:700; color:var(--text-muted); margin-bottom:8px;">CAMERA PRESETS</div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">
            <button class="btn btn-primary" onclick="set3DCamera('ant')">Anterior</button>
            <button class="btn" onclick="set3DCamera('post')">Posterior</button>
            <button class="btn" onclick="set3DCamera('lat')">Left Lateral</button>
            <button class="btn" onclick="set3DCamera('4ch')">4-Chamber</button>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Center Interactive 3D Cardiac Stage -->
    <div class="stage-3d" id="stage-3d">
      <div id="webgl-container"></div>
      
      <div style="position:absolute; bottom:14px; left:50%; transform:translateX(-50%); font-size:11px; color:#cbd5e1; background:rgba(0,0,0,0.6); padding:4px 12px; border-radius:20px; backdrop-filter:blur(6px); pointer-events:none; z-index:10;">
        🖱️ Left-Click & Drag to Orbit 3D Heart Mesh | Right-Click to Pan | Mouse Wheel to Zoom
      </div>
    </div>
  </div>

</div>

<script>
// Embedded Real Cohort Payload with Marching-Cubes 3D Meshes
const cohortData = {patients_json};

let currentPatientId = Object.keys(cohortData)[0] || "0674";
let currentTabId = 'B';
let currentAxialSlice = 0;
let currentCoronalSlice = 0;
let currentSagittalSlice = 0;
let curtainActive = false;
let curtainX = 0.5;

// Three.js State
let threeScene, threeCamera, threeRenderer, threeControls;
let threeMeshes = {{}};
let isThreeInitialized = false;

function initApp() {{
  const selectEl = document.getElementById('patient-select');
  selectEl.innerHTML = '';
  Object.keys(cohortData).forEach(pid => {{
    const opt = document.createElement('option');
    opt.value = pid;
    const pMeta = (cohortData[pid] && cohortData[pid].metrics) ? cohortData[pid].metrics : {{}};
    opt.textContent = `Patient ${{pid}} (${{pMeta.age || '?'}}y, ${{pMeta.sex || '?'}})`;
    selectEl.appendChild(opt);
  }});

  if (cohortData[currentPatientId]) {{
    selectEl.value = currentPatientId;
    currentAxialSlice = cohortData[currentPatientId].default_axial || 0;
    const corRange = cohortData[currentPatientId].coronal_range || [0, 0];
    const sagRange = cohortData[currentPatientId].sagittal_range || [0, 0];
    currentCoronalSlice = Math.floor((corRange[0] + corRange[1]) / 2);
    currentSagittalSlice = Math.floor((sagRange[0] + sagRange[1]) / 2);
  }}

  initCurtainSlider();
  initViewportScrubber();
  updatePatientView();
}}

function setVariantById(tabId) {{
  currentTabId = tabId;
  document.querySelectorAll('.top-tab-btn').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById(`tab-btn-${{tabId.toLowerCase()}}`);
  if (activeBtn) activeBtn.classList.add('active');
  
  document.querySelectorAll('.variant-view').forEach(el => el.classList.remove('active'));
  const activeView = document.getElementById(`variant-${{tabId.toLowerCase()}}`);
  if (activeView) activeView.classList.add('active');
  
  if (tabId === 'C') {{
    setTimeout(init3DStudio, 50);
  }}
}}

function selectPatient(pid) {{
  if (!cohortData[pid]) return;
  currentPatientId = pid;
  const selectEl = document.getElementById('patient-select');
  if (selectEl) selectEl.value = pid;
  currentAxialSlice = cohortData[pid].default_axial || 0;
  updatePatientView();
  if (isThreeInitialized) {{
    loadPatient3DMeshes();
  }}
}}

function updatePatientView() {{
  const p = cohortData[currentPatientId];
  if (!p) return;
  const m = p.metrics || {{}};

  const laVolAdaptive = m.la_vol_adaptive ?? m.la_fat_volume_ml ?? 0.0;
  const laVolStd = m.la_vol_std ?? m.la_conservative_volume_ml ?? 0.0;
  const highFlags = m.high_flags ?? ((m.quality_flags || []).filter(f => (f.severity || '').toLowerCase() === 'high').length);
  const medFlags = m.med_flags ?? ((m.quality_flags || []).filter(f => ['med', 'medium'].includes((f.severity || '').toLowerCase())).length);
  const fittedMu = m.fitted_mu_hu ?? (m.gaussian_fit ? m.gaussian_fit.mu : null);
  const fittedSigma = m.fitted_sigma_hu ?? (m.gaussian_fit ? m.gaussian_fit.sigma : null);
  const purity = m.primary_component_purity ?? 1.0;
  
  // Header stats
  const statLa = document.getElementById('stat-la-vol');
  if (statLa) statLa.textContent = `${{Number(laVolAdaptive).toFixed(1)}} mL`;
  const statStd = document.getElementById('stat-std-vol');
  if (statStd) statStd.textContent = `${{m.scanner_la_eat_ml ? Number(m.scanner_la_eat_ml).toFixed(1) : Number(laVolStd).toFixed(1)}} mL`;
  
  const badgeEl = document.getElementById('stat-badge');
  if (badgeEl) {{
    if (highFlags > 0) {{
      badgeEl.textContent = 'HIGH CONCERN';
      badgeEl.className = 'badge badge-fail';
    }} else if (medFlags > 0) {{
      badgeEl.textContent = 'REVIEW';
      badgeEl.className = 'badge badge-warn';
    }} else {{
      badgeEl.textContent = 'PASSED';
      badgeEl.className = 'badge badge-pass';
    }}
  }}

  // HUD
  const hudP = document.getElementById('hud-patient');
  if (hudP) hudP.textContent = p.id;
  const hudLa = document.getElementById('hud-la-vol');
  if (hudLa) hudLa.textContent = `${{Number(laVolAdaptive).toFixed(1)}} mL`;
  
  // Ranges
  const minZ = (p.axial_range && p.axial_range[0] !== undefined) ? p.axial_range[0] : 0;
  const maxZ = (p.axial_range && p.axial_range[1] !== undefined) ? p.axial_range[1] : 0;
  currentAxialSlice = Math.max(minZ, Math.min(maxZ, currentAxialSlice));
  
  const minY = (p.coronal_range && p.coronal_range[0] !== undefined) ? p.coronal_range[0] : 0;
  const maxY = (p.coronal_range && p.coronal_range[1] !== undefined) ? p.coronal_range[1] : 0;
  currentCoronalSlice = Math.max(minY, Math.min(maxY, currentCoronalSlice));
  
  const minX = (p.sagittal_range && p.sagittal_range[0] !== undefined) ? p.sagittal_range[0] : 0;
  const maxX = (p.sagittal_range && p.sagittal_range[1] !== undefined) ? p.sagittal_range[1] : 0;
  currentSagittalSlice = Math.max(minX, Math.min(maxX, currentSagittalSlice));
  
  // Sliders config
  const sliderA = document.getElementById('slice-slider-a');
  if (sliderA) {{
    sliderA.min = minZ; sliderA.max = maxZ; sliderA.value = currentAxialSlice;
  }}
  const sliceNumA = document.getElementById('slice-num-a');
  if (sliceNumA) sliceNumA.textContent = currentAxialSlice;
  const hudZ = document.getElementById('hud-z');
  if (hudZ) hudZ.textContent = currentAxialSlice;
  const hudZB = document.getElementById('hud-z-b');
  if (hudZB) hudZB.textContent = `${{currentAxialSlice}} / ${{maxZ}}`;

  const sliderCor = document.getElementById('slider-coronal');
  if (sliderCor) {{
    sliderCor.min = minY; sliderCor.max = maxY; sliderCor.value = currentCoronalSlice;
  }}
  const coronalNum = document.getElementById('coronal-num');
  if (coronalNum) coronalNum.textContent = currentCoronalSlice;

  const sliderSag = document.getElementById('slider-sagittal');
  if (sliderSag) {{
    sliderSag.min = minX; sliderSag.max = maxX; sliderSag.value = currentSagittalSlice;
  }}
  const sagittalNum = document.getElementById('sagittal-num');
  if (sagittalNum) sagittalNum.textContent = currentSagittalSlice;

  // Render Slices
  renderActiveSlices();

  // Variant B Sidebar
  const bLaVol = document.getElementById('b-la-vol');
  if (bLaVol) bLaVol.textContent = Number(laVolAdaptive).toFixed(1);
  const bScannerVol = document.getElementById('b-scanner-vol');
  if (bScannerVol) bScannerVol.textContent = m.scanner_la_eat_ml ? Number(m.scanner_la_eat_ml).toFixed(1) : '--';
  const bScannerDelta = document.getElementById('b-scanner-delta');
  if (bScannerDelta) {{
    bScannerDelta.textContent = m.delta_la_adaptive_pct ? `Δ vs Scanner: ${{Number(m.delta_la_adaptive_pct).toFixed(1)}}% (${{Number(m.delta_la_adaptive_ml).toFixed(2)}} mL)` : (m.scanner_la_eat_ml ? `Scanner ref: ${{Number(m.scanner_la_eat_ml).toFixed(1)}} mL` : 'Scanner ref not available');
  }}

  const bGaussFit = document.getElementById('b-gauss-fit');
  if (bGaussFit) {{
    bGaussFit.textContent = (fittedMu !== null && fittedSigma !== null) ? `μ = ${{Number(fittedMu).toFixed(1)}} HU, σ = ${{Number(fittedSigma).toFixed(1)}} HU` : 'Fallback window used';
  }}
  const bGaussWin = document.getElementById('b-gauss-win');
  if (bGaussWin) {{
    bGaussWin.textContent = (fittedMu !== null && fittedSigma !== null) ? `[${{(Number(fittedMu) - 2*Number(fittedSigma)).toFixed(1)}}, 0.0] HU` : '[-190.0, -30.0] HU';
  }}

  // Checklist
  const bConcernList = document.getElementById('b-concern-list');
  if (bConcernList) {{
    bConcernList.innerHTML = `
      <div style="display:flex; justify-content:space-between;">
        <span>Pericardium Sac Hull:</span>
        <span style="color:#4ade80">🟢 TS Direct Solid</span>
      </div>
      <div style="display:flex; justify-content:space-between;">
        <span>Gaussian HU Fit:</span>
        <span style="color:${{fittedMu !== null ? '#4ade80' : '#fbbf24'}}">${{fittedMu !== null ? '🟢 Converged' : '🟡 Fallback'}}</span>
      </div>
      <div style="display:flex; justify-content:space-between;">
        <span>Topological Purity:</span>
        <span style="color:${{purity > 0.7 ? '#4ade80' : '#fbbf24'}}">${{(purity * 100).toFixed(1)}}%</span>
      </div>
    `;
  }}

  // Landmark Filmstrip
  const filmstripEl = document.getElementById('filmstrip-b');
  if (filmstripEl) {{
    filmstripEl.innerHTML = '';
    const labels = ["Apex", "Mid-LV", "Mid-LA", "Mitral", "Aorta"];
    const landmarkList = m.landmark_slices || [];
    landmarkList.forEach((lz, idx) => {{
      const lzClamped = Math.max(minZ, Math.min(maxZ, lz));
      const sliceObj = getClosestSlice(p.slices ? p.slices.axial : null, lzClamped);
      if (sliceObj && sliceObj.la_fat) {{
        const div = document.createElement('div');
        div.className = `filmstrip-thumb ${{lzClamped === currentAxialSlice ? 'active' : ''}}`;
        div.innerHTML = `<img src="${{sliceObj.la_fat}}"><div class="filmstrip-label">${{labels[idx] || 'Slice'}} (Z:${{lzClamped}})</div>`;
        div.onclick = () => {{
          currentAxialSlice = lzClamped;
          updatePatientView();
        }};
        filmstripEl.appendChild(div);
      }}
    }});
  }}

  renderCohortTable();
}}

function getClosestSlice(sliceDict, targetIdx) {{
  if (!sliceDict || Object.keys(sliceDict).length === 0) return null;
  if (sliceDict[targetIdx]) return sliceDict[targetIdx];
  const keys = Object.keys(sliceDict).map(Number);
  keys.sort((a, b) => Math.abs(a - targetIdx) - Math.abs(b - targetIdx));
  return sliceDict[keys[0]];
}}

function renderActiveSlices() {{
  const p = cohortData[currentPatientId];
  if (!p || !p.slices) return;

  const axObj = getClosestSlice(p.slices.axial, currentAxialSlice);
  if (axObj) {{
    const setSrc = (id, src) => {{ const el = document.getElementById(id); if (el && src) el.src = src; }};
    setSrc('img-ct-a', axObj.ct);
    setSrc('img-peri-a', axObj.peri);
    setSrc('img-anchors-a', axObj.anchors);
    setSrc('img-partition-a', axObj.partition);
    setSrc('img-la-fat-a', axObj.la_fat);
    setSrc('img-pv-a', axObj.pv_zone);

    setSrc('img-ct-b', axObj.ct);
    setSrc('img-peri-b', axObj.peri);
    setSrc('img-la-fat-b', axObj.la_fat);
    setSrc('img-pv-b', axObj.pv_zone);
  }}

  const corObj = getClosestSlice(p.slices.coronal, currentCoronalSlice);
  if (corObj) {{
    const el = document.getElementById('img-coronal-a');
    if (el && corObj.ct) el.src = corObj.ct;
  }}

  const sagObj = getClosestSlice(p.slices.sagittal, currentSagittalSlice);
  if (sagObj) {{
    const el = document.getElementById('img-sagittal-a');
    if (el && sagObj.ct) el.src = sagObj.ct;
  }}
}}

function onAxialSlider(val) {{
  currentAxialSlice = parseInt(val, 10);
  const el1 = document.getElementById('slice-num-a'); if (el1) el1.textContent = currentAxialSlice;
  const el2 = document.getElementById('hud-z'); if (el2) el2.textContent = currentAxialSlice;
  const el3 = document.getElementById('hud-z-b'); if (el3) el3.textContent = currentAxialSlice;
  renderActiveSlices();
}}

function onCoronalSlider(val) {{
  currentCoronalSlice = parseInt(val, 10);
  const el = document.getElementById('coronal-num'); if (el) el.textContent = currentCoronalSlice;
  renderActiveSlices();
}}

function onSagittalSlider(val) {{
  currentSagittalSlice = parseInt(val, 10);
  const el = document.getElementById('sagittal-num'); if (el) el.textContent = currentSagittalSlice;
  renderActiveSlices();
}}

function updateLayerVisibility() {{
  const updateEl = (id, checkId) => {{
    const el = document.getElementById(id);
    const cb = document.getElementById(checkId);
    if (el && cb) el.style.display = cb.checked ? 'block' : 'none';
  }};
  updateEl('img-ct-a', 'layer-ct');
  updateEl('img-peri-a', 'layer-peri');
  updateEl('img-anchors-a', 'layer-anchors');
  updateEl('img-partition-a', 'layer-partition');
  updateEl('img-la-fat-a', 'layer-la-fat');
  updateEl('img-pv-a', 'layer-pv');
}}

function toggleAllLayers() {{
  const cbPart = document.getElementById('layer-partition');
  const target = cbPart ? !cbPart.checked : true;
  ['layer-ct', 'layer-peri', 'layer-anchors', 'layer-partition', 'layer-la-fat', 'layer-pv'].forEach(id => {{
    const cb = document.getElementById(id);
    if (cb) cb.checked = target;
  }});
  updateLayerVisibility();
}}

function renderCohortTable() {{
  const tbody = document.getElementById('cohort-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';
  
  Object.values(cohortData).forEach(p => {{
    const tr = document.createElement('tr');
    if (p.id === currentPatientId) tr.className = 'active-patient';
    tr.onclick = () => selectPatient(p.id);

    const m = p.metrics || {{}};
    const laVolAdaptive = m.la_vol_adaptive ?? m.la_fat_volume_ml ?? 0.0;
    const laVolStd = m.la_vol_std ?? m.la_conservative_volume_ml ?? null;
    const totalEAT = m.total_eat_vol ?? m.total_eat_volume_ml ?? null;
    const highFlags = m.high_flags ?? ((m.quality_flags || []).filter(f => (f.severity || '').toLowerCase() === 'high').length);
    const medFlags = m.med_flags ?? ((m.quality_flags || []).filter(f => ['med', 'medium'].includes((f.severity || '').toLowerCase())).length);
    const fittedMu = m.fitted_mu_hu ?? (m.gaussian_fit ? m.gaussian_fit.mu : null);
    const fittedSigma = m.fitted_sigma_hu ?? (m.gaussian_fit ? m.gaussian_fit.sigma : null);

    const scannerLA = m.scanner_la_eat_ml ? Number(m.scanner_la_eat_ml).toFixed(1) : '--';
    const pipeLA = Number(laVolAdaptive).toFixed(1);
    const pipeStd = (laVolStd !== null && laVolStd !== undefined) ? Number(laVolStd).toFixed(1) : '--';
    const deltaPct = m.delta_la_adaptive_pct ? `${{Number(m.delta_la_adaptive_pct) > 0 ? '+' : ''}}${{Number(m.delta_la_adaptive_pct).toFixed(1)}}%` : '--';
    const totalEATStr = (totalEAT !== null && totalEAT !== undefined) ? Number(totalEAT).toFixed(1) : '--';
    const gaussStr = (fittedMu !== null && fittedSigma !== null) ? `${{Number(fittedMu).toFixed(0)}} ± ${{Number(fittedSigma).toFixed(0)}} HU` : 'Fallback';

    const statusBadge = (highFlags > 0) ? '<span class="badge badge-fail">Concern</span>' : (medFlags > 0 ? '<span class="badge badge-warn">Review</span>' : '<span class="badge badge-pass">Passed</span>');

    tr.innerHTML = `
      <td><strong>${{p.id}}</strong></td>
      <td>${{m.age || '?'}}y, ${{m.sex || '?'}}</td>
      <td>${{scannerLA}}</td>
      <td><strong style="color:var(--accent-gold);">${{pipeLA}}</strong></td>
      <td>${{pipeStd}}</td>
      <td style="color:${{m.delta_la_adaptive_pct && Math.abs(m.delta_la_adaptive_pct) < 15 ? '#4ade80' : '#fbbf24'}};">${{deltaPct}}</td>
      <td>${{totalEATStr}}</td>
      <td>${{gaussStr}}</td>
      <td>${{statusBadge}}</td>
    `;
    tbody.appendChild(tr);
  }});
}}

function initViewportScrubber() {{
  const stage = document.getElementById('main-stage-a');
  stage.addEventListener('wheel', (e) => {{
    e.preventDefault();
    const p = cohortData[currentPatientId];
    if (!p) return;
    const dir = e.deltaY > 0 ? 1 : -1;
    const nextZ = Math.max(p.axial_range[0], Math.min(p.axial_range[1], currentAxialSlice + dir));
    onAxialSlider(nextZ);
  }}, {{ passive: false }});
}}

function initCurtainSlider() {{
  const handle = document.getElementById('curtain-handle-a');
  const stack = document.getElementById('canvas-stack-a');
  const layers = ['img-peri-a', 'img-la-fat-a', 'img-anchors-a', 'img-partition-a', 'img-pv-a'];
  
  document.getElementById('btn-curtain-toggle').onclick = () => {{
    curtainActive = !curtainActive;
    handle.style.display = curtainActive ? 'block' : 'none';
    if (!curtainActive) {{
      layers.forEach(id => {{
        const el = document.getElementById(id);
        if (el) el.style.clipPath = 'none';
      }});
    }} else {{
      updateCurtain(0.5);
    }}
  }};
  
  function updateCurtain(pct) {{
    curtainX = Math.max(0, Math.min(1, pct));
    handle.style.left = `${{curtainX * 100}}%`;
    layers.forEach(id => {{
      const el = document.getElementById(id);
      if (el) el.style.clipPath = `polygon(0 0, ${{curtainX * 100}}% 0, ${{curtainX * 100}}% 100%, 0 100%)`;
    }});
  }}
  
  let draggingCurtain = false;
  handle.onmousedown = (e) => {{
    draggingCurtain = true;
    e.stopPropagation();
  }};
  
  window.addEventListener('mousemove', (e) => {{
    if (!draggingCurtain || !curtainActive) return;
    const rect = stack.getBoundingClientRect();
    const pct = (e.clientX - rect.left) / rect.width;
    updateCurtain(pct);
  }});
  
  window.addEventListener('mouseup', () => {{ draggingCurtain = false; }});
}}

// =========================================================================
// Real 3D WebGL Three.js Engine
// =========================================================================
function init3DStudio() {{
  if (typeof THREE === 'undefined') {{
    const container = document.getElementById('webgl-container');
    if (container) {{
      container.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#94a3b8;font-size:14px;padding:20px;text-align:center;">Interactive 3D WebGL requires internet access to load Three.js from CDN. When online, 3D surface meshes render automatically.</div>';
    }}
    return;
  }}

  if (isThreeInitialized) {{
    on3DResize();
    return;
  }}

  const container = document.getElementById('webgl-container');
  if (!container) return;

  const width = container.clientWidth || 800;
  const height = container.clientHeight || 600;

  // Scene
  threeScene = new THREE.Scene();

  // Camera
  threeCamera = new THREE.PerspectiveCamera(45, width / height, 1, 2000);
  threeCamera.position.set(0, 0, 260);

  // Renderer
  threeRenderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
  threeRenderer.setSize(width, height);
  threeRenderer.setPixelRatio(window.devicePixelRatio || 1);
  threeRenderer.shadowMap.enabled = true;
  container.appendChild(threeRenderer.domElement);

  // Controls
  if (typeof THREE.OrbitControls !== 'undefined') {{
    threeControls = new THREE.OrbitControls(threeCamera, threeRenderer.domElement);
    threeControls.enableDamping = true;
    threeControls.dampingFactor = 0.05;
  }}

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  threeScene.add(ambientLight);

  const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight1.position.set(150, 200, 200);
  threeScene.add(dirLight1);

  const dirLight2 = new THREE.DirectionalLight(0x90b0ff, 0.4);
  dirLight2.position.set(-150, -100, -200);
  threeScene.add(dirLight2);

  isThreeInitialized = true;
  window.addEventListener('resize', on3DResize);

  loadPatient3DMeshes();
  animate3D();
}}

function on3DResize() {{
  const container = document.getElementById('webgl-container');
  if (!container || !threeRenderer || !threeCamera) return;
  const w = container.clientWidth;
  const h = container.clientHeight;
  threeCamera.aspect = w / h;
  threeCamera.updateProjectionMatrix();
  threeRenderer.setSize(w, h);
}}

function loadPatient3DMeshes() {{
  if (!isThreeInitialized || typeof THREE === 'undefined') return;
  const p = cohortData[currentPatientId];
  if (!p || !p.meshes) return;

  // Remove old meshes
  Object.values(threeMeshes).forEach(m => {{
    if (m && m.parent) threeScene.remove(m);
    if (m.geometry) m.geometry.dispose();
  }});
  threeMeshes = {{}};

  const meshConfigs = {{
    la_fat: {{ color: 0xfacc15, opacity: 0.95, roughness: 0.25, metalness: 0.15, renderOrder: 10 }},
    la:     {{ color: 0xef4444, opacity: 0.85, roughness: 0.35, metalness: 0.05, renderOrder: 3 }},
    lv:     {{ color: 0x3b82f6, opacity: 0.85, roughness: 0.35, metalness: 0.05, renderOrder: 3 }},
    ra:     {{ color: 0x06b6d4, opacity: 0.85, roughness: 0.35, metalness: 0.05, renderOrder: 3 }},
    rv:     {{ color: 0xf97316, opacity: 0.85, roughness: 0.35, metalness: 0.05, renderOrder: 3 }},
    ao:     {{ color: 0xd946ef, opacity: 0.80, roughness: 0.35, metalness: 0.05, renderOrder: 3 }},
    pa:     {{ color: 0xeab308, opacity: 0.80, roughness: 0.35, metalness: 0.05, renderOrder: 3 }},
    peri:   {{ color: 0x22c55e, opacity: 0.20, roughness: 0.60, metalness: 0.0, transparent: true, side: THREE.DoubleSide, renderOrder: 1 }},
  }};

  Object.keys(meshConfigs).forEach(key => {{
    const rawMesh = p.meshes[key];
    if (rawMesh && rawMesh.v && rawMesh.v.length > 0) {{
      const geom = new THREE.BufferGeometry();
      geom.setAttribute('position', new THREE.Float32BufferAttribute(rawMesh.v, 3));
      geom.setIndex(rawMesh.f);
      geom.computeVertexNormals();

      const cfg = meshConfigs[key];
      const mat = new THREE.MeshStandardMaterial({{
        color: cfg.color,
        roughness: cfg.roughness,
        metalness: cfg.metalness,
        transparent: true,
        opacity: cfg.opacity,
        side: cfg.side || THREE.FrontSide,
        depthWrite: key !== 'peri',
      }});

      const mesh = new THREE.Mesh(geom, mat);
      mesh.renderOrder = cfg.renderOrder || 0;
      threeScene.add(mesh);
      threeMeshes[key] = mesh;
    }}
  }});

  update3DMeshVisibility();
}}

function update3DMeshVisibility() {{
  const toggles = {{
    la_fat: document.getElementById('m3d-la-fat').checked,
    peri: document.getElementById('m3d-peri').checked,
    la: document.getElementById('m3d-la').checked,
    lv: document.getElementById('m3d-lv').checked,
    ra: document.getElementById('m3d-ra') ? document.getElementById('m3d-ra').checked : true,
    rv: document.getElementById('m3d-rv') ? document.getElementById('m3d-rv').checked : true,
    ao: document.getElementById('m3d-ao').checked,
    pa: document.getElementById('m3d-pa') ? document.getElementById('m3d-pa').checked : true,
  }};

  Object.keys(toggles).forEach(k => {{
    if (threeMeshes[k]) {{
      threeMeshes[k].visible = toggles[k];
    }}
  }});
}}

function update3DOpacity(val) {{
  const pct = val / 100.0;
  ['la_fat', 'la', 'lv', 'ra', 'rv', 'ao', 'pa'].forEach(k => {{
    if (threeMeshes[k]) threeMeshes[k].material.opacity = pct * (k === 'la_fat' ? 1.0 : 0.9);
  }});
  if (threeMeshes.peri) threeMeshes.peri.material.opacity = pct * 0.25;
}}

function set3DCamera(preset) {{
  if (!threeCamera || !threeControls) return;
  const dist = 260;
  if (preset === 'ant') {{
    threeCamera.position.set(0, 0, dist);
  }} else if (preset === 'post') {{
    threeCamera.position.set(0, 0, -dist);
  }} else if (preset === 'lat') {{
    threeCamera.position.set(dist, 0, 0);
  }} else if (preset === '4ch') {{
    threeCamera.position.set(-130, 140, 180);
  }}
  threeCamera.lookAt(0, 0, 0);
  threeControls.target.set(0, 0, 0);
  threeControls.update();
}}

function animate3D() {{
  requestAnimationFrame(animate3D);
  if (threeControls) threeControls.update();
  if (threeRenderer && threeScene && threeCamera) {{
    threeRenderer.render(threeScene, threeCamera);
  }}
}}

// Keyboard navigation
window.addEventListener('keydown', (e) => {{
  const pids = Object.keys(cohortData);
  const curIdx = pids.indexOf(currentPatientId);
  if (e.key === 'j') {{
    const nextIdx = (curIdx + 1) % pids.length;
    selectPatient(pids[nextIdx]);
  }} else if (e.key === 'k') {{
    const prevIdx = (curIdx - 1 + pids.length) % pids.length;
    selectPatient(pids[prevIdx]);
  }}
}});

window.onload = initApp;
</script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[+] Saved interactive HTML5 Cohort QA Viewer to: {output_html_path}")
