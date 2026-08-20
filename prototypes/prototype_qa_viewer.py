"""
Prototype generator for Ticket 5: Lightweight Zero-Footprint QA Slice Viewer UI.
Generates realistic multi-patient cardiac CT slices, anatomical masks, Gaussian fit curves,
and packages everything into a self-contained, rich, interactive HTML prototype with top tabs,
synchronized multi-planar crosshairs, and rich 3D inspection controls.
"""

import json
import base64
import os
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw

def create_synthetic_patient_volume(patient_id: str, shape=(60, 80, 80)):
    """
    Creates realistic 3D synthetic CT volume and multi-channel masks.
    shape: (Z, Y, X) -> Z slices (axial), Y (coronal), X (sagittal)
    """
    nz, ny, nx = shape
    z_coords, y_coords, x_coords = np.ogrid[:nz, :ny, :nx]
    
    # Center coordinates
    cz, cy, cx = nz // 2, ny // 2, nx // 2
    
    # Variations per patient
    if patient_id == "0674": # Standard ideal case
        la_rad = (10, 14, 14)
        fat_thickness = 3.5
        gaussian_mu, gaussian_sigma = -88.0, 18.5
        flags = {"high": 0, "med": 0, "low": 0}
        la_vol = 22.4
        std_vol = 18.1
        total_eat = 94.2
    elif patient_id == "1512": # Dilated LA / High EAT
        la_rad = (14, 18, 18)
        fat_thickness = 5.0
        gaussian_mu, gaussian_sigma = -82.0, 22.0
        flags = {"high": 0, "med": 1, "low": 1} # Med: High LV/LA ratio
        la_vol = 38.6
        std_vol = 31.2
        total_eat = 142.8
    elif patient_id == "2996": # Low EAT / Thin layer
        la_rad = (9, 12, 12)
        fat_thickness = 2.0
        gaussian_mu, gaussian_sigma = -94.0, 15.0
        flags = {"high": 0, "med": 1, "low": 0} # Med: Low EAT ratio (<8%)
        la_vol = 9.8
        std_vol = 7.9
        total_eat = 41.5
    else: # 8841: Loose Pericardium / Fallback trigger
        la_rad = (11, 15, 15)
        fat_thickness = 4.0
        gaussian_mu, gaussian_sigma = -76.0, 28.0
        flags = {"high": 1, "med": 1, "low": 1} # High: Pericardium fallback
        la_vol = 27.1
        std_vol = 21.0
        total_eat = 112.0

    # Base CT HU initialization: Air (-1000 HU), Mediastinum/Soft tissue (40 HU), Lungs (-700 HU)
    ct = np.full(shape, -700.0, dtype=np.float32)
    
    # Thoracic cavity & mediastinum background
    body_mask = np.broadcast_to(((y_coords - cy)**2 / (ny*0.45)**2 + (x_coords - cx)**2 / (nx*0.45)**2) <= 1.0, shape)
    ct[body_mask] = 40.0 # Soft tissue
    
    # Pericardium ellipsoid
    peri_rz, peri_ry, peri_rx = nz*0.38, ny*0.34, nx*0.34
    peri_dist = np.sqrt(((z_coords - cz)/peri_rz)**2 + ((y_coords - (cy+2))/peri_ry)**2 + ((x_coords - cx)/peri_rx)**2)
    pericardium = peri_dist <= 1.0
    
    # Chamber ellipsoids
    # 1: LA (Posterior Superior)
    la_cz, la_cy, la_cx = cz + 4, cy - 8, cx
    la_mask = ((z_coords - la_cz)**2 / la_rad[0]**2 + 
               (y_coords - la_cy)**2 / la_rad[1]**2 + 
               (x_coords - la_cx)**2 / la_rad[2]**2) <= 1.0
    
    # 2: LV (Anterior-Inferior-Left)
    lv_cz, lv_cy, lv_cx = cz - 6, cy + 8, cx + 10
    lv_mask = (((z_coords - lv_cz)/13)**2 + ((y_coords - lv_cy)/14)**2 + ((x_coords - lv_cx)/14)**2) <= 1.0
    
    # 3: RA (Right-Superior-Anterior)
    ra_cz, ra_cy, ra_cx = cz + 2, cy - 2, cx - 14
    ra_mask = (((z_coords - ra_cz)/11)**2 + ((y_coords - ra_cy)/11)**2 + ((x_coords - ra_cx)/11)**2) <= 1.0
    
    # 4: RV (Anterior-Right-Inferior)
    rv_cz, rv_cy, rv_cx = cz - 5, cy + 10, cx - 8
    rv_mask = (((z_coords - rv_cz)/12)**2 + ((y_coords - rv_cy)/12)**2 + ((x_coords - rv_cx)/12)**2) <= 1.0
    
    # 5: Aorta (Superior Center)
    ao_cz, ao_cy, ao_cx = cz + 14, cy - 2, cx + 2
    ao_mask = (((z_coords - ao_cz)/10)**2 + ((y_coords - ao_cy)/7)**2 + ((x_coords - ao_cx)/7)**2) <= 1.0
    
    # 6: Pulmonary Artery (Superior Anterior Left)
    pa_cz, pa_cy, pa_cx = cz + 15, cy + 7, cx - 4
    pa_mask = (((z_coords - pa_cz)/9)**2 + ((y_coords - pa_cy)/7)**2 + ((x_coords - pa_cx)/7)**2) <= 1.0
    
    # Combine heart chambers (Blood pool: 75 HU in non-contrast/native)
    chambers = la_mask | lv_mask | ra_mask | rv_mask | ao_mask | pa_mask
    ct[chambers] = 75.0
    
    # Epicardial Fat layer (Inside pericardium, outside chambers)
    fat_space = pericardium & (~chambers)
    
    # Distance transform from chamber surfaces for partition
    la_surface_dist = np.sqrt((z_coords - la_cz)**2 + (y_coords - la_cy)**2 + (x_coords - la_cx)**2) - la_rad[1]
    lv_surface_dist = np.sqrt((z_coords - lv_cz)**2 + (y_coords - lv_cy)**2 + (x_coords - lv_cx)**2) - 14
    ra_surface_dist = np.sqrt((z_coords - ra_cz)**2 + (y_coords - ra_cy)**2 + (x_coords - ra_cx)**2) - 11
    rv_surface_dist = np.sqrt((z_coords - rv_cz)**2 + (y_coords - rv_cy)**2 + (x_coords - rv_cx)**2) - 12
    ao_surface_dist = np.sqrt((z_coords - ao_cz)**2 + (y_coords - ao_cy)**2 + (x_coords - ao_cx)**2) - 7
    pa_surface_dist = np.sqrt((z_coords - pa_cz)**2 + (y_coords - pa_cy)**2 + (x_coords - pa_cx)**2) - 7
    
    min_dist_to_any_chamber = np.minimum.reduce([
        la_surface_dist, lv_surface_dist, ra_surface_dist, rv_surface_dist, ao_surface_dist, pa_surface_dist
    ])
    
    fat_layer = fat_space & (min_dist_to_any_chamber <= fat_thickness)
    
    # CT Fat HU values
    np.random.seed(42)
    noise = np.random.normal(0, 8.0, shape)
    fat_hu = gaussian_mu + noise
    ct[fat_layer] = fat_hu[fat_layer]
    
    # Partitioned anchors
    assigned_anchor = np.zeros(shape, dtype=np.uint8) # 1: LA, 2: LV, 3: RA, 4: RV, 5: Aorta, 6: PA
    stack_dists = np.stack([
        la_surface_dist, lv_surface_dist, ra_surface_dist, rv_surface_dist, ao_surface_dist, pa_surface_dist
    ], axis=-1)
    nearest_idx = np.argmin(stack_dists, axis=-1) + 1
    assigned_anchor[fat_layer] = nearest_idx[fat_layer]
    
    # Final LA Fat mask
    la_fat_mask = (assigned_anchor == 1) & (ct >= (gaussian_mu - 2*gaussian_sigma)) & (ct <= 0.0)
    
    # Partial volume zone (-30 to 0 HU)
    partial_volume_mask = (assigned_anchor == 1) & (ct > -30.0) & (ct <= 0.0)
    
    return {
        "id": patient_id,
        "ct": ct,
        "pericardium": pericardium,
        "la": la_mask, "lv": lv_mask, "ra": ra_mask, "rv": rv_mask, "ao": ao_mask, "pa": pa_mask,
        "assigned_anchor": assigned_anchor,
        "la_fat": la_fat_mask,
        "pv_zone": partial_volume_mask,
        "metrics": {
            "la_vol_adaptive": la_vol,
            "la_vol_std": std_vol,
            "total_eat_vol": total_eat,
            "la_ratio": round((la_vol / total_eat) * 100, 1),
            "gaussian_mu": gaussian_mu,
            "gaussian_sigma": gaussian_sigma,
            "gaussian_bounds": [round(gaussian_mu - 2*gaussian_sigma, 1), min(0.0, round(gaussian_mu + 2*gaussian_sigma, 1))],
            "flags": flags,
            "landmark_slices": [cz - 8, cz - 2, cz + 4, cz + 8, cz + 14] # Apex, Mid-LV, Mid-LA, Mitral/PV, Aorta
        }
    }

def hu_to_grayscale(ct_slice, window_center=40, window_width=350):
    """Applies medical CT window/level transform to 8-bit grayscale."""
    min_hu = window_center - window_width / 2.0
    max_hu = window_center + window_width / 2.0
    scaled = np.clip((ct_slice - min_hu) / (max_hu - min_hu), 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)

def render_slice_layers(patient_data, slice_idx, plane="axial"):
    """
    Renders the CT image and individual transparent overlay masks for a slice.
    plane: 'axial' (along Z), 'coronal' (along Y), 'sagittal' (along X)
    Returns dictionary of base64 data URLs for CT and each layer.
    """
    def extract_slice(arr, p, idx):
        if p == "axial": return arr[idx, :, :]
        elif p == "coronal": return arr[:, idx, :]
        else: return arr[:, :, idx]

    ct_s = extract_slice(patient_data["ct"], plane, slice_idx)
    peri_s = extract_slice(patient_data["pericardium"], plane, slice_idx)
    la_s = extract_slice(patient_data["la"], plane, slice_idx)
    lv_s = extract_slice(patient_data["lv"], plane, slice_idx)
    ra_s = extract_slice(patient_data["ra"], plane, slice_idx)
    rv_s = extract_slice(patient_data["rv"], plane, slice_idx)
    ao_s = extract_slice(patient_data["ao"], plane, slice_idx)
    pa_s = extract_slice(patient_data["pa"], plane, slice_idx)
    anchor_s = extract_slice(patient_data["assigned_anchor"], plane, slice_idx)
    la_fat_s = extract_slice(patient_data["la_fat"], plane, slice_idx)
    pv_s = extract_slice(patient_data["pv_zone"], plane, slice_idx)
    
    h, w = ct_s.shape
    
    # 1. CT Base Grayscale (Mediastinal default)
    ct_gray = hu_to_grayscale(ct_s, 40, 350)
    img_ct = Image.fromarray(ct_gray, mode='L').convert('RGB')
    
    # 2. Pericardium Outline Mask (Lime Green #22c55e outline)
    peri_contour = np.zeros((h, w, 4), dtype=np.uint8)
    if np.any(peri_s):
        from scipy.ndimage import binary_dilation, binary_erosion
        try:
            outline = binary_dilation(peri_s) ^ binary_erosion(peri_s)
        except Exception:
            outline = peri_s
        peri_contour[outline] = [34, 197, 94, 240] # #22c55e
    img_peri = Image.fromarray(peri_contour, mode='RGBA')
    
    # 3. TS 6 Anchors Fill
    anchors_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    anchors_rgba[la_s] = [239, 68, 68, 140]
    anchors_rgba[lv_s] = [59, 130, 246, 140]
    anchors_rgba[ra_s] = [6, 182, 212, 140]
    anchors_rgba[rv_s] = [249, 115, 22, 140]
    anchors_rgba[ao_s] = [217, 70, 239, 140]
    anchors_rgba[pa_s] = [234, 179, 8, 140]
    img_anchors = Image.fromarray(anchors_rgba, mode='RGBA')
    
    # 4. Partitioned EAT (Color-coded by assigned anchor)
    partition_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    partition_rgba[anchor_s == 1] = [239, 68, 68, 180] # LA Fat
    partition_rgba[anchor_s == 2] = [59, 130, 246, 180] # LV Fat
    partition_rgba[anchor_s == 3] = [6, 182, 212, 180] # RA Fat
    partition_rgba[anchor_s == 4] = [249, 115, 22, 180] # RV Fat
    partition_rgba[anchor_s == 5] = [217, 70, 239, 180] # Ao Fat
    partition_rgba[anchor_s == 6] = [234, 179, 8, 180] # PA Fat
    img_partition = Image.fromarray(partition_rgba, mode='RGBA')
    
    # 5. Final LA Fat (Bright Amber Gold #facc15 with solid border)
    la_fat_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    la_fat_rgba[la_fat_s] = [250, 204, 21, 210] # #facc15
    img_la_fat = Image.fromarray(la_fat_rgba, mode='RGBA')
    
    # 6. Partial Volume Tail (0 to -30 HU) Electric Purple (#a855f7)
    pv_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    pv_rgba[pv_s] = [168, 85, 247, 200]
    img_pv = Image.fromarray(pv_rgba, mode='RGBA')
    
    def to_b64(img):
        buf = BytesIO()
        img.save(buf, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return {
        "ct": to_b64(img_ct),
        "peri": to_b64(img_peri),
        "anchors": to_b64(img_anchors),
        "partition": to_b64(img_partition),
        "la_fat": to_b64(img_la_fat),
        "pv_zone": to_b64(img_pv)
    }

def generate_all_patients_payload():
    """Generates slice data and metadata for 4 patients."""
    patient_ids = ["0674", "1512", "2996", "8841"]
    patients = {}
    
    for pid in patient_ids:
        print(f"Generating synthetic 3D cardiac volume for Patient {pid}...")
        pdata = create_synthetic_patient_volume(pid)
        
        # Dense sampling for smooth scrolling across all 3 planes
        axial_indices = list(range(15, 45))
        coronal_indices = list(range(20, 60))
        sagittal_indices = list(range(20, 60))
        
        slices_axial = {}
        for idx in axial_indices:
            slices_axial[idx] = render_slice_layers(pdata, idx, "axial")
            
        slices_coronal = {}
        for idx in coronal_indices:
            slices_coronal[idx] = render_slice_layers(pdata, idx, "coronal")
            
        slices_sagittal = {}
        for idx in sagittal_indices:
            slices_sagittal[idx] = render_slice_layers(pdata, idx, "sagittal")
            
        patients[pid] = {
            "id": pid,
            "metrics": pdata["metrics"],
            "axial_range": [axial_indices[0], axial_indices[-1]],
            "coronal_range": [coronal_indices[0], coronal_indices[-1]],
            "sagittal_range": [sagittal_indices[0], sagittal_indices[-1]],
            "default_axial": pdata["metrics"]["landmark_slices"][2], # Mid-LA
            "slices": {
                "axial": slices_axial,
                "coronal": slices_coronal,
                "sagittal": slices_sagittal
            }
        }
    return patients

def build_prototype_html(patients_data):
    """Builds the standalone HTML prototype containing all 3 UI variants with top navigation tabs."""
    patients_json = json.dumps(patients_data)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LA-FAT QA Viewer — Verified Production Layout</title>
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
  grid-template-rows: 150px 1fr;
  grid-template-columns: 320px 1fr;
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
   VARIANT C: Colleague 3D Presentation Studio
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

.stage-3d-canvas {{
  width: 100%;
  height: 100%;
  cursor: grab;
}}
.stage-3d-canvas:active {{ cursor: grabbing; }}

.stage-3d-controls {{
  position: absolute;
  top: 14px;
  left: 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  z-index: 10;
}}
.angle-btn-group {{
  display: flex;
  background: var(--bg-glass);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  overflow: hidden;
}}
.angle-btn {{
  background: transparent;
  color: var(--text-main);
  border: none;
  padding: 4px 8px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
}}
.angle-btn:hover, .angle-btn.active {{
  background: var(--accent-blue);
  color: white;
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
      <select id="patient-select">
        <option value="0674">Patient 0674 (Ideal Baseline)</option>
        <option value="1512">Patient 1512 (High EAT / Dilated LA)</option>
        <option value="2996">Patient 2996 (Low EAT / Thin Sac)</option>
        <option value="8841">Patient 8841 (Loose Pericardium Trigger)</option>
      </select>
    </div>
    
    <div class="header-stats">
      <div class="stat-pill">Adaptive LA EAT: <span class="stat-val" id="stat-la-vol">22.4 mL</span></div>
      <span style="color:var(--border-color);">|</span>
      <div class="stat-pill">Std [-190,-30]: <span class="stat-val" id="stat-std-vol">18.1 mL</span></div>
      <span style="color:var(--border-color);">|</span>
      <div class="stat-pill">Quality: <span id="stat-badge" class="badge badge-pass">PASSED</span></div>
    </div>
    
    <button class="btn" id="btn-export-qa">
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
        <button class="btn" style="padding:2px 6px; font-size:10px;" id="btn-all-layers">All ON</button>
      </div>
      <div class="panel-content">
        <div class="layer-toggle-group">
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#fff;"></span>
              <span>Base CT Image</span>
            </div>
            <input type="checkbox" id="layer-ct" checked>
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#22c55e;"></span>
              <span>Pericardium Sac (Outline)</span>
            </div>
            <input type="checkbox" id="layer-peri" checked>
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#ef4444;"></span>
              <span>TS 6 Anchors (Solid)</span>
            </div>
            <input type="checkbox" id="layer-anchors">
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#3b82f6;"></span>
              <span>Partitioned EAT (6-Color)</span>
            </div>
            <input type="checkbox" id="layer-partition">
          </label>
          
          <label class="layer-row" style="background:rgba(250,204,21,0.1); border-color:rgba(250,204,21,0.3);">
            <div class="layer-label">
              <span class="layer-dot" style="background:#facc15;"></span>
              <span style="font-weight:700; color:#facc15;">Final LA Fat Mask</span>
            </div>
            <input type="checkbox" id="layer-la-fat" checked>
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#a855f7;"></span>
              <span>Partial Volume Tail (0 to -30 HU)</span>
            </div>
            <input type="checkbox" id="layer-pv">
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
            <input type="range" id="slice-slider-a" style="flex:1;" min="15" max="44" value="34">
            <span id="slice-num-a" style="font-family:var(--font-mono); font-size:12px; font-weight:bold;">34</span>
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
          <div>PATIENT: <span id="hud-patient">0674</span> (AXIAL Z: <span id="hud-z">34</span>)</div>
          <div>WINDOW: <span id="hud-wl">L: 40, W: 350 HU</span></div>
          <div>LA-EAT: <span id="hud-la-vol">22.4 mL</span></div>
        </div>
        
        <div class="pacs-overlay-tools">
          <button class="btn" id="btn-curtain-toggle"><span>⇹</span> Curtain Wipe</button>
          <button class="btn" id="btn-zoom-fit"><span>🔍</span> Reset Zoom</button>
        </div>
      </div>
    </div>
    
    <!-- Right Orthogonal Multi-Planar Views (Coronal & Sagittal) -->
    <div class="orthogonal-side-stack">
      <div class="ortho-subview">
        <span class="ortho-title">CORONAL PLANE (Y: <span id="coronal-num">40</span>)</span>
        <img id="img-coronal-a" src="" alt="Coronal">
        <div class="ortho-scrubber">
          <span style="font-size:10px; color:#cbd5e1;">Y:</span>
          <input type="range" id="slider-coronal" min="20" max="59" value="40">
        </div>
      </div>
      <div class="ortho-subview">
        <span class="ortho-title">SAGITTAL PLANE (X: <span id="sagittal-num">40</span>)</span>
        <img id="img-sagittal-a" src="" alt="Sagittal">
        <div class="ortho-scrubber">
          <span style="font-size:10px; color:#cbd5e1;">X:</span>
          <input type="range" id="slider-sagittal" min="20" max="59" value="40">
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
              <th>Adaptive Volume (mL)</th>
              <th>Std [-190,-30] (mL)</th>
              <th>PV Difference</th>
              <th>LA Share (%)</th>
              <th>Gaussian Fit (μ ± 2σ)</th>
              <th>Quality Concerns</th>
              <th>Status</th>
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
        <div class="metric-big"><span id="b-la-vol">22.4</span> <span class="metric-unit">mL</span></div>
        <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">
          Includes 1-2 voxel partial volume boundary layer (0 to -30 HU).
        </div>
      </div>
      
      <div class="metric-card">
        <div class="metric-header">Gaussian HU Threshold Fit</div>
        <div style="font-family:var(--font-mono); font-size:12px; color:var(--accent-gold); margin-bottom:4px;" id="b-gauss-fit">
          μ = -88.0 HU, σ = 18.5 HU
        </div>
        <div style="font-size:11px; color:var(--text-muted);">
          Optimal Window: <strong style="color:#fff;" id="b-gauss-win">[-125.0, 0.0] HU</strong>
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
        <div class="pacs-overlay-hud">AXIAL SLICE: <span id="hud-z-b">34</span> / 44</div>
      </div>
    </div>
  </div>

  <!-- =========================================================================
       VARIANT C: Colleague 3D Presentation Studio
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
            <input type="checkbox" id="m3d-la-fat" checked>
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#22c55e;"></span>
              <span>Pericardium Sac Envelope</span>
            </div>
            <input type="checkbox" id="m3d-peri" checked>
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#ef4444;"></span>
              <span>Left Atrium (LA) Surface</span>
            </div>
            <input type="checkbox" id="m3d-la" checked>
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#3b82f6;"></span>
              <span>Left Ventricle (LV) Surface</span>
            </div>
            <input type="checkbox" id="m3d-lv" checked>
          </label>
          
          <label class="layer-row">
            <div class="layer-label">
              <span class="layer-dot" style="background:#d946ef;"></span>
              <span>Aorta / Great Vessels</span>
            </div>
            <input type="checkbox" id="m3d-ao" checked>
          </label>
        </div>
        
        <div style="margin-top:20px;">
          <div style="font-size:11px; font-weight:700; color:var(--text-muted); margin-bottom:8px;">SURFACE OPACITY</div>
          <input type="range" id="m3d-opacity" style="width:100%;" min="20" max="100" value="85">
        </div>
        
        <div style="margin-top:20px;">
          <div style="font-size:11px; font-weight:700; color:var(--text-muted); margin-bottom:8px;">CAMERA PRESETS</div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">
            <button class="btn btn-primary" id="btn-cam-ant">Anterior</button>
            <button class="btn" id="btn-cam-post">Posterior</button>
            <button class="btn" id="btn-cam-lat">Left Lateral</button>
            <button class="btn" id="btn-cam-4ch">4-Chamber</button>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Center Interactive 3D Cardiac Stage -->
    <div class="stage-3d" id="stage-3d">
      <canvas id="canvas-3d" class="stage-3d-canvas"></canvas>
      
      <div style="position:absolute; bottom:14px; left:50%; transform:translateX(-50%); font-size:11px; color:#cbd5e1; background:rgba(0,0,0,0.6); padding:4px 10px; border-radius:20px; backdrop-filter:blur(6px);">
        🖱️ Click & Drag to Orbit 3D Heart Mesh | Mouse Wheel to Zoom
      </div>
    </div>
  </div>

</div>

<script>
// Load generated synthetic multi-patient cohort data
const cohortData = {patients_json};

let currentPatientId = "0674";
let currentAxialSlice = 34;
let currentCoronalSlice = 40;
let currentSagittalSlice = 40;
let currentTabId = "B";

let curtainActive = false;
let curtainX = 0.5;

// 3D Mesh Visibility States
let meshState = {{
  laFat: true,
  peri: true,
  la: true,
  lv: true,
  ao: true,
  opacity: 0.85
}};

function setVariantById(tabId) {{
  currentTabId = tabId;
  
  // Highlight top tab buttons
  document.querySelectorAll('.top-tab-btn').forEach(btn => btn.classList.remove('active'));
  const activeBtn = document.getElementById(`tab-btn-${{tabId.toLowerCase()}}`);
  if (activeBtn) activeBtn.classList.add('active');
  
  // Switch view
  document.querySelectorAll('.variant-view').forEach(el => el.classList.remove('active'));
  const activeView = document.getElementById(`variant-${{tabId.toLowerCase()}}`);
  if (activeView) activeView.classList.add('active');
  
  // Update URL param
  const url = new URL(window.location);
  url.searchParams.set('tab', tabId);
  window.history.replaceState({{}}, '', url);
  
  if (tabId === 'C') {{
    init3DStage();
  }}
}}

// Update displayed slices and metadata
function updatePatientView() {{
  const p = cohortData[currentPatientId];
  if (!p) return;
  
  // Update header stats
  document.getElementById('stat-la-vol').textContent = `${{p.metrics.la_vol_adaptive}} mL`;
  document.getElementById('stat-std-vol').textContent = `${{p.metrics.la_vol_std}} mL`;
  const badgeEl = document.getElementById('stat-badge');
  if (p.metrics.flags.high > 0) {{
    badgeEl.textContent = 'HIGH CONCERN';
    badgeEl.className = 'badge badge-fail';
  }} else if (p.metrics.flags.med > 0) {{
    badgeEl.textContent = 'MED CONCERN';
    badgeEl.className = 'badge badge-warn';
  }} else {{
    badgeEl.textContent = 'PASSED';
    badgeEl.className = 'badge badge-pass';
  }}
  
  // Update HUD
  document.getElementById('hud-patient').textContent = p.id;
  document.getElementById('hud-la-vol').textContent = `${{p.metrics.la_vol_adaptive}} mL`;
  
  // Clamp slice ranges
  const minZ = p.axial_range[0], maxZ = p.axial_range[1];
  currentAxialSlice = Math.max(minZ, Math.min(maxZ, currentAxialSlice));
  
  const minY = p.coronal_range[0], maxY = p.coronal_range[1];
  currentCoronalSlice = Math.max(minY, Math.min(maxY, currentCoronalSlice));
  
  const minX = p.sagittal_range[0], maxX = p.sagittal_range[1];
  currentSagittalSlice = Math.max(minX, Math.min(maxX, currentSagittalSlice));
  
  // Axial Slice Slices
  const sliceObj = p.slices.axial[currentAxialSlice];
  if (sliceObj) {{
    // Variant A images
    document.getElementById('img-ct-a').src = sliceObj.ct;
    document.getElementById('img-peri-a').src = sliceObj.peri;
    document.getElementById('img-anchors-a').src = sliceObj.anchors;
    document.getElementById('img-partition-a').src = sliceObj.partition;
    document.getElementById('img-la-fat-a').src = sliceObj.la_fat;
    document.getElementById('img-pv-a').src = sliceObj.pv_zone;
    
    // Variant B images
    document.getElementById('img-ct-b').src = sliceObj.ct;
    document.getElementById('img-peri-b').src = sliceObj.peri;
    document.getElementById('img-la-fat-b').src = sliceObj.la_fat;
    document.getElementById('img-pv-b').src = sliceObj.pv_zone;
  }}
  
  // Coronal & Sagittal Views in Variant A
  const corObj = p.slices.coronal[currentCoronalSlice];
  if (corObj) {{
    document.getElementById('img-coronal-a').src = corObj.ct;
    document.getElementById('coronal-num').textContent = currentCoronalSlice;
    document.getElementById('slider-coronal').value = currentCoronalSlice;
  }}
  
  const sagObj = p.slices.sagittal[currentSagittalSlice];
  if (sagObj) {{
    document.getElementById('img-sagittal-a').src = sagObj.ct;
    document.getElementById('sagittal-num').textContent = currentSagittalSlice;
    document.getElementById('slider-sagittal').value = currentSagittalSlice;
  }}
  
  // Sliders and HUD
  document.getElementById('slice-slider-a').value = currentAxialSlice;
  document.getElementById('slice-num-a').textContent = currentAxialSlice;
  document.getElementById('hud-z').textContent = currentAxialSlice;
  document.getElementById('hud-z-b').textContent = currentAxialSlice;
  
  // Update Variant B metrics
  document.getElementById('b-la-vol').textContent = p.metrics.la_vol_adaptive;
  document.getElementById('b-gauss-fit').textContent = `μ = ${{p.metrics.gaussian_mu}} HU, σ = ${{p.metrics.gaussian_sigma}} HU`;
  document.getElementById('b-gauss-win').textContent = `[${{p.metrics.gaussian_bounds[0]}}, ${{p.metrics.gaussian_bounds[1]}}] HU`;
  
  // Checklist
  const concernListEl = document.getElementById('b-concern-list');
  concernListEl.innerHTML = `
    <div style="display:flex; justify-content:space-between;">
      <span>Pericardium Sac Hull:</span>
      <span style="color:${{p.metrics.flags.high ? '#f87171':'#4ade80'}}">${{p.metrics.flags.high ? '🔴 Fallback' : '🟢 TS Solid'}}</span>
    </div>
    <div style="display:flex; justify-content:space-between;">
      <span>Gaussian HU Fit:</span>
      <span style="color:#4ade80">🟢 Converged (No clamp)</span>
    </div>
    <div style="display:flex; justify-content:space-between;">
      <span>LV/LA Fat Ratio:</span>
      <span style="color:${{p.metrics.flags.med ? '#fbbf24':'#4ade80'}}">${{p.metrics.flags.med ? '🟡 Review Ratio' : '🟢 Normal'}}</span>
    </div>
  `;
  
  // Filmstrip
  const filmstripEl = document.getElementById('filmstrip-b');
  filmstripEl.innerHTML = '';
  const landmarkLabels = ["Apex", "Mid-LV", "Mid-LA", "Mitral Plane", "Aorta"];
  p.metrics.landmark_slices.forEach((lz, idx) => {{
    const lzClamped = Math.max(minZ, Math.min(maxZ, lz));
    const thumbObj = p.slices.axial[lzClamped];
    if (thumbObj) {{
      const div = document.createElement('div');
      div.className = `filmstrip-thumb ${{lzClamped === currentAxialSlice ? 'active' : ''}}`;
      div.innerHTML = `<img src="${{thumbObj.la_fat}}"><div class="filmstrip-label">${{landmarkLabels[idx]}} (Z:${{lzClamped}})</div>`;
      div.onclick = () => {{
        currentAxialSlice = lzClamped;
        updatePatientView();
      }};
      filmstripEl.appendChild(div);
    }}
  }});
  
  renderCohortTable();
}}

function renderCohortTable() {{
  const tbody = document.getElementById('cohort-tbody');
  tbody.innerHTML = '';
  
  Object.values(cohortData).forEach(p => {{
    const tr = document.createElement('tr');
    if (p.id === currentPatientId) tr.className = 'active-patient';
    
    let badgeHtml = '<span class="badge badge-pass">PASSED</span>';
    if (p.metrics.flags.high > 0) badgeHtml = '<span class="badge badge-fail">HIGH CONCERN</span>';
    else if (p.metrics.flags.med > 0) badgeHtml = '<span class="badge badge-warn">MED CONCERN</span>';
    
    const diffMl = (p.metrics.la_vol_adaptive - p.metrics.la_vol_std).toFixed(1);
    const diffPct = (((p.metrics.la_vol_adaptive / p.metrics.la_vol_std) - 1) * 100).toFixed(0);
    
    tr.innerHTML = `
      <td><strong>${{p.id}}</strong></td>
      <td><span style="color:var(--accent-gold); font-weight:bold;">${{p.metrics.la_vol_adaptive}} mL</span></td>
      <td>${{p.metrics.la_vol_std}} mL</td>
      <td><span style="color:var(--accent-purple); font-weight:bold;">+${{diffMl}} mL (+${{diffPct}}%)</span></td>
      <td>${{p.metrics.la_ratio}}%</td>
      <td>${{p.metrics.gaussian_mu}} ± ${{p.metrics.gaussian_sigma*2}}</td>
      <td>${{p.metrics.flags.high ? '🔴 Fallback Pericardium' : (p.metrics.flags.med ? '🟡 Fat Ratio' : '🟢 None')}}</td>
      <td>${{badgeHtml}}</td>
    `;
    
    tr.onclick = () => {{
      currentPatientId = p.id;
      document.getElementById('patient-select').value = p.id;
      updatePatientView();
    }};
    tbody.appendChild(tr);
  }});
}}

// 3D Canvas Raycasting / Rotating Mockup
let rotX = 0.3, rotY = -0.6;
let scale3D = 1.0;
function init3DStage() {{
  const canvas = document.getElementById('canvas-3d');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;
  
  function draw3D() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const baseSize = Math.min(canvas.width, canvas.height) * 0.28 * scale3D;
    
    ctx.save();
    ctx.translate(cx, cy);
    
    // Pericardium sac (Translucent Green)
    if (meshState.peri) {{
      ctx.beginPath();
      ctx.ellipse(0, 0, baseSize * 1.3, baseSize * 1.5, rotY * 0.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(34, 197, 94, ${{meshState.opacity * 0.15}})`;
      ctx.fill();
      ctx.strokeStyle = 'rgba(34, 197, 94, 0.7)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }}
    
    // LA Chamber (Red)
    const laX = Math.sin(rotY) * baseSize * 0.4;
    const laY = -Math.cos(rotX) * baseSize * 0.5;
    if (meshState.la) {{
      ctx.beginPath();
      ctx.arc(laX, laY, baseSize * 0.45, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(239, 68, 68, ${{meshState.opacity * 0.65}})`;
      ctx.fill();
      ctx.strokeStyle = '#ef4444';
      ctx.stroke();
    }}
    
    // Final LA Fat Mesh (Bright Amber Gold)
    if (meshState.laFat) {{
      ctx.beginPath();
      ctx.arc(laX + 10 * Math.cos(rotY), laY - 10, baseSize * 0.52, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(250, 204, 21, ${{meshState.opacity * 0.85}})`;
      ctx.fill();
      ctx.strokeStyle = '#facc15';
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }}
    
    // LV Chamber (Blue)
    if (meshState.lv) {{
      const lvX = Math.sin(rotY + 0.8) * baseSize * 0.6;
      const lvY = baseSize * 0.35;
      ctx.beginPath();
      ctx.arc(lvX, lvY, baseSize * 0.55, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(59, 130, 246, ${{meshState.opacity * 0.55}})`;
      ctx.fill();
      ctx.strokeStyle = '#3b82f6';
      ctx.stroke();
    }}
    
    // Aorta (Magenta)
    if (meshState.ao) {{
      const aoX = Math.sin(rotY - 0.5) * baseSize * 0.3;
      const aoY = -baseSize * 0.75;
      ctx.beginPath();
      ctx.arc(aoX, aoY, baseSize * 0.30, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(217, 70, 239, ${{meshState.opacity * 0.6}})`;
      ctx.fill();
      ctx.strokeStyle = '#d946ef';
      ctx.stroke();
    }}
    
    ctx.restore();
  }}
  
  draw3D();
  
  let isDragging = false;
  let startX, startY;
  
  canvas.onmousedown = (e) => {{
    isDragging = true;
    startX = e.clientX;
    startY = e.clientY;
  }};
  
  window.onmousemove = (e) => {{
    if (!isDragging) return;
    const dx = e.clientX - startX;
    const dy = e.clientY - startY;
    rotY += dx * 0.01;
    rotX += dy * 0.01;
    startX = e.clientX;
    startY = e.clientY;
    draw3D();
  }};
  
  window.onmouseup = () => {{ isDragging = false; }};
  
  canvas.onwheel = (e) => {{
    e.preventDefault();
    scale3D += e.deltaY * -0.001;
    scale3D = Math.max(0.5, Math.min(2.0, scale3D));
    draw3D();
  }};
  
  // Camera buttons
  document.getElementById('btn-cam-ant').onclick = () => {{ rotX = 0; rotY = 0; draw3D(); }};
  document.getElementById('btn-cam-post').onclick = () => {{ rotX = 0; rotY = Math.PI; draw3D(); }};
  document.getElementById('btn-cam-lat').onclick = () => {{ rotX = 0; rotY = Math.PI/2; draw3D(); }};
  document.getElementById('btn-cam-4ch').onclick = () => {{ rotX = 0.3; rotY = -0.6; draw3D(); }};
  
  // Layer toggles in 3D
  document.getElementById('m3d-la-fat').onchange = (e) => {{ meshState.laFat = e.target.checked; draw3D(); }};
  document.getElementById('m3d-peri').onchange = (e) => {{ meshState.peri = e.target.checked; draw3D(); }};
  document.getElementById('m3d-la').onchange = (e) => {{ meshState.la = e.target.checked; draw3D(); }};
  document.getElementById('m3d-lv').onchange = (e) => {{ meshState.lv = e.target.checked; draw3D(); }};
  document.getElementById('m3d-ao').onchange = (e) => {{ meshState.ao = e.target.checked; draw3D(); }};
  document.getElementById('m3d-opacity').oninput = (e) => {{ meshState.opacity = e.target.value / 100; draw3D(); }};
}}

// Curtain Slider logic
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

// Event Listeners & Initialization
window.addEventListener('DOMContentLoaded', () => {{
  // Layer toggles
  document.getElementById('layer-ct').onchange = (e) => {{
    document.getElementById('img-ct-a').style.display = e.target.checked ? 'block' : 'none';
  }};
  document.getElementById('layer-peri').onchange = (e) => {{
    document.getElementById('img-peri-a').style.display = e.target.checked ? 'block' : 'none';
  }};
  document.getElementById('layer-anchors').onchange = (e) => {{
    document.getElementById('img-anchors-a').style.display = e.target.checked ? 'block' : 'none';
  }};
  document.getElementById('layer-partition').onchange = (e) => {{
    document.getElementById('img-partition-a').style.display = e.target.checked ? 'block' : 'none';
  }};
  document.getElementById('layer-la-fat').onchange = (e) => {{
    document.getElementById('img-la-fat-a').style.display = e.target.checked ? 'block' : 'none';
  }};
  document.getElementById('layer-pv').onchange = (e) => {{
    document.getElementById('img-pv-a').style.display = e.target.checked ? 'block' : 'none';
  }};
  
  // Patient picker
  document.getElementById('patient-select').onchange = (e) => {{
    currentPatientId = e.target.value;
    updatePatientView();
  }};
  
  // Axial Slice Slider
  document.getElementById('slice-slider-a').oninput = (e) => {{
    currentAxialSlice = parseInt(e.target.value);
    updatePatientView();
  }};
  
  // Coronal Slider
  document.getElementById('slider-coronal').oninput = (e) => {{
    currentCoronalSlice = parseInt(e.target.value);
    updatePatientView();
  }};
  
  // Sagittal Slider
  document.getElementById('slider-sagittal').oninput = (e) => {{
    currentSagittalSlice = parseInt(e.target.value);
    updatePatientView();
  }};
  
  // Mousewheel scrolling
  document.getElementById('canvas-stack-a').onwheel = (e) => {{
    e.preventDefault();
    if (e.deltaY > 0) currentAxialSlice++;
    else currentAxialSlice--;
    updatePatientView();
  }};
  
  // Keyboard navigation
  window.addEventListener('keydown', (e) => {{
    if (['input', 'select', 'textarea'].includes(e.target.tagName.toLowerCase())) return;
    
    if (e.key === '1') setVariantById('B');
    else if (e.key === '2') setVariantById('A');
    else if (e.key === '3') setVariantById('C');
    else if (e.key === 'j' || e.key === 'ArrowDown') {{
      const keys = Object.keys(cohortData);
      const currIdx = keys.indexOf(currentPatientId);
      currentPatientId = keys[(currIdx + 1) % keys.length];
      document.getElementById('patient-select').value = currentPatientId;
      updatePatientView();
    }} else if (e.key === 'k' || e.key === 'ArrowUp') {{
      const keys = Object.keys(cohortData);
      const currIdx = keys.indexOf(currentPatientId);
      currentPatientId = keys[(currIdx - 1 + keys.length) % keys.length];
      document.getElementById('patient-select').value = currentPatientId;
      updatePatientView();
    }}
  }});
  
  initCurtainSlider();
  
  // Check URL param or default to Cohort Scorecard (B)
  const urlParams = new URLSearchParams(window.location.search);
  const initialTab = urlParams.get('tab') || 'B';
  setVariantById(initialTab);
  updatePatientView();
}});
</script>
</body>
</html>
"""
    return html_content

if __name__ == "__main__":
    print("Building QA Viewer Prototype data and standalone HTML...")
    data = generate_all_patients_payload()
    html = build_prototype_html(data)
    
    out_path = os.path.join(os.path.dirname(__file__), "qa_viewer_prototype.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Successfully generated prototype at: {out_path} ({len(html)} bytes)")
