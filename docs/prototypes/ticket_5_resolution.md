# Ticket 5 Resolution: Lightweight Zero-Footprint QA Slice Viewer UI

**Issue:** [#35 — [Prototype] Lightweight Zero-Footprint QA Slice Viewer UI](https://github.com/marmor123/la-fat-segmentation/issues/35)  
**Parent:** [#30 — Wayfinder Map](https://github.com/marmor123/la-fat-segmentation/issues/30)  
**Artifacts:**
- Prototype HTML: `prototypes/qa_viewer_prototype.html`
- Generator Script: `prototypes/prototype_qa_viewer.py`
- Architectural Graphic: `docs/prototypes/ticket_5_qa_viewer_demo.png`

---

## 1. The Question Settled
> *What is the most effective, zero-dependency layout for a researcher to review 10+ patient segmentations (Multi-planar Orthogonal SVG/HTML Slice Gallery vs Canvas vs in-browser NiiVue)?*

### The Verdict:
A **unified single-page application with 3 top-level navigation tabs**, operating 100% offline via `file://` double-click without requiring local server processes or WebGL/WASM compilation overhead:
1. **`[ 📊 Cohort Scorecard ]` Tab**: A persistent 10-patient biometric scorecard matrix with discrete Quality Concern badges (🔴 High, 🟡 Medium, 🟢 Low), Gaussian fit parameters ($\mu \pm 2\sigma$), and a 5-landmark anatomical filmstrip for 1-click slice jumping across key cardiac levels. Supports fast keyboard triage via `j` / `k` scan switching.
2. **`[ 🩻 Multi-Planar PACS ]` Tab**: 3-panel synchronized orthogonal matrix (Axial primary, Coronal, Sagittal) with dedicated plane scrubbers (`Z`, `Y`, `X`), 6 independent multi-channel layer opacity toggles, and a draggable split-screen **curtain wipe** (`⇹`) to visually verify zero tissue bleed against native CT.
3. **`[ 🧊 3D Colleague Studio ]` Tab**: Clean full-stage 3D rotatable cardiac volume mesh viewport with independent surface toggles (*Final LA Fat*, *Pericardium sac*, *Left Atrium*, *Left Ventricle*, *Aorta*), surface opacity slider, mouse orbit/zoom, and camera angle presets (*Anterior*, *Posterior*, *Left Lateral*, *4-Chamber*).

---

## 2. Specifications Locked for Production (Ticket 9 QA Dashboard)
- **Zero Runtime Dependencies:** Pure HTML5/Canvas/JS embedding compact WebP slice stacks (~1.9 MB for 4-patient multi-slice, 3-plane dataset). Zero server, zero npm, zero external CDN reliance.
- **Top-Level Navigation:** Clean top tab bar embedded in the application header for seamless switching between Cohort Scorecard, Multi-Planar PACS, and 3D Studio views.
- **Canonical Color Palette:**
  - Pericardium Sac: Lime Green outline (`#22c55e`)
  - Left Atrium (LA): Coral Red (`#ef4444`)
  - Left Ventricle (LV): Royal Blue (`#3b82f6`)
  - Right Atrium (RA): Cyan/Teal (`#06b6d4`)
  - Right Ventricle (RV): Orange (`#f97316`)
  - Aorta / Great Vessels: Magenta (`#d946ef`)
  - Pulmonary Artery (PA): Bright Yellow (`#eab308`)
  - Final LA-EAT Mask: Bright Amber Gold (`#facc15`) with semi-transparent fill
  - Partial Volume Zone (0 to -30 HU): Electric Purple (`#a855f7`)
- **Clinical Presentation Tools:** 1-click Window/Level presets (`Mediastinal 40/350`, `Fat -70/200`, `Wide -100/500`) and Split Curtain Wipe.
