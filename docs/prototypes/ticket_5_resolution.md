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
A **hybrid single-page HTML5 Canvas & WebP architecture** that operates 100% offline via `file://` double-click, combining three complementary viewing modalities:
1. **Instant 2D Multi-Planar Orthogonal Scrubber (Variant A)** with multi-channel layer opacity toggles, synchronized crosshairs, and a draggable split-screen curtain wipe to visually verify zero tissue bleed against native CT.
2. **Cohort Scorecard & Triage Inspector (Variant B)** featuring a persistent 10-patient biometric matrix with quality concern badges (🔴 High, 🟡 Medium, 🟢 Low), Gaussian fit parameters, and a 5-landmark anatomical filmstrip for 1-click slice jumping.
3. **Colleague 3D Presentation Studio (Variant C)** with rotatable 3D cardiac volume meshes, presentation takeaway cards, and 1-click clinical summary figure export.

---

## 2. Structural Comparison of the 3 Prototype Variants

| Feature | Variant A: Radiology PACS | Variant B: Cohort Scorecard | Variant C: 3D Colleague Studio |
| :--- | :--- | :--- | :--- |
| **Primary Focus** | Deep anatomical boundary QA | Rapid 10-scan cohort triage | Colleague & clinical meetings |
| **Layout** | 3-Panel Orthogonal (Axial + Coronal + Sagittal) | Top Scorecard + Center Scroller + Filmstrip | 3D Rotatable Heart Stage + Slide Deck |
| **Layer Toggles** | 6 Independent Channels (CT, Sac, Anchors, EAT, Final LA, PV) | Preset combinations + Filmstrip | 3D mesh surface toggles |
| **Verification Tool** | Draggable Split-Screen Curtain Wipe | Discrete Concern Badges & Histograms | Presentation summary cards |
| **Navigation** | Mouse-wheel continuous scrub | `j` / `k` scan switching + Landmark pills | Camera presets (Anterior/Posterior/Lateral) |

---

## 3. Key Specifications Locked for Production (Ticket 9 QA Generator)
- **Zero Runtime Dependencies:** Pure HTML/CSS/JS embedding compact WebP slice stacks (~1.1 MB for full 4-patient multi-layer cohort). Zero server, zero npm, zero external CDN reliance.
- **Canonical Palette:**
  - Pericardium Sac: Lime Green outline (`#22c55e`)
  - Left Atrium (LA): Coral Red (`#ef4444`)
  - Left Ventricle (LV): Royal Blue (`#3b82f6`)
  - Right Atrium (RA): Cyan/Teal (`#06b6d4`)
  - Right Ventricle (RV): Orange (`#f97316`)
  - Aorta: Magenta (`#d946ef`)
  - Pulmonary Artery (PA): Bright Yellow (`#eab308`)
  - Final LA-EAT Mask: Bright Amber Gold (`#facc15`) with semi-transparent fill
  - Partial Volume Zone (0 to -30 HU): Electric Purple (`#a855f7`)
- **Presentation Features:** 1-click Window/Level presets (`Mediastinal 40/350`, `Fat -70/200`, `Wide -100/500`) and Split Curtain Wipe.
