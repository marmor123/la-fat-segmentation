# Research: Clinical Non-Contrast HU Windows & TotalSegmentator v2 Label Contracts

**Ticket:** [Ticket 1: [Research] Clinical Non-Contrast HU Windows & TS v2 Label Contracts](https://github.com/marmor123/la-fat-segmentation/issues/31)  
**Date:** 2026-08-18  
**Author:** Antigravity (Wayfinder Session)

---

## 1. Clinical Non-Contrast Hounsfield Unit (HU) Windows

### 1.1 Consensus & Literature Standards for EAT Quantification

Epicardial adipose tissue (EAT) quantification on non-contrast cardiac CT relies on attenuation windowing to isolate fat voxels from surrounding myocardium, blood pool, and pericardial structures. Multiple threshold intervals exist in clinical literature:

| Interval (HU) | Primary Literature & Usage | Characteristics & Rationale |
| :--- | :--- | :--- |
| **`[-190, -30]` HU** | **SCCT Consensus / Mahabadi et al. / Dey et al. / Commandeur et al.** | **Standard clinical convention** for non-contrast cardiac CT EAT quantification. Captures full physiological fat range while preventing myocardial (>0 HU) contamination. Widely used across automated CT software. |
| **`[-195, -45]` HU** | **MESA Study (Ding et al.) / Calcium Scoring Protocols** | Tighter threshold designed to suppress partial-volume voxels at the boundary between myocardium and adipose tissue. |
| **`[-135, -35]` HU** | **Gorter et al. / Contrast-Enhanced CT Variants** | Narrowed window historically used in select CTA protocols to avoid high-attenuation contrast leakage or blood-pool partial volume effects. |
| **`[-200, 0]` HU** | **Broad Adipose Studies** | Generous window; prone to including partial-volume myocardial and fibrous tissue if unmasked by anatomical boundaries. |

### 1.2 Pipeline Decision: Peak-Centered Gaussian Fit with Fallback

1. **Per-Patient Adaptive Fit:** Fit a single Gaussian $\mathcal{N}(\mu, \sigma^2)$ on sub-0 HU voxels strictly within the pericardial ROI. In non-contrast cardiac CT, the fat peak consistently centers at $\mu \approx -100 \text{ to } -75\text{ HU}$ with $\sigma \approx 20 \text{ to } 30\text{ HU}$.
2. **Adaptive Range:** Set $\text{Range} = [\mu - 2\sigma, \mu + 2\sigma]$.
3. **Clamping & Fallback:** Clamp the resulting range to the consensus fallback bounds `[-190, -30]` HU. If the Gaussian fit fails (e.g., $\sigma > 100\text{ HU}$, too few voxels, inverted bounds), fall back to `[-190, -30]` HU and raise a high-severity quality flag.

---

## 2. TotalSegmentator v2 Label Contracts

### 2.1 Task: `heartchambers_highres`
TotalSegmentator v2 provides high-resolution 7-class cardiac segmentation via `--task heartchambers_highres`.

| Label ID | Structure Name | Pipeline Role |
| :--- | :--- | :--- |
| `1` | `heart_myocardium` | Myocardial wall reference |
| `2` | `heart_atrium_left` | **Partition Anchor 1 (LA)** — Target Chamber |
| `3` | `heart_ventricle_left` | **Partition Anchor 2 (LV)** |
| `4` | `heart_atrium_right` | **Partition Anchor 3 (RA)** |
| `5` | `heart_ventricle_right` | **Partition Anchor 4 (RV)** |
| `6` | `aorta` | **Partition Anchor 5 (Aorta)** |
| `7` | `pulmonary_artery` | **Partition Anchor 6 (Pulmonary Artery)** |

### 2.2 Task: `trunk_cavities`
TotalSegmentator v2 provides anatomical cavity segmentation via `--task trunk_cavities` (Note: `--fast` mode is not supported for this task in TS v2).

| Label ID | Structure Name | Pipeline Role |
| :--- | :--- | :--- |
| `1` | `abdominal_cavity` | Non-cardiac |
| `2` | `thoracic_cavity` | Non-cardiac |
| `3` | `pericardium` | **Primary Pericardial Envelope (Solid 3D ROI)** |
| `4` | `mediastinum` | Non-cardiac |

*Fallback Note:* When TotalSegmentator fails to detect a valid pericardium ($V_{\text{pericardium}} < 50\text{ mL}$), the pipeline falls back to the morphological convex hull of the 6 cardiac anchor masks with a configurable dilation.

---

## 3. Coordinate Spaces, Affines, & Resampling Contracts

1. **TotalSegmentator Spatial Invariance:** TotalSegmentator outputs binary/multi-label NIfTI volumes in the **exact coordinate space and affine grid** of the input CT image (`sform`, `qform`, voxel dimensions, and origin are strictly preserved).
2. **Reference-Locked Isotropic Grid (1.5 mm):**
   - The pipeline resamples all inputs and masks onto an isotropic $1.5 \times 1.5 \times 1.5\text{ mm}^3$ grid.
   - **CT Image Resampling:** Uses 3rd-order spline (or trilinear) interpolation with $-1000\text{ HU}$ (air) constant padding to prevent boundary edge distortion.
   - **Binary Anchor & Pericardium Masks Resampling:** Uses Nearest-Neighbor interpolation to preserve binary integer topology without creating pseudo-volume artifacts.
   - All distance transform calculations for the multi-anchor partition are executed in physical millimeter space on this unified grid.

---

## Primary References
1. Mahabadi AA, et al. *Association of Epicardial Fat Volume With Characteristics of Plaque and Coronary Artery Disease*. JACC: Cardiovascular Imaging.
2. Dey D, et al. *Automated quantification of epicardial adipose tissue from non-contrast CT*. RSNA Radiology.
3. Commandeur F, et al. *Deep learning for quantification of epicardial adipose tissue on non-contrast CT*. IEEE TMI.
4. Ding J, et al. *Pericardial and visceral adipose tissue in MESA*. Am J Clin Nutr.
5. Wasserth J, et al. *TotalSegmentator: Robust Segmentation of 104 Anatomical Structures in CT Images*. Radiology: Artificial Intelligence 2023. GitHub: `wasserth/TotalSegmentator`.
