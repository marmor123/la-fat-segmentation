## Destination

A scientifically validated, production-grade, deep-module Left Atrial Epicardial Adipose Tissue (LA EAT) segmentation pipeline that accurately extracts topologically sound 3D binary masks and volume metrics from non-contrast cardiac CT, verified on the 10 real patient scans with zero fragile dependencies.

## Notes

- **Working Branch:** All development, testing, prototypes, and tickets MUST be executed on the `rebuild-v2` branch.
- **Domain:** Cardiac CT image processing, left atrial epicardial fat quantification, radiomics preparation.
- **Guiding Skills:** `/codebase-design/`, `/code-review/`, `/prototype/`, `/tdd/`, `/research/`, `/grilling/`.
- **Architectural Rules:**
  1. Deep modules with small interfaces; the interface is the test surface.
  2. Reference-locked grid resampling with -1000 HU air padding.
  3. Pure CPU fat extraction & QA (no GPU or Docker required for local development).
  4. Throwaway prototypes before production implementation.
  5. **Dual-Grid Radiomics Output:** Pipeline supports fast 1.5mm isotropic screening and native-resolution (512x512, ~0.35mm) binary mask export for IBSI-compliant PyRadiomics texture analysis.
  6. **Forensic Volume Safeguards:** Adaptive Gaussian upper bound clamped at 0.0 HU (capturing the 1-2 voxel partial volume boundary layer), with dual reporting of both adaptive and conservative [-190, -30] HU volumes.

## Decisions so far

- [Modality & Target] — Non-contrast cardiac CT; primary output is 3D binary NIfTI masks with full spatial headers (radiomics-ready) and volume (mL).
- [Thresholding Strategy] — Robust peak-centered trimmed Gaussian fit with fallback to standard clinical window [-190, -30] HU.
- [Pericardial Geometry] — TotalSegmentator v2 solid 3D pericardial cavity is the primary envelope; fallback uses chamber-bounded exclusion.
- [Partition Principle] — Multi-anchor 3D Euclidean surface distance transform across 6 canonical chambers.
- [Eliminated Artifacts] — Discard heavy Marching Cubes .ply pre-computation and Panel/PyVista dashboard.
- [Ticket 1: [Research] Clinical Non-Contrast HU Windows & TS v2 Label Contracts](https://github.com/marmor123/la-fat-segmentation/issues/31) — Validated [-190, -30] HU clinical baseline, Gaussian fit boundaries, and TS v2 heartchambers_highres (7 classes) / trunk_cavities (pericardium ID 3) contract.
- [Ticket 2: [Grilling] Ingestion & Structure of the 10 Real Scans](https://github.com/marmor123/la-fat-segmentation/issues/32) — Standardized 4-digit canonical patient IDs, external NIfTI/mask cache storage with repo manifest, Flash CT acquisition specs, and physiological vs scanner reference sanity bounds.
- [Ticket 3: [Prototype] Trimmed-Gaussian Peak Fitting Logic Demo](https://github.com/marmor123/la-fat-segmentation/issues/33) — Verified prominence-based mode detection, asymmetric tail trimming, multi-tiered quality flags, and safe fallback across 7 synthetic and clinical CT scenarios.
- [Ticket 4: [Prototype] Surface Distance Partition on Synthetic Phantom](https://github.com/marmor123/la-fat-segmentation/issues/34) — Validated 3D multi-anchor solid EDT across non-convex AV groove saddle concavities and thin septal boundaries with 100% primary component purity, 0 septal bleed, and 35mm distance clamping.
- [Ticket 5: [Prototype] Lightweight Zero-Footprint QA Slice Viewer UI](https://github.com/marmor123/la-fat-segmentation/issues/35) — Validated zero-dependency offline HTML5/WebP multi-planar viewer combining cohort triage scorecard, 6-channel layer toggles, split-screen curtain wipe, and 3D colleague presentation studio.

## Not yet specified

- Automated PyRadiomics feature extraction module (IBSI-standardized GLCM, GLRLM, GLSZM, Wavelet textures at native CT resolution).
- Distal pulmonary vein sleeve ostial boundary refinement heuristics.
- TotalSegmentator v2 native-grid re-inference on raw 512x512 CT scans (for Ticket 9 benchmark cohort).
- Long-term Option C: Training a direct 3D nnU-Net on the verified cohort.

## Out of scope

- Contrast-enhanced CTA blood pool handling (deferred to future phase).
- Marching Cubes 3D `.ply` mesh disk generation for dashboarding.
