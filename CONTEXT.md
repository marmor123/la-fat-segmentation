# CONTEXT — LA Fat Segmentation

## Glossary

- **LA Fat** — Left Atrial Epicardial Adipose Tissue. All epicardial fat partitioned to the left atrium. Shorthand: "LA fat."
- **Epicardial Fat** — Adipose tissue between the heart muscle (myocardium) and the pericardium (the sac around the heart). Not mediastinal fat, not visceral fat.
- **TotalSegmentator (TS)** — An off-the-shelf deep-learning model that segments 100+ anatomical structures from CT scans. Used here for anatomical localization, not for fat segmentation itself.
- **Pericardium** — The fibrous sac surrounding the heart. The outer boundary for epicardial fat — fat outside this is not epicardial.
- **Partition Anchors** — The six structures whose surfaces form the basis for assigning fat voxels: {LA, LV, RA, RV, Aorta, Pulmonary Artery}. Pulmonary veins are excluded as a partition target (their surrounding fat belongs to LA).
- **Partition** — The assignment of each epicardial fat voxel to the nearest anchor structure, based on distance from the voxel to each anchor's surface. Replaces explicit anatomical exclusions, mitral plane cutoffs, and superior buffers with a single distance-based principle.
- **HU (Hounsfield Units)** — CT intensity scale. Fat typically sits in the negative range (~-190 to -30 HU), but the exact range varies by scanner, contrast protocol, and patient habitus.
- **Per-Patient Fat Threshold** — A single Gaussian fitted to the sub-0 HU voxel distribution within the pericardium. The fat range = mean ± 2σ, clamped to a wide fixed fallback (-190 to -30 HU). Simpler than GMM: one peak, no class-swapping, no seed initialization.
- **TS Pre-Compute** — TotalSegmentator runs as a separate, GPU-dependent pre-processing step. Masks are saved to disk once. The fat extraction pipeline is pure CPU and can be re-run with different parameters without re-running TS. Matches the "validate TS first" goal.
- **QA Dashboard** — Per-scan output includes: (1) slice gallery with all 6 anchors + pericardium overlaid on CT in distinct colors, (2) fat overlay color-coded by assigned anchor, (3) numeric table with volume per partition, fat thresholds, failsafes fired, and confidence, (4) 3D rotatable view of LA, pericardium, and EAT.
- **Quality Flags** (not a collapsed score) — **High concern:** pericardium fallback triggered, anchor mask missing or below volume threshold, fat threshold fell back to fixed range. **Medium concern:** LA fat volume outside 2–150ml, LV captures more total fat than LA, >80% of pericardial fat unassigned. **Low concern:** wide Gaussian σ, small islands cleaned up. Each flag reported separately; the user evaluates, not a magic number.
