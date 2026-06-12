# PRD: LA Fat Segmentation — Ground-Up Rebuild

## Problem Statement

The existing LA EAT segmentation pipeline evolved through trial-and-error experimentation with an AI coding agent. Over many iterations, the codebase accumulated technical debt: dead-end algorithm variants, inconsistent error handling, memory blowups (55 GiB allocations), API mismatches, and fragile heuristics that fail on different anatomical variants. The core idea — segment left atrial epicardial fat from CT scans — got buried under implementation patches. The researcher needs a clean rebuild from first principles, preserving the hard-won research insights while replacing the architecture with something simpler and more robust.

## Solution

A ground-up rebuild of the LA fat segmentation pipeline that replaces four fragile heuristic steps (SVM mitral plane, dilate-and-subtract exclusions, superior buffer, vein plug) with a single distance-based partition principle: every epicardial fat voxel is assigned to whichever anatomical anchor surface it is nearest to. The rebuild separates concerns cleanly — TotalSegmentator runs as a pre-compute step, and the fat extraction pipeline is pure CPU with no GPU dependency. Every intermediate decision is instrumented and visualized so the researcher can debug failures directly.

## User Stories

1. As a researcher, I want to run TotalSegmentator once on a batch of CT scans and save all anatomical masks to disk, so that I can iterate on the fat extraction logic without re-running the GPU step.
2. As a researcher, I want the pipeline to detect fat using a per-patient Gaussian fit to the HU distribution within the pericardium, so that thresholding adapts to scanner and protocol differences without the complexity and instability of GMM.
3. As a researcher, I want epicardial fat partitioned to the nearest anatomical anchor surface (LA, LV, RA, RV, Aorta, Pulmonary Artery), so that the fat assignment respects the patient's actual heart geometry rather than fixed cutoff planes.
4. As a researcher, I want the pericardium to serve as the outer boundary for epicardial fat, with a convex-hull-of-chambers fallback when TotalSegmentator fails to detect the pericardium, so that no scan is silently wrong.
5. As a researcher, I want every quality concern reported as a discrete flag (not collapsed into a single score), so that I can triage results by what actually went wrong.
6. As a researcher, I want a per-scan QA dashboard that includes: a multi-anchor slice gallery, a fat overlay color-coded by assigned anchor, a numeric table of volumes and thresholds and flags, and a 3D rotatable view, so that I can visually validate every result.
7. As a researcher, I want the pipeline to flag and skip scans where an anchor mask is missing or below a volume threshold, but also attempt estimation where possible, so that I know which results need manual review.
8. As a researcher, I want all spatial computations to work in physical millimeters (via resampling to isotropic spacing), so that distance-based partition is consistent across scans with different native resolutions.
9. As a researcher, I want the fat extraction pipeline to be a pure-CPU module with no GPU dependency, so that I can run it on any machine and re-run with different parameters cheaply.
10. As a researcher, I want the codebase structured with clear module boundaries and testable seams, so that future changes don't require understanding the entire pipeline.

## Implementation Decisions

### Architecture

- **TS Pre-Compute step**: A separate script (GPU-dependent) that runs TotalSegmentator on raw CT scans, extracts 8 anatomical masks plus pericardium, resamples to 1.5mm isotropic spacing, and saves everything to `data/intermediate/{patient_id}/`. This step is run once per batch.
- **Fat Extraction pipeline**: A pure-CPU pipeline that reads TS masks from disk, computes the fat partition, and generates outputs. No GPU required. Can be re-run with different parameters without re-running TS.
- **Two distinct executables** rather than a two-phase monolith. The boundary between them is the filesystem.

### Modules

- **Config module**: Dataclass-based configuration loaded from YAML. Parameters for spacing, HU fallback range, Gaussian sigma multiplier, minimum anchor volumes, quality flag thresholds. All tunable without editing source.
- **Preprocessor**: Resample CT volumes to isotropic spacing. Extract and preserve spatial metadata (origin, direction, spacing). Handle non-orthonormal direction cosines.
- **TS Runner**: Wrap the TotalSegmentator Python API. Extract the 6 partition anchors plus pericardium and pulmonary veins. Handle the scout-phase optimization (crop to heart bounding box before full segmentation).
- **Pericardium Resolver**: Given TS mask dict, return a valid pericardium mask. If TS pericardium volume < 50ml, fall back to convex hull of {LA, LV, RA, RV, Aorta} with configurable dilation. Reports whether fallback was triggered.
- **Fat Thresholder**: Given CT array and a pericardium ROI, fit a single Gaussian to sub-0 HU voxels. Return fat range = mean ± 2σ, clamped to a configurable fixed fallback range. Reports whether fallback was used.
- **Partition Engine**: The core module. Given CT array, pericardium mask, HU range, and 6 anchor masks, compute 3D distance transforms from each anchor surface. Assign each fat voxel inside the pericardium to the nearest anchor. LA-assigned voxels are the output. Returns the LA fat mask and stats (volume per anchor, partition shares).
- **Cleanup**: Remove small connected components (< configurable mm³ threshold). Optional morphological opening and vessel filling.
- **QA Generator**: Produce per-scan debug outputs: multi-anchor axial/coronal/sagittal slice gallery, fat-over-CT overlay color-coded by assigned anchor, numeric summary table, 3D rotatable visualization (GIF + HTML).
- **Quality Flagger**: Evaluate the result against concern thresholds. Return a list of flags at three severity levels (high, medium, low). No score collapsing.

### Key Design Decisions

- **Distance metric**: Surface distance via 3D distance transforms, not centroid distance. Fat near the LA appendage should be assigned to LA even if the aorta centroid is geometrically closer. Surface distance respects actual chamber geometry.
- **Resampling**: All computations on 1.5mm isotropic grid, configurable. Native anisotropic spacing would complicate distance transforms (one voxel step means different physical distances in different directions).
- **No GMM**: Single Gaussian on sub-0 HU voxels replaces the 2-component GMM. No class-swapping risk, no seed initialization, simpler debugging.
- **No collapsed confidence score**: Each quality concern is a discrete flag. The researcher decides what matters, not a magic number that blends a pericardium failure with a wide Gaussian.

## Testing Decisions

### What makes a good test

- Tests exercise module boundaries at the defined seams — input goes in, output comes out, assertions check the output shape/stats/flags.
- Synthetic 3D volumes (ellipsoids, spheres at known HU values) are preferred over real CT scans for unit tests. They're deterministic, fast, and can be constructed to exercise specific edge cases.
- Tests do NOT assert exact fat volumes (that depends on real data). They assert that: the module returned the right data types, fallback flags fired when expected, quality flags triggered on extreme inputs, output masks have the right dimensions and spatial metadata.

### Modules to test

| Module | Test type | What to verify |
|---|---|---|
| Config | Unit | YAML parsing, defaults, missing file handling |
| Preprocessor | Unit | Resampling preserves spatial metadata, handles non-orthonormal direction |
| Pericardium Resolver | Unit | Fallback triggers below 50ml, convex hull produces valid mask, normal path passes through |
| Fat Thresholder | Unit | Gaussian fit on synthetic HU distribution, fallback on degenerate input |
| Partition Engine | Integration | Synthetic CT with known fat distribution near synthetic anchors — verify LA gets expected voxels |
| Cleanup | Unit | Small islands removed, spatial metadata preserved on export |
| Quality Flagger | Unit | Each concern fires independently, edge thresholds trigger correctly |
| QA Generator | Visual | Output images exist, have expected dimensions, aren't blank |
| TS Runner | Integration (GPU) | Requires real TS install; skip on CI |

### Prior art

The existing project's `tests/test_heuristic_core_e2e.py` already uses synthetic CT volumes with ellipsoid chambers. That approach carries forward. The existing `test_config.py` and `test_qc_engine.py` provide patterns for config and quality testing.

## Out of Scope

- Neural model training (nnU-Net on pseudo-labels). This is Pathway B→C from the advancement plan and comes after the heuristic pipeline is stable and validated.
- Ground truth validation against manual segmentations. Requires clinical annotation effort; not part of the engineering rebuild.
- DICOM SR structured report generation.
- Cloud deployment or containerization.
- Real-time or interactive use — the pipeline is batch-oriented.
- Handling of non-standard patient positioning. All scans are assumed to use the same standard orientation.

## Further Notes

- The rebuild preserves the research knowledge captured in `PROJECT_CONTEXT.md` and `ADVANCEMENT_PLAN.md`. Those documents remain as reference for the old pipeline's findings.
- ADR-0001 (`docs/adr/0001-distance-partition-replaces-heuristics.md`) records the key architectural decision.
- The domain glossary in `CONTEXT.md` defines the canonical vocabulary for the rebuild. All module names, function names, and variable names should use these terms.
- The old `data/intermediate/` and `data/outputs/` directories from the previous pipeline should not be assumed compatible. The rebuild may change file formats and mask naming.
