# ADR-0002: Zero-Footprint HTML5/WebGL PACS and 3D Mesh QA Studio

## Status
Accepted (Supersedes previous PyVista/Panel architecture)

## Context
Quality assurance (QA) is critical for clinical adoption and verification of cardiac CT adipose tissue segmentations. The initial implementation explored Python-based interactive dashboards using PyVista, VTK, and Panel. However, that stack introduced heavy C++ OpenGL system dependencies, complex local web-server lifecycles, cross-platform display driver incompatibilities, and slow startup times.

Researchers and clinicians require an instant, zero-setup, portable viewer that runs directly in any modern web browser without requiring a running Python kernel or npm server.

## Decision
We implemented a zero-footprint, single-file HTML5/Canvas/WebGL PACS QA Studio (`cohort_qa_viewer.html` and per-patient `qa_report.html`) generated automatically by `la_fat.cohort_qa_generator`:

1. **Multi-Planar Reconstruction (MPR) PACS Viewer (Tab B)**:
   - Synchronized orthogonal 2D slice scrubbers across Axial, Coronal, and Sagittal planes.
   - 6-channel layer toggles (CT grayscale, Pericardial envelope, 6 Partition Anchors, LA Fat, Unassigned Fat).
   - Interactive curtain wipe comparing raw CT against full multi-label segmentation overlays.
   - Window/Level presets (Cardiac Fat `[-190, 30]`, Soft Tissue `[-100, 200]`, Mediastinal `[-50, 350]`).

2. **3D WebGL Mesh Studio (Tab C)**:
   - Direct client-side WebGL rendering using embedded 3D isosurface geometry across all 8 cardiac structures.
   - Smooth trackball orbit, pan, and zoom controls.
   - Anatomy layer visibility and opacity sliders.
   - Camera presets (Anterior, Posterior, Left Lateral, Mitral Valve View, Superior).

3. **Cohort Scorecard & Triage Matrix (Tab A)**:
   - Tabular summary of volumes (Adaptive, GMM Bayes, Conservative), HU threshold boundaries, and quality flags.
   - Interactive 5-slice axial filmstrip thumbnails with hover enlargement.
   - Direct deep-linking into individual patient multi-planar inspection.

4. **100% Offline & Zero Dependency**:
   - All shaders, styles, and WebP slice textures are embedded directly in the standalone HTML file.
   - Zero background Python servers, zero node/npm dependencies, and instant load time.

## Consequences
- Eliminates `pyvista`, `panel`, and `vtk` from pipeline dependencies.
- Facilitates seamless sharing of QA reports with clinicians via email, USB, or static file servers.
- Provides immediate visual debugging for quality flags and threshold fallback edge cases.
