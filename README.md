# LA Fat Segmentation

Automated quantification of **Left Atrial Epicardial Adipose Tissue** from cardiac CT scans.

Given a chest CT, the pipeline:

1. Runs **TotalSegmentator** to segment 100+ anatomical structures (heart chambers, pericardium, great vessels)
2. Resolves the **pericardial boundary** (the outer limit of epicardial fat)
3. Computes a **per-patient fat HU threshold** via Gaussian fitting
4. **Partitions** every epicardial fat voxel to the nearest anchor surface (LA, LV, RA, RV, Aorta, Pulmonary Artery)
5. Cleans the LA fat mask, **extracts 3D meshes**, and generates a per-scan **interactive QA dashboard**

---

## For Researchers

> You don't need Python. You don't need a command line. You need Docker Desktop and the distribution package.

### Installation

You'll receive a folder (USB drive or network share) containing:

```
la-fat/
├── la-fat-image.tar          (~11 GB Docker image)
├── Install.bat               Windows installer
├── install.sh                Linux installer
├── Process Scans.bat         Windows shortcut
├── View Results.bat          Windows shortcut
├── Process Scans.desktop     Linux shortcut
└── View Results.desktop      Linux shortcut
```

**Windows:**

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (one-time)
2. Double-click **`Install.bat`**
3. Two shortcuts appear on your Desktop: **Process Scans** and **View Results**

**Linux:**

```bash
chmod +x install.sh && bash install.sh
```

Two `.desktop` files appear on your Desktop.

> **Safe to re-run:** `Install.bat` / `install.sh` can be re-run to update shortcuts or the Docker image. Your data folder is never touched.

### Usage

#### 1. Drop your CT scans

Place `.nii.gz` or `.nii` files into:

```
Desktop/la-fat-data/data/raw/
```

```
Desktop/
└── la-fat-data/
    └── data/
        └── raw/
            ├── patient_001.nii.gz
            ├── patient_002.nii.gz
            └── ...
```

#### 2. Process Scans

Double-click **`Process Scans`** on your Desktop.

```
============================================================
  LA FAT SEGMENTATION — BATCH PROCESSING
  47 patient(s) found
============================================================

  PATIENT_001           SKIPPED — already processed
  PATIENT_002           SKIPPED — already processed
  [1/43] PATIENT_003    TotalSegmentator (generating masks)...
  DONE (8 masks, 124s)
  DONE (LA Fat: 14.32 ml)
  [2/43] PATIENT_004    TotalSegmentator (generating masks)...
  DONE (8 masks, 98s)
  DONE (LA Fat: 22.17 ml)
  [3/43] PATIENT_005    processing...
  DONE (LA Fat: 8.45 ml)

------------------------------------------------------------
  SUMMARY
  Processed: 41 succeeded, 2 failed, 4 skipped
  Failed patients: PATIENT_022, PATIENT_039
============================================================
```

**What happens:**
- Already-processed patients are **skipped** automatically
- For new patients, **TotalSegmentator** runs first (GPU if available, CPU otherwise)
- The fat extraction pipeline runs next
- Progress is printed to the terminal
- Non-fatal errors on individual patients don't stop the batch

> **First run is slow.** TotalSegmentator must segment every new scan — expect 2–15 minutes per patient on CPU, or 1–3 minutes with an NVIDIA GPU. Once masks are saved, re-running the pipeline for the same patient takes seconds.

#### 3. View Results

Double-click **`View Results`** on your Desktop.

Your browser opens to `http://localhost:5006` with an interactive QA dashboard:

- **Patient list** with severity indicators (green/yellow/red dots)
- **Key Numbers** card: LA Fat volume, total epicardial fat, quality flag counts
- **Three 3D viewports** you can rotate, zoom, and toggle individual structures:
  - Anchors + Pericardium
  - Fat Partition (color-coded by anchor)
  - Final LA Fat (cleaned mask)
- **Quality Flags** panel with per-flag detail

Close the terminal window to stop the dashboard.

### Output Files

Each patient produces a directory under `Desktop/la-fat-data/outputs/<patient_id>/`:

| File | Description |
|---|---|
| `pipeline_result.json` | All numeric results (volumes, flags, warnings) |
| `la_fat_mask.nii.gz` | Final cleaned LA fat binary mask |
| `quality_flags.json` | Quality concerns with severity, detail, and thresholds |
| `dashboard.html` | Standalone QA dashboard |
| `slice_gallery.png` | Gallery of all anchor masks overlaid on CT |
| `fat_overlay.png` | Epicardial fat color-coded by anchor assignment |
| `summary.csv` | Machine-readable numeric summary |
| `summary.txt` | Human-readable numeric summary |
| `meshes/` | 3D surface meshes (`.ply` files) for the dashboard |

### How It Works

```
CT scan (.nii.gz)
  │
  ├─ [TotalSegmentator]  ← runs once per patient, masks cached
  │   ├─ heartchambers_highres  →  LA, LV, RA, RV, Aorta, Pulmonary Artery
  │   ├─ total (heart ROI)      →  Pulmonary Veins
  │   └─ trunk_cavities         →  Pericardium
  │
  └─ [Fat Extraction Pipeline]  ← runs every time (seconds when masks exist)
      ├─ Resample CT to isotropic 1.5 mm
      ├─ Resolve pericardium (direct or fallback)
      ├─ Compute per-patient fat HU threshold
      ├─ Partition fat to nearest anchor surface
      ├─ Clean LA fat mask (island removal)
      ├─ Extract 3D meshes (marching cubes)
      ├─ Generate quality flags
      └─ Generate QA dashboard
```

### Troubleshooting

| Problem | Solution |
|---|---|
| "Docker is not installed" | Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) |
| Docker Desktop won't start | Restart your computer, then start Docker Desktop from the Start Menu |
| Port 5006 already in use | Close any other program using that port, or the other dashboard terminal window |
| Dashboard opens but shows "No patients found" | Run **Process Scans** first |
| Processing is very slow | This is normal for CPU. A GPU speeds up TotalSegmentator significantly |
| "No CT scans found" | Make sure your `.nii.gz` files are in `la-fat-data/data/raw/` |

---

### Requirements (minimum)

| | |
|---|---|
| OS | Windows 10+ or Linux |
| Docker | Docker Desktop (Windows) or Docker Engine (Linux) |
| RAM | 8 GB (16 GB recommended) |
| Disk | 25 GB free (image + data + outputs) |
| GPU | Optional — NVIDIA GPU with ≥6 GB VRAM for faster TotalSegmentator |

---

## For Developers

### Install from Source

```bash
git clone https://github.com/marmor123/la-fat-segmentation.git
cd la-fat-segmentation
pip install -e .
```

Requires Python ≥3.9 and a working TotalSegmentator installation.

### Running a Single Patient

```bash
# From the command line:
python run_pipeline.py --patient 0674 --data-dir data --output-dir outputs

# Or as a console script:
la-fat --patient 0674
```

### Running in Batch Mode

```bash
python -m la_fat.batch_pipeline --data-dir data --output-dir outputs
```

### Running the Dashboard

```bash
python run_dashboard.py --output-dir outputs
# Dashboard at http://localhost:5006
```

### Configuration

Edit `config.yaml` (or pass `--config config.yaml`):

```yaml
# Resampling
spacing_mm: 1.5

# Fat HU threshold (fallback range)
hu_fallback_low: -190.0
hu_fallback_high: -30.0
gaussian_sigma_multiplier: 2.0

# Pericardium
min_pericardium_volume_ml: 50.0
pericardium_dilation_mm: 5.0

# Anchors
min_anchor_volume_ml: 5.0

# Cleanup
min_fat_island_volume_mm3: 100.0

# Quality flag thresholds
la_fat_volume_low_ml: 2.0
la_fat_volume_high_ml: 150.0
max_unassigned_fat_pct: 80.0
max_gaussian_sigma: 100.0
max_lv_la_ratio: 4.0
min_fat_fraction_pct: 8.0

# Paths (relative to working directory)
data_dir: data
output_dir: outputs
intermediate_subdir: intermediate
raw_subdir: raw
```

All parameters have sensible defaults — you only need to set what you want to override.

### Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests use synthetic NIfTI data and mock TotalSegmentator — no GPU or TS license needed.

### Building the Docker Image

The image is built from source for distribution to researchers.

```bash
# 1. Set your TotalSegmentator license
export TOTALSEG_LICENSE=aca_XXXXXXXXXXXXXX

# 2. Build (clones repo, installs deps, downloads TS model weights)
bash docker/rebuild.sh

# 3. Distribute
# Copy la-fat-image.tar + docker/{Install.bat,install.sh,Process Scans.bat,View Results.bat} to USB/network
```

The rebuild script:
- Builds from the current GitHub `master` branch
- Bakes in the TS license via `--build-arg TOTALSEG_LICENSE=...`
- Pre-downloads all 11 TS model weights (~2 GB) so users never wait for downloads
- Exports to `la-fat-image.tar` (~11 GB)

To force a fresh clone (bypass Docker cache):
```bash
docker build --build-arg CACHEBUST=$(date +%s) ...
```

### Architecture

```
src/la_fat/
├── anatomy.py              Canonical anchor definitions, colors, TS name mappings
├── batch_pipeline.py       Discovers CT scans, runs TS pre-compute + fat extraction
├── cleanup.py              Island removal from LA fat mask
├── cli.py                  Single-patient CLI entry point
├── config.py               PipelineConfig frozen dataclass (YAML-loadable)
├── interactive_dashboard.py Panel + PyVista 3D QA dashboard
├── mesh_extractor.py       Marching cubes mesh extraction
├── nifti_io.py             NIfTI read/write helpers
├── partition_engine.py     Distance-based fat-to-anchor partition
├── pericardium_resolver.py Pericardium mask resolution (direct + fallback)
├── pipeline.py             12-step fat extraction orchestrator
├── pipeline_result.py      Typed pipeline result serialization
├── pipeline_types.py       Shared types (SurfaceSpec, ViewportPreset)
├── preprocessor.py         CT resampling to isotropic spacing
├── qa_dashboard.py         Static QA dashboard generation
├── quality_flagger.py      Quality flag generation
└── ts_runner.py            TotalSegmentator pre-compute runner

docker/
├── Dockerfile              PyTorch CUDA base, clones repo, bakes weights + license
├── entrypoint.sh           Writes TS config, dispatches pipeline | dashboard
├── download_weights.py     Pre-downloads all TS model weights during build
├── rebuild.sh              Maintainer: docker build + docker save
├── Install.bat / install.sh       User installers
├── Process Scans.bat / .desktop   Pipeline shortcuts
└── View Results.bat / .desktop    Dashboard shortcuts

tests/
├── test_batch_pipeline.py   Batch wrapper tests (discovery, skip, TS integration)
├── test_pipeline.py         Full pipeline integration tests
├── test_entrypoint.py       CLI + entrypoint tests
├── test_distribution.py     Distribution script validation
└── ...                      Per-module unit tests
```

### Data Flow

```
data/raw/<patient>.nii.gz
  │
  ├─[TS Pre-Compute]──→ data/intermediate/<patient>/
  │                       ├── <patient>_LA.nii.gz
  │                       ├── <patient>_LV.nii.gz
  │                       ├── ... (8 structures)
  │                       └── <patient>_ct_resampled.nii.gz
  │
  └─[Fat Extraction]──→ outputs/<patient>/
                           ├── pipeline_result.json
                           ├── la_fat_mask.nii.gz
                           ├── quality_flags.json
                           ├── dashboard.html
                           ├── slice_gallery.png
                           ├── fat_overlay.png
                           ├── summary.csv / summary.txt
                           └── meshes/
                               ├── step2_anchors/
                               ├── step5_partition/
                               └── step7_final/
```

### TotalSegmentator License

This pipeline uses three TotalSegmentator tasks, two of which are gated:

| Task | Gated | Purpose |
|---|---|---|
| `heartchambers_highres` | Yes — requires license | Heart chambers + great vessels |
| `total` (ROI subset) | No | Pulmonary veins |
| `trunk_cavities` | Yes — requires license | Pericardium |

To obtain a license: [totalsegmentator.com/license-academic](https://backend.totalsegmentator.com/license-academic/)

The license is stored in `~/.totalsegmentator/config.json` as:
```json
{
  "totalseg_id": "totalseg_XXXXXXXX",
  "license_number": "aca_XXXXXXXXXXXXXX"
}
```

For Docker builds, the license is passed via `--build-arg TOTALSEG_LICENSE=aca_XXXXXXXXXXXXXX` and baked into the image. End users never see or provide the license.

---

## Domain Glossary

See [CONTEXT.md](CONTEXT.md) for the full domain glossary covering terms like Partition Anchors, Per-Patient Fat Threshold, Epicardial Fat, and Quality Flag severity levels.
