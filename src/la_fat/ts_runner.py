"""TotalSegmentator pre-computation for the LA Fat Segmentation pipeline.

This module provides the GPU-dependent pre-processing step that runs
TotalSegmentator (TS) on raw CT scans and saves all anatomical masks
to disk.  Downstream CPU-only pipeline stages read these masks from
disk and never call TS themselves.

Output directory convention::

    <output_dir>/<patient_id>/
        <patient_id>_<structure>.nii.gz   … resampled isotropic mask
        _ts_raw/                           … raw TS output (intermediate)
            <ts_structure>.nii.gz

Where ``output_dir`` defaults to ``data/intermediate/`` as configured
via :class:`~la_fat.config.PipelineConfig`, ``patient_id`` is derived
from the input CT filename, and ``structure`` is one of the eight
domain names (LA, LV, RA, RV, Aorta, Pulmonary Artery, Pericardium,
Pulmonary Veins).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import subprocess
import time

import numpy as np
import SimpleITK as sitk

from la_fat.config import PipelineConfig

logger = logging.getLogger(__name__)

# ── Public constants ────────────────────────────────────────────────────────

#: Mapping from our domain names to the filenames TotalSegmentator produces.
#: The keys are the short names used throughout the pipeline; the values are
#: the stem (without ``.nii.gz``) that TS writes to its output directory.
TS_STRUCTURE_NAMES: dict[str, str] = {
    "LA": "heart_atrium_left",
    "LV": "heart_ventricle_left",
    "RA": "heart_atrium_right",
    "RV": "heart_ventricle_right",
    "Aorta": "aorta",
    "Pulmonary Artery": "pulmonary_artery",
    "Pericardium": "pericardium",
    "Pulmonary Veins": "pulmonary_vein",
}

#: Each TS run needed to collect all 8 structures.  Structures are spread
#: across different TotalSegmentator models so we must run several tasks.
#: Each entry is a dict of keyword arguments passed to the TS Python API.
_TS_RUNS: list[dict[str, object]] = [
    # Run 1: heartchambers_highres → LA, LV, RA, RV, Aorta, Pulmonary Artery
    {"task": "heartchambers_highres"},
    # Run 2: total model cropped to heart + pulmonary_vein → Pulmonary Vein
    {"roi_subset": ["heart", "pulmonary_vein"]},
    # Run 3: trunk_cavities → Pericardium (--fast not supported here)
    {"task": "trunk_cavities"},
]


# ── Public result type ──────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class TsPrecomputeResult:
    """Result of running TotalSegmentator pre-computation on a single scan.

    Attributes
    ----------
    patient_id:
        Patient identifier derived from the input CT filename.
    output_dir:
        Absolute path to the directory where masks were saved.
    masks_saved:
        Mapping of domain structure name (e.g. ``"LA"``, ``"LV"``) to
        the absolute path of the saved NIfTI mask file.
    mask_volumes_ml:
        Mapping of domain structure name to computed volume in ml.
    errors:
        List of structure names that could not be segmented or whose
        masks could not be processed.
    total_runtime_seconds:
        Wall-clock duration of the pre-computation in seconds.
    """

    patient_id: str
    output_dir: str
    masks_saved: dict[str, str]
    mask_volumes_ml: dict[str, float]
    errors: list[str]
    total_runtime_seconds: float


# ── Public helpers ──────────────────────────────────────────────────────────


def extract_patient_id(ct_path: str) -> str:
    """Derive a patient identifier from a CT file path.

    The identifier is the filename without common NIfTI extensions
    (``.nii.gz``, ``.nii``).  Examples::

        /data/raw/001.nii.gz   → "001"
        scan_42.nii            → "scan_42"
        patient.nii.gz         → "patient"
    """
    basename = os.path.basename(ct_path)
    if basename.endswith(".gz"):  # .nii.gz → strip .gz first
        basename = os.path.splitext(basename)[0]
    if basename.endswith(".nii"):
        basename = os.path.splitext(basename)[0]
    return basename


def is_ts_available() -> bool:
    """Return ``True`` if a TotalSegmentator executable is reachable.

    Checks both the Python API and the CLI.
    """
    # Check Python API
    try:
        import totalsegmentator  # noqa: F401
        return True
    except ImportError:
        pass

    # Check CLI
    try:
        result = subprocess.run(
            ["TotalSegmentator", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ── Internal helpers ────────────────────────────────────────────────────────


def _run_ts_api(
    ct_path: str,
    output_dir: str,
    **kwargs: object,
) -> None:
    """Run a single TotalSegmentator invocation via the Python API.

    Parameters
    ----------
    ct_path:
        Path to the input CT NIfTI file.
    output_dir:
        Directory where TS will write its output masks.
    **kwargs:
        Additional keyword arguments forwarded to ``totalsegmentator()``
        (e.g. ``task``, ``roi_subset``, ``fast``).

    Raises
    ------
    RuntimeError
        If the API call fails.
    """
    from totalsegmentator.python_api import (  # type: ignore[import-untyped]
        totalsegmentator as _ts_func,
    )

    merged: dict[str, object] = {
        "input": ct_path,
        "output": output_dir,
        **kwargs,
    }
    logger.info(
        "Running TS (task=%s, roi_subset=%s, fast=%s)",
        merged.get("task", "total"),
        merged.get("roi_subset", None),
        merged.get("fast", False),
    )
    _ts_func(**merged)  # type: ignore[arg-type]


def _run_ts_cli(
    ct_path: str,
    output_dir: str,
    task: str = "total",
    roi_subset: list[str] | None = None,
) -> None:
    """Run a single TotalSegmentator invocation via the CLI (fallback).

    Parameters
    ----------
    ct_path:
        Path to the input CT NIfTI file.
    output_dir:
        Directory where TS will write its output masks.
    task:
        TS task name (default ``"total"``).
    roi_subset:
        Optional list of ROI names for region-restricted inference.

    Raises
    ------
    RuntimeError
        If the CLI is not available or fails.
    """
    cmd: list[str] = [
        "TotalSegmentator",
        "-i", ct_path,
        "-o", output_dir,
        "-t", task,
    ]
    if roi_subset:
        cmd.extend(["--roi_subset", ",".join(roi_subset)])

    logger.info("Running TotalSegmentator (CLI): %s", " ".join(cmd))
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "TotalSegmentator CLI not found on PATH. "
            "Install TotalSegmentator (pip install TotalSegmentator) "
            "or verify the executable is accessible."
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr[-500:] if exc.stderr else "(no stderr)"
        raise RuntimeError(
            f"TotalSegmentator failed (exit {exc.returncode}): {stderr}"
        ) from exc
    except subprocess.TimeoutExpired:
        raise RuntimeError("TotalSegmentator timed out after 3600 seconds")


def _resample_mask_to_isotropic(
    mask_path: str,
    target_spacing_mm: float,
) -> sitk.Image:
    """Resample a binary mask to isotropic spacing (nearest neighbour).

    Parameters
    ----------
    mask_path:
        Path to a NIfTI mask file.
    target_spacing_mm:
        Desired isotropic voxel spacing in mm.

    Returns
    -------
    sitk.Image
        Resampled mask.
    """
    image: sitk.Image = sitk.ReadImage(mask_path)

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = tuple(
        int(np.ceil(sz * osp / target_spacing_mm))
        for sz, osp in zip(original_size, original_spacing, strict=True)
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(
        (target_spacing_mm, target_spacing_mm, target_spacing_mm)
    )
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)

    return resampler.Execute(image)


def _compute_volume_ml(mask: sitk.Image, spacing_mm: float) -> float:
    """Compute the volume of a binary mask in millilitres.

    Volume (ml) = voxel_count × voxel_volume_mm³ / 1000.
    """
    array: np.ndarray = sitk.GetArrayFromImage(mask)
    voxel_count = int(np.count_nonzero(array))
    voxel_volume_mm3 = spacing_mm ** 3
    return float(voxel_count * voxel_volume_mm3 / 1000.0)


# ── Main public function ────────────────────────────────────────────────────


def run_ts_precompute(
    ct_path: str,
    output_dir: str,
    config: PipelineConfig,
) -> TsPrecomputeResult:
    """Run TotalSegmentator on a raw CT scan and save resampled masks.

    Steps
    -----
    1. Derive a patient ID from the CT filename.
    2. Create the output directory ``{output_dir}/{patient_id}/``.
    3. Run several TotalSegmentator models (total with heart roi_subset,
       heartchambers_highres, trunk_cavities), saving raw masks to
       per-run temp dirs under ``_ts_raw/``.
    4. For each of the 8 structures:
       a. Locate the TS output mask across all run directories.
       b. Resample to isotropic spacing with nearest-neighbour
          interpolation.
       c. Save as ``{patient_id}_{structure}.nii.gz``.
       d. Compute and log volume in ml.
    5. Report any structures that were not found or failed processing.
    6. Clean up the raw TS output directory.

    Parameters
    ----------
    ct_path:
        Path to the raw CT NIfTI (``.nii`` or ``.nii.gz``) file.
    output_dir:
        Base output directory (e.g. the ``intermediate`` subdir from
        ``PipelineConfig``).
    config:
        Pipeline configuration; primarily used for ``spacing_mm``.

    Returns
    -------
    TsPrecomputeResult
        Summary of saved masks, volumes, and any errors.
    """
    start_time = time.perf_counter()

    patient_id = extract_patient_id(ct_path)
    patient_out_dir = os.path.join(output_dir, patient_id)
    os.makedirs(patient_out_dir, exist_ok=True)

    # Sub-directory for raw TS output (cleaned up at the end)
    ts_raw_dir = os.path.join(patient_out_dir, "_ts_raw")
    os.makedirs(ts_raw_dir, exist_ok=True)

    # ---- Step 3: run TS (possibly multiple models) ------------------------
    logger.info(
        "TS pre-compute for %s (output: %s)", patient_id, patient_out_dir
    )

    # Check whether the Python API is available.
    _use_api = True
    try:
        from totalsegmentator.python_api import totalsegmentator  # noqa: F401
    except ImportError:
        _use_api = False

    for idx, run_kwargs in enumerate(_TS_RUNS):
        run_label = run_kwargs.get("task", "total")
        run_raw_dir = os.path.join(ts_raw_dir, f"_run{idx:02d}")
        os.makedirs(run_raw_dir, exist_ok=True)

        logger.info(
            "TS run %d/%d (task=%s)", idx + 1, len(_TS_RUNS), run_label
        )
        try:
            if _use_api:
                _run_ts_api(ct_path, run_raw_dir, **run_kwargs)
            else:
                _run_ts_cli(
                    ct_path,
                    run_raw_dir,
                    task=str(run_kwargs.get("task", "total")),
                    roi_subset=run_kwargs.get("roi_subset", None),  # type: ignore[arg-type]
                )
        except Exception as exc:
            logger.error(
                "TS run %d/%d (task=%s) failed: %s",
                idx + 1, len(_TS_RUNS), run_label, exc,
            )
            _cleanup_dir(run_raw_dir)
            continue

        # Move all .nii.gz files from run dir up to ts_raw_dir so the
        # structure-discovery loop below can find them in one place.
        for fname in os.listdir(run_raw_dir):
            if fname.endswith(".nii.gz"):
                src = os.path.join(run_raw_dir, fname)
                dst = os.path.join(ts_raw_dir, fname)
                if not os.path.exists(dst):
                    os.rename(src, dst)
        _cleanup_dir(run_raw_dir)

    # ---- Step 4: extract & resample each structure -----------------------
    masks_saved: dict[str, str] = {}
    mask_volumes_ml: dict[str, float] = {}
    errors: list[str] = []

    for domain_name, ts_stem in TS_STRUCTURE_NAMES.items():
        ts_mask_path = os.path.join(ts_raw_dir, f"{ts_stem}.nii.gz")

        if not os.path.isfile(ts_mask_path):
            logger.warning("TS output not found for %s (expected: %s)", domain_name, ts_stem)
            errors.append(domain_name)
            continue

        logger.info("Processing %s  ← %s", domain_name, ts_mask_path)

        try:
            resampled = _resample_mask_to_isotropic(ts_mask_path, config.spacing_mm)

            out_path = os.path.join(
                patient_out_dir,
                f"{patient_id}_{domain_name}.nii.gz",
            )
            sitk.WriteImage(resampled, out_path)

            volume_ml = _compute_volume_ml(resampled, config.spacing_mm)

            masks_saved[domain_name] = out_path
            mask_volumes_ml[domain_name] = volume_ml

            logger.info(
                "  ✓ %s — %.2f ml → %s",
                domain_name,
                volume_ml,
                os.path.basename(out_path),
            )
        except Exception as exc:
            logger.error("  ✗ %s failed: %s", domain_name, exc)
            errors.append(domain_name)

    # ---- Step 6: cleanup raw TS output -----------------------------------
    _cleanup_dir(ts_raw_dir)

    total_runtime = time.perf_counter() - start_time

    logger.info(
        "TS pre-complete for %s: %d masks, %d errors, %.1f s",
        patient_id,
        len(masks_saved),
        len(errors),
        total_runtime,
    )

    return TsPrecomputeResult(
        patient_id=patient_id,
        output_dir=patient_out_dir,
        masks_saved=masks_saved,
        mask_volumes_ml=mask_volumes_ml,
        errors=errors,
        total_runtime_seconds=total_runtime,
    )


def _cleanup_dir(dir_path: str) -> None:
    """Recursively remove a directory tree."""
    try:
        import shutil
        shutil.rmtree(dir_path, ignore_errors=True)
    except Exception:
        pass
