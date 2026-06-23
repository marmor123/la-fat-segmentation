"""Batch pipeline wrapper for LA Fat Segmentation.

Discovers all CT scans in ``data/raw/``, derives patient IDs, runs
TotalSegmentator pre-compute when masks are missing, skips
already-processed patients, and runs the pipeline sequentially for
each new patient.

This sits above :func:`la_fat.pipeline.run_fat_extraction_pipeline` and
:func:`la_fat.ts_runner.run_ts_precompute` — no pipeline internals are
modified.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
import time

from la_fat.config import PipelineConfig
from la_fat.pipeline import run_fat_extraction_pipeline
from la_fat.ts_runner import extract_patient_id, run_ts_precompute

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_batch_pipeline(
    data_dir: str | None = None,
    output_dir: str | None = None,
    config: PipelineConfig | None = None,
    config_path: str | None = None,
) -> dict:
    """Run the fat extraction pipeline for all new CT scans in ``data/raw/``.

    Discovers ``.nii.gz`` and ``.nii`` files, derives patient IDs,
    checks for existing ``outputs/<patient_id>/pipeline_result.json``
    to skip completed patients, and processes the rest sequentially.

    Parameters
    ----------
    data_dir:
        Path to the data directory (contains ``raw/`` subdirectory).
        If not provided, the config default is used.
    output_dir:
        Path to the output directory (contains per-patient results).
        If not provided, the config default is used.
    config:
        Optional pre-built :class:`PipelineConfig`.  Overrides *data_dir*
        and *output_dir* if provided.
    config_path:
        Path to a YAML configuration file to load.  Ignored if *config*
        is provided directly.

    Returns
    -------
    dict
        Summary with keys ``total``, ``succeeded``, ``failed``,
        ``skipped``, and ``failed_ids``.
    """
    # Resolve configuration
    if config is not None:
        cfg = config
    elif config_path is not None:
        cfg = PipelineConfig.from_yaml(config_path)
    else:
        cfg = PipelineConfig()

    resolved_data_dir = data_dir or cfg.data_dir
    resolved_output_dir = output_dir or cfg.output_dir

    # Override config paths so the pipeline uses the resolved directories
    cfg = dataclasses.replace(
        cfg,
        data_dir=resolved_data_dir,
        output_dir=resolved_output_dir,
    )

    # Discover CT files
    ct_files = _discover_ct_files(resolved_data_dir)
    if not ct_files:
        msg = f"No CT scans found in {resolved_data_dir}/raw/"
        logger.info(msg)
        _print_header(0)
        _print_summary(0, 0, 0, [])
        return {
            "total": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped": 0,
            "failed_ids": [],
        }

    # Build patient list
    patient_ids = [extract_patient_id(f) for f in ct_files]

    # Separate new vs. completed
    new_patients: list[str] = []
    skipped: list[str] = []
    for pid in patient_ids:
        if _is_completed(resolved_output_dir, pid):
            skipped.append(pid)
        else:
            new_patients.append(pid)

    total = len(patient_ids)
    to_process = len(new_patients)

    if to_process == 0:
        _print_header(total)
        for pid in skipped:
            _print_skipped(pid)
        _print_summary(0, 0, len(skipped), [])
        return {
            "total": total,
            "succeeded": 0,
            "failed": 0,
            "skipped": len(skipped),
            "failed_ids": [],
        }

    # Resolve intermediate directory (where TS masks live)
    intermediate_dir = os.path.join(resolved_data_dir, cfg.intermediate_subdir)

    # Process new patients
    succeeded_count = 0
    failed_count = 0
    failed_ids: list[str] = []

    _print_header(total)

    for pid in skipped:
        _print_skipped(pid)

    for idx, pid in enumerate(new_patients, start=1):
        ct_file = ct_files[patient_ids.index(pid)]

        # ── Step A: TS pre-compute (if masks missing) ─────────────────────
        if not _masks_exist(intermediate_dir, pid):
            _print_ts_start(idx, to_process, pid)
            try:
                ts_result = run_ts_precompute(
                    ct_file, intermediate_dir, cfg,
                )
                mask_count = len(ts_result.masks_saved)
                ts_time = ts_result.total_runtime_seconds
                if ts_result.errors:
                    _print_ts_failed(ts_result.errors)
                    failed_count += 1
                    failed_ids.append(pid)
                    continue
                _print_ts_done(mask_count, ts_time)
            except Exception as exc:
                _print_ts_failed([str(exc)])
                failed_count += 1
                failed_ids.append(pid)
                continue
        else:
            _print_progress(idx, to_process, pid)

        # ── Step B: Fat extraction pipeline ───────────────────────────────
        try:
            result = run_fat_extraction_pipeline(
                patient_id=pid,
                config=cfg,
            )
            if result.success:
                la_vol = (
                    result.partition_result.anchor_volumes_ml.get("LA", 0.0)
                    if result.partition_result is not None
                    else 0.0
                )
                succeeded_count += 1
                _print_done(la_vol)
            else:
                failed_count += 1
                failed_ids.append(pid)
                _print_failed(result.errors)
        except Exception as exc:
            failed_count += 1
            failed_ids.append(pid)
            _print_failed([str(exc)])

    _print_summary(succeeded_count, failed_count, len(skipped), failed_ids)

    return {
        "total": total,
        "succeeded": succeeded_count,
        "failed": failed_count,
        "skipped": len(skipped),
        "failed_ids": failed_ids,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _discover_ct_files(data_dir: str) -> list[str]:
    """Discover CT scan files in ``<data_dir>/raw/``.

    Returns absolute paths to ``.nii.gz`` and ``.nii`` files, sorted
    alphabetically.
    """
    raw_dir = os.path.join(data_dir, "raw")
    if not os.path.isdir(raw_dir):
        return []

    files: list[str] = []
    for fname in os.listdir(raw_dir):
        if fname.endswith(".nii.gz") or fname.endswith(".nii"):
            files.append(os.path.join(raw_dir, fname))

    files.sort()
    return files


def _is_completed(output_dir: str, patient_id: str) -> bool:
    """Return ``True`` if *patient_id* already has a pipeline result."""
    result_path = os.path.join(output_dir, patient_id, "pipeline_result.json")
    return os.path.isfile(result_path)


def _masks_exist(intermediate_dir: str, patient_id: str) -> bool:
    """Return ``True`` if TS masks exist for *patient_id*.

    Checks for any ``.nii.gz`` file inside the patient's intermediate
    directory (e.g. ``intermediate/<patient_id>/``).
    """
    patient_dir = os.path.join(intermediate_dir, patient_id)
    if not os.path.isdir(patient_dir):
        return False
    return any(
        fname.endswith(".nii.gz")
        for fname in os.listdir(patient_dir)
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def _print_header(total: int) -> None:
    """Print the batch processing header."""
    print()
    print("=" * 60)
    print(f"  LA FAT SEGMENTATION — BATCH PROCESSING")
    print(f"  {total} patient(s) found")
    print("=" * 60)
    print()


def _print_skipped(patient_id: str) -> None:
    """Print a skip message for an already-processed patient."""
    print(f"  {patient_id:<20}  SKIPPED — already processed")


def _print_progress(idx: int, total: int, patient_id: str) -> None:
    """Print the start of processing for a patient."""
    print(f"  [{idx}/{total}] {patient_id:<20}  processing...", end="")
    sys.stdout.flush()


def _print_done(la_fat_volume_ml: float) -> None:
    """Print completion with LA fat volume."""
    print(f"\r  DONE (LA Fat: {la_fat_volume_ml:.2f} ml)" + " " * 20)


def _print_failed(errors: list[str]) -> None:
    """Print failure with the first error message."""
    first_error = errors[0] if errors else "unknown error"
    print(f"\r  FAILED — {first_error}" + " " * 20)


def _print_ts_start(idx: int, total: int, patient_id: str) -> None:
    """Print that TotalSegmentator is starting for a patient."""
    print(
        f"  [{idx}/{total}] {patient_id:<20}  TotalSegmentator (generating masks)...",
        end="",
    )
    sys.stdout.flush()


def _print_ts_done(mask_count: int, seconds: float) -> None:
    """Print TS completion with mask count and runtime."""
    print(
        f"\r  DONE ({mask_count} masks, {seconds:.0f}s)" + " " * 30
    )


def _print_ts_failed(errors: list[str]) -> None:
    """Print TS failure."""
    first_error = errors[0] if errors else "unknown error"
    print(f"\r  FAILED (TS) — {first_error}" + " " * 20)


def _print_summary(
    succeeded: int,
    failed: int,
    skipped: int,
    failed_ids: list[str],
) -> None:
    """Print the final batch summary."""
    print()
    print("-" * 60)
    print(f"  SUMMARY")
    print(f"  Processed: {succeeded} succeeded, {failed} failed, {skipped} skipped")
    if failed_ids:
        print(f"  Failed patients: {', '.join(failed_ids)}")
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# CLI entry point (python -m la_fat.batch_pipeline)
# ---------------------------------------------------------------------------


def _main_cli() -> None:
    """Parse CLI arguments and run the batch pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="LA Fat Segmentation — Batch Pipeline",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Path to data directory (default: value in config, or 'data')",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Path to output directory (default: value in config, or 'outputs')",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    run_batch_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_path=args.config,
    )


if __name__ == "__main__":
    _main_cli()
