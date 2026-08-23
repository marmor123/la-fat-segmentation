"""Batch & Cohort pipeline orchestrator for LA Fat Segmentation.

Discovers CT scans in ``data/raw/``, derives canonical patient IDs, runs
TotalSegmentator pre-compute when masks are missing, skips already-processed
patients, executes the deep ``run_fat_extraction`` pipeline for each scan,
and generates the consolidated multi-tab ``cohort_qa_dashboard.html``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from la_fat.cohort_qa_generator import generate_cohort_qa_html
from la_fat.config import PipelineConfig
from la_fat.pipeline import (
    SegmentationResult,
    run_fat_extraction,
    run_fat_extraction_pipeline,
)
from la_fat.ts_runner import extract_patient_id, run_ts_precompute

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public Entry Points
# ---------------------------------------------------------------------------


def run_batch_pipeline(
    data_dir: str | None = None,
    output_dir: str | None = None,
    config: PipelineConfig | None = None,
    config_path: str | None = None,
    force_recompute: bool = False,
) -> dict:
    """Run the fat extraction pipeline for all CT scans in ``data/raw/``.

    Discovers ``.nii.gz`` and ``.nii`` files, derives canonical patient IDs,
    checks for existing results, processes new patients sequentially, and
    compiles a unified cohort QA dashboard.

    Parameters
    ----------
    data_dir:
        Path to data root (containing ``raw/`` subdirectory).
    output_dir:
        Path to output root.
    config:
        Optional pre-built PipelineConfig.
    config_path:
        Optional path to YAML config file.
    force_recompute:
        Whether to re-process scans that already have existing results.

    Returns
    -------
    dict
        Summary with keys ``total``, ``succeeded``, ``failed``, ``skipped``,
        ``failed_ids``, and ``cohort_dashboard_path``.
    """
    if config is not None:
        cfg = config
    elif config_path is not None:
        cfg = PipelineConfig.from_yaml(config_path)
    else:
        cfg = PipelineConfig()

    resolved_data_dir = data_dir or cfg.data_dir
    resolved_output_dir = output_dir or cfg.output_dir

    cfg = dataclasses.replace(
        cfg,
        data_dir=resolved_data_dir,
        output_dir=resolved_output_dir,
    )

    ct_files = _discover_ct_files(resolved_data_dir)
    if not ct_files:
        logger.info("No CT scans found in %s/raw/", resolved_data_dir)
        _print_header(0)
        _print_summary(0, 0, 0, [])
        return {
            "total": 0,
            "succeeded": 0,
            "failed": 0,
            "skipped": 0,
            "failed_ids": [],
            "cohort_dashboard_path": None,
        }

    patient_ids = [extract_patient_id(f) for f in ct_files]

    # Separate new vs. completed
    new_patients: list[str] = []
    skipped: list[str] = []
    for pid in patient_ids:
        if not force_recompute and _is_completed(resolved_output_dir, pid):
            skipped.append(pid)
        else:
            new_patients.append(pid)

    total = len(patient_ids)
    to_process = len(new_patients)

    intermediate_dir = os.path.join(resolved_data_dir, cfg.intermediate_subdir)
    succeeded_count = 0
    failed_count = 0
    failed_ids: list[str] = []
    cohort_records: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    _print_header(total)
    for pid in skipped:
        _print_skipped(pid)
        # Load cached record if available
        cached_result_path = os.path.join(resolved_output_dir, pid, "pipeline_result.json")
        if os.path.isfile(cached_result_path):
            try:
                with open(cached_result_path, "r", encoding="utf-8") as fh:
                    res_data = json.load(fh)
                    summary_rows.append(res_data)
            except Exception:
                pass

    for idx, pid in enumerate(new_patients, start=1):
        ct_file = ct_files[patient_ids.index(pid)]

        # TS pre-compute if masks missing
        if not _masks_exist(intermediate_dir, pid):
            _print_ts_start(idx, to_process, pid)
            try:
                ts_result = run_ts_precompute(ct_file, intermediate_dir, cfg)
                if ts_result.errors:
                    _print_ts_failed(ts_result.errors)
                    failed_count += 1
                    failed_ids.append(pid)
                    continue
                _print_ts_done(len(ts_result.masks_saved), ts_result.total_runtime_seconds)
            except Exception as exc:
                _print_ts_failed([str(exc)])
                failed_count += 1
                failed_ids.append(pid)
                continue
        else:
            _print_progress(idx, to_process, pid)

        # Execute deep extraction pipeline
        try:
            result = run_fat_extraction_pipeline(
                patient_id=pid,
                config=cfg,
            )
            if result.success:
                succeeded_count += 1
                _print_done(result.la_fat_volume_adaptive_ml)
                if result.qa_record:
                    cohort_records[pid] = result.qa_record
                summary_rows.append(result.to_dict())
            else:
                failed_count += 1
                failed_ids.append(pid)
                _print_failed(result.errors)
        except Exception as exc:
            failed_count += 1
            failed_ids.append(pid)
            _print_failed([str(exc)])

    # Compile Cohort QA Dashboard if records exist
    cohort_dashboard_path: Optional[str] = None
    if cohort_records:
        cohort_dashboard_path = os.path.join(resolved_output_dir, "cohort_qa_dashboard.html")
        try:
            generate_cohort_qa_html(cohort_records, cohort_dashboard_path)
            logger.info("Compiled multi-patient cohort QA dashboard at: %s", cohort_dashboard_path)
        except Exception as exc:
            logger.warning("Failed to generate cohort QA dashboard: %s", exc)

    # Save Cohort Summary CSV
    if summary_rows:
        try:
            csv_path = os.path.join(resolved_output_dir, "cohort_summary.csv")
            pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
            logger.info("Saved cohort summary metrics to: %s", csv_path)
        except Exception as exc:
            logger.warning("Failed to save cohort summary CSV: %s", exc)

    _print_summary(succeeded_count, failed_count, len(skipped), failed_ids)

    return {
        "total": total,
        "succeeded": succeeded_count,
        "failed": failed_count,
        "skipped": len(skipped),
        "failed_ids": failed_ids,
        "cohort_dashboard_path": cohort_dashboard_path,
    }


run_cohort_pipeline = run_batch_pipeline


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------


def _discover_ct_files(data_dir: str) -> list[str]:
    """Discover CT scan files in ``<data_dir>/raw/`` or ``<data_dir>/``."""
    raw_dir = os.path.join(data_dir, "raw")
    target_dir = raw_dir if os.path.isdir(raw_dir) else data_dir
    if not os.path.isdir(target_dir):
        return []

    files: list[str] = []
    for fname in os.listdir(target_dir):
        if fname.endswith(".nii.gz") or fname.endswith(".nii"):
            files.append(os.path.join(target_dir, fname))

    files.sort()
    return files


def _is_completed(output_dir: str, patient_id: str) -> bool:
    """Return ``True`` if patient already has a successful pipeline result."""
    result_path = os.path.join(output_dir, patient_id, "pipeline_result.json")
    return os.path.isfile(result_path)


def _masks_exist(intermediate_dir: str, patient_id: str) -> bool:
    """Return ``True`` if TS masks exist for *patient_id*."""
    patient_dir = os.path.join(intermediate_dir, patient_id)
    if not os.path.isdir(patient_dir):
        return False
    return any(fname.endswith(".nii.gz") for fname in os.listdir(patient_dir))


# ---------------------------------------------------------------------------
# Terminal Output Formatting
# ---------------------------------------------------------------------------


def _print_header(total: int) -> None:
    print(f"\n=======================================================")
    print(f"[*] LA Fat Segmentation — Cohort Pipeline ({total} Scans)")
    print(f"=======================================================")


def _print_progress(idx: int, total: int, patient_id: str) -> None:
    print(f"[{idx}/{total}] Processing {patient_id}... ", end="", flush=True)


def _print_ts_start(idx: int, total: int, patient_id: str) -> None:
    print(f"[{idx}/{total}] Running TotalSegmentator for {patient_id}... ", end="", flush=True)


def _print_ts_done(mask_count: int, runtime: float) -> None:
    print(f"Done ({mask_count} masks in {runtime:.1f}s)")


def _print_ts_failed(errors: list[str]) -> None:
    print(f"FAILED (TS errors: {', '.join(errors)})")


def _print_done(la_fat_volume: float) -> None:
    print(f"DONE (LA fat: {la_fat_volume:.2f} mL)")


def _print_failed(errors: list[str]) -> None:
    print(f"FAILED ({'; '.join(errors)})")


def _print_skipped(patient_id: str) -> None:
    print(f"[-] Patient {patient_id} already processed — SKIPPED")


def _print_summary(
    succeeded: int,
    failed: int,
    skipped: int,
    failed_ids: list[str],
) -> None:
    print(f"\n=======================================================")
    print(f"[*] Batch Pipeline Complete: {succeeded} Succeeded, {failed} Failed, {skipped} Skipped")
    if failed_ids:
        print(f"[-] Failed IDs: {', '.join(failed_ids)}")
    print(f"=======================================================\n")


def main() -> None:
    """CLI entrypoint for cohort batch pipeline."""
    parser = argparse.ArgumentParser(
        description="Run LA Fat Segmentation across a cohort of patient CT scans.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root data directory containing raw/ and intermediate/ subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root output directory for patient segmentation outputs and cohort dashboard.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--force",
        "--force-recompute",
        action="store_true",
        help="Recompute all patients even if existing outputs are found.",
    )
    args = parser.parse_args()

    # Discover and inform
    resolved_data_dir = args.data_dir or PipelineConfig().data_dir
    ct_files = _discover_ct_files(resolved_data_dir)
    print(f"Found {len(ct_files)} patient(s) to process in {resolved_data_dir}")

    if not ct_files:
        print(f"No CT scans found in {os.path.join(resolved_data_dir, 'raw')}. Exiting gracefully (0 patient(s)).")
        sys.exit(0)

    summary = run_batch_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        force_recompute=args.force,
    )
    if summary.get("failed", 0) > 0 and summary.get("succeeded", 0) == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
