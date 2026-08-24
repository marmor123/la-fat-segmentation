"""Unified Command-Line Interface for LA Fat Segmentation.

Provides the comprehensive ``la-fat`` executable with multi-command dispatch:
  - ``la-fat run``: Single-scan fat extraction with tri-track radiomics output.
  - ``la-fat batch``: Directory-level and cohort batch segmentation & QA dashboard.
  - ``la-fat precompute``: TotalSegmentator mask generation with GPU / CPU selection.
  - ``la-fat dashboard``: Launch zero-footprint WebGL PACS QA Studio in default browser.
  - ``la-fat benchmark``: Run 10-patient clinical correlation analysis against scanner baseline.
  - ``la-fat check``: Diagnostic environment check (PyTorch, CUDA, TS, SimpleITK).
  - ``la-fat --help`` / ``la-fat help``: Rich help menu with usage examples.

Direct patient shorthand:
  ``la-fat 0674`` or ``la-fat --patient 0674`` automatically delegates to ``la-fat run``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import platform
import subprocess
import sys
import webbrowser
from typing import Any, List, Optional

from la_fat.batch_pipeline import run_batch_pipeline
from la_fat.config import PipelineConfig
from la_fat.pipeline import SegmentationResult, run_fat_extraction
from la_fat.ts_runner import extract_patient_id, is_ts_available, run_ts_precompute

logger = logging.getLogger("la_fat")


# ---------------------------------------------------------------------------
# Formatting & Logging Helpers
# ---------------------------------------------------------------------------


def _configure_cli_logging(verbose: bool = False) -> None:
    """Configure console logging format and severity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
    )


def _print_cli_summary(result: SegmentationResult) -> None:
    """Print a structured clinical summary of single-scan segmentation results."""
    lines: list[str] = []
    add = lines.append

    add("=" * 68)
    add(f"  LA FAT SEGMENTATION SUMMARY -- PATIENT {result.patient_id}")
    add("=" * 68)
    add("")

    if result.success:
        add("  Status:                   SUCCESS")
    else:
        add(f"  Status:                   FAILED  ({len(result.errors)} error(s))")
    add(f"  Execution Runtime:        {result.total_runtime_seconds:.2f} s")
    add("")

    if result.success:
        add(
            f"  LA Fat (Adaptive Gauss):  {result.la_fat_volume_adaptive_ml:.2f} mL "
            f"(window: [{result.fat_hu_range_adaptive[0]:.1f}, {result.fat_hu_range_adaptive[1]:.1f}] HU)"
        )
        add(
            f"  LA Fat (GMM Bayes P>=0.5): {result.la_fat_volume_gmm_bayes_ml:.2f} mL "
            f"(window: [{result.fat_hu_range_gmm_bayes[0]:.1f}, {result.fat_hu_range_gmm_bayes[1]:.1f}] HU)"
        )
        add(
            f"  LA Fat (Conservative):    {result.la_fat_volume_conservative_ml:.2f} mL "
            f"(window: [-190.0, -30.0] HU)"
        )
        add(f"  Total Epicardial Fat:     {result.total_eat_volume_adaptive_ml:.2f} mL")
        add(f"  Pericardium Volume:       {result.pericardium_volume_ml:.2f} mL")
        add(
            f"  Unassigned Fat:           {result.unassigned_volume_ml:.2f} mL "
            f"({result.unassigned_fat_pct:.1f}%)"
        )

        if result.islands_removed > 0:
            add(
                f"  Island Cleanup:           {result.islands_removed} island(s) removed "
                f"({result.total_removed_volume_mm3:.1f} mm3)"
            )

        add("")
        add("  Generated Radiomics Artifacts:")
        if result.mask_native_path:
            add(f"    - Native Adaptive Mask: {result.mask_native_path}")
        if result.mask_gmm_bayes_native_path:
            add(f"    - Native GMM Bayes Mask:{result.mask_gmm_bayes_native_path}")
        if result.mask_conservative_native_path:
            add(f"    - Native Cons. Mask:    {result.mask_conservative_native_path}")
        if result.qa_report_path:
            add(f"    - QA PACS Studio HTML:  {result.qa_report_path}")

    if result.quality_flags:
        add("")
        add("  Quality Audit Flags:")
        for flag in result.quality_flags:
            c = flag.concern or getattr(flag, "flag_id", "")
            d = flag.detail or getattr(flag, "message", "")
            add(f"    [{flag.severity.upper():<6}] {c}: {d}")

    if result.errors:
        add("")
        add("  Errors:")
        for err in result.errors:
            add(f"    [-] {err}")

    if result.warnings:
        add("")
        add("  Warnings:")
        for warn in result.warnings:
            add(f"    [!] {warn}")

    add("")
    add("=" * 68)

    sys.stdout.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Subcommand Handlers
# ---------------------------------------------------------------------------


def handle_run(args: argparse.Namespace) -> int:
    """Execute single patient scan fat extraction."""
    patient_id = getattr(args, "patient", None)
    if not patient_id and getattr(args, "input_file", None):
        patient_id = extract_patient_id(args.input_file)

    if not patient_id:
        print("Error: Patient ID is required. Specify --patient <ID> (e.g. --patient 0674) or input file.", file=sys.stderr)
        return 2

    config: PipelineConfig | None = None
    if getattr(args, "config", None):
        config = PipelineConfig.from_yaml(args.config)

    if getattr(args, "data_dir", None) is not None or getattr(args, "output_dir", None) is not None:
        if config is None:
            config = PipelineConfig()
        overrides: dict[str, str] = {}
        if args.data_dir is not None:
            overrides["data_dir"] = args.data_dir
        if args.output_dir is not None:
            overrides["output_dir"] = args.output_dir
        config = dataclasses.replace(config, **overrides)

    logger.info("[*] Starting LA Fat extraction for patient %s...", patient_id)
    result = run_fat_extraction(
        patient_id=patient_id,
        config=config,
        generate_qa=not getattr(args, "no_qa", False),
    )

    _print_cli_summary(result)
    return 0 if result.success else 1


def handle_batch(args: argparse.Namespace) -> int:
    """Execute folder-level and cohort batch processing."""
    patient_ids: list[str] | None = None
    if getattr(args, "patient_ids", None):
        patient_ids = [p.strip() for p in args.patient_ids.split(",") if p.strip()]

    summary = run_batch_pipeline(
        data_dir=getattr(args, "data_dir", None),
        output_dir=getattr(args, "output_dir", None),
        input_dir=getattr(args, "input_dir", None),
        patient_ids=patient_ids,
        config_path=getattr(args, "config", None),
        force_recompute=getattr(args, "force_recompute", False),
        device=getattr(args, "device", "auto"),
        fast=getattr(args, "fast", False),
    )

    if summary.get("cohort_dashboard_path") and not getattr(args, "no_open", False):
        try:
            webbrowser.open(f"file://{os.path.abspath(summary['cohort_dashboard_path'])}")
        except Exception:
            pass

    return 0 if summary.get("failed", 0) == 0 or summary.get("succeeded", 0) > 0 else 1


def handle_precompute(args: argparse.Namespace) -> int:
    """Execute TotalSegmentator mask precomputation on CT scans."""
    if getattr(args, "set_license", None):
        lic = args.set_license.strip()
        ts_home = os.path.expanduser("~/.totalsegmentator")
        os.makedirs(ts_home, exist_ok=True)
        cfg_file = os.path.join(ts_home, "config.json")
        cfg_data = {}
        if os.path.isfile(cfg_file):
            try:
                with open(cfg_file, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
            except Exception:
                pass
        cfg_data["totalseg_id"] = lic
        cfg_data["license_number"] = lic
        cfg_data["statistics_disclaimer_shown"] = True
        with open(cfg_file, "w", encoding="utf-8") as f:
            json.dump(cfg_data, f, indent=2)
        print(f"[+] TotalSegmentator license configured successfully: {lic[:7]}...")
        if not getattr(args, "patient", None) and not getattr(args, "input_dir", None) and not getattr(args, "all", False) and not getattr(args, "input_file", None):
            return 0

    config = PipelineConfig.from_yaml(args.config) if getattr(args, "config", None) else PipelineConfig()
    output_dir = getattr(args, "output_dir", None) or os.path.join(config.data_dir, config.intermediate_subdir)

    # Collect files to precompute
    ct_files: list[str] = []
    if getattr(args, "input_file", None):
        ct_files.append(args.input_file)
    elif getattr(args, "patient", None):
        cand_gz = os.path.join(config.data_dir, config.raw_subdir, f"{args.patient}.nii.gz")
        cand_nii = os.path.join(config.data_dir, config.raw_subdir, f"{args.patient}.nii")
        if os.path.isfile(cand_gz):
            ct_files.append(cand_gz)
        elif os.path.isfile(cand_nii):
            ct_files.append(cand_nii)
        else:
            print(f"[-] CT scan for patient {args.patient} not found in {config.data_dir}/{config.raw_subdir}", file=sys.stderr)
            return 1
    elif getattr(args, "input_dir", None) or getattr(args, "all", False):
        scan_dir = getattr(args, "input_dir", None) or os.path.join(config.data_dir, config.raw_subdir)
        if os.path.isdir(scan_dir):
            for fname in sorted(os.listdir(scan_dir)):
                if fname.endswith(".nii.gz") or fname.endswith(".nii"):
                    ct_files.append(os.path.join(scan_dir, fname))

    if not ct_files:
        print("[-] No CT scan files found to precompute. Specify --patient, --input-file, or --input-dir.", file=sys.stderr)
        return 1

    print(f"[*] Starting TotalSegmentator pre-computation for {len(ct_files)} scan(s)...")
    success_count = 0
    fail_count = 0

    for idx, ct_path in enumerate(ct_files, start=1):
        pid = extract_patient_id(ct_path)
        print(f"\n[{idx}/{len(ct_files)}] Precomputing anatomical masks for {pid} ({ct_path})...")
        try:
            res = run_ts_precompute(
                ct_path=ct_path,
                output_dir=output_dir,
                config=config,
                device=getattr(args, "device", "auto"),
                fast=getattr(args, "fast", False),
            )
            if res.errors:
                print(f"  [-] Completed with errors for {pid}: {', '.join(res.errors)}")
                fail_count += 1
            else:
                print(f"  [+] Successfully saved {len(res.masks_saved)} masks ({res.total_runtime_seconds:.1f}s)")
                success_count += 1
        except Exception as exc:
            print(f"  [-] Failed precomputing {pid}: {exc}", file=sys.stderr)
            fail_count += 1

    print(f"\n[*] Precompute complete: {success_count} succeeded, {fail_count} failed.")
    return 0 if fail_count == 0 else 1


def handle_dashboard(args: argparse.Namespace) -> int:
    """Open the zero-footprint standalone HTML5/WebGL QA Studio in default browser."""
    output_dir = os.path.abspath(getattr(args, "output_dir", None) or "data/outputs")

    target_html: str | None = None
    if getattr(args, "patient", None):
        cand = os.path.join(output_dir, args.patient, "qa_report.html")
        if os.path.isfile(cand):
            target_html = cand
    else:
        for name in ["cohort_qa_viewer.html", "cohort_qa_dashboard.html"]:
            cand = os.path.join(output_dir, name)
            if os.path.isfile(cand):
                target_html = cand
                break
        if not target_html and os.path.isdir(output_dir):
            for item in sorted(os.listdir(output_dir)):
                cand = os.path.join(output_dir, item, "qa_report.html")
                if os.path.isfile(cand):
                    target_html = cand
                    break

    if not target_html or not os.path.isfile(target_html):
        print(f"[-] QA Dashboard HTML not found in output directory: {output_dir}", file=sys.stderr)
        print("[-] Run the pipeline first: `la-fat run --patient <ID>` or `la-fat batch`", file=sys.stderr)
        return 1

    print(f"[+] Opening QA Studio: {target_html}")
    webbrowser.open(f"file://{os.path.abspath(target_html)}")
    return 0


def handle_benchmark(args: argparse.Namespace) -> int:
    """Run the 10-patient cohort correlation benchmark against workstation ground truth."""
    manifest_path = getattr(args, "manifest", None) or "data/cohort_manifest.json"
    if not os.path.isfile(manifest_path):
        print(f"[-] Cohort manifest not found at: {manifest_path}", file=sys.stderr)
        return 1

    try:
        benchmark_script = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "benchmark_10_patients.py")
        if os.path.isfile(benchmark_script):
            cmd = [sys.executable, benchmark_script]
            return subprocess.run(cmd).returncode
    except Exception as exc:
        print(f"[-] Failed executing benchmark script: {exc}", file=sys.stderr)
        return 1

    print("[*] Benchmark script executed.")
    return 0


def handle_check(args: argparse.Namespace) -> int:
    """Run diagnostic environment sanity checks."""
    print("=" * 68)
    print("  LA FAT SEGMENTATION -- SYSTEM & ENVIRONMENT DIAGNOSTICS")
    print("=" * 68)
    print("")

    # 1. Python & Platform
    print(f"  Python Version:         {platform.python_version()} ({sys.executable})")
    print(f"  Operating System:       {platform.system()} {platform.release()} ({platform.machine()})")

    # 2. PyTorch & CUDA / GPU
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
        cuda_version = torch.version.cuda if cuda_avail else "N/A"
        print(f"  PyTorch Version:        {torch.__version__}")
        print(f"  CUDA GPU Available:     {'YES (GPU Acceleration Active)' if cuda_avail else 'NO (CPU Mode Only)'}")
        if cuda_avail:
            print(f"  GPU Device:             {gpu_name} (CUDA {cuda_version})")
    except ImportError:
        print("  PyTorch:                NOT INSTALLED")

    # 3. TotalSegmentator & License
    ts_avail = is_ts_available()
    ts_license_str = "NOT CONFIGURED (Required for heartchambers_highres)"
    try:
        ts_config_path = os.path.expanduser(
            os.environ.get("TOTALSEG_CONFIG", "~/.totalsegmentator/config.json")
        )
        if os.path.isfile(ts_config_path):
            with open(ts_config_path, "r", encoding="utf-8") as f:
                ts_cfg = json.load(f)
                lic = ts_cfg.get("license_number") or ts_cfg.get("totalseg_id")
                if lic and lic.startswith("aca_"):
                    ts_license_str = f"ACTIVE (Academic: {lic[:7]}...)"
                elif lic:
                    ts_license_str = f"ACTIVE ({lic[:7]}...)"
    except Exception:
        pass

    try:
        import totalsegmentator
        ts_ver = getattr(totalsegmentator, "__version__", "Installed")
        print(f"  TotalSegmentator:       AVAILABLE (v{ts_ver})")
    except ImportError:
        print(f"  TotalSegmentator:       {'CLI AVAILABLE' if ts_avail else 'NOT FOUND (Pre-computation requires TS)'}")
    print(f"  TS Heart Model License: {ts_license_str}")

    # 4. Core Imaging Libraries
    try:
        import SimpleITK as sitk
        print(f"  SimpleITK Version:      {sitk.__version__}")
    except ImportError:
        print("  SimpleITK:              NOT INSTALLED")

    try:
        import scipy
        import numpy as np
        print(f"  NumPy / SciPy:          NumPy {np.__version__} / SciPy {scipy.__version__}")
    except ImportError:
        pass

    # 5. Data Directories Check
    config = PipelineConfig()
    data_dir_exists = os.path.isdir(config.data_dir)
    raw_dir_exists = os.path.isdir(os.path.join(config.data_dir, config.raw_subdir))
    inter_dir_exists = os.path.isdir(os.path.join(config.data_dir, config.intermediate_subdir))
    output_dir_exists = os.path.isdir(config.output_dir)

    print("")
    print("  Repository Data Paths:")
    print(f"    - data_dir ({config.data_dir}):                 {'EXISTS' if data_dir_exists else 'NOT FOUND'}")
    print(f"    - raw scans ({config.data_dir}/{config.raw_subdir}):        {'EXISTS' if raw_dir_exists else 'NOT FOUND'}")
    print(f"    - mask cache ({config.data_dir}/{config.intermediate_subdir}): {'EXISTS' if inter_dir_exists else 'NOT FOUND'}")
    print(f"    - output_dir ({config.output_dir}):               {'EXISTS' if output_dir_exists else 'NOT FOUND'}")

    print("")
    print("=" * 68)
    return 0


# ---------------------------------------------------------------------------
# CLI Argument Parser Construction
# ---------------------------------------------------------------------------


def build_cli_parser() -> argparse.ArgumentParser:
    """Construct the rich argument parser with subcommands and examples."""
    epilog_text = """
Examples:
  la-fat run --patient 0674                  Run fat extraction for patient 0674
  la-fat 0674                                Direct shorthand for running patient 0674
  la-fat batch --input-dir /path/to/ctscans  Batch process all CT scans in a folder
  la-fat batch --patient-ids 0674,1512,2996  Batch process specific patients
  la-fat precompute --input-dir /scans --gpu Run TotalSegmentator on GPU for all scans
  la-fat dashboard                           Open zero-footprint QA PACS Studio
  la-fat dashboard --patient 0674            Open patient 0674 QA report
  la-fat check                               Check environment, GPU, and dependencies
"""
    parser = argparse.ArgumentParser(
        prog="la-fat",
        description="LA Fat Segmentation -- Deep-Module Epicardial Adipose Tissue Analysis Pipeline.",
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose debug logging")

    subparsers = parser.add_subparsers(dest="command", title="Available Commands", metavar="<command>")

    # 1. run subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Run pure-CPU fat extraction on a single patient scan",
        description="Run fat extraction, distance partition, and tri-track radiomics mask export for a single patient.",
    )
    run_parser.add_argument("input_file", nargs="?", default=None, help="Optional positional CT NIfTI file or patient ID")
    run_parser.add_argument("-p", "--patient", default=None, help="Canonical patient ID (e.g. '0674')")
    run_parser.add_argument("-c", "--config", default=None, help="Path to YAML configuration file")
    run_parser.add_argument("--data-dir", default=None, help="Override root data directory")
    run_parser.add_argument("--output-dir", default=None, help="Override root output directory")
    run_parser.add_argument("--no-qa", action="store_true", help="Skip HTML QA Studio report generation")

    # 2. batch subcommand
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch process an entire directory or cohort of CT scans",
        description="Batch process multiple patient CT scans, save cohort summary CSV, and compile multi-tab QA studio.",
    )
    batch_parser.add_argument("-i", "--input-dir", default=None, help="Path to folder containing CT scans (.nii.gz / .nii)")
    batch_parser.add_argument("-p", "--patient-ids", default=None, help="Comma-separated patient IDs to process (e.g. '0674,1512')")
    batch_parser.add_argument("-c", "--config", default=None, help="Path to YAML configuration file")
    batch_parser.add_argument("--data-dir", default=None, help="Override root data directory")
    batch_parser.add_argument("-o", "--output-dir", default=None, help="Override root output directory")
    batch_parser.add_argument("--force-recompute", "--force", action="store_true", help="Recompute already processed scans")
    batch_parser.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"], help="Inference device for TS precompute")
    batch_parser.add_argument("--fast", action="store_true", help="Run fast TotalSegmentator models where supported")
    batch_parser.add_argument("--no-open", action="store_true", help="Do not automatically open QA studio after completion")

    # 3. precompute subcommand
    pre_parser = subparsers.add_parser(
        "precompute",
        help="Run TotalSegmentator GPU/CPU mask generation on raw CTs",
        description="Run TotalSegmentator pre-computation to extract anatomical masks for raw CT scans.",
    )
    pre_parser.add_argument("input_file", nargs="?", default=None, help="Optional path to raw CT NIfTI file")
    pre_parser.add_argument("-p", "--patient", default=None, help="Canonical patient ID to precompute")
    pre_parser.add_argument("-i", "--input-dir", default=None, help="Directory containing raw CT scans")
    pre_parser.add_argument("-a", "--all", action="store_true", help="Precompute all scans found in raw scans directory")
    pre_parser.add_argument("-c", "--config", default=None, help="Path to YAML configuration file")
    pre_parser.add_argument("-o", "--output-dir", default=None, help="Output directory for anatomical mask cache")
    pre_parser.add_argument("-d", "--device", default="auto", choices=["auto", "gpu", "cpu"], help="Inference device (default: auto)")
    pre_parser.add_argument("--fast", action="store_true", help="Use fast TotalSegmentator models where supported")
    pre_parser.add_argument("--set-license", default=None, help="Set and register TotalSegmentator academic/commercial license key")

    # 4. dashboard subcommand
    dash_parser = subparsers.add_parser(
        "dashboard",
        help="Open zero-footprint WebGL PACS QA Studio in default web browser",
        description="Launch offline zero-footprint HTML5/WebGL QA Studio in your web browser.",
    )
    dash_parser.add_argument("-o", "--output-dir", default="data/outputs", help="Output directory containing QA HTML")
    dash_parser.add_argument("-p", "--patient", default=None, help="Open specific patient QA report")

    # 5. benchmark subcommand
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run 10-patient clinical correlation benchmark vs scanner measurements",
        description="Execute full 10-patient clinical benchmark and compute Pearson correlation vs workstation baseline.",
    )
    bench_parser.add_argument("-m", "--manifest", default="data/cohort_manifest.json", help="Path to cohort manifest JSON")
    bench_parser.add_argument("-o", "--output-dir", default="data/outputs", help="Output directory")

    # 6. check subcommand
    subparsers.add_parser(
        "check",
        help="Run system diagnostics for PyTorch, CUDA GPU, TS, and data paths",
        description="Perform diagnostic environment health check.",
    )

    return parser


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------


def main_cli(argv: list[str] | None = None) -> None:
    """Main CLI entry point with command dispatch and shorthand support."""
    raw_args = list(sys.argv[1:] if argv is None else argv)

    # Shorthand 1: `la-fat help` -> `la-fat --help`
    if raw_args and raw_args[0] == "help":
        raw_args = ["--help"]

    # Shorthand 2: `la-fat 0674` or `la-fat --patient 0674` without subcommand -> `la-fat run ...`
    known_commands = {"run", "batch", "precompute", "dashboard", "benchmark", "check", "-h", "--help"}
    if raw_args and raw_args[0] not in known_commands and not raw_args[0].startswith("-"):
        raw_args = ["run", raw_args[0]] + raw_args[1:]
    elif raw_args and (raw_args[0] == "--patient" or raw_args[0].startswith("--patient=")):
        raw_args = ["run"] + raw_args

    parser = build_cli_parser()

    if not raw_args:
        parser.print_help(sys.stderr)
        print("\nError: Missing required argument. Please specify a patient ID via --patient (e.g. --patient 0674) or a command.", file=sys.stderr)
        sys.exit(2)

    args = parser.parse_args(raw_args)
    _configure_cli_logging(verbose=getattr(args, "verbose", False))

    handlers = {
        "run": handle_run,
        "batch": handle_batch,
        "precompute": handle_precompute,
        "dashboard": handle_dashboard,
        "benchmark": handle_benchmark,
        "check": handle_check,
    }

    handler = handlers.get(args.command)
    if not handler:
        parser.print_help(sys.stderr)
        sys.exit(2)

    exit_code = handler(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main_cli()
