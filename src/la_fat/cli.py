"""Command-line interface for the LA Fat Segmentation pipeline.

Provides the ``main_cli`` entry point (``la-fat`` console script) and the
``_print_cli_summary`` / ``_configure_cli_logging`` helpers.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys

from la_fat.config import PipelineConfig
from la_fat.pipeline import SegmentationResult, run_fat_extraction

logger = logging.getLogger(__name__)


def main_cli() -> None:
    """Console-script entry point for ``la-fat``."""
    parser = argparse.ArgumentParser(
        description="LA Fat Segmentation Pipeline — "
        "extract epicardial adipose tissue from Cardiac CT scans.",
    )
    parser.add_argument(
        "--patient",
        required=True,
        help="Canonical patient identifier (e.g. '0674')",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Override data directory (default: value in config)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: value in config)",
    )
    parser.add_argument(
        "--no-qa",
        action="store_true",
        help="Skip QA HTML report generation",
    )
    args = parser.parse_args()

    _configure_cli_logging()

    config: PipelineConfig | None = None
    config_path: str | None = args.config

    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    if args.data_dir is not None or args.output_dir is not None:
        if config is None:
            config = PipelineConfig()
        kwargs: dict[str, str] = {}
        if args.data_dir is not None:
            kwargs["data_dir"] = args.data_dir
        if args.output_dir is not None:
            kwargs["output_dir"] = args.output_dir
        if kwargs:
            config = dataclasses.replace(config, **kwargs)

    logger.info("Starting LA Fat extraction pipeline for patient %s", args.patient)
    result = run_fat_extraction(
        patient_id=args.patient,
        config=config,
        config_path=None if config is not None else config_path,
        generate_qa=not args.no_qa,
    )

    _print_cli_summary(result)
    sys.exit(0 if result.success else 1)


def _configure_cli_logging() -> None:
    """Configure logging for the CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )


def _print_cli_summary(result: SegmentationResult) -> None:
    """Print a human-readable summary of pipeline results to stdout."""
    lines: list[str] = []
    add = lines.append

    add("=" * 64)
    add(f"  LA FAT SEGMENTATION PIPELINE — {result.patient_id}")
    add("=" * 64)
    add("")

    if result.success:
        add("  Status:       SUCCESS")
    else:
        add(f"  Status:       FAILED  ({len(result.errors)} error(s))")
    add(f"  Runtime:      {result.total_runtime_seconds:.1f} s")
    add("")

    if result.success:
        add(f"  LA Fat (Adaptive):      {result.la_fat_volume_adaptive_ml:.2f} mL (window: {result.fat_hu_range_adaptive[0]:.1f} to {result.fat_hu_range_adaptive[1]:.1f} HU)")
        add(f"  LA Fat (Conservative):  {result.la_fat_volume_conservative_ml:.2f} mL (window: [-190, -30] HU)")
        add(f"  Total Epicardial Fat:   {result.total_eat_volume_adaptive_ml:.2f} mL")
        add(f"  Pericardium Volume:     {result.pericardium_volume_ml:.2f} mL")
        add(f"  Unassigned Fat:         {result.unassigned_volume_ml:.2f} mL ({result.unassigned_fat_pct:.1f}%)")

        if result.islands_removed > 0:
            add(f"  Island Cleanup:         {result.islands_removed} island(s) removed ({result.total_removed_volume_mm3:.1f} mm³)")

        if result.mask_1_5mm_path:
            add(f"  1.5mm Mask:             {result.mask_1_5mm_path}")
        if result.mask_native_path:
            add(f"  Native Grid Mask:       {result.mask_native_path}")
        if result.qa_report_path:
            add(f"  QA Studio HTML:         {result.qa_report_path}")

    if result.quality_flags:
        add("")
        add("  Quality Audit Flags:")
        for flag in result.quality_flags:
            c = flag.concern or getattr(flag, "flag_id", "")
            d = flag.detail or getattr(flag, "message", "")
            add(f"    [{flag.severity.upper()}] {c}: {d}")

    if result.errors:
        add("")
        add("  Errors:")
        for err in result.errors:
            add(f"    - {err}")

    if result.warnings:
        add("")
        add("  Warnings:")
        for warn in result.warnings:
            add(f"    - {warn}")

    add("")
    add("=" * 64)

    sys.stdout.write("\n".join(lines) + "\n")
