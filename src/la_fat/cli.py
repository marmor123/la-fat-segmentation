"""Command-line interface for the LA Fat Segmentation pipeline.

Provides the ``main_cli`` entry point (``la-fat`` console script) and the
``_print_cli_summary`` / ``_configure_cli_logging`` helpers that were
previously inlined in ``pipeline.py``.
"""

from __future__ import annotations

import dataclasses
import logging
import sys

from la_fat.config import PipelineConfig
from la_fat.pipeline import PipelineResult, run_fat_extraction_pipeline

logger = logging.getLogger(__name__)


def main_cli() -> None:
    """Console-script entry point for ``la-fat``.

    Parses command-line arguments and runs
    :func:`~la_fat.pipeline.run_fat_extraction_pipeline`.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="LA Fat Segmentation Pipeline — "
        "extract epicardial adipose tissue from CT scans.",
    )
    parser.add_argument(
        "--patient",
        required=True,
        help="Patient identifier (e.g. '0674')",
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
    args = parser.parse_args()

    # Configure logging to stderr so stdout stays clean for the summary.
    _configure_cli_logging()

    # Build config, optionally overriding paths.
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
            # Frozen dataclass — replace with a new instance.
            config = dataclasses.replace(config, **kwargs)

    logger.info("Starting LA Fat extraction pipeline for patient %s", args.patient)
    result = run_fat_extraction_pipeline(
        patient_id=args.patient,
        config=config,
        config_path=None if config is not None else config_path,
    )

    # Print summary to stdout
    _print_cli_summary(result)

    sys.exit(0 if result.success else 1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _configure_cli_logging() -> None:
    """Configure logging for the CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stderr,
    )


def _print_cli_summary(result: PipelineResult) -> None:
    """Print a human-readable summary of pipeline results to stdout."""
    lines: list[str] = []
    add = lines.append

    add("=" * 56)
    add(f"  LA FAT SEGMENTATION PIPELINE — {result.patient_id}")
    add("=" * 56)
    add("")

    if result.success:
        add("  Status:   SUCCESS")
    else:
        add("  Status:   FAILED  ({0} error(s))".format(len(result.errors)))
    add("  Runtime:  {0:.1f} s".format(result.total_runtime_seconds))
    add("")

    if result.pericardium_result is not None:
        add("  Pericardium:  {0}".format(result.pericardium_result.method))
        if result.pericardium_result.fallback_triggered:
            add("    (fallback: {0})".format(
                result.pericardium_result.fallback_reason or "",
            ))

    if result.fat_threshold_result is not None:
        add("  Fat range:    [{0:.1f}, {1:.1f}] HU  ({2})".format(
            result.fat_threshold_result.hu_low,
            result.fat_threshold_result.hu_high,
            result.fat_threshold_result.method,
        ))

    if result.partition_result is not None:
        la_vol = result.partition_result.anchor_volumes_ml.get("LA", 0.0)
        total = result.partition_result.total_fat_volume_ml
        unassigned = result.partition_result.unassigned_volume_ml
        add("  LA Fat:       {0:.2f} ml  (total epicardial: {1:.2f} ml)".format(
            la_vol, total,
        ))
        add("  Unassigned:   {0:.2f} ml".format(unassigned))
        if result.partition_result.excluded_anchors:
            add("  Excluded:     {0}".format(
                ", ".join(result.partition_result.excluded_anchors),
            ))

    if result.cleanup_result is not None:
        add("  Cleanup:      {0} island(s) removed ({1:.1f} mm³)".format(
            result.cleanup_result.islands_removed,
            result.cleanup_result.total_removed_volume_mm3,
        ))

    if result.mesh_paths is not None:
        total_meshes = sum(len(files) for files in result.mesh_paths.values())
        add("  Meshes:       {0} saved ({1} step(s))".format(
            total_meshes, len(result.mesh_paths),
        ))
    else:
        add("  Meshes:       not extracted")

    if result.dashboard_output is not None:
        add("  Dashboard:    {0}".format(result.dashboard_output.output_dir))

    if result.quality_flags:
        add("")
        add("  Quality Flags:")
        for flag in result.quality_flags:
            add("    [{0}] {1}: {2}".format(
                flag.severity, flag.concern, flag.detail,
            ))

    if result.errors:
        add("")
        add("  Errors:")
        for err in result.errors:
            add("    - {0}".format(err))

    if result.warnings:
        add("")
        add("  Warnings:")
        for warn in result.warnings:
            add("    - {0}".format(warn))

    add("")
    add("=" * 56)

    sys.stdout.write("\n".join(lines) + "\n")
