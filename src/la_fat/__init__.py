"""LA Fat Segmentation — GPU-accelerated epicardial adipose tissue analysis."""

from la_fat.cleanup import CleanupResult, cleanup_la_fat_mask
from la_fat.config import PipelineConfig
from la_fat.interactive_dashboard import (
    PatientSummary,
    create_dashboard,
    discover_patients,
)
from la_fat.mesh_extractor import (
    extract_interactive_meshes,
    extract_meshes_for_step,
)
from la_fat.partition_engine import PartitionResult, partition_fat
from la_fat.pipeline import PipelineResult, run_fat_extraction_pipeline
from la_fat.preprocessor import ResampleResult, resample_to_isotropic
from la_fat.qa_dashboard import DashboardOutput, generate_dashboard
from la_fat.quality_flagger import QualityFlag, generate_quality_flags
from la_fat.ts_runner import (
    TsPrecomputeResult,
    is_ts_available,
    resolve_ts_mask_path,
    run_ts_precompute,
)

__all__ = [
    "CleanupResult",
    "DashboardOutput",
    "PatientSummary",
    "PipelineConfig",
    "PipelineResult",
    "PartitionResult",
    "QualityFlag",
    "ResampleResult",
    "TsPrecomputeResult",
    "cleanup_la_fat_mask",
    "create_dashboard",
    "discover_patients",
    "extract_interactive_meshes",
    "extract_meshes_for_step",
    "generate_dashboard",
    "generate_quality_flags",
    "is_ts_available",
    "partition_fat",
    "resample_to_isotropic",
    "resolve_ts_mask_path",
    "run_fat_extraction_pipeline",
    "run_ts_precompute",
]
