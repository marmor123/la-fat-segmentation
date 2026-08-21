"""LA Fat Segmentation — GPU-accelerated epicardial adipose tissue analysis."""

from la_fat.cleanup import CleanupConfig, CleanupResult, cleanup_la_fat_mask
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
from la_fat.pipeline_types import (
    PipelineArtifacts,
    QualityFlag,
    QualitySeverity,
    SurfaceSpec,
    ViewportPreset,
)
from la_fat.pipeline_result import (
    PipelineResultData,
    load_pipeline_result,
    save_pipeline_result,
)
from la_fat.image_ops import (
    GridGeometry,
    ResampleResult,
    apply_grid_geometry,
    get_grid_geometry,
    resample_to_isotropic,
    resample_to_reference,
)
from la_fat.qa_dashboard import DashboardOutput, generate_dashboard
from la_fat.quality_flagger import generate_quality_flags
from la_fat.thresholding import (
    ThresholdConfig,
    ThresholdResult,
    compute_fat_threshold,
    create_fat_mask,
    fit_trimmed_gaussian,
)
from la_fat.ts_runner import (
    TsPrecomputeResult,
    is_ts_available,
    resolve_ts_mask_path,
    run_ts_precompute,
)

__all__ = [
    "CleanupConfig",
    "CleanupResult",
    "DashboardOutput",
    "GridGeometry",
    "PatientSummary",
    "PipelineArtifacts",
    "PipelineConfig",
    "PipelineResult",
    "PipelineResultData",
    "PartitionResult",
    "QualityFlag",
    "QualitySeverity",
    "ResampleResult",
    "SurfaceSpec",
    "ThresholdConfig",
    "ThresholdResult",
    "TsPrecomputeResult",
    "ViewportPreset",
    "apply_grid_geometry",
    "cleanup_la_fat_mask",
    "compute_fat_threshold",
    "create_dashboard",
    "create_fat_mask",
    "discover_patients",
    "extract_interactive_meshes",
    "extract_meshes_for_step",
    "fit_trimmed_gaussian",
    "generate_dashboard",
    "generate_quality_flags",
    "get_grid_geometry",
    "is_ts_available",
    "load_pipeline_result",
    "partition_fat",
    "resample_to_isotropic",
    "resample_to_reference",
    "resolve_ts_mask_path",
    "run_fat_extraction_pipeline",
    "run_ts_precompute",
    "save_pipeline_result",
]
