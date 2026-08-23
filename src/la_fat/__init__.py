"""LA Fat Segmentation — Deep Modules for Epicardial Adipose Tissue Analysis."""

from la_fat.anatomy import (
    ANCHOR_COLORS,
    ANCHOR_LABELS,
    ANCHOR_ORDINALS,
    CANONICAL_ANCHORS,
    PERICARDIUM_COLOR,
    LA_FAT_COLOR_3D,
    voxel_volume_ml,
)
from la_fat.cleanup import CleanupConfig, CleanupResult, cleanup_la_fat_mask
from la_fat.cohort_qa_generator import (
    extract_patient_qa_record,
    generate_cohort_qa_html,
)
from la_fat.config import PipelineConfig
from la_fat.image_ops import (
    GridGeometry,
    ResampleResult,
    apply_grid_geometry,
    get_grid_geometry,
    resample_to_isotropic,
    resample_to_reference,
)
from la_fat.partition_engine import (
    PartitionConfig,
    PartitionMetrics,
    PartitionResult,
    partition_fat,
)
from la_fat.pericardium_resolver import PericardiumResult, resolve_pericardium
from la_fat.pipeline import (
    PipelineResult,
    SegmentationResult,
    load_and_resample_masks,
    run_fat_extraction,
    run_fat_extraction_pipeline,
)
from la_fat.quality_flagger import QualityFlag, QualitySeverity, generate_quality_flags
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
    "ANCHOR_COLORS",
    "ANCHOR_LABELS",
    "ANCHOR_ORDINALS",
    "CANONICAL_ANCHORS",
    "CleanupConfig",
    "CleanupResult",
    "GridGeometry",
    "LA_FAT_COLOR_3D",
    "PERICARDIUM_COLOR",
    "PartitionConfig",
    "PartitionMetrics",
    "PartitionResult",
    "PericardiumResult",
    "PipelineConfig",
    "PipelineResult",
    "QualityFlag",
    "QualitySeverity",
    "ResampleResult",
    "SegmentationResult",
    "ThresholdConfig",
    "ThresholdResult",
    "TsPrecomputeResult",
    "apply_grid_geometry",
    "cleanup_la_fat_mask",
    "compute_fat_threshold",
    "create_fat_mask",
    "extract_patient_qa_record",
    "fit_trimmed_gaussian",
    "generate_cohort_qa_html",
    "generate_quality_flags",
    "get_grid_geometry",
    "is_ts_available",
    "load_and_resample_masks",
    "partition_fat",
    "resample_to_isotropic",
    "resample_to_reference",
    "resolve_pericardium",
    "resolve_ts_mask_path",
    "run_fat_extraction",
    "run_fat_extraction_pipeline",
    "run_ts_precompute",
    "voxel_volume_ml",
]
