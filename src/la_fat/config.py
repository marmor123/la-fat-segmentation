"""Configuration module for the LA Fat Segmentation pipeline.

Provides a dataclass-based configuration loaded from YAML with
type validation and default values.
"""

from __future__ import annotations

import dataclasses
import os
import typing as t

import yaml

# Type map for validation: field_name -> expected type
_FIELD_TYPES: dict[str, type] = {
    "spacing_mm": float,
    "hu_fallback_low": float,
    "hu_fallback_high": float,
    "gaussian_sigma_multiplier": float,
    "min_pericardium_volume_ml": float,
    "pericardium_dilation_mm": float,
    "min_anchor_volume_ml": float,
    "min_fat_island_volume_mm3": float,
    "min_sub_zero_voxels_for_fit": int,
    "la_fat_volume_low_ml": float,
    "la_fat_volume_high_ml": float,
    "max_unassigned_fat_pct": float,
    "max_gaussian_sigma": float,
    "max_lv_la_ratio": float,
    "min_fat_fraction_pct": float,
    "data_dir": str,
    "output_dir": str,
    "intermediate_subdir": str,
    "raw_subdir": str,
}


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    """Immutable configuration for the LA Fat Segmentation pipeline.

    All parameters have sensible defaults.  Use ``from_yaml`` to load
    an overlay from a YAML file; missing keys in the file retain the
    defaults.
    """

    # --- Resampling -----------------------------------------------------------
    spacing_mm: float = 1.5

    # --- HU / fat-threshold --------------------------------------------------
    hu_fallback_low: float = -190.0
    hu_fallback_high: float = -30.0
    gaussian_sigma_multiplier: float = 2.0

    # --- Pericardium ---------------------------------------------------------
    min_pericardium_volume_ml: float = 50.0
    pericardium_dilation_mm: float = 5.0

    # --- Anatomical anchors --------------------------------------------------
    min_anchor_volume_ml: float = 5.0

    # --- Fat cleanup / filtering ---------------------------------------------
    min_fat_island_volume_mm3: float = 100.0
    min_sub_zero_voxels_for_fit: int = 1000

    # --- Quality flags thresholds --------------------------------------------
    la_fat_volume_low_ml: float = 2.0
    la_fat_volume_high_ml: float = 150.0
    max_unassigned_fat_pct: float = 80.0
    max_gaussian_sigma: float = 100.0
    max_lv_la_ratio: float = 4.0
    min_fat_fraction_pct: float = 8.0

    # --- File-system paths ---------------------------------------------------
    data_dir: str = "data"
    output_dir: str = "outputs"
    intermediate_subdir: str = "intermediate"
    raw_subdir: str = "raw"

    # ── Public API ───────────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from a YAML file.

        Parameters
        ----------
        path:
            Filesystem path to a ``.yaml`` file.

        Returns
        -------
        PipelineConfig
            A frozen config instance with values merged from the YAML
            on top of built-in defaults.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        yaml.YAMLError
            If the file cannot be parsed as YAML.
        TypeError
            If a field in the YAML has an unexpected type.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as fh:
            try:
                raw: dict = yaml.safe_load(fh) or {}
            except yaml.YAMLError as exc:
                raise yaml.YAMLError(
                    f"Failed to parse YAML configuration: {path}"
                ) from exc

        # Build kwargs from defaults, then overlay YAML values.
        kwargs = {}
        for field in dataclasses.fields(cls):
            if field.name in raw:
                value = raw[field.name]
                expected = _FIELD_TYPES.get(field.name)
                if expected is not None and not isinstance(value, expected):
                    raise TypeError(
                        f"Field '{field.name}' expected type {expected.__name__}, "
                        f"got {type(value).__name__} (value: {value!r})"
                    )
                kwargs[field.name] = value
            # Omitted fields use the dataclass default.

        return cls(**kwargs)
