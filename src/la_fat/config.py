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
_FIELD_TYPES: dict[str, t.Any] = {
    "spacing_mm": (float, int),
    "fat_hu_low": (float, int),
    "fat_hu_high": (float, int),
    "fat_clamping_max_hu": (float, int),
    "fat_sigma_multiplier": (float, int),
    "fat_smoothing_sigma_hu": (float, int),
    "min_fat_voxels": int,
    "fat_peak_prominence_ratio": (float, int),
    "fat_wide_sigma_warn_hu": (float, int),
    "min_pericardium_volume_ml": (float, int),
    "pericardium_dilation_mm": (float, int),
    "min_anchor_volume_ml": (float, int),
    "min_fat_island_volume_mm3": (float, int),
    "la_fat_volume_low_ml": (float, int),
    "la_fat_volume_high_ml": (float, int),
    "max_unassigned_fat_pct": (float, int),
    "max_lv_la_ratio": (float, int),
    "min_fat_fraction_pct": (float, int),
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
    fat_hu_low: float = -190.0
    fat_hu_high: float = -30.0
    fat_clamping_max_hu: float = 0.0
    fat_sigma_multiplier: float = 2.0
    fat_smoothing_sigma_hu: float = 2.5
    min_fat_voxels: int = 500
    fat_peak_prominence_ratio: float = 0.003
    fat_wide_sigma_warn_hu: float = 25.0

    # --- Pericardium ---------------------------------------------------------
    min_pericardium_volume_ml: float = 50.0
    pericardium_dilation_mm: float = 5.0

    # --- Anatomical anchors --------------------------------------------------
    min_anchor_volume_ml: float = 5.0

    # --- Fat cleanup / filtering ---------------------------------------------
    min_fat_island_volume_mm3: float = 100.0

    # --- Quality flags thresholds --------------------------------------------
    la_fat_volume_low_ml: float = 2.0
    la_fat_volume_high_ml: float = 150.0
    max_unassigned_fat_pct: float = 80.0
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
                    exp_name = (
                        expected.__name__
                        if hasattr(expected, "__name__")
                        else str(expected)
                    )
                    raise TypeError(
                        f"Field '{field.name}' expected type {exp_name}, "
                        f"got {type(value).__name__} (value: {value!r})"
                    )
                kwargs[field.name] = value
            # Omitted fields use the dataclass default.

        return cls(**kwargs)
