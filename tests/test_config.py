"""Tests for the la_fat.config module (PipelineConfig)."""

import os
import tempfile
import yaml

import pytest

from la_fat.config import PipelineConfig


class TestPipelineConfigDefaults:
    """Default instantiation should produce the documented defaults."""

    def test_default_spacing(self):
        cfg = PipelineConfig()
        assert cfg.spacing_mm == 1.5

    def test_default_fat_hu_low(self):
        cfg = PipelineConfig()
        assert cfg.fat_hu_low == -190.0

    def test_default_fat_hu_high(self):
        cfg = PipelineConfig()
        assert cfg.fat_hu_high == -30.0

    def test_default_min_pericardium_volume_ml(self):
        cfg = PipelineConfig()
        assert cfg.min_pericardium_volume_ml == 50.0

    def test_default_pericardium_dilation_mm(self):
        cfg = PipelineConfig()
        assert cfg.pericardium_dilation_mm == 5.0

    def test_default_min_anchor_volume_ml(self):
        cfg = PipelineConfig()
        assert cfg.min_anchor_volume_ml == 5.0

    def test_default_min_fat_island_volume_mm3(self):
        cfg = PipelineConfig()
        assert cfg.min_fat_island_volume_mm3 == 100.0

    def test_default_la_fat_volume_low_ml(self):
        cfg = PipelineConfig()
        assert cfg.la_fat_volume_low_ml == 2.0

    def test_default_la_fat_volume_high_ml(self):
        cfg = PipelineConfig()
        assert cfg.la_fat_volume_high_ml == 150.0

    def test_default_max_unassigned_fat_pct(self):
        cfg = PipelineConfig()
        assert cfg.max_unassigned_fat_pct == 80.0

    def test_default_data_dir(self):
        cfg = PipelineConfig()
        assert cfg.data_dir == "data"

    def test_default_output_dir(self):
        cfg = PipelineConfig()
        assert cfg.output_dir == "outputs"

    def test_default_intermediate_subdir(self):
        cfg = PipelineConfig()
        assert cfg.intermediate_subdir == "intermediate"

    def test_default_raw_subdir(self):
        cfg = PipelineConfig()
        assert cfg.raw_subdir == "raw"

    def test_default_max_lv_la_ratio(self):
        cfg = PipelineConfig()
        assert cfg.max_lv_la_ratio == 4.0

    def test_default_min_fat_fraction_pct(self):
        cfg = PipelineConfig()
        assert cfg.min_fat_fraction_pct == 8.0


class TestPipelineConfigFromYaml:
    """Tests for the from_yaml class method."""

    def test_load_valid_yaml(self):
        """Loading a valid YAML file should return a PipelineConfig with those values."""
        data = {
            "spacing_mm": 2.0,
            "fat_hu_low": -200.0,
            "fat_hu_high": -40.0,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = PipelineConfig.from_yaml(path)
            assert cfg.spacing_mm == 2.0
            assert cfg.fat_hu_low == -200.0
            assert cfg.fat_hu_high == -40.0
        finally:
            os.unlink(path)

    def test_missing_file_error(self):
        """Loading a non-existent YAML file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PipelineConfig.from_yaml("/nonexistent/path/config.yaml")

    def test_malformed_yaml_error(self):
        """Malformed YAML content should raise a yaml.YAMLError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("{invalid: yaml: unbalanced")
            path = f.name
        try:
            with pytest.raises(yaml.YAMLError):
                PipelineConfig.from_yaml(path)
        finally:
            os.unlink(path)

    def test_defaults_applied_for_missing_keys(self):
        """Keys not present in the YAML should fall back to defaults."""
        data = {"spacing_mm": 1.0}  # only one key provided
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            cfg = PipelineConfig.from_yaml(path)
            # Provided value
            assert cfg.spacing_mm == 1.0
            # Defaults for everything else
            assert cfg.fat_hu_low == -190.0
            assert cfg.min_pericardium_volume_ml == 50.0
            assert cfg.data_dir == "data"
        finally:
            os.unlink(path)

    def test_type_validation_invalid_type(self):
        """If a YAML value has the wrong type, from_yaml should raise TypeError."""
        data = {"spacing_mm": "not_a_number"}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(data, f)
            path = f.name
        try:
            with pytest.raises(TypeError):
                PipelineConfig.from_yaml(path)
        finally:
            os.unlink(path)

    def test_empty_yaml_file(self):
        """An empty YAML file should result in all defaults."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            path = f.name
        try:
            cfg = PipelineConfig.from_yaml(path)
            assert cfg.spacing_mm == 1.5
            assert cfg.fat_hu_low == -190.0
        finally:
            os.unlink(path)
