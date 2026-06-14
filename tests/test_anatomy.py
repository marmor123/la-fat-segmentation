"""Tests for the la_fat.anatomy module.

Validates the single source of truth for Partition Anchor constants
and the voxel_volume_ml utility function.
"""

from __future__ import annotations

import numpy as np
import pytest

from la_fat.anatomy import (
    ANCHOR_COLORS,
    ANCHOR_LABELS,
    ANCHOR_ORDINALS,
    CANONICAL_ANCHORS,
    LA_FAT_COLOR_3D,
    PERICARDIUM_COLOR,
    TS_STRUCTURE_NAMES,
    voxel_volume_ml,
)


# ===================================================================
# 1. CANONICAL_ANCHORS
# ===================================================================


class TestCanonicalAnchors:
    """CANONICAL_ANCHORS must have exactly 6 entries in a fixed order."""

    def test_has_six_entries(self):
        assert len(CANONICAL_ANCHORS) == 6

    def test_order_is_la_lv_ra_rv_aorta_pa(self):
        expected = ["LA", "LV", "RA", "RV", "Aorta", "Pulmonary_Artery"]
        assert list(CANONICAL_ANCHORS) == expected

    def test_all_entries_are_strings(self):
        for anchor in CANONICAL_ANCHORS:
            assert isinstance(anchor, str)

    def test_no_duplicates(self):
        assert len(set(CANONICAL_ANCHORS)) == len(CANONICAL_ANCHORS)


# ===================================================================
# 2. ANCHOR_ORDINALS
# ===================================================================


class TestAnchorOrdinals:
    """ANCHOR_ORDINALS maps each anchor to its 0-based index."""

    def test_ordinals_are_sequential(self):
        expected = {
            "LA": 0,
            "LV": 1,
            "RA": 2,
            "RV": 3,
            "Aorta": 4,
            "Pulmonary_Artery": 5,
        }
        assert ANCHOR_ORDINALS == expected

    def test_every_anchor_has_ordinal(self):
        for anchor in CANONICAL_ANCHORS:
            assert anchor in ANCHOR_ORDINALS

    def test_ordinal_values_are_ints_0_to_5(self):
        for anchor, ordinal in ANCHOR_ORDINALS.items():
            assert isinstance(ordinal, int)
            assert 0 <= ordinal <= 5


# ===================================================================
# 3. ANCHOR_LABELS
# ===================================================================


class TestAnchorLabels:
    """ANCHOR_LABELS provides human-readable display names."""

    def test_all_anchors_have_labels(self):
        for anchor in CANONICAL_ANCHORS:
            assert anchor in ANCHOR_LABELS

    def test_labels_are_readable_strings(self):
        assert ANCHOR_LABELS["LA"] == "Left Atrium"
        assert ANCHOR_LABELS["LV"] == "Left Ventricle"
        assert ANCHOR_LABELS["RA"] == "Right Atrium"
        assert ANCHOR_LABELS["RV"] == "Right Ventricle"
        assert ANCHOR_LABELS["Aorta"] == "Aorta"
        assert ANCHOR_LABELS["Pulmonary_Artery"] == "Pulmonary Artery"

    def test_no_duplicate_labels(self):
        assert len(set(ANCHOR_LABELS.values())) == len(ANCHOR_LABELS)


# ===================================================================
# 4. ANCHOR_COLORS
# ===================================================================


class TestAnchorColors:
    """ANCHOR_COLORS provides RGB colour tuples for rendering."""

    def test_all_anchors_have_colors(self):
        for anchor in CANONICAL_ANCHORS:
            assert anchor in ANCHOR_COLORS

    def test_colors_are_rgb_tuples(self):
        for anchor, color in ANCHOR_COLORS.items():
            assert isinstance(color, tuple)
            assert len(color) == 3
            for channel in color:
                assert 0.0 <= channel <= 1.0

    def test_known_colors(self):
        assert ANCHOR_COLORS["LA"] == (1.0, 0.0, 0.0)  # red
        assert ANCHOR_COLORS["LV"] == (0.0, 0.0, 1.0)  # blue
        assert ANCHOR_COLORS["RA"] == (0.0, 0.8, 0.0)  # green
        assert ANCHOR_COLORS["RV"] == (1.0, 0.65, 0.0)  # orange
        assert ANCHOR_COLORS["Aorta"] == (1.0, 1.0, 0.0)  # yellow
        assert ANCHOR_COLORS["Pulmonary_Artery"] == (0.6, 0.0, 0.6)  # purple


# ===================================================================
# 5. TS_STRUCTURE_NAMES
# ===================================================================


class TestTsStructureNames:
    """TS_STRUCTURE_NAMES maps anchors to TS v2 output folder names."""

    def test_all_anchors_have_ts_names(self):
        for anchor in CANONICAL_ANCHORS:
            assert anchor in TS_STRUCTURE_NAMES

    def test_known_ts_names(self):
        assert TS_STRUCTURE_NAMES["LA"] == "heart_atrium_left"
        assert TS_STRUCTURE_NAMES["LV"] == "heart_ventricle_left"
        assert TS_STRUCTURE_NAMES["RA"] == "heart_atrium_right"
        assert TS_STRUCTURE_NAMES["RV"] == "heart_ventricle_right"
        assert TS_STRUCTURE_NAMES["Aorta"] == "aorta"
        assert TS_STRUCTURE_NAMES["Pulmonary_Artery"] == "pulmonary_artery"


# ===================================================================
# 6. Voxel volume utility
# ===================================================================


class TestVoxelVolumeMl:
    """voxel_volume_ml computes volume in ml for a given spacing."""

    def test_isotropic_spacing_1_5(self):
        # (1.5 * 1.5 * 1.5) / 1000 = 0.003375
        vol = voxel_volume_ml((1.5, 1.5, 1.5))
        assert abs(vol - 0.003375) < 1e-10

    def test_isotropic_spacing_1_0(self):
        # (1.0 * 1.0 * 1.0) / 1000 = 0.001
        vol = voxel_volume_ml((1.0, 1.0, 1.0))
        assert abs(vol - 0.001) < 1e-10

    def test_non_isotropic_spacing(self):
        # (0.5 * 0.5 * 1.0) / 1000 = 0.00025
        vol = voxel_volume_ml((0.5, 0.5, 1.0))
        assert abs(vol - 0.00025) < 1e-10

    def test_returns_float(self):
        vol = voxel_volume_ml((1.5, 1.5, 1.5))
        assert isinstance(vol, float)

    def test_raises_on_empty_tuple(self):
        with pytest.raises((IndexError, TypeError)):
            voxel_volume_ml(())

    def test_large_spacing(self):
        # (5.0 * 5.0 * 5.0) / 1000 = 0.125
        vol = voxel_volume_ml((5.0, 5.0, 5.0))
        assert abs(vol - 0.125) < 1e-10


# ===================================================================
# 7. Shared color constants
# ===================================================================


class TestSharedColors:
    """PERICARDIUM_COLOR and LA_FAT_COLOR_3D are available."""

    def test_pericardium_color_present(self):
        assert isinstance(PERICARDIUM_COLOR, tuple)
        assert len(PERICARDIUM_COLOR) == 3
        assert PERICARDIUM_COLOR == (0.0, 0.75, 0.75)  # cyan

    def test_la_fat_color_3d_present(self):
        assert isinstance(LA_FAT_COLOR_3D, tuple)
        assert len(LA_FAT_COLOR_3D) == 3
        assert LA_FAT_COLOR_3D == (1.0, 0.84, 0.0)  # gold
