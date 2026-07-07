"""Tests for the la_fat.qa_dashboard module.

Exercises the generation of the QA dashboard including slice gallery,
fat overlay, numeric summary, and combined HTML.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest
from PIL import Image

from la_fat.config import PipelineConfig
from la_fat.cleanup import CleanupResult
from la_fat.partition_engine import PartitionResult
from la_fat.pericardium_resolver import PericardiumResult
from la_fat.quality_flagger import QualityFlag
from la_fat.qa_dashboard import DashboardOutput, generate_dashboard

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

SHAPE = (32, 32, 32)
SPACING = (1.5, 1.5, 1.5)
CFG = PipelineConfig()
PATIENT_ID = "TEST001"
_VOXEL_VOL_ML = SPACING[0] * SPACING[1] * SPACING[2] / 1000.0

# Canonical order matches partition_engine._CANONICAL_ANCHORS
_CANONICAL_ANCHORS: list[str] = [
    "LA",
    "LV",
    "RA",
    "RV",
    "Aorta",
    "Pulmonary_Artery",
]
_LABELS: dict[str, int] = {
    "LA": 1,
    "LV": 2,
    "RA": 3,
    "RV": 4,
    "Aorta": 5,
    "Pulmonary_Artery": 6,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sphere(
    shape: tuple[int, ...],
    centre: tuple[float, float, float],
    radius: float,
) -> np.ndarray:
    """Return a binary sphere mask (bool)."""
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    dist = np.sqrt(
        (z - centre[0]) ** 2
        + (y - centre[1]) ** 2
        + (x - centre[2]) ** 2
    )
    return dist <= radius


def _build_shared_data(output_dir: str) -> dict:
    """Build the standard synthetic pipeline results dict.

    Separated from the fixture so both ``tmp_path`` (per-function) and
    ``tmp_path_factory`` (per-class/session) can use it.
    """
    ct_array = np.zeros(SHAPE, dtype=np.float32)
    ct_array[6:26, 6:26, 6:26] = 50.0
    ct_array[8:24, 8:24, 8:24] = -80.0

    anchor_masks: dict[str, np.ndarray] = {}
    positions = {
        "LA": (12, 16, 16),
        "LV": (20, 16, 16),
        "RA": (12, 16, 12),
        "RV": (20, 16, 12),
        "Aorta": (16, 16, 22),
        "Pulmonary_Artery": (16, 16, 10),
    }
    for name, pos in positions.items():
        anchor_masks[name] = _make_sphere(SHAPE, pos, 5)

    pericardium_mask = _make_sphere(SHAPE, (16, 16, 16), 13)

    pericardium_result = PericardiumResult(
        mask=pericardium_mask,
        fallback_triggered=False,
        fallback_reason=None,
        method="ts_direct",
    )

    from scipy.ndimage import binary_dilation

    in_hu = (ct_array >= CFG.fat_hu_low) & (
        ct_array <= CFG.fat_hu_high
    )
    fat_mask = pericardium_mask & in_hu

    anchor_assignments = np.zeros(SHAPE, dtype=np.int32)

    for name, lbl in _LABELS.items():
        mask = anchor_masks[name].astype(bool)
        dilated = binary_dilation(mask, iterations=2)
        region = dilated & fat_mask & (anchor_assignments == 0)
        anchor_assignments[region] = lbl

    remaining = fat_mask & (anchor_assignments == 0)
    anchor_assignments[remaining] = 1

    total_fat_voxels = int(np.count_nonzero(fat_mask))

    anchor_volumes_ml: dict[str, float] = {}
    anchor_shares: dict[str, float] = {}
    for name, lbl in _LABELS.items():
        nv = int(np.count_nonzero(anchor_assignments == lbl))
        vol = nv * _VOXEL_VOL_ML
        anchor_volumes_ml[name] = vol
        anchor_shares[name] = (
            (nv / total_fat_voxels * 100.0) if total_fat_voxels > 0 else 0.0
        )

    la_fat_mask = anchor_assignments == 1

    cleanup_result = CleanupResult(
        cleaned_mask=la_fat_mask.copy(),
        islands_removed=1,
        island_volumes_mm3=[10.0],
        total_removed_volume_mm3=10.0,
        morphological_opening_applied=False,
        vessel_filling_applied=False,
    )

    quality_flags = [
        QualityFlag(
            severity="low",
            concern="islands_cleaned",
            detail="1 small island(s) removed (total volume: 10.0 mm3)",
            threshold_value=None,
            actual_value=1.0,
        ),
    ]

    partition_result = PartitionResult(
        la_fat_mask=la_fat_mask,
        all_fat_mask=fat_mask,
        anchor_assignments=anchor_assignments,
        anchor_volumes_ml=anchor_volumes_ml,
        anchor_shares=anchor_shares,
        unassigned_volume_ml=0.0,
        total_fat_volume_ml=total_fat_voxels * _VOXEL_VOL_ML,
        excluded_anchors=[],
        exclusion_reasons={},
    )

    return {
        "ct_array": ct_array,
        "anchor_masks": anchor_masks,
        "pericardium_result": pericardium_result,
        "partition_result": partition_result,
        "cleanup_result": cleanup_result,
        "quality_flags": quality_flags,
        "config": CFG,
        "patient_id": PATIENT_ID,
        "output_dir": output_dir,
        "spacing": SPACING,
    }


# ---------------------------------------------------------------------------
# Session-scoped fixtures (dashboard generated once for all shared tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def shared_dashboard(tmp_path_factory) -> DashboardOutput:
    """Generate a single dashboard shared by all non-mutating test classes."""
    d = tmp_path_factory.mktemp("dashboard_shared")
    data = _build_shared_data(str(d))
    return generate_dashboard(**data)


# ===================================================================
# 1. Dashboard output directory created
# ===================================================================


class TestOutputDirectory:
    """generate_dashboard creates the output directory."""

    def test_directory_created(self, shared_dashboard):
        assert os.path.isdir(shared_dashboard.output_dir)


# ===================================================================
# 2. All output files exist
# ===================================================================


class TestAllOutputFilesExist:
    """All expected output files are created."""

    def test_all_five_files_exist(self, shared_dashboard):
        paths = [
            shared_dashboard.slice_gallery_path,
            shared_dashboard.fat_overlay_path,
            shared_dashboard.summary_table_path,
            shared_dashboard.summary_html_path,
        ]
        for path in paths:
            assert os.path.isfile(path), f"File not found: {path}"

    def test_csv_file_exists(self, shared_dashboard):
        csv_dir = os.path.dirname(shared_dashboard.summary_table_path)
        csv_path = os.path.join(csv_dir, "summary.csv")
        assert os.path.isfile(csv_path)


# ===================================================================
# 3. Slice gallery image — non-blank
# ===================================================================


class TestSliceGallery:
    """Slice gallery image has expected properties."""

    def test_image_not_blank(self, shared_dashboard):
        img = Image.open(shared_dashboard.slice_gallery_path)
        assert img.size[0] > 0 and img.size[1] > 0
        # 3-row × 2-column layout → wider than tall
        assert img.size[0] > img.size[1]


# ===================================================================
# 4. Fat overlay image — non-blank
# ===================================================================


class TestFatOverlay:
    """Fat overlay image has expected properties."""

    def test_image_not_blank(self, shared_dashboard):
        img = Image.open(shared_dashboard.fat_overlay_path)
        assert img.size[0] > 0 and img.size[1] > 0
        assert img.size[0] > img.size[1]


# ===================================================================
# 5. Summary table contains key values
# ===================================================================


class TestSummaryTable:
    """Summary table text file contains expected content."""

    def test_contains_patient_id(self, shared_dashboard):
        with open(shared_dashboard.summary_table_path) as f:
            text = f.read()
        assert PATIENT_ID in text

    def test_contains_pericardium_info(self, shared_dashboard):
        with open(shared_dashboard.summary_table_path) as f:
            text = f.read()
        assert "Pericardium" in text

    def test_contains_la_volume(self, shared_dashboard):
        with open(shared_dashboard.summary_table_path) as f:
            text = f.read()
        assert "LA" in text

    def test_contains_volume_ml(self, shared_dashboard):
        with open(shared_dashboard.summary_table_path) as f:
            text = f.read()
        assert "ml" in text.lower()

    def test_csv_contains_values(self, shared_dashboard):
        csv_dir = os.path.dirname(shared_dashboard.summary_table_path)
        csv_path = os.path.join(csv_dir, "summary.csv")
        with open(csv_path) as f:
            content = f.read()
        assert PATIENT_ID in content
        assert "LA" in content
        assert "volume" in content.lower()


# ===================================================================
# 6. Summary HTML contains section headers
# ===================================================================


class TestSummaryHTML:
    """Combined HTML contains sections for each component."""

    def test_has_slice_gallery_section(self, shared_dashboard):
        with open(shared_dashboard.summary_html_path) as f:
            html = f.read()
        assert "Slice Gallery" in html

    def test_has_fat_overlay_section(self, shared_dashboard):
        with open(shared_dashboard.summary_html_path) as f:
            html = f.read()
        assert "Fat Overlay" in html

    def test_has_numeric_summary_section(self, shared_dashboard):
        with open(shared_dashboard.summary_html_path) as f:
            html = f.read()
        assert "Numeric Summary" in html

    def test_3d_view_section_removed(self, shared_dashboard):
        with open(shared_dashboard.summary_html_path) as f:
            html = f.read()
        assert "3D View" not in html

    def test_has_patient_id_in_html(self, shared_dashboard):
        with open(shared_dashboard.summary_html_path) as f:
            html = f.read()
        assert PATIENT_ID in html

    def test_is_self_contained_no_cdn(self, shared_dashboard):
        with open(shared_dashboard.summary_html_path) as f:
            html = f.read()
        assert "http://" not in html
        assert "https://" not in html
        assert "cdn" not in html.lower()


# ===================================================================
# 7. DashboardOutput dataclass


class TestDashboardOutputDataclass:
    """DashboardOutput fields, types, and immutability."""

    def test_fields_present(self):
        d = DashboardOutput(
            output_dir="/tmp",
            slice_gallery_path="/tmp/gallery.png",
            fat_overlay_path="/tmp/fat.png",
            summary_table_path="/tmp/summary.txt",
            summary_html_path="/tmp/dashboard.html",
        )
        assert isinstance(d.output_dir, str)
        assert isinstance(d.slice_gallery_path, str)
        assert isinstance(d.fat_overlay_path, str)
        assert isinstance(d.summary_table_path, str)
        assert isinstance(d.summary_html_path, str)

    def test_frozen_immutable(self):
        d = DashboardOutput(
            output_dir="/tmp",
            slice_gallery_path="/tmp/gallery.png",
            fat_overlay_path="/tmp/fat.png",
            summary_table_path="/tmp/summary.txt",
            summary_html_path="/tmp/dashboard.html",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            d.output_dir = "/other"  # type: ignore

    def test_repr(self):
        d = DashboardOutput(
            output_dir="/tmp",
            slice_gallery_path="/tmp/gallery.png",
            fat_overlay_path="/tmp/fat.png",
            summary_table_path="/tmp/summary.txt",
            summary_html_path="/tmp/dashboard.html",
        )
        assert "DashboardOutput" in repr(d)
        assert "slice_gallery_path" in repr(d)


# ===================================================================
# 8. Handles empty fat mask
# ===================================================================


class TestEmptyFatMask:
    """Dashboard handles case where LA fat is empty."""

    @pytest.fixture(scope="class")
    def empty_fat_dashboard(
        self, tmp_path_factory
    ) -> DashboardOutput:
        d = tmp_path_factory.mktemp("dashboard_empty_fat")
        data = _build_shared_data(str(d))
        data["cleanup_result"] = dataclasses.replace(
            data["cleanup_result"],
            cleaned_mask=np.zeros(SHAPE, dtype=bool),
        )
        data["partition_result"] = dataclasses.replace(
            data["partition_result"],
            la_fat_mask=np.zeros(SHAPE, dtype=bool),
        )
        return generate_dashboard(**data)

    def test_no_crash_with_empty_fat(self, empty_fat_dashboard):
        assert os.path.isfile(empty_fat_dashboard.slice_gallery_path)
        assert os.path.isfile(empty_fat_dashboard.fat_overlay_path)
        assert os.path.isfile(empty_fat_dashboard.summary_html_path)


# ===================================================================
# 9. Handles excluded anchor
# ===================================================================


class TestExcludedAnchor:
    """Dashboard handles excluded anchors without crash."""

    @pytest.fixture(scope="class")
    def excluded_dashboard(self, tmp_path_factory) -> DashboardOutput:
        d = tmp_path_factory.mktemp("dashboard_excluded")
        data = _build_shared_data(str(d))
        sr = data["partition_result"]
        excl = ["RV"]
        reasons = {"RV": "mask is empty"}
        vols = dict(sr.anchor_volumes_ml)
        shares = dict(sr.anchor_shares)
        vols["RV"] = 0.0
        shares["RV"] = 0.0
        data["partition_result"] = dataclasses.replace(
            sr,
            excluded_anchors=excl,
            exclusion_reasons=reasons,
            anchor_volumes_ml=vols,
            anchor_shares=shares,
        )
        return generate_dashboard(**data)

    def test_no_crash_with_excluded_anchor(self, excluded_dashboard):
        assert os.path.isfile(excluded_dashboard.slice_gallery_path)
        assert os.path.isfile(excluded_dashboard.fat_overlay_path)

    def test_excluded_anchor_gallery_valid(self, excluded_dashboard):
        img = Image.open(excluded_dashboard.slice_gallery_path)
        assert img.size[0] > 0
