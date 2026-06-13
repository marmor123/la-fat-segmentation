"""Tests for the la_fat.quality_flagger module.

Exercises the generation of discrete quality flags from pipeline results.
Each flag is evaluated independently — no short-circuiting or collapsed
scores.
"""

from __future__ import annotations

import dataclasses
import typing as t

import numpy as np
import pytest

from la_fat.config import PipelineConfig
from la_fat.cleanup import CleanupResult
from la_fat.fat_thresholder import FatThresholdResult
from la_fat.partition_engine import PartitionResult
from la_fat.pericardium_resolver import PericardiumResult
from la_fat.quality_flagger import QualityFlag, generate_quality_flags

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------

CFG = PipelineConfig()
SHAPE = (16, 16, 16)


# ---------------------------------------------------------------------------
# Helpers — build dummy result objects
# ---------------------------------------------------------------------------


def _dummy_partition(
    *,
    excluded_anchors: t.Optional[list[str]] = None,
    exclusion_reasons: t.Optional[dict[str, str]] = None,
    la_volume_ml: float = 10.0,
    lv_volume_ml: float = 5.0,
    unassigned_volume_ml: float = 1.0,
    total_fat_volume_ml: float = 16.0,
) -> PartitionResult:
    """Create a PartitionResult with controllable fields."""
    excluded = excluded_anchors or []
    reasons = exclusion_reasons or {}
    # Build volumes dict for all 6 anchors + any excluded.
    volumes: dict[str, float] = {
        "LA": la_volume_ml,
        "LV": lv_volume_ml,
        "RA": 2.0,
        "RV": 2.0,
        "Aorta": 1.0,
        "Pulmonary_Artery": 1.0,
    }
    shares: dict[str, float] = {
        k: v / max(total_fat_volume_ml, 0.001) * 100.0 for k, v in volumes.items()
    }
    for a in excluded:
        volumes[a] = 0.0
        shares[a] = 0.0

    return PartitionResult(
        la_fat_mask=np.zeros(SHAPE, dtype=bool),
        all_fat_mask=np.zeros(SHAPE, dtype=bool),
        anchor_assignments=np.zeros(SHAPE, dtype=np.int32),
        anchor_volumes_ml=volumes,
        anchor_shares=shares,
        unassigned_volume_ml=unassigned_volume_ml,
        total_fat_volume_ml=total_fat_volume_ml,
        excluded_anchors=excluded,
        exclusion_reasons=reasons,
    )


def _dummy_fat_threshold(
    *,
    fallback_triggered: bool = False,
    sigma_hu: float = 50.0,
) -> FatThresholdResult:
    return FatThresholdResult(
        hu_low=-190.0,
        hu_high=-30.0,
        mean_hu=-110.0,
        sigma_hu=sigma_hu,
        fallback_triggered=fallback_triggered,
        fallback_reason="test fallback" if fallback_triggered else None,
        method="fixed_fallback" if fallback_triggered else "gaussian_fit",
        num_voxels_fit=5000,
    )


def _dummy_pericardium(
    *,
    fallback_triggered: bool = False,
) -> PericardiumResult:
    return PericardiumResult(
        mask=np.zeros(SHAPE, dtype=bool),
        fallback_triggered=fallback_triggered,
        fallback_reason="test fallback" if fallback_triggered else None,
        method="convex_hull_fallback" if fallback_triggered else "ts_direct",
    )


def _dummy_cleanup(
    *,
    islands_removed: int = 0,
) -> CleanupResult:
    return CleanupResult(
        cleaned_mask=np.zeros(SHAPE, dtype=bool),
        islands_removed=islands_removed,
        island_volumes_mm3=[10.0] * islands_removed if islands_removed > 0 else [],
        total_removed_volume_mm3=10.0 * islands_removed,
        morphological_opening_applied=False,
        vessel_filling_applied=False,
    )


# ===================================================================
# 1. All flags fire independently
# ===================================================================


class TestAllFlagsIndependent:
    """All 8 flags appear when all conditions are triggered."""

    def test_all_eight_flags_present(self):
        part = _dummy_partition(
            excluded_anchors=["RA"],
            exclusion_reasons={"RA": "volume too low"},
            la_volume_ml=1.0,  # below 2 ml
            lv_volume_ml=3.0,  # exceeds LA
            unassigned_volume_ml=15.0,
            total_fat_volume_ml=16.0,  # ~93.75% unassigned > 80%
        )
        fat = _dummy_fat_threshold(
            fallback_triggered=True,
            sigma_hu=150.0,  # > 100
        )
        peri = _dummy_pericardium(fallback_triggered=True)
        clean = _dummy_cleanup(islands_removed=3)

        flags = generate_quality_flags(part, fat, peri, clean, CFG)

        concerns = [f.concern for f in flags]
        expected = [
            "pericardium_fallback",
            "anchor_excluded",
            "fat_threshold_fallback",
            "la_volume_out_of_range",
            "lv_exceeds_la",
            "high_unassigned_fat",
            "wide_gaussian_sigma",
            "islands_cleaned",
        ]
        for concern in expected:
            assert concern in concerns, (
                f"Expected flag '{concern}' not found in {concerns}"
            )
        assert len(flags) == len(expected), (
            f"Expected {len(expected)} flags, got {len(flags)}: {concerns}"
        )


# ===================================================================
# 2. No flags — everything perfect
# ===================================================================


class TestNoFlags:
    """Normal inputs with no issues — empty list."""

    def test_all_normal_returns_empty_list(self):
        part = _dummy_partition(
            la_volume_ml=50.0,   # within 2-150
            lv_volume_ml=30.0,   # less than LA
            unassigned_volume_ml=1.0,
            total_fat_volume_ml=100.0,  # 1% unassigned
        )
        fat = _dummy_fat_threshold(
            fallback_triggered=False,
            sigma_hu=50.0,  # <= 100
        )
        peri = _dummy_pericardium(fallback_triggered=False)
        clean = _dummy_cleanup(islands_removed=0)

        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        assert flags == []


# ===================================================================
# 3. High — pericardium fallback
# ===================================================================


class TestPericardiumFallback:
    """PericardiumResult with fallback_triggered=True → flag."""

    def test_pericardium_fallback_flag(self):
        part = _dummy_partition()
        fat = _dummy_fat_threshold()
        peri = _dummy_pericardium(fallback_triggered=True)
        clean = _dummy_cleanup()

        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "pericardium_fallback" in concerns

    def test_no_flag_when_no_fallback(self):
        part = _dummy_partition()
        fat = _dummy_fat_threshold()
        peri = _dummy_pericardium(fallback_triggered=False)
        clean = _dummy_cleanup()

        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "pericardium_fallback" not in concerns


# ===================================================================
# 4. High — anchor excluded
# ===================================================================


class TestAnchorExcluded:
    """PartitionResult with excluded anchors → flag."""

    def test_anchor_excluded_flag(self):
        part = _dummy_partition(
            excluded_anchors=["RA", "RV"],
            exclusion_reasons={
                "RA": "volume 1.0 ml < 5.0 ml threshold",
                "RV": "mask not provided",
            },
        )
        fat = _dummy_fat_threshold()
        peri = _dummy_pericardium()
        clean = _dummy_cleanup()

        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "anchor_excluded" in concerns

    def test_no_flag_when_no_exclusions(self):
        part = _dummy_partition(excluded_anchors=[])
        fat = _dummy_fat_threshold()
        peri = _dummy_pericardium()
        clean = _dummy_cleanup()

        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "anchor_excluded" not in concerns


# ===================================================================
# 5. High — fat threshold fallback
# ===================================================================


class TestFatThresholdFallback:
    """FatThresholdResult with fallback_triggered=True → flag."""

    def test_fat_threshold_fallback_flag(self):
        part = _dummy_partition()
        fat = _dummy_fat_threshold(fallback_triggered=True)
        peri = _dummy_pericardium()
        clean = _dummy_cleanup()

        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "fat_threshold_fallback" in concerns

    def test_no_flag_when_no_fallback(self):
        part = _dummy_partition()
        fat = _dummy_fat_threshold(fallback_triggered=False)
        peri = _dummy_pericardium()
        clean = _dummy_cleanup()

        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "fat_threshold_fallback" not in concerns


# ===================================================================
# 6. Medium — LA volume range
# ===================================================================


class TestLaVolumeRange:
    """Volume outside 2-150 ml → flag. At boundary → no flag."""

    @pytest.fixture
    def fat(self):
        return _dummy_fat_threshold()

    @pytest.fixture
    def peri(self):
        return _dummy_pericardium()

    @pytest.fixture
    def clean(self):
        return _dummy_cleanup()

    def test_volume_below_low(self, fat, peri, clean):
        part = _dummy_partition(la_volume_ml=1.0)
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "la_volume_out_of_range" in concerns

    def test_volume_above_high(self, fat, peri, clean):
        part = _dummy_partition(la_volume_ml=200.0)
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "la_volume_out_of_range" in concerns

    def test_volume_at_low_boundary_no_flag(self, fat, peri, clean):
        """Volume exactly at 2.0 is not flagged (>= threshold)."""
        part = _dummy_partition(la_volume_ml=2.0)
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "la_volume_out_of_range" not in concerns

    def test_volume_at_high_boundary_no_flag(self, fat, peri, clean):
        """Volume exactly at 150.0 is not flagged (<= threshold)."""
        part = _dummy_partition(la_volume_ml=150.0)
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "la_volume_out_of_range" not in concerns

    def test_volume_within_range_no_flag(self, fat, peri, clean):
        part = _dummy_partition(la_volume_ml=75.0)
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "la_volume_out_of_range" not in concerns


# ===================================================================
# 7. Medium — LV exceeds LA
# ===================================================================


class TestLvExceedsLa:
    """LV volume > LA volume → flag."""

    @pytest.fixture
    def fat(self):
        return _dummy_fat_threshold()

    @pytest.fixture
    def peri(self):
        return _dummy_pericardium()

    @pytest.fixture
    def clean(self):
        return _dummy_cleanup()

    def test_lv_exceeds_la_flag(self, fat, peri, clean):
        part = _dummy_partition(la_volume_ml=5.0, lv_volume_ml=10.0)
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "lv_exceeds_la" in concerns

    def test_no_flag_when_la_exceeds_lv(self, fat, peri, clean):
        part = _dummy_partition(la_volume_ml=10.0, lv_volume_ml=5.0)
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "lv_exceeds_la" not in concerns

    def test_no_flag_when_equal(self, fat, peri, clean):
        """Equal volumes should not trigger."""
        part = _dummy_partition(la_volume_ml=5.0, lv_volume_ml=5.0)
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "lv_exceeds_la" not in concerns

    def test_no_flag_when_lv_zero(self, fat, peri, clean):
        part = _dummy_partition(la_volume_ml=10.0, lv_volume_ml=0.0)
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "lv_exceeds_la" not in concerns


# ===================================================================
# 8. Medium — high unassigned
# ===================================================================


class TestHighUnassigned:
    """More than 80% unassigned → flag. Exactly 80% → no flag."""

    @pytest.fixture
    def fat(self):
        return _dummy_fat_threshold()

    @pytest.fixture
    def peri(self):
        return _dummy_pericardium()

    @pytest.fixture
    def clean(self):
        return _dummy_cleanup()

    def test_above_80_percent(self, fat, peri, clean):
        """81% unassigned → flag."""
        part = _dummy_partition(
            unassigned_volume_ml=81.0,
            total_fat_volume_ml=100.0,
        )
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "high_unassigned_fat" in concerns

    def test_exactly_80_percent_no_flag(self, fat, peri, clean):
        """Exactly 80% should not trigger (> threshold, not >=)."""
        part = _dummy_partition(
            unassigned_volume_ml=80.0,
            total_fat_volume_ml=100.0,
        )
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "high_unassigned_fat" not in concerns

    def test_below_80_percent_no_flag(self, fat, peri, clean):
        part = _dummy_partition(
            unassigned_volume_ml=30.0,
            total_fat_volume_ml=100.0,
        )
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "high_unassigned_fat" not in concerns

    def test_zero_unassigned_no_flag(self, fat, peri, clean):
        part = _dummy_partition(
            unassigned_volume_ml=0.0,
            total_fat_volume_ml=100.0,
        )
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "high_unassigned_fat" not in concerns


# ===================================================================
# 9. Low — wide Gaussian sigma
# ===================================================================


class TestWideGaussianSigma:
    """sigma > 100 → flag."""

    @pytest.fixture
    def part(self):
        return _dummy_partition()

    @pytest.fixture
    def peri(self):
        return _dummy_pericardium()

    @pytest.fixture
    def clean(self):
        return _dummy_cleanup()

    def test_sigma_above_threshold(self, part, peri, clean):
        fat = _dummy_fat_threshold(sigma_hu=150.0)
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "wide_gaussian_sigma" in concerns

    def test_sigma_at_threshold_no_flag(self, part, peri, clean):
        """sigma exactly 100 should not trigger (> threshold, not >=)."""
        fat = _dummy_fat_threshold(sigma_hu=100.0)
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "wide_gaussian_sigma" not in concerns

    def test_sigma_below_threshold_no_flag(self, part, peri, clean):
        fat = _dummy_fat_threshold(sigma_hu=50.0)
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "wide_gaussian_sigma" not in concerns


# ===================================================================
# 10. Low — islands cleaned
# ===================================================================


class TestIslandsCleaned:
    """islands_removed > 0 → flag."""

    @pytest.fixture
    def part(self):
        return _dummy_partition()

    @pytest.fixture
    def fat(self):
        return _dummy_fat_threshold()

    @pytest.fixture
    def peri(self):
        return _dummy_pericardium()

    def test_islands_removed_flag(self, part, fat, peri):
        clean = _dummy_cleanup(islands_removed=3)
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "islands_cleaned" in concerns

    def test_no_islands_removed_no_flag(self, part, fat, peri):
        clean = _dummy_cleanup(islands_removed=0)
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "islands_cleaned" not in concerns


# ===================================================================
# 11. Severity field correctness
# ===================================================================


class TestSeverityCorrectness:
    """Each flag has the correct severity string."""

    def test_high_severity_flags(self):
        part = _dummy_partition(
            excluded_anchors=["RA"],
            exclusion_reasons={"RA": "too small"},
        )
        fat = _dummy_fat_threshold(fallback_triggered=True)
        peri = _dummy_pericardium(fallback_triggered=True)
        clean = _dummy_cleanup()

        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        for f in flags:
            if f.concern in ("pericardium_fallback", "anchor_excluded", "fat_threshold_fallback"):
                assert f.severity == "high", (
                    f"Expected '{f.concern}' to be 'high', got '{f.severity}'"
                )

    def test_medium_severity_flags(self):
        part = _dummy_partition(
            la_volume_ml=1.0,
            lv_volume_ml=3.0,
            unassigned_volume_ml=81.0,
            total_fat_volume_ml=100.0,
        )
        fat = _dummy_fat_threshold()
        peri = _dummy_pericardium()
        clean = _dummy_cleanup()

        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        for f in flags:
            assert f.severity == "medium", (
                f"Expected '{f.concern}' to be 'medium', got '{f.severity}'"
            )

    def test_low_severity_flags(self):
        part = _dummy_partition()
        fat = _dummy_fat_threshold(sigma_hu=150.0)
        peri = _dummy_pericardium()
        clean = _dummy_cleanup(islands_removed=1)

        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        for f in flags:
            assert f.severity == "low", (
                f"Expected '{f.concern}' to be 'low', got '{f.severity}'"
            )


# ===================================================================
# 12. QualityFlag dataclass
# ===================================================================


class TestQualityFlagDataclass:
    """QualityFlag fields, types."""

    def test_fields_present(self):
        flag = QualityFlag(
            severity="high",
            concern="pericardium_fallback",
            detail="Pericardium fallback was triggered",
            threshold_value=None,
            actual_value=None,
        )
        assert isinstance(flag.severity, str)
        assert isinstance(flag.concern, str)
        assert isinstance(flag.detail, str)
        assert flag.threshold_value is None
        assert flag.actual_value is None

    def test_with_values(self):
        flag = QualityFlag(
            severity="medium",
            concern="la_volume_out_of_range",
            detail="LA fat volume 1.0 ml outside range [2.0, 150.0]",
            threshold_value=2.0,
            actual_value=1.0,
        )
        assert flag.severity == "medium"
        assert flag.concern == "la_volume_out_of_range"
        assert flag.threshold_value == 2.0
        assert flag.actual_value == 1.0

    def test_frozen_immutable(self):
        flag = QualityFlag(
            severity="low",
            concern="islands_cleaned",
            detail="3 small islands removed",
            threshold_value=None,
            actual_value=3.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            flag.severity = "high"  # type: ignore

    def test_repr(self):
        flag = QualityFlag(
            severity="high",
            concern="pericardium_fallback",
            detail="test",
            threshold_value=None,
            actual_value=None,
        )
        assert "QualityFlag" in repr(flag)
        assert "pericardium_fallback" in repr(flag)


# ===================================================================
# 13. Edge case — zero total fat volume
# ===================================================================


class TestZeroTotalFatVolume:
    """Avoid division by zero in unassigned percentage calculation."""

    def test_zero_total_fat_does_not_crash(self):
        """When total_fat_volume_ml is 0, unassigned % should be 0."""
        part = _dummy_partition(
            unassigned_volume_ml=0.0,
            total_fat_volume_ml=0.0,
        )
        fat = _dummy_fat_threshold()
        peri = _dummy_pericardium()
        clean = _dummy_cleanup()

        # Should not raise ZeroDivisionError.
        flags = generate_quality_flags(part, fat, peri, clean, CFG)
        concerns = [f.concern for f in flags]
        assert "high_unassigned_fat" not in concerns
