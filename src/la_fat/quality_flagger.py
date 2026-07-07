"""Quality flagger module for LA Fat Segmentation.

Generates discrete quality flags from pipeline results.  Each flag is
evaluated independently — no short-circuiting or collapsed scores.

Domain
------
**Quality Flags** — three severity levels (high, medium, low).  Each
concern is reported separately for the researcher to evaluate.
"""

from __future__ import annotations

import dataclasses
import typing as t

from la_fat.config import PipelineConfig
from la_fat.cleanup import CleanupResult
from la_fat.partition_engine import PartitionResult
from la_fat.pericardium_resolver import PericardiumResult


@dataclasses.dataclass(frozen=True)
class QualityFlag:
    """A single quality flag with severity and contextual details.

    Attributes
    ----------
    severity:
        One of ``"high"``, ``"medium"``, or ``"low"``.
    concern:
        Short name identifying the specific concern (e.g.
        ``"pericardium_fallback"``, ``"la_volume_out_of_range"``).
    detail:
        Human-readable description with actual values.
    threshold_value:
        The threshold that was checked (or ``None`` if not applicable).
    actual_value:
        The value observed (or ``None`` if not applicable).
    """

    severity: str
    concern: str
    detail: str
    threshold_value: float | None
    actual_value: float | None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_quality_flags(
    partition_result: PartitionResult,
    pericardium_result: PericardiumResult,
    cleanup_result: CleanupResult,
    config: PipelineConfig,
) -> list[QualityFlag]:
    """Generate quality flags from pipeline results.

    Each flag is evaluated independently — all applicable flags are
    returned.  No short-circuiting.

    Parameters
    ----------
    partition_result:
        Result from ``partition_fat``.
    pericardium_result:
        Result from ``resolve_pericardium``.
    cleanup_result:
        Result from ``cleanup_la_fat_mask``.
    config:
        Pipeline configuration providing threshold values.

    Returns
    -------
    list[QualityFlag]
        Sorted by severity: high concern flags first, then medium, then
        low.
    """
    flags: list[QualityFlag] = []

    # ── HIGH CONCERN ─────────────────────────────────────────────────────────

    # Pericardium fallback triggered.
    if pericardium_result.fallback_triggered:
        flags.append(
            QualityFlag(
                severity="high",
                concern="pericardium_fallback",
                detail=(
                    f"Pericardium resolution used fallback method: "
                    f"{pericardium_result.fallback_reason or 'unknown'}"
                ),
                threshold_value=None,
                actual_value=None,
            )
        )

    # Anchor(s) excluded.
    if len(partition_result.excluded_anchors) > 0:
        excluded_list = ", ".join(
            f"{a} ({partition_result.exclusion_reasons.get(a, 'no reason given')})"
            for a in partition_result.excluded_anchors
        )
        flags.append(
            QualityFlag(
                severity="high",
                concern="anchor_excluded",
                detail=(
                    f"Anchor(s) excluded: {excluded_list}"
                ),
                threshold_value=config.min_anchor_volume_ml,
                actual_value=None,
            )
        )

    # ── MEDIUM CONCERN ───────────────────────────────────────────────────────

    # LA fat volume outside expected range.
    la_volume = partition_result.anchor_volumes_ml.get("LA", 0.0)
    low_bound = config.la_fat_volume_low_ml
    high_bound = config.la_fat_volume_high_ml
    if la_volume < low_bound or la_volume > high_bound:
        flags.append(
            QualityFlag(
                severity="medium",
                concern="la_volume_out_of_range",
                detail=(
                    f"LA fat volume {la_volume:.2f} ml outside expected range "
                    f"[{low_bound}, {high_bound}] ml"
                ),
                threshold_value=low_bound,
                actual_value=la_volume,
            )
        )

    # LV/LA ratio exceeds threshold.
    lv_volume = partition_result.anchor_volumes_ml.get("LV", 0.0)
    la_safe = max(la_volume, 0.001)
    lv_la_ratio = lv_volume / la_safe
    if lv_la_ratio > config.max_lv_la_ratio:
        flags.append(
            QualityFlag(
                severity="medium",
                concern="lv_la_ratio_high",
                detail=(
                    f"LV/LA fat volume ratio {lv_la_ratio:.1f} exceeds "
                    f"threshold {config.max_lv_la_ratio} "
                    f"(LA={la_volume:.2f} ml, LV={lv_volume:.2f} ml)"
                ),
                threshold_value=config.max_lv_la_ratio,
                actual_value=lv_la_ratio,
            )
        )

    # More than max_unassigned_fat_pct of pericardial fat unassigned.
    total = max(partition_result.total_fat_volume_ml, 0.001)
    unassigned_pct = partition_result.unassigned_volume_ml / total * 100.0
    if unassigned_pct > config.max_unassigned_fat_pct:
        flags.append(
            QualityFlag(
                severity="medium",
                concern="high_unassigned_fat",
                detail=(
                    f"{unassigned_pct:.1f}% of epicardial fat is unassigned "
                    f"(threshold: {config.max_unassigned_fat_pct}%)"
                ),
                threshold_value=config.max_unassigned_fat_pct,
                actual_value=unassigned_pct,
            )
        )

    # Total epicardial fat fraction of pericardium volume too low.
    # Indicates either genuinely low fat or possibly a loose pericardium
    # mask / contrast-impaired fat threshold.
    peri_vol = pericardium_result.volume_ml
    if peri_vol > 0.0:
        fat_fraction_pct = total / peri_vol * 100.0
        if fat_fraction_pct < config.min_fat_fraction_pct:
            flags.append(
                QualityFlag(
                    severity="medium",
                    concern="low_fat_fraction",
                    detail=(
                        f"Epicardial fat ({total:.2f} ml) is only "
                        f"{fat_fraction_pct:.1f}% of pericardium volume "
                        f"({peri_vol:.2f} ml) — below {config.min_fat_fraction_pct}% "
                        f"threshold"
                    ),
                    threshold_value=config.min_fat_fraction_pct,
                    actual_value=fat_fraction_pct,
                )
            )

    # Return flags sorted by severity order.
    severity_order = {"high": 0, "medium": 1, "low": 2}
    flags.sort(key=lambda f: severity_order.get(f.severity, 99))

    return flags
