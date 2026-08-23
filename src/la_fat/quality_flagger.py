"""Quality flagger module for LA Fat Segmentation.

Generates discrete quality audit flags from pipeline results. Each flag is
evaluated independently — no short-circuiting or collapsed scores.

Domain
------
**Quality Flags** — three severity levels (high, medium, low). Each
concern is reported separately for the researcher to evaluate.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from la_fat.cleanup import CleanupResult
    from la_fat.config import PipelineConfig
    from la_fat.partition_engine import PartitionResult
    from la_fat.pericardium_resolver import PericardiumResult


class QualitySeverity(str, Enum):
    """Severity tier for quality audit flags."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_str(cls, value: str | QualitySeverity) -> QualitySeverity:
        if isinstance(value, cls):
            return value
        val = str(value).strip().lower()
        if val in ("high", "h"):
            return cls.HIGH
        if val in ("medium", "med", "m"):
            return cls.MEDIUM
        if val in ("low", "l"):
            return cls.LOW
        raise ValueError(f"Unknown QualitySeverity: {value}")


@dataclasses.dataclass(frozen=True)
class QualityFlag:
    """An auditable quality flag with severity and clinical description.

    Attributes
    ----------
    severity:
        One of ``"high"``, ``"medium"``, or ``"low"`` (or QualitySeverity enum).
    concern:
        Short name identifying the specific concern.
    detail:
        Human-readable description with actual values.
    threshold_value:
        The threshold that was checked (or ``None`` if not applicable).
    actual_value:
        The value observed (or ``None`` if not applicable).
    flag_id:
        Optional alias for concern.
    message:
        Optional alias for detail.
    metric_value:
        Optional alias for actual_value.
    """

    severity: QualitySeverity | str
    concern: str = ""
    detail: str = ""
    threshold_value: float | None = None
    actual_value: float | None = None
    flag_id: str | None = None
    message: str | None = None
    metric_value: float | None = None

    def __post_init__(self) -> None:
        sev_str = str(
            self.severity.value if isinstance(self.severity, Enum) else self.severity
        ).lower()
        object.__setattr__(self, "severity", sev_str)

        eff_concern = self.concern or self.flag_id or ""
        object.__setattr__(self, "concern", eff_concern)
        if not self.flag_id:
            object.__setattr__(self, "flag_id", eff_concern)

        eff_detail = self.detail or self.message or ""
        object.__setattr__(self, "detail", eff_detail)
        if not self.message:
            object.__setattr__(self, "message", eff_detail)

        eff_val = self.actual_value if self.actual_value is not None else self.metric_value
        object.__setattr__(self, "actual_value", eff_val)
        if self.metric_value is None:
            object.__setattr__(self, "metric_value", eff_val)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_quality_flags(
    partition_result: Any,
    pericardium_result: Any,
    cleanup_result: Any,
    config: Any,
) -> list[QualityFlag]:
    """Generate quality flags from pipeline results.

    Evaluates each condition independently according to the domain rules:

    High Concern:
      - Pericardium fallback triggered.
      - Canonical anchor mask excluded during partitioning.

    Medium Concern:
      - LA fat volume outside physiological range (2–150 ml).
      - LV/LA fat ratio exceeds 4.0.
      - More than 80% of pericardial fat unassigned.
      - Total epicardial fat below 8% of pericardium volume.
    """
    flags: list[QualityFlag] = []

    # ── HIGH CONCERN ────────────────────────────────────────────────────────

    if getattr(pericardium_result, "fallback_triggered", False):
        flags.append(
            QualityFlag(
                severity="high",
                concern="pericardium_fallback",
                detail=(
                    f"Pericardium fallback triggered: "
                    f"{getattr(pericardium_result, 'fallback_reason', '')}"
                ),
            ),
        )

    excluded_anchors = getattr(partition_result, "excluded_anchors", [])
    if excluded_anchors:
        flags.append(
            QualityFlag(
                severity="high",
                concern="anchor_excluded",
                detail=f"Anchor(s) excluded: {', '.join(excluded_anchors)}",
            ),
        )

    # ── MEDIUM CONCERN ──────────────────────────────────────────────────────

    anchor_volumes = getattr(partition_result, "anchor_volumes_ml", {})
    la_vol = anchor_volumes.get("LA", 0.0)
    la_min = getattr(config, "la_volume_min_ml", 2.0)
    la_max = getattr(config, "la_volume_max_ml", 150.0)
    if la_vol < la_min or la_vol > la_max:
        flags.append(
            QualityFlag(
                severity="medium",
                concern="la_volume_out_of_range",
                detail=(
                    f"LA fat volume ({la_vol:.2f} ml) outside expected "
                    f"range [{la_min:.1f}, {la_max:.1f}] ml"
                ),
                threshold_value=la_min if la_vol < la_min else la_max,
                actual_value=la_vol,
            ),
        )

    lv_vol = anchor_volumes.get("LV", 0.0)
    lv_la_max = getattr(config, "lv_la_ratio_max", 4.0)
    ratio = lv_vol / max(la_vol, 0.001)
    if ratio > lv_la_max:
        flags.append(
            QualityFlag(
                severity="medium",
                concern="lv_la_ratio_high",
                detail=(
                    f"LV/LA fat ratio ({ratio:.2f}) exceeds max "
                    f"threshold ({lv_la_max:.1f})"
                ),
                threshold_value=lv_la_max,
                actual_value=ratio,
            ),
        )

    total_fat = getattr(partition_result, "total_fat_volume_ml", 0.0)
    unassigned = getattr(partition_result, "unassigned_volume_ml", 0.0)
    unassigned_max_frac = getattr(config, "unassigned_fat_max_fraction", 0.8)
    if total_fat > 0:
        unassigned_frac = unassigned / total_fat
        if unassigned_frac > unassigned_max_frac:
            flags.append(
                QualityFlag(
                    severity="medium",
                    concern="high_unassigned_fat",
                    detail=(
                        f"Unassigned fat fraction ({unassigned_frac * 100:.1f}%) "
                        f"exceeds threshold "
                        f"({unassigned_max_frac * 100:.1f}%)"
                    ),
                    threshold_value=unassigned_max_frac * 100.0,
                    actual_value=unassigned_frac * 100.0,
                ),
            )

    peri_vol = getattr(pericardium_result, "volume_ml", 0.0)
    min_eat_frac = getattr(config, "min_epicardial_fat_fraction", 0.08)
    if peri_vol > 0:
        fat_fraction = total_fat / peri_vol
        if fat_fraction < min_eat_frac:
            flags.append(
                QualityFlag(
                    severity="medium",
                    concern="low_fat_fraction",
                    detail=(
                        f"Total epicardial fat ({total_fat:.2f} ml) is "
                        f"{fat_fraction * 100:.1f}% of pericardium volume "
                        f"({peri_vol:.2f} ml), below threshold "
                        f"({min_eat_frac * 100:.1f}%)"
                    ),
                    threshold_value=min_eat_frac * 100.0,
                    actual_value=fat_fraction * 100.0,
                ),
            )

    return flags
