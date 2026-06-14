"""Single source of truth for Partition Anchors and shared metadata.

This module owns the canonical list of the six Partition Anchors and ALL
associated metadata (ordinals, display labels, colours, TotalSegmentator
filenames) so that every consumer imports from a single place.

Partition Anchors
=================
The six canonical structures closest to which epicardial fat is partitioned:

  - LA   (Left Atrium)
  - LV   (Left Ventricle)
  - RA   (Right Atrium)
  - RV   (Right Ventricle)
  - Aorta
  - Pulmonary_Artery

Pulmonary Veins are **not** Partition Anchors — their fat belongs to LA.

Domain
------
See ``CONTEXT.md`` and ADR-0001 for the domain model.
"""

from __future__ import annotations

import typing as t

# ---------------------------------------------------------------------------
# Canonical anchor identifiers
# ---------------------------------------------------------------------------

#: Ordered list of the six Partition Anchor keys.
#: The order defines the integer labels (1-based) used in anchor_assignments maps.
CANONICAL_ANCHORS: list[str] = [
    "LA",
    "LV",
    "RA",
    "RV",
    "Aorta",
    "Pulmonary_Artery",
]

#: 0-based ordinal index for each anchor (matches enumerate position).
ANCHOR_ORDINALS: dict[str, int] = {
    "LA": 0,
    "LV": 1,
    "RA": 2,
    "RV": 3,
    "Aorta": 4,
    "Pulmonary_Artery": 5,
}

# ---------------------------------------------------------------------------
# Display labels
# ---------------------------------------------------------------------------

#: Human-readable display name for each Partition Anchor.
ANCHOR_LABELS: dict[str, str] = {
    "LA": "Left Atrium",
    "LV": "Left Ventricle",
    "RA": "Right Atrium",
    "RV": "Right Ventricle",
    "Aorta": "Aorta",
    "Pulmonary_Artery": "Pulmonary Artery",
}

# ---------------------------------------------------------------------------
# Rendering colours
# ---------------------------------------------------------------------------

#: RGB colour (0.0–1.0) for each Partition Anchor in dashboard rendering.
ANCHOR_COLORS: dict[str, tuple[float, float, float]] = {
    "LA": (1.0, 0.0, 0.0),  # red
    "LV": (0.0, 0.0, 1.0),  # blue
    "RA": (0.0, 0.8, 0.0),  # green
    "RV": (1.0, 0.65, 0.0),  # orange
    "Aorta": (1.0, 1.0, 0.0),  # yellow
    "Pulmonary_Artery": (0.6, 0.0, 0.6),  # purple
}

#: Shared colour for pericardium rendering (cyan).
PERICARDIUM_COLOR: tuple[float, float, float] = (0.0, 0.75, 0.75)

#: Shared colour for final LA Fat 3D rendering (gold).
LA_FAT_COLOR_3D: tuple[float, float, float] = (1.0, 0.84, 0.0)

# ---------------------------------------------------------------------------
# TotalSegmentator filename mappings
# ---------------------------------------------------------------------------

#: Mapping from anchor key to TotalSegmentator v2 output folder name.
#: Used by ts_runner to locate TS output masks.
TS_STRUCTURE_NAMES: dict[str, str] = {
    "LA": "heart_atrium_left",
    "LV": "heart_ventricle_left",
    "RA": "heart_atrium_right",
    "RV": "heart_ventricle_right",
    "Aorta": "aorta",
    "Pulmonary_Artery": "pulmonary_artery",
}

#: Mapping from anchor key to old TS v1 native filename (backward compat).
#: Used by pipeline to load masks from older TS runs.
TS_NATIVE_FILENAMES: dict[str, str] = {
    "LA": "heart_atrium_left",
    "LV": "heart_ventricle_left",
    "RA": "heart_atrium_right",
    "RV": "heart_ventricle_right",
    "Aorta": "aorta",
    "Pulmonary_Artery": "pulmonary_artery",
}

# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def voxel_volume_ml(spacing: tuple[float, float, float]) -> float:
    """Compute the volume of a single voxel in millilitres.

    Parameters
    ----------
    spacing:
        Voxel spacing ``(sx, sy, sz)`` in mm.

    Returns
    -------
    float
        Volume of one voxel in ml.
    """
    return spacing[0] * spacing[1] * spacing[2] / 1000.0
