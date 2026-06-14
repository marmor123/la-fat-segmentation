"""Mesh extraction module for LA Fat Segmentation.

Provides marching cubes mesh extraction and PLY file output for
intermediate and final pipeline results.
"""

from __future__ import annotations

import os
import typing as t

import numpy as np
from skimage.measure import marching_cubes

from la_fat.anatomy import CANONICAL_ANCHORS
from la_fat import nifti_io

__all__ = [
    "extract_meshes_for_step",
    "extract_interactive_meshes",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_meshes_for_step(
    masks: dict[str, np.ndarray],
    spacing: tuple[float, float, float],
) -> dict[str, tuple[np.ndarray, np.ndarray] | None]:
    """Run marching cubes on each mask and return ``{name: (verts, faces)}``.

    Parameters
    ----------
    masks:
        Dictionary mapping surface names to binary mask arrays.
    spacing:
        Voxel spacing ``(sx, sy, sz)`` in mm.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray] | None]
        Each entry is ``(verts, faces)`` from
        ``skimage.measure.marching_cubes`` or ``None`` if the mask is
        empty or extraction fails.
    """
    result: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}

    for name, mask in masks.items():
        if np.any(mask):
            try:
                verts, faces, _, _ = marching_cubes(
                    mask.astype(float), level=0.5, spacing=spacing
                )
                result[name] = (verts, faces)
            except (ValueError, RuntimeError):
                result[name] = None
        else:
            result[name] = None

    return result


def extract_interactive_meshes(
    pipeline_state: dict[str, t.Any],
    output_dir: str,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray] | None]]:
    """Orchestrate mesh extraction for all three Dashboard Steps.

    For each step, runs ``extract_meshes_for_step``, saves each surface
    as a ``.ply`` file to ``output_dir/meshes/<step_name>/<surface_name>.ply``,
    and saves the corresponding binary masks as ``.nii.gz`` alongside for
    debugging.

    Parameters
    ----------
    pipeline_state:
        Dict containing:

        - ``anchor_masks``: ``{anchor_name: binary_mask}`` for LA, LV, RA,
          RV, Aorta, Pulmonary_Artery
        - ``pericardium_mask``: binary mask array
        - ``partition_result``: object with ``anchor_assignments`` (int
          array), ``all_fat_mask`` (bool array)
        - ``cleanup_result``: object with ``cleaned_mask`` (the final LA
          fat mask)
        - ``spacing``: tuple ``(sx, sy, sz)``
    output_dir:
        Root output directory; meshes are written to
        ``output_dir/meshes/<step_name>/``.

    Returns
    -------
    dict[str, dict[str, tuple[np.ndarray, np.ndarray] | None]]
        Nested dict keyed by step name, then surface name.
    """
    anchor_masks: dict[str, np.ndarray] = pipeline_state["anchor_masks"]
    pericardium_mask: np.ndarray = pipeline_state["pericardium_mask"]
    partition_result = pipeline_state["partition_result"]
    cleanup_result = pipeline_state["cleanup_result"]
    spacing: tuple[float, float, float] = pipeline_state["spacing"]

    results: dict[str, dict[str, tuple[np.ndarray, np.ndarray] | None]] = {}

    # ---- Step 2: Anchors (segmented cardiac structures) ---------------------
    step2_masks: dict[str, np.ndarray] = {}
    for name in CANONICAL_ANCHORS:
        if name in anchor_masks:
            step2_masks[name] = anchor_masks[name]
    step2_masks["Pericardium"] = pericardium_mask

    step2_dir = os.path.join(output_dir, "meshes", "step2_anchors")
    step2_result = extract_meshes_for_step(step2_masks, spacing)
    _save_meshes_and_masks(step2_result, step2_masks, step2_dir, spacing)
    results["step2_anchors"] = step2_result

    # ---- Step 5: Partition (fat assigned to each anchor) --------------------
    anchor_assignments: np.ndarray = partition_result.anchor_assignments
    all_fat_mask: np.ndarray = partition_result.all_fat_mask

    step5_masks: dict[str, np.ndarray] = {}
    anchor_labels = {name: idx + 1 for idx, name in enumerate(CANONICAL_ANCHORS)}
    for anchor_name, label in anchor_labels.items():
        fat_for_anchor = all_fat_mask & (anchor_assignments == label)
        step5_masks[anchor_name] = fat_for_anchor
    step5_masks["Pericardium"] = pericardium_mask

    step5_dir = os.path.join(output_dir, "meshes", "step5_partition")
    step5_result = extract_meshes_for_step(step5_masks, spacing)
    _save_meshes_and_masks(step5_result, step5_masks, step5_dir, spacing)
    results["step5_partition"] = step5_result

    # ---- Step 7: Final (LA chamber, Pericardium, LA fat) --------------------
    la_mask = anchor_masks.get("LA")
    if la_mask is None:
        la_mask = np.zeros_like(pericardium_mask, dtype=bool)

    cleaned_mask: np.ndarray = cleanup_result.cleaned_mask

    step7_masks: dict[str, np.ndarray] = {
        "LA_chamber": la_mask,
        "Pericardium": pericardium_mask,
        "LA_fat": cleaned_mask,
    }

    step7_dir = os.path.join(output_dir, "meshes", "step7_final")
    step7_result = extract_meshes_for_step(step7_masks, spacing)
    _save_meshes_and_masks(step7_result, step7_masks, step7_dir, spacing)
    results["step7_final"] = step7_result

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_ply(filepath: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write a text-format PLY file.

    Parameters
    ----------
    filepath:
        Output path (should end in ``.ply``).
    verts:
        ``(N, 3)`` array of vertex coordinates.
    faces:
        ``(M, 3)`` array of triangle face indices.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    n_verts = len(verts)
    n_faces = len(faces)

    lines: list[str] = []
    _add = lines.append
    _add("ply")
    _add("format ascii 1.0")
    _add(f"element vertex {n_verts}")
    _add("property float x")
    _add("property float y")
    _add("property float z")
    _add(f"element face {n_faces}")
    _add("property list uchar int vertex_indices")
    _add("end_header")

    for v in verts:
        _add(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    for f in faces:
        _add(f"3 {int(f[0])} {int(f[1])} {int(f[2])}")

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")


def _save_meshes_and_masks(
    meshes: dict[str, tuple[np.ndarray, np.ndarray] | None],
    masks: dict[str, np.ndarray],
    output_dir: str,
    spacing: tuple[float, float, float],
) -> None:
    """Save extracted meshes as PLY files and masks as NIfTI files.

    Parameters
    ----------
    meshes:
        Result from ``extract_meshes_for_step``.
    masks:
        Original binary masks (same keys).
    output_dir:
        Directory to write files into.
    spacing:
        Voxel spacing ``(sx, sy, sz)`` in mm.
    """
    os.makedirs(output_dir, exist_ok=True)

    for name in meshes:
        mesh = meshes[name]
        if mesh is not None:
            verts, faces = mesh
            ply_path = os.path.join(output_dir, f"{name}.ply")
            _save_ply(ply_path, verts, faces)

        # Always save the mask NIfTI for debugging.
        if name in masks:
            nii_path = os.path.join(output_dir, f"{name}.nii.gz")
            nifti_io.save_nifti(
                masks[name].astype(np.uint8),
                nii_path,
                spacing=spacing,
            )
