"""Tests for the la_fat.mesh_extractor module.

Exercises marching cubes mesh extraction and PLY file output.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from la_fat.mesh_extractor import extract_meshes_for_step, extract_interactive_meshes


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


SHAPE = (32, 32, 32)
SPACING = (1.5, 1.5, 1.5)

_CANONICAL_ANCHORS: list[str] = [
    "LA",
    "LV",
    "RA",
    "RV",
    "Aorta",
    "Pulmonary_Artery",
]


def _make_pipeline_state() -> dict:
    """Build a synthetic pipeline_state dict."""
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

    # Build anchor_assignments: int array like partition_engine produces
    anchor_assignments = np.zeros(SHAPE, dtype=np.int32)
    all_fat_mask = _make_sphere(SHAPE, (16, 16, 16), 11) & ~_make_sphere(
        SHAPE, (16, 16, 16), 6
    )
    # Assign some fat voxels to each anchor label
    from scipy.ndimage import binary_dilation

    for idx, name in enumerate(_CANONICAL_ANCHORS, start=1):
        mask = anchor_masks[name].astype(bool)
        dilated = binary_dilation(mask, iterations=2)
        region = dilated & all_fat_mask & (anchor_assignments == 0)
        anchor_assignments[region] = idx

    la_fat_mask = anchor_assignments == 1

    # Use dataclass-like objects to simulate partition_result and cleanup_result
    from dataclasses import dataclass

    @dataclass
    class PartitionResult:
        anchor_assignments: np.ndarray
        all_fat_mask: np.ndarray
        la_fat_mask: np.ndarray

    @dataclass
    class CleanupResult:
        cleaned_mask: np.ndarray

    partition_result = PartitionResult(
        anchor_assignments=anchor_assignments,
        all_fat_mask=all_fat_mask,
        la_fat_mask=la_fat_mask,
    )
    cleanup_result = CleanupResult(
        cleaned_mask=la_fat_mask.copy(),
    )

    return {
        "anchor_masks": anchor_masks,
        "pericardium_mask": pericardium_mask,
        "partition_result": partition_result,
        "cleanup_result": cleanup_result,
        "spacing": SPACING,
    }


# ===================================================================
# 1. Basic happy path: single sphere -> dict of (verts, faces)
# ===================================================================


class TestExtractMeshesForStepBasic:
    """extract_meshes_for_step returns correct structure with a single mask."""

    def test_returns_dict_of_verts_faces(self):
        mask = _make_sphere(SHAPE, (16, 16, 16), 8)
        result = extract_meshes_for_step({"test": mask}, SPACING)
        assert isinstance(result, dict)
        assert "test" in result
        verts, faces = result["test"]
        assert isinstance(verts, np.ndarray)
        assert isinstance(faces, np.ndarray)
        assert verts.ndim == 2 and verts.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3


# ===================================================================
# 2. Empty mask returns None
# ===================================================================


class TestExtractMeshesForStepEmptyMask:
    """extract_meshes_for_step returns None for empty masks."""

    def test_empty_mask_returns_none(self):
        mask = np.zeros(SHAPE, dtype=bool)
        result = extract_meshes_for_step({"empty": mask}, SPACING)
        assert result["empty"] is None


# ===================================================================
# 3. Multiple masks — all keys present
# ===================================================================


class TestExtractMeshesForStepMultiple:
    """extract_meshes_for_step handles multiple masks."""

    def test_multiple_masks(self):
        masks = {
            "LA": _make_sphere(SHAPE, (12, 16, 16), 5),
            "LV": _make_sphere(SHAPE, (20, 16, 16), 5),
            "Pericardium": _make_sphere(SHAPE, (16, 16, 16), 13),
        }
        result = extract_meshes_for_step(masks, SPACING)
        assert set(result.keys()) == {"LA", "LV", "Pericardium"}
        for name in ("LA", "LV", "Pericardium"):
            verts, faces = result[name]
            assert verts.ndim == 2 and verts.shape[1] == 3
            assert faces.ndim == 2 and faces.shape[1] == 3
            assert len(verts) > 0
            assert len(faces) > 0


# ===================================================================
# 4. extract_interactive_meshes creates PLY files
# ===================================================================


class TestExtractInteractiveMeshesPLY:
    """extract_interactive_meshes creates all expected .ply files."""

    def test_creates_ply_files(self, tmp_path):
        state = _make_pipeline_state()
        result = extract_interactive_meshes(state, str(tmp_path))

        assert isinstance(result, dict)
        assert set(result.keys()) == {"step2_anchors", "step5_partition", "step7_final"}

        # step2_anchors: 6 anchors + Pericardium = 7 .ply files
        step2_dir = os.path.join(str(tmp_path), "meshes", "step2_anchors")
        for name in _CANONICAL_ANCHORS + ["Pericardium"]:
            ply_path = os.path.join(step2_dir, f"{name}.ply")
            assert os.path.isfile(ply_path), f"Missing PLY: {ply_path}"

        # step5_partition: 6 anchors + Pericardium = 7 .ply files
        step5_dir = os.path.join(str(tmp_path), "meshes", "step5_partition")
        for name in _CANONICAL_ANCHORS + ["Pericardium"]:
            ply_path = os.path.join(step5_dir, f"{name}.ply")
            assert os.path.isfile(ply_path), f"Missing PLY: {ply_path}"

        # step7_final: LA_chamber + Pericardium + LA_fat = 3 .ply files
        step7_dir = os.path.join(str(tmp_path), "meshes", "step7_final")
        for name in ["LA_chamber", "Pericardium", "LA_fat"]:
            ply_path = os.path.join(step7_dir, f"{name}.ply")
            assert os.path.isfile(ply_path), f"Missing PLY: {ply_path}"


# ===================================================================
# 5. extract_interactive_meshes creates NIfTI files
# ===================================================================


class TestExtractInteractiveMeshesNifti:
    """extract_interactive_meshes creates .nii.gz files alongside .ply files."""

    def test_creates_nifti_files(self, tmp_path):
        state = _make_pipeline_state()
        extract_interactive_meshes(state, str(tmp_path))

        # step2_anchors
        step2_dir = os.path.join(str(tmp_path), "meshes", "step2_anchors")
        for name in _CANONICAL_ANCHORS + ["Pericardium"]:
            nii_path = os.path.join(step2_dir, f"{name}.nii.gz")
            assert os.path.isfile(nii_path), f"Missing NIfTI: {nii_path}"

        # step5_partition
        step5_dir = os.path.join(str(tmp_path), "meshes", "step5_partition")
        for name in _CANONICAL_ANCHORS + ["Pericardium"]:
            nii_path = os.path.join(step5_dir, f"{name}.nii.gz")
            assert os.path.isfile(nii_path), f"Missing NIfTI: {nii_path}"

        # step7_final
        step7_dir = os.path.join(str(tmp_path), "meshes", "step7_final")
        for name in ["LA_chamber", "Pericardium", "LA_fat"]:
            nii_path = os.path.join(step7_dir, f"{name}.nii.gz")
            assert os.path.isfile(nii_path), f"Missing NIfTI: {nii_path}"


# ===================================================================
# 6. PLY files are valid (correct header + vertex/face counts)
# ===================================================================


class TestPLYFileValidity:
    """Generated PLY files have correct format and content."""

    def test_ply_files_are_valid(self, tmp_path):
        state = _make_pipeline_state()
        extract_interactive_meshes(state, str(tmp_path))

        # Check one representative PLY from each step.
        ply_paths = [
            os.path.join(str(tmp_path), "meshes", "step2_anchors", "LA.ply"),
            os.path.join(str(tmp_path), "meshes", "step5_partition", "LA.ply"),
            os.path.join(str(tmp_path), "meshes", "step7_final", "LA_fat.ply"),
        ]

        for ply_path in ply_paths:
            assert os.path.isfile(ply_path)
            with open(ply_path) as f:
                lines = f.read().splitlines()

            # Header checks
            assert lines[0] == "ply"
            assert lines[1] == "format ascii 1.0"
            # Find element counts
            vertex_line = next(l for l in lines if l.startswith("element vertex "))
            face_line = next(l for l in lines if l.startswith("element face "))
            n_verts = int(vertex_line.split()[-1])
            n_faces = int(face_line.split()[-1])

            end_header_idx = lines.index("end_header")
            data_lines = lines[end_header_idx + 1:]

            # Vertex count matches
            vertex_data = data_lines[:n_verts]
            assert len(vertex_data) == n_verts
            for vline in vertex_data:
                parts = vline.split()
                assert len(parts) == 3
                float(parts[0]), float(parts[1]), float(parts[2])

            # Face count matches
            face_data = data_lines[n_verts:n_verts + n_faces]
            assert len(face_data) == n_faces
            for fline in face_data:
                parts = fline.split()
                assert len(parts) == 4
                assert parts[0] == "3"
                int(parts[1]), int(parts[2]), int(parts[3])

            # No extra data lines
            assert len(data_lines) == n_verts + n_faces


# ===================================================================
# 7. Module is importable from package
# ===================================================================


class TestImportable:
    """Functions are importable from the la_fat.mesh_extractor module."""

    def test_importable_from_package(self):
        from la_fat.mesh_extractor import extract_meshes_for_step

        assert callable(extract_meshes_for_step)
