"""Integration tests verifying all consumers import from la_fat.anatomy.

Ensures no module has its own private copy of the canonical anchor list
or other constants that should be sourced from the single anatomy module.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

from la_fat import anatomy

# Modules that should import from anatomy instead of defining local constants.
_CONSUMER_MODULES = [
    "la_fat.partition_engine",
    "la_fat.mesh_extractor",
    "la_fat.qa_dashboard",
    "la_fat.pipeline",
    "la_fat.ts_runner",
    "la_fat.interactive_dashboard",
    "la_fat.pericardium_resolver",
]


class TestNoLocalAnchorLists:
    """Consumers must not define their own CANONICAL_ANCHORS list."""

    @pytest.mark.parametrize("module_name", _CONSUMER_MODULES)
    def test_no_private_canonical_anchors(self, module_name: str):
        """Module should not have a private _CANONICAL_ANCHORS list."""
        try:
            mod = importlib.import_module(module_name)
        except ImportError as exc:
            pytest.skip(f"Cannot import {module_name}: {exc}")

        # Check no module-level constant named _CANONICAL_ANCHORS
        for name, val in inspect.getmembers(mod):
            if name == "_CANONICAL_ANCHORS":
                pytest.fail(
                    f"{module_name} defines its own {name} — "
                    "should import from la_fat.anatomy.CANONICAL_ANCHORS"
                )

    @pytest.mark.parametrize("module_name", _CONSUMER_MODULES)
    def test_imports_from_anatomy(self, module_name: str):
        """Module should import CANONICAL_ANCHORS from la_fat.anatomy."""
        try:
            mod = importlib.import_module(module_name)
        except ImportError as exc:
            pytest.skip(f"Cannot import {module_name}: {exc}")

        source = inspect.getsource(mod)

        if "CANONICAL_ANCHORS" in source:
            # If it references CANONICAL_ANCHORS, it must import it from anatomy.
            if "from la_fat.anatomy import" not in source:
                pytest.fail(
                    f"{module_name} references CANONICAL_ANCHORS but "
                    "does not import from la_fat.anatomy"
                )


class TestAnatomyCanonicalSource:
    """The anatomy module is the single source of truth."""

    def test_every_consumer_references_anatomy(self):
        """All consumer modules depend on la_fat.anatomy in their imports.

        This test verifies that the anatomy module is imported by each
        consumer — either directly or transitively.
        """
        for module_name in _CONSUMER_MODULES:
            try:
                mod = importlib.import_module(module_name)
            except ImportError as exc:
                pytest.skip(f"Cannot import {module_name}: {exc}")

            source = inspect.getsource(mod)
            # Every consumer must either import from anatomy directly
            # or import a constant that itself originates from anatomy.
            # At minimum, the module source should mention "anatomy".
            if "from la_fat.anatomy" not in source:
                pytest.fail(
                    f"{module_name} does not import from la_fat.anatomy"
                )


class TestVoxelVolumeMlSingleSource:
    """voxel_volume_ml utility must be imported from anatomy, not duplicated."""

    def test_no_inline_voxel_volume_ml_formula(self):
        """No module should have the inline formula spacing[0]*spacing[1]*spacing[2]/1000."""
        for module_name in _CONSUMER_MODULES:
            try:
                mod = importlib.import_module(module_name)
            except ImportError as exc:
                continue

            source = inspect.getsource(mod)
            # The raw formula pattern should not appear (only the import of the function).
            if "spacing[0] * spacing[1] * spacing[2] / 1000.0" in source:
                pytest.fail(
                    f"{module_name} has the inline voxel volume formula — "
                    "should call la_fat.anatomy.voxel_volume_ml(spacing) instead"
                )

    def test_anatomy_defines_voxel_volume_ml(self):
        assert hasattr(anatomy, "voxel_volume_ml")
        assert callable(anatomy.voxel_volume_ml)
