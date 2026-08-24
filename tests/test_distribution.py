"""Tests for the package packaging and distribution.

Verifies that pyproject.toml, setup.py, Dockerfile, entrypoint.sh, and requirements.txt
are well-formed, contain the expected configuration, and define standard entry points.
"""

from __future__ import annotations

import os
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCKER_DIR = os.path.join(REPO_ROOT, "docker")
DOCKERFILE = os.path.join(DOCKER_DIR, "Dockerfile")
ENTRYPOINT_SH = os.path.join(DOCKER_DIR, "entrypoint.sh")
PYPROJECT_TOML = os.path.join(REPO_ROOT, "pyproject.toml")
SETUP_PY = os.path.join(REPO_ROOT, "setup.py")
REQUIREMENTS_TXT = os.path.join(REPO_ROOT, "requirements.txt")


class TestDistributionFilesExist:
    """Core distribution files are present."""

    def test_dockerfile_exists(self):
        assert os.path.isfile(DOCKERFILE)

    def test_entrypoint_sh_exists(self):
        assert os.path.isfile(ENTRYPOINT_SH)

    def test_pyproject_toml_exists(self):
        assert os.path.isfile(PYPROJECT_TOML)

    def test_setup_py_exists(self):
        assert os.path.isfile(SETUP_PY)

    def test_requirements_txt_exists(self):
        assert os.path.isfile(REQUIREMENTS_TXT)


class TestPyProjectToml:
    """pyproject.toml is well-formed and defines console_scripts."""

    def test_defines_la_fat_script(self):
        with open(PYPROJECT_TOML, "r", encoding="utf-8") as f:
            content = f.read()
        assert 'la-fat = "la_fat.cli:main_cli"' in content
        assert 'name = "la_fat"' in content

    def test_requires_totalsegmentator(self):
        with open(PYPROJECT_TOML, "r", encoding="utf-8") as f:
            content = f.read()
        assert "TotalSegmentator" in content
        assert "SimpleITK" in content


class TestSetupPy:
    """setup.py is well-formed."""

    def test_defines_entry_point(self):
        with open(SETUP_PY, "r", encoding="utf-8") as f:
            content = f.read()
        assert "la-fat=la_fat.cli:main_cli" in content

    def test_no_legacy_gui_dependencies(self):
        with open(SETUP_PY, "r", encoding="utf-8") as f:
            content = f.read()
        assert "pyvista" not in content.lower()
        assert "panel" not in content.lower()


class TestRequirementsTxt:
    """requirements.txt contains core dependencies and no legacy visualization packages."""

    def test_core_dependencies_present(self):
        with open(REQUIREMENTS_TXT, "r", encoding="utf-8") as f:
            content = f.read()
        assert "SimpleITK" in content
        assert "numpy" in content
        assert "scipy" in content

    def test_no_legacy_gui_dependencies(self):
        with open(REQUIREMENTS_TXT, "r", encoding="utf-8") as f:
            content = f.read()
        assert "pyvista" not in content
        assert "panel" not in content


class TestDockerfile:
    """Dockerfile is well-formed."""

    def test_dockerfile_base_image(self):
        with open(DOCKERFILE, "r", encoding="utf-8") as f:
            content = f.read()
        assert "pytorch/pytorch" in content
        assert "ENTRYPOINT" in content


class TestEntrypointSh:
    """entrypoint.sh dispatches to la-fat commands."""

    def test_entrypoint_commands(self):
        with open(ENTRYPOINT_SH, "r", encoding="utf-8") as f:
            content = f.read()
        assert "la-fat" in content
        assert "batch" in content
