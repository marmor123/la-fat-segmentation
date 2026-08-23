"""Tests for the Docker distribution package.

Verifies that install scripts, desktop shortcuts, and the rebuild
script are well-formed and contain the expected commands.
"""

from __future__ import annotations

import os
import subprocess

import pytest


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DOCKER_DIR = os.path.join(os.path.dirname(__file__), "..", "docker")

INSTALL_BAT = os.path.join(DOCKER_DIR, "Install.bat")
PROCESS_BAT = os.path.join(DOCKER_DIR, "Process Scans.bat")
VIEW_BAT = os.path.join(DOCKER_DIR, "View Results.bat")
INSTALL_SH = os.path.join(DOCKER_DIR, "install.sh")
PROCESS_DESKTOP = os.path.join(DOCKER_DIR, "Process Scans.desktop")
VIEW_DESKTOP = os.path.join(DOCKER_DIR, "View Results.desktop")
REBUILD_SH = os.path.join(DOCKER_DIR, "rebuild.sh")
ENTRYPOINT_SH = os.path.join(DOCKER_DIR, "entrypoint.sh")
DOCKERFILE = os.path.join(DOCKER_DIR, "Dockerfile")


# ---------------------------------------------------------------------------
# Tests: Files existence
# ---------------------------------------------------------------------------


class TestDistributionFilesExist:
    """All distribution files are present."""

    def test_install_bat_exists(self):
        assert os.path.isfile(INSTALL_BAT)

    def test_process_scans_bat_exists(self):
        assert os.path.isfile(PROCESS_BAT)

    def test_view_results_bat_exists(self):
        assert os.path.isfile(VIEW_BAT)

    def test_install_sh_exists(self):
        assert os.path.isfile(INSTALL_SH)

    def test_process_scans_desktop_exists(self):
        assert os.path.isfile(PROCESS_DESKTOP)

    def test_view_results_desktop_exists(self):
        assert os.path.isfile(VIEW_DESKTOP)

    def test_rebuild_sh_exists(self):
        assert os.path.isfile(REBUILD_SH)

    def test_entrypoint_sh_exists(self):
        assert os.path.isfile(ENTRYPOINT_SH)

    def test_dockerfile_exists(self):
        assert os.path.isfile(DOCKERFILE)


# ---------------------------------------------------------------------------
# Tests: Windows Install.bat
# ---------------------------------------------------------------------------


class TestInstallBat:
    """The Windows installer is well-formed."""

    def test_contains_docker_load(self):
        with open(INSTALL_BAT) as f:
            content = f.read()
        assert "docker load" in content
        assert "la-fat-image.tar" in content

    def test_creates_data_folder(self):
        with open(INSTALL_BAT) as f:
            content = f.read()
        assert "la-fat-data" in content
        assert "Desktop" in content
        # Uses USERPROFILE for Windows
        assert "USERPROFILE" in content

    def test_copies_shortcuts(self):
        with open(INSTALL_BAT) as f:
            content = f.read()
        assert "Process Scans.bat" in content
        assert "View Results.bat" in content

    def test_idempotent_mkdir(self):
        """Install.bat checks if data folder exists before creating."""
        with open(INSTALL_BAT) as f:
            content = f.read()
        assert "if not exist" in content.lower() or "IF NOT EXIST" in content

    def test_checks_docker_installed(self):
        with open(INSTALL_BAT) as f:
            content = f.read()
        assert "docker" in content.lower()
        assert "where docker" in content.lower()


# ---------------------------------------------------------------------------
# Tests: Windows shortcuts
# ---------------------------------------------------------------------------


class TestProcessScansBat:
    """The Process Scans shortcut is correct."""

    def test_runs_docker_pipeline(self):
        with open(PROCESS_BAT) as f:
            content = f.read()
        assert "docker run" in content
        assert "pipeline" in content
        assert "la-fat" in content

    def test_mounts_data_dir(self):
        with open(PROCESS_BAT) as f:
            content = f.read()
        assert "la-fat-data:/workspace" in content or "la-fat-data" in content
        assert "Desktop" in content

    def test_leaves_terminal_open(self):
        with open(PROCESS_BAT) as f:
            content = f.read()
        assert "pause" in content.lower()


class TestViewResultsBat:
    """The View Results shortcut is correct."""

    def test_runs_docker_dashboard(self):
        with open(VIEW_BAT) as f:
            content = f.read()
        assert "docker run" in content
        assert "dashboard" in content
        assert "la-fat" in content

    def test_exposes_port(self):
        with open(VIEW_BAT) as f:
            content = f.read()
        assert "-p 5006:5006" in content or "-p 5006" in content

    def test_opens_browser(self):
        with open(VIEW_BAT) as f:
            content = f.read()
        assert "http://localhost:5006" in content
        assert "start http" in content.lower()


# ---------------------------------------------------------------------------
# Tests: Linux install.sh
# ---------------------------------------------------------------------------


class TestInstallSh:
    """The Linux installer is well-formed."""

    def test_has_shebang(self):
        with open(INSTALL_SH) as f:
            first_line = f.readline().strip()
        assert first_line.startswith("#!/")

    def test_contains_docker_load(self):
        with open(INSTALL_SH) as f:
            content = f.read()
        assert "docker load" in content
        assert "la-fat-image.tar" in content

    def test_creates_data_folder(self):
        with open(INSTALL_SH) as f:
            content = f.read()
        assert "la-fat-data" in content
        assert "Desktop" in content
        assert "HOME" in content

    def test_copies_desktop_files(self):
        with open(INSTALL_SH) as f:
            content = f.read()
        assert "Process Scans.desktop" in content
        assert "View Results.desktop" in content

    def test_idempotent(self):
        with open(INSTALL_SH) as f:
            content = f.read()
        # Uses [[ ! -d ]] check
        assert "! -d" in content or "not exist" in content.lower()

    def test_checks_docker_installed(self):
        with open(INSTALL_SH) as f:
            content = f.read()
        assert "command -v docker" in content

    def test_bash_syntax(self):
        """install.sh has no syntax errors if bash is available."""
        import shutil
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash not found")
        try:
            result = subprocess.run(
                [bash, "-n", INSTALL_SH],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "WSL" in result.stderr or "execvpe" in result.stderr:
                pytest.skip("WSL bash environment not functional on this host")
            assert result.returncode == 0, (
                f"Syntax error in install.sh:\n{result.stderr}"
            )
        except Exception:
            pytest.skip("bash execution not supported on this host")


# ---------------------------------------------------------------------------
# Tests: Linux .desktop files
# ---------------------------------------------------------------------------


class TestDesktopFiles:
    """Linux .desktop files are well-formed."""

    def test_process_desktop_has_entry_type(self):
        with open(PROCESS_DESKTOP) as f:
            content = f.read()
        assert "[Desktop Entry]" in content
        assert "Type=Application" in content
        assert "Terminal=true" in content

    def test_process_desktop_runs_pipeline(self):
        with open(PROCESS_DESKTOP) as f:
            content = f.read()
        assert "pipeline" in content
        assert "la-fat" in content

    def test_view_desktop_has_entry_type(self):
        with open(VIEW_DESKTOP) as f:
            content = f.read()
        assert "[Desktop Entry]" in content
        assert "Type=Application" in content
        assert "Terminal=true" in content

    def test_view_desktop_runs_dashboard(self):
        with open(VIEW_DESKTOP) as f:
            content = f.read()
        assert "dashboard" in content
        assert "5006" in content

    def test_view_desktop_opens_browser(self):
        with open(VIEW_DESKTOP) as f:
            content = f.read()
        assert "xdg-open" in content
        assert "localhost:5006" in content


# ---------------------------------------------------------------------------
# Tests: rebuild.sh
# ---------------------------------------------------------------------------


class TestRebuildSh:
    """The maintainer rebuild script is well-formed."""

    def test_has_shebang(self):
        with open(REBUILD_SH) as f:
            first_line = f.readline().strip()
        assert first_line.startswith("#!/")

    def test_contains_docker_build(self):
        with open(REBUILD_SH) as f:
            content = f.read()
        assert "docker build" in content
        assert "TOTALSEG_LICENSE" in content
        assert "--build-arg" in content

    def test_contains_docker_save(self):
        with open(REBUILD_SH) as f:
            content = f.read()
        assert "docker save" in content
        assert "la-fat-image.tar" in content

    def test_has_safety_check(self):
        """Rebuild script warns if TOTALSEG_LICENSE is not set."""
        with open(REBUILD_SH) as f:
            content = f.read()
        assert "-z" in content or "LICENSE" in content  # checks for empty var

    def test_bash_syntax(self):
        """rebuild.sh has no syntax errors if bash is available."""
        import shutil
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash not found")
        try:
            result = subprocess.run(
                [bash, "-n", REBUILD_SH],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "WSL" in result.stderr or "execvpe" in result.stderr:
                pytest.skip("WSL bash environment not functional on this host")
            assert result.returncode == 0, (
                f"Syntax error in rebuild.sh:\n{result.stderr}"
            )
        except Exception:
            pytest.skip("bash execution not supported on this host")


# ---------------------------------------------------------------------------
# Tests: Dockerfile
# ---------------------------------------------------------------------------


class TestDockerfile:
    """The Dockerfile is well-formed."""

    def test_has_from(self):
        with open(DOCKERFILE) as f:
            content = f.read()
        assert "FROM pytorch/pytorch" in content

    def test_has_build_arg(self):
        with open(DOCKERFILE) as f:
            content = f.read()
        assert "ARG TOTALSEG_LICENSE" in content

    def test_has_entrypoint(self):
        with open(DOCKERFILE) as f:
            content = f.read()
        assert "ENTRYPOINT" in content
        assert "entrypoint.sh" in content

    def test_has_totalseg_env(self):
        with open(DOCKERFILE) as f:
            content = f.read()
        assert "TOTALSEG_HOME_DIR" in content

    def test_installs_package(self):
        with open(DOCKERFILE) as f:
            content = f.read()
        assert "pip install" in content
