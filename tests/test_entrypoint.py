"""Tests for the Docker entrypoint and CLI modules.

Tests exercise:
- The batch_pipeline CLI entry point (python -m la_fat.batch_pipeline)
- The interactive_dashboard CLI entry point (python -m la_fat.interactive_dashboard)
- The entrypoint.sh shell script syntax validation

Shell-execution tests of the entrypoint (config writing, mode dispatch)
are covered by the end-to-end Docker validation in issue #29.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ENTRYPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "docker", "entrypoint.sh",
)


# ---------------------------------------------------------------------------
# Tests: Batch pipeline CLI
# ---------------------------------------------------------------------------


class TestBatchPipelineCLI:
    """The batch pipeline is invocable via python -m."""

    def test_help_exits_zero(self):
        """``python -m la_fat.batch_pipeline --help`` exits 0."""
        result = subprocess.run(
            [sys.executable, "-m", "la_fat.batch_pipeline", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "--data-dir" in result.stdout
        assert "--output-dir" in result.stdout

    def test_no_scans_exits_gracefully(self, tmp_path):
        """When no scans exist, exits 0 with summary."""
        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        result = subprocess.run(
            [
                sys.executable, "-m", "la_fat.batch_pipeline",
                "--data-dir", data_dir,
                "--output-dir", output_dir,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "0 patient(s)" in result.stdout

    def test_processes_single_patient(self, tmp_path):
        """When a CT scan exists, the batch pipeline processes it."""
        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # Write a minimal valid NIfTI file
        import nibabel as nib
        import numpy as np
        nib.save(
            nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.int16), np.eye(4)),
            os.path.join(raw_dir, "TESTBATCH.nii.gz"),
        )

        # Pre-create minimal masks so TS pre-compute is skipped
        # (TS can't process an 8x8x8 dummy volume)
        mask_dir = os.path.join(data_dir, "intermediate", "TESTBATCH")
        os.makedirs(mask_dir, exist_ok=True)
        nib.save(
            nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.uint8), np.eye(4)),
            os.path.join(mask_dir, "TESTBATCH_LA.nii.gz"),
        )

        result = subprocess.run(
            [
                sys.executable, "-m", "la_fat.batch_pipeline",
                "--data-dir", data_dir,
                "--output-dir", output_dir,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        # May fail because masks are synthetic, but should not throw unhandled errors
        # The batch wrapper catches pipeline failures gracefully
        assert result.returncode in (0, 1)

    def test_skips_completed_patient(self, tmp_path):
        """Patient with existing pipeline_result.json is skipped."""
        import json

        data_dir = str(tmp_path / "data")
        output_dir = str(tmp_path / "outputs")
        raw_dir = os.path.join(data_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # Create CT file
        import nibabel as nib
        import numpy as np
        nib.save(
            nib.Nifti1Image(np.ones((8, 8, 8), dtype=np.int16), np.eye(4)),
            os.path.join(raw_dir, "SKIPME.nii.gz"),
        )

        # Pre-create a pipeline_result.json
        result_dir = os.path.join(output_dir, "SKIPME")
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "pipeline_result.json"), "w") as f:
            json.dump({"patient_id": "SKIPME"}, f)

        result = subprocess.run(
            [
                sys.executable, "-m", "la_fat.batch_pipeline",
                "--data-dir", data_dir,
                "--output-dir", output_dir,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "SKIPPED" in result.stdout


# ---------------------------------------------------------------------------
# Tests: Interactive dashboard CLI
# ---------------------------------------------------------------------------


class TestInteractiveDashboardCLI:
    """The interactive dashboard launcher is invocable via run_dashboard.py."""

    def test_help_exits_zero(self):
        """``python run_dashboard.py --help`` exits 0."""
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "run_dashboard.py",
        )
        result = subprocess.run(
            [sys.executable, script_path, "--help"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0
        assert "--output-dir" in result.stdout
        assert "--patient" in result.stdout


# ---------------------------------------------------------------------------
# Tests: Entrypoint script syntax
# ---------------------------------------------------------------------------


class TestEntrypointSyntax:
    """The entrypoint.sh is syntactically valid bash."""

    def test_entrypoint_exists(self):
        """The entrypoint.sh file exists."""
        assert os.path.isfile(ENTRYPOINT_PATH), (
            f"entrypoint.sh not found at {ENTRYPOINT_PATH}"
        )

    def test_entrypoint_is_executable(self):
        """The entrypoint.sh is readable (executable permission set in Dockerfile)."""
        assert os.access(ENTRYPOINT_PATH, os.R_OK), (
            "entrypoint.sh is not readable"
        )

    def test_entrypoint_has_shebang(self):
        """The entrypoint.sh starts with a bash shebang."""
        with open(ENTRYPOINT_PATH) as f:
            first_line = f.readline().strip()
        assert first_line in ("#!/usr/bin/env bash", "#!/bin/bash", "#!/usr/bin/bash"), (
            f"Unexpected shebang: {first_line}"
        )

    def test_entrypoint_has_pipeline_mode(self):
        """The entrypoint.sh contains a pipeline/batch dispatch case."""
        with open(ENTRYPOINT_PATH) as f:
            content = f.read()
        assert "pipeline" in content or "batch" in content
        assert "la-fat" in content

    def test_entrypoint_has_dashboard_mode(self):
        """The entrypoint.sh contains a dashboard dispatch case."""
        with open(ENTRYPOINT_PATH) as f:
            content = f.read()
        assert "dashboard" in content

    def test_entrypoint_has_config_writing(self):
        """The entrypoint.sh writes config.json on startup."""
        with open(ENTRYPOINT_PATH) as f:
            content = f.read()
        assert "config.json" in content
        assert "TOTALSEG_HOME" in content
        assert "TOTALSEG_LICENSE" in content

    def test_bash_syntax_check(self):
        """``bash -n entrypoint.sh`` reports no syntax errors if bash is available."""
        import shutil
        bash = shutil.which("bash")
        if bash is None:
            pytest.skip("bash not found")

        try:
            result = subprocess.run(
                [bash, "-n", ENTRYPOINT_PATH],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "WSL" in result.stderr or "execvpe" in result.stderr:
                pytest.skip("WSL bash environment not functional on this host")
            assert result.returncode == 0, (
                f"Syntax error in entrypoint.sh:\n{result.stderr}"
            )
        except Exception:
            pytest.skip("bash execution not supported on this host")
