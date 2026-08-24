"""Tests for the unified LA Fat Segmentation CLI (src/la_fat/cli.py)."""

from __future__ import annotations

import argparse
import io
import sys
from unittest import mock

import pytest

from la_fat.cli import (
    build_cli_parser,
    handle_check,
    handle_dashboard,
    handle_run,
    main_cli,
)
from la_fat.pipeline import SegmentationResult


def test_build_cli_parser():
    """Verify that parser builds with all subcommands and rich help."""
    parser = build_cli_parser()
    assert isinstance(parser, argparse.ArgumentParser)
    assert parser.prog == "la-fat"

    # Verify subcommands exist
    subparser_actions = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(subparser_actions) == 1
    choices = subparser_actions[0].choices
    assert "run" in choices
    assert "batch" in choices
    assert "precompute" in choices
    assert "dashboard" in choices
    assert "benchmark" in choices
    assert "check" in choices


def test_cli_help_stdout(capsys):
    """Verify that --help outputs usage examples and subcommand descriptions."""
    parser = build_cli_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "la-fat run" in captured.out or "Available Commands" in captured.out
    assert "batch" in captured.out
    assert "dashboard" in captured.out


def test_cli_check_handler(capsys):
    """Verify that la-fat check executes without error and prints system info."""
    args = argparse.Namespace(command="check")
    exit_code = handle_check(args)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "SYSTEM & ENVIRONMENT DIAGNOSTICS" in captured.out
    assert "Python Version:" in captured.out
    assert "SimpleITK Version:" in captured.out


def test_cli_shorthand_patient_dispatch():
    """Verify that `la-fat 0674` or `la-fat --patient 0674` delegates to `run`."""
    mock_result = SegmentationResult(patient_id="0674", success=True)
    with mock.patch("la_fat.cli.run_fat_extraction", return_value=mock_result) as mock_run:
        with pytest.raises(SystemExit) as exc:
            main_cli(["0674"])
        assert exc.value.code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["patient_id"] == "0674"


def test_cli_run_subcommand():
    """Verify that `la-fat run --patient 1512 --no-qa` calls run_fat_extraction with correct flags."""
    mock_result = SegmentationResult(patient_id="1512", success=True)
    with mock.patch("la_fat.cli.run_fat_extraction", return_value=mock_result) as mock_run:
        with pytest.raises(SystemExit) as exc:
            main_cli(["run", "--patient", "1512", "--no-qa"])
        assert exc.value.code == 0
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["patient_id"] == "1512"
        assert mock_run.call_args.kwargs["generate_qa"] is False


def test_cli_batch_subcommand():
    """Verify that `la-fat batch --input-dir /test/dir` calls run_batch_pipeline."""
    mock_summary = {"total": 2, "succeeded": 2, "failed": 0, "cohort_dashboard_path": None}
    with mock.patch("la_fat.cli.run_batch_pipeline", return_value=mock_summary) as mock_batch:
        with pytest.raises(SystemExit) as exc:
            main_cli(["batch", "--input-dir", "/test/dir", "--no-open"])
        assert exc.value.code == 0
        mock_batch.assert_called_once()
        assert mock_batch.call_args.kwargs["input_dir"] == "/test/dir"


def test_cli_precompute_subcommand(tmp_path):
    """Verify that `la-fat precompute --patient 0674` invokes TS precompute."""
    from la_fat.ts_runner import TsPrecomputeResult
    mock_res = TsPrecomputeResult(
        patient_id="0674",
        output_dir=str(tmp_path),
        masks_saved={"LA": "la.nii.gz"},
        mask_volumes_ml={"LA": 45.0},
        errors=[],
        total_runtime_seconds=10.0,
    )
    # Create dummy raw CT file
    ct_file = tmp_path / "0674.nii.gz"
    ct_file.write_text("dummy")

    with mock.patch("la_fat.cli.run_ts_precompute", return_value=mock_res) as mock_ts:
        with pytest.raises(SystemExit) as exc:
            main_cli(["precompute", str(ct_file), "--device", "cpu"])
        assert exc.value.code == 0
        mock_ts.assert_called_once()


def test_cli_dashboard_subcommand(tmp_path):
    """Verify that `la-fat dashboard` locates HTML and calls webbrowser.open."""
    qa_html = tmp_path / "cohort_qa_viewer.html"
    qa_html.write_text("<html></html>")

    with mock.patch("webbrowser.open") as mock_open:
        with pytest.raises(SystemExit) as exc:
            main_cli(["dashboard", "--output-dir", str(tmp_path)])
        assert exc.value.code == 0
        mock_open.assert_called_once()
        assert str(qa_html) in mock_open.call_args[0][0]
