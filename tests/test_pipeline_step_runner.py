"""Tests for the step-runner pattern (PipelineState, _run_steps).

These tests verify the new infrastructure that replaces the 540-line
inline step pattern with a declarative step list. Each step function
mutates a PipelineState object, and _run_steps orchestrates the loop
with error handling.
"""

from __future__ import annotations

import pytest

from la_fat.pipeline import PipelineState, _run_steps


class TestPipelineState:
    """PipelineState is a mutable dataclass for pipeline intermediate state."""

    def test_can_create_with_patient_id(self):
        """PipelineState can be created with just a patient_id."""
        state = PipelineState(patient_id="SYNTH001")
        assert state.patient_id == "SYNTH001"
        assert state.errors == []
        assert state.warnings == []
        assert state.step == 0

    def test_state_is_mutable(self):
        """PipelineState is not frozen — fields can be mutated by steps."""
        state = PipelineState(patient_id="test")
        state.errors.append("Some error")
        state.warnings.append("Some warning")
        state.patient_id = "changed"
        assert state.errors == ["Some error"]
        assert state.warnings == ["Some warning"]
        assert state.patient_id == "changed"


class TestStepRunner:
    """Tests for the _run_steps orchestrator."""

    def test_non_fatal_step_error_is_caught(self):
        """A non-fatal step that raises is caught, error is accumulated,
        and _run_steps returns None (pipeline continues)."""
        state = PipelineState(patient_id="test")

        def _failing_step(_state: PipelineState) -> None:
            raise ValueError("Something went wrong")

        result = _run_steps(state, [
            ("Test step", _failing_step, "Test context", False),
        ])

        assert result is None, (
            "_run_steps should return None when no fatal step fails"
        )
        assert len(state.errors) == 1
        assert "Test context failed: Something went wrong" in state.errors[0]

    def test_non_fatal_step_does_not_stop_pipeline(self):
        """A non-fatal step failure does not prevent subsequent steps
        from executing."""
        state = PipelineState(patient_id="test")
        marker: list[str] = []

        def _failing_step(_state: PipelineState) -> None:
            raise ValueError("Boom")

        def _good_step(_state: PipelineState) -> None:
            marker.append("executed")

        result = _run_steps(state, [
            ("Fail", _failing_step, "Failing step", False),
            ("Good", _good_step, "Good step", False),
        ])

        assert result is None
        assert len(state.errors) == 1  # only the failing step error
        assert marker == ["executed"], "Second step should have run"

    def test_fatal_step_stops_pipeline(self):
        """A fatal step failure causes _run_steps to return a partial
        PipelineResult immediately."""
        state = PipelineState(patient_id="test")
        marker: list[str] = []

        def _fatal_step(_state: PipelineState) -> None:
            raise ValueError("Fatal error")

        def _should_not_run(_state: PipelineState) -> None:
            marker.append("should_not_run")

        result = _run_steps(state, [
            ("Fatal", _fatal_step, "Fatal step", True),
            ("After fatal", _should_not_run, "After fatal", False),
        ])

        assert result is not None, "Fatal step should return early result"
        assert isinstance(result, object)  # it's a PipelineResult
        assert len(state.errors) == 1
        assert "Fatal step failed: Fatal error" in state.errors[0]
        assert marker == [], "Step after fatal should NOT run"

    def test_step_counter_increments(self):
        """The step counter in the state matches step index."""
        state = PipelineState(patient_id="test")
        counters: list[int] = []

        def _track_step(_state: PipelineState) -> None:
            counters.append(_state.step)

        _run_steps(state, [
            ("Step A", _track_step, "A", False),
            ("Step B", _track_step, "B", False),
            ("Step C", _track_step, "C", False),
        ])

        assert counters == [1, 2, 3]

    def test_step_total_is_set(self):
        """Step total on state matches the number of steps."""
        state = PipelineState(patient_id="test")

        _run_steps(state, [
            ("S1", lambda s: None, "S1", False),
            ("S2", lambda s: None, "S2", False),
        ])

        assert state.step_total == 2
