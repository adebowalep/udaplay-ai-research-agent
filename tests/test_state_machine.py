"""Tests for udaplay.state_machine — StateMachine, Step, Run, Snapshot."""
from typing import TypedDict
import pytest

from udaplay.state_machine import (
    StateMachine, Step, EntryPoint, Termination, Run, Snapshot, Resource,
)


# ---------------------------------------------------------------------------
# Minimal state schema for testing
# ---------------------------------------------------------------------------

class CounterState(TypedDict):
    value: int
    log: list


def _increment(state: CounterState) -> dict:
    return {"value": state.get("value", 0) + 1, "log": state.get("log", []) + ["increment"]}


def _double(state: CounterState) -> dict:
    return {"value": state.get("value", 0) * 2, "log": state.get("log", []) + ["double"]}


def _build_machine(*step_fns):
    """Build a linear state machine from a sequence of step functions."""
    machine = StateMachine[CounterState](CounterState)
    entry = EntryPoint[CounterState]()
    term = Termination[CounterState]()

    steps = [Step[CounterState](f"step_{i}", fn) for i, fn in enumerate(step_fns)]
    all_steps = [entry] + steps + [term]

    machine.add_steps(all_steps)
    machine.connect(entry, steps[0])
    for a, b in zip(steps, steps[1:]):
        machine.connect(a, b)
    machine.connect(steps[-1], term)
    return machine


class TestStateMachineExecution:
    def test_single_step_executes(self):
        machine = _build_machine(_increment)
        run = machine.run({"value": 0, "log": []})
        assert run.get_final_state()["value"] == 1

    def test_two_steps_execute_in_order(self):
        machine = _build_machine(_increment, _double)
        run = machine.run({"value": 3, "log": []})
        # 3 → +1 = 4 → *2 = 8
        assert run.get_final_state()["value"] == 8

    def test_step_log_captures_order(self):
        machine = _build_machine(_increment, _double)
        run = machine.run({"value": 0, "log": []})
        assert run.get_final_state()["log"] == ["increment", "double"]

    def test_state_accumulates(self):
        machine = _build_machine(_increment, _increment, _increment)
        run = machine.run({"value": 0, "log": []})
        assert run.get_final_state()["value"] == 3

    def test_no_entry_point_raises(self):
        machine = StateMachine[CounterState](CounterState)
        s = Step[CounterState]("s", _increment)
        t = Termination[CounterState]()
        machine.add_steps([s, t])
        machine.connect(s, t)
        with pytest.raises(Exception, match="EntryPoint"):
            machine.run({"value": 0, "log": []})

    def test_missing_transition_raises(self):
        machine = StateMachine[CounterState](CounterState)
        entry = EntryPoint[CounterState]()
        s = Step[CounterState]("s", _increment)
        machine.add_steps([entry, s])
        machine.connect(entry, s)
        # No transition from s — should raise
        with pytest.raises(Exception):
            machine.run({"value": 0, "log": []})

    def test_empty_initial_state_raises(self):
        machine = _build_machine(_increment)
        with pytest.raises(ValueError):
            machine.run({})     # no keys match schema


class TestRunObject:
    def test_run_has_snapshots(self):
        machine = _build_machine(_increment)
        run = machine.run({"value": 0, "log": []})
        assert len(run.snapshots) > 0

    def test_run_has_timestamps(self):
        machine = _build_machine(_increment)
        run = machine.run({"value": 0, "log": []})
        assert run.start_timestamp is not None
        assert run.end_timestamp is not None
        assert run.end_timestamp >= run.start_timestamp

    def test_run_id_is_string(self):
        machine = _build_machine(_increment)
        run = machine.run({"value": 0, "log": []})
        assert isinstance(run.run_id, str)
        assert len(run.run_id) > 0

    def test_get_final_state(self):
        machine = _build_machine(_increment)
        run = machine.run({"value": 5, "log": []})
        fs = run.get_final_state()
        assert fs is not None
        assert fs["value"] == 6

    def test_empty_run_final_state_is_none(self):
        run = Run.create()
        assert run.get_final_state() is None


class TestConditionalTransition:
    def test_conditional_routing(self):
        """State machine routes to different steps based on state value."""

        class RouteState(TypedDict):
            path: str
            value: int

        def set_low(state):
            return {"path": "low"}

        def set_high(state):
            return {"path": "high"}

        machine = StateMachine[RouteState](RouteState)
        entry = EntryPoint[RouteState]()
        low  = Step[RouteState]("low",  set_low)
        high = Step[RouteState]("high", set_high)
        term = Termination[RouteState]()

        machine.add_steps([entry, low, high, term])
        machine.connect(entry, [low, high], lambda s: low if s["value"] < 5 else high)
        machine.connect(low, term)
        machine.connect(high, term)

        run_low  = machine.run({"path": "", "value": 2})
        run_high = machine.run({"path": "", "value": 9})

        assert run_low.get_final_state()["path"] == "low"
        assert run_high.get_final_state()["path"] == "high"


class TestResourceInjection:
    def test_step_receives_resource(self):
        """Steps that accept (state, resource) should get the resource vars."""

        class ResState(TypedDict):
            result: str

        def use_resource(state: ResState, resource: Resource) -> dict:
            multiplier = resource.vars.get("multiplier", 1)
            return {"result": f"x{multiplier}"}

        machine = StateMachine[ResState](ResState)
        entry = EntryPoint[ResState]()
        s     = Step[ResState]("s", use_resource)
        term  = Termination[ResState]()

        machine.add_steps([entry, s, term])
        machine.connect(entry, s)
        machine.connect(s, term)

        resource = Resource(vars={"multiplier": 7})
        run = machine.run({"result": ""}, resource=resource)
        assert run.get_final_state()["result"] == "x7"
