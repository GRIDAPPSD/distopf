"""Regression tests for agent-based ADMM boundary coordination."""

import pandas as pd
import pytest

import distopf as opf
from distopf.distributed.spatial.admm_agents import (
    ADMMAgent,
    _average_boundary_frames,
    create_admm_agents,
)
from distopf.distributed.spatial.decompose import decompose
from distopf.distributed.spatial.messaging import AreaAgent, BoundaryMessage, safe_area_solve
from distopf.distributed.spatial.execution import _finalize_result


AREA_INFO = {
    "area1": {
        "up_areas": [],
        "down_areas": ["area2", "area3"],
        "up_buses": ["150"],
        "down_buses": ["152", "135"],
    },
    "area2": {
        "up_areas": ["area1"],
        "down_areas": ["area4"],
        "up_buses": ["152"],
        "down_buses": ["160"],
    },
    "area3": {
        "up_areas": ["area1"],
        "down_areas": [],
        "up_buses": ["135"],
        "down_buses": [],
    },
    "area4": {
        "up_areas": ["area2"],
        "down_areas": [],
        "up_buses": ["160"],
        "down_buses": [],
    },
}


def test_area_agent_solve_uses_neutral_safe_solver(monkeypatch):
    calls = {}

    def fake_solver(name, case, objective, **kwargs):
        calls.update(name=name, case=case, objective=objective, kwargs=kwargs)
        return "result"

    monkeypatch.setattr("distopf.distributed.spatial.messaging.safe_area_solve", fake_solver)
    agent = AreaAgent("area", object(), [])

    assert agent.solve("objective", limit=1) == "result"
    assert calls["name"] == "area"
    assert calls["objective"] == "objective"


def test_messaging_owns_area_agent_types():
    assert AreaAgent is not None
    assert BoundaryMessage is not None
    assert safe_area_solve is not None


def test_area_iteration_summaries_are_attached_by_finalizer():
    summaries = pd.DataFrame(
        {
            "iteration": [1, 1],
            "area": ["area1", "area2"],
            "objective": [3.0, 2.5],
            "solve_time": [0.2, 0.3],
            "converged": [True, False],
            "solve_failed": [False, True],
            "solver_status": ["optimal", None],
            "termination_condition": ["optimal", None],
            "has_result": [True, False],
        }
    )

    result = _finalize_result(
        None,
        case=object(),
        objective_value=3.0,
        converged=False,
        iterations=1,
        runtime=0.5,
        solver_status="max_iterations",
        termination_condition="max_iterations",
        area_results={},
        boundary_errors=[],
        area_iteration_summaries=summaries,
        parallel_used=False,
        area_solve_failed=True,
        failed_areas={"area2"},
    )

    pd.testing.assert_frame_equal(result.area_iteration_summaries, summaries)
    pd.testing.assert_frame_equal(
        result.raw_result["area_iteration_summaries"], summaries
    )


def test_iteration_summaries_are_attached_by_finalizer():
    summaries = pd.DataFrame(
        {
            "iteration": [1, 2],
            "objective": [3.0, 2.5],
            "solve_time": [0.2, 0.3],
            "boundary_error": [float("nan"), 0.01],
            "primal_consensus_residual": [0.2, 0.01],
        }
    )

    result = _finalize_result(
        None,
        case=object(),
        objective_value=2.5,
        converged=False,
        iterations=2,
        runtime=0.5,
        solver_status="max_iterations",
        termination_condition="max_iterations",
        area_results={},
        boundary_errors=[0.01],
        iteration_summaries=summaries,
        parallel_used=False,
        area_solve_failed=False,
        failed_areas=set(),
    )

    pd.testing.assert_frame_equal(result.iteration_summaries, summaries)
    pd.testing.assert_frame_equal(result.raw_result["iteration_summaries"], summaries)
    assert result.boundary_error_per_iter == [0.01]


def test_admm_target_actual_residual_uses_maximum_frame_difference():
    local = pd.DataFrame(
        {
            "name": ["area"],
            "t": [0],
            "a": [1.0],
            "b": [0.9],
            "c": [1.1],
        }
    )
    target = local.copy()
    target.loc[0, "b"] = 0.7

    assert ADMMAgent._frame_residual(local, target) == pytest.approx(0.2)


def test_unscaled_admm_uses_consensus_without_updating_dual():
    local = pd.DataFrame(
        {
            "name": ["area"],
            "t": [0],
            "a": [1.0],
            "b": [0.9],
            "c": [1.1],
        }
    )
    remote = local.copy()
    remote.loc[0, ["a", "b", "c"]] = [0.8, 1.0, 1.3]
    written = []
    dual_updates = []

    agent = ADMMAgent("area", object(), [], scaled=False)
    pair = type(
        "Pair",
        (),
        {
            "neighbor": "remote",
            "variable": "v",
            "local": local,
            "remote": remote,
            "dual": local.copy(),
            "set_dual": dual_updates.append,
            "write_target": written.append,
        },
    )()

    agent._process_interface_pair(pair)

    pd.testing.assert_frame_equal(written[0], _average_boundary_frames(local, remote))
    assert dual_updates == []
    assert agent.pending_targets[("remote", "v")].equals(written[0])


def test_algorithm_neutral_finalizer_preserves_identity():
    result = _finalize_result(
        None,
        case=object(),
        objective_value=1.0,
        converged=True,
        iterations=1,
        runtime=0.1,
        solver_status="optimal",
        termination_condition="converged",
        area_results={},
        boundary_errors=[],
        parallel_used=False,
        area_solve_failed=False,
        failed_areas=set(),
        solver="admm",
        backend="admm",
        metadata_prefix="admm",
    )

    assert result.solver == "admm"
    assert result.backend == "admm"
    assert result.raw_result["admm_iterations"] == 1


def test_admm_upstream_voltage_target_updates_child_schedule():
    """A parent-boundary target must update the child's IN-bus schedule."""
    case = opf.create_case(
        opf.CASES_DIR / "csv" / "ieee123_30der",
        n_steps=1,
        ignore_bat=True,
        ignore_schedule=False,
        ignore_gen=False,
    )
    cases = decompose(
        case,
        {name: info["up_buses"][0] for name, info in AREA_INFO.items()},
    )
    agents = create_admm_agents(cases, AREA_INFO)
    child = agents["area4"]

    target = pd.DataFrame(
        {
            "name": ["area2"],
            "t": [0],
            "a": [0.991],
            "b": [0.992],
            "c": [0.993],
        }
    )

    child._write_v_up_target(target)

    schedule = child.case.schedules.set_index("time")
    assert schedule.loc[0, ["v_a", "v_b", "v_c"]].tolist() == [
        0.991,
        0.992,
        0.993,
    ]
