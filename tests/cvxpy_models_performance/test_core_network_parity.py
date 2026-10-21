"""Parity of the vectorized network core against ``distopf.cvxpy_models``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import distopf as opf
from distopf import cvxpy_models as scalar
from distopf import cvxpy_models_performance as perf
from distopf.cvxpy_models.common import factory as sfac
from distopf.cvxpy_models.common import objectives as sobj
from distopf.cvxpy_models.network.bfm import BFMProvider as SBFM
from distopf.cvxpy_models_performance.common import factory as pfac
from distopf.cvxpy_models_performance.common import objectives as pobj

TOL = 1e-5
CASES = ["ieee13", "ieee123_30der"]


def load(name):
    return opf.create_case(opf.CASES_DIR / "csv" / name)


def scalar_model(case, linear=True, **kw):
    return sfac.create_model(
        case, core=SBFM(), devices=sfac._standard_devices(), linear=linear, **kw
    )


def perf_model(case, linear=True, **kw):
    return pfac.create_model(
        case,
        core=pfac.BFMProvider(),
        devices=pfac._standard_devices(),
        linear=linear,
        **kw,
    )


def thermal_case(margin=1.2, tight_branches=2, tight_factor=1.0):
    """ieee13 with limits ``margin`` above a baseline solve; the heaviest branches sit on it."""
    case = load("ieee13")
    case.bus_data["v_min"] = 0.5
    case.bus_data["v_max"] = 1.5
    base = perf_model(case, equality_only=True)
    perf.set_objective(base, pobj.loss_objective_rule)
    res = perf.solve(base, solver="CLARABEL", duals=False)
    apparent = {}
    for ph in ("a", "b", "c"):
        s_val = np.hypot(res.p_flow[ph].to_numpy(float), res.q_flow[ph].to_numpy(float))
        for fb, tb, value in zip(res.p_flow["fb"], res.p_flow["tb"], s_val):
            if not np.isnan(value):
                apparent[(int(fb), int(tb), ph)] = float(value)
    ranked = sorted(apparent, key=apparent.get, reverse=True)[:tight_branches]
    for ph in ("a", "b", "c"):
        case.branch_data[f"s_{ph}_max"] = np.nan
    for (fb, tb, ph), value in apparent.items():
        factor = tight_factor if (fb, tb, ph) in ranked else margin
        mask = (case.branch_data.fb == fb) & (case.branch_data.tb == tb)
        case.branch_data.loc[mask, f"s_{ph}_max"] = value * factor
    return case


def sorted_frame(df: pd.DataFrame) -> pd.DataFrame:
    keys = [c for c in df.columns if not pd.api.types.is_float_dtype(df[c])]
    return df.sort_values(keys).reset_index(drop=True)


def assert_frames_close(a, b, tol=TOL):
    assert a.shape == b.shape, (a.shape, b.shape)
    a, b = sorted_frame(a), sorted_frame(b)
    for col in a.columns:
        if pd.api.types.is_float_dtype(a[col]):
            np.testing.assert_allclose(
                a[col].to_numpy(float),
                b[col].to_numpy(float),
                atol=tol,
                rtol=tol,
                err_msg=col,
            )
        else:
            assert (a[col].astype(str) == b[col].astype(str)).all(), col


def solve_pair(case, linear=True, objective="loss_objective_rule", **kw):
    sm = scalar_model(case, linear=linear, **kw)
    pm = perf_model(case, linear=linear, **kw)
    scalar.set_objective(sm, getattr(sobj, objective))
    perf.set_objective(pm, getattr(pobj, objective))
    return (
        sm,
        pm,
        scalar.solve(sm, solver="CLARABEL"),
        perf.solve(pm, solver="CLARABEL"),
    )


def compare(rs, rp, tol=TOL):
    np.testing.assert_allclose(
        rs.objective_value, rp.objective_value, rtol=TOL, atol=TOL
    )
    for name in ("voltages", "p_flow", "q_flow", "p_load", "q_load"):
        assert_frames_close(getattr(rs, name), getattr(rp, name), tol=tol)


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize(
    "objective", ["loss_objective_rule", "voltage_min_objective_rule"]
)
def test_lindist_equality_only(case_name, objective):
    _, _, rs, rp = solve_pair(load(case_name), objective=objective, equality_only=True)
    compare(rs, rp, tol=1e-3 if objective == "voltage_min_objective_rule" else TOL)


def test_structure_matches_scalar_model():
    case = load("ieee13")
    sm, pm = scalar_model(case, linear=True), perf_model(case, linear=True)
    assert set(sm.constraint_names) == set(pm.constraint_names)
    for name in sm.constraint_names:
        scalar_rows = {
            (k if isinstance(k, tuple) else (k,))[:-1] for k in getattr(sm, name)
        }
        perf_rows = {
            k if isinstance(k, tuple) else (k,) for k in getattr(pm, name).rows
        }
        assert scalar_rows == perf_rows, name


def test_far_fewer_cvxpy_objects():
    case = load("ieee13")
    sm, pm = scalar_model(case), perf_model(case)
    assert len(pm.all_constraints()) * 20 < len(sm.all_constraints())
    assert set(pm.var_names) == set(sm.var_names)


def test_duals_match():
    _, _, rs, rp = solve_pair(load("ieee13"), equality_only=True)
    for name in ("power_balance_p", "power_balance_q", "voltage_drop"):
        assert_frames_close(rs.get_dual(name), rp.get_dual(name), tol=1e-6)


@pytest.mark.parametrize(
    "kw",
    [
        {"circular_constraints": True, "thermal_constraints": True},
        {"circular_constraints": False, "thermal_constraints": True},
        {"thermal_constraints": True, "thermal_slacks": True},
        {"voltage_slacks": True},
        {"free_swing_voltage": True},
    ],
)
def test_thermal_and_slack_options(kw):
    octagon = kw.get("circular_constraints") is False
    case = thermal_case(tight_factor=1.1 if octagon else 1.0)
    sm, pm, rs, rp = solve_pair(case, **kw)
    assert set(sm.constraint_names) == set(pm.constraint_names)
    np.testing.assert_allclose(
        rs.objective_value, rp.objective_value, rtol=TOL, atol=TOL
    )
    assert_frames_close(rs.voltages, rp.voltages)
    assert_frames_close(rs.p_flow, rp.p_flow)


def test_free_boundary_loads_structure():
    case = load("ieee13")
    sm = scalar_model(case, free_boundary_loads=True)
    pm = perf_model(case, free_boundary_loads=True)
    assert len(pm.cvr_p_load.rows) == len(sm.cvr_p_load)


def test_socp_matches_scalar():
    _, _, rs, rp = solve_pair(
        load("ieee13"), linear=False, socp_relaxation=True, equality_only=True
    )
    np.testing.assert_allclose(
        rs.objective_value, rp.objective_value, rtol=TOL, atol=TOL
    )
    assert_frames_close(rs.l_flow, rp.l_flow, tol=1e-4)


def test_sdp_matches_scalar():
    _, _, rs, rp = solve_pair(
        load("ieee13"), linear=False, sdp_relaxation=True, equality_only=True
    )
    np.testing.assert_allclose(
        rs.objective_value, rp.objective_value, rtol=1e-4, atol=1e-4
    )


def test_exact_branchflow_is_dnlp():
    case = load("ieee13")
    pm = perf_model(case, linear=False, equality_only=True)
    perf.set_objective(pm, pobj.loss_objective_rule)
    from distopf.cvxpy_models_performance.common.solvers import build_problem

    assert build_problem(pm).is_dnlp()


def test_set_current_angles_updates_parameters():
    from distopf.cvxpy_models_performance.network.bfm import set_current_angles

    pm = perf_model(load("ieee13"), linear=False, equality_only=True)
    key = next(iter(pm.branch_angle_phase_pair_set))
    set_current_angles(pm, {key: 0.5})
    i = pm.branch_angle_phase_pair_set.pos[key]
    np.testing.assert_allclose(pm.cos_d.value[i], np.cos(0.5))
    np.testing.assert_allclose(pm.sin_d.value[i], np.sin(0.5))
