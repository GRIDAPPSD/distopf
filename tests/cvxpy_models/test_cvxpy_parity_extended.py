"""Extended numerical parity between pyomo_models and cvxpy_models.

Covers options beyond the basic LinDist objectives: penalty and cost
objectives, ADMM terms, slacks, thermal limit forms, equality_only, mixed
integer cap/reg, and multi-period runs with batteries and schedules.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import pytest

import distopf as opf
from distopf import cvxpy_models as cvx
from distopf import pyomo_models as pyo_models
from distopf.cvxpy_models.common import objectives as cvx_obj
from distopf.pyomo_models.common import objectives as pyo_obj

warnings.filterwarnings("ignore", category=FutureWarning)

TOL = 1e-4


def load(name, **kwargs):
    return opf.create_case(opf.CASES_DIR / "csv" / name, **kwargs)


def build_pair(case, model_kwargs=None, builder="lindist"):
    model_kwargs = model_kwargs or {}
    create_p = (
        pyo_models.create_lindist_model
        if builder == "lindist"
        else pyo_models.create_nl_branchflow_model
    )
    create_c = (
        cvx.create_lindist_model
        if builder == "lindist"
        else cvx.create_nl_branchflow_model
    )
    return create_p(case, **model_kwargs), create_c(case, **model_kwargs)


def solve_pair(pm, cm, pyo_rule, cvx_rule, pyo_solver="ipopt", cvx_solver=None, duals=False):
    pyo_models.set_objective(pm, pyo.Objective(rule=pyo_rule, sense=pyo.minimize))
    cvx.set_objective(cm, cvx_rule)
    return (
        pyo_models.solve(pm, solver=pyo_solver, duals=duals),
        cvx.solve(cm, solver=cvx_solver, duals=duals),
    )


def sorted_values(frame: pd.DataFrame) -> np.ndarray:
    keys = [c for c in ("fb", "tb", "device_name", "id", "t") if c in frame.columns]
    frame = frame.sort_values(keys).reset_index(drop=True)
    cols = [c for c in ("a", "b", "c", "s1", "s2", "value") if c in frame.columns]
    return frame[cols].to_numpy(dtype=float)


def assert_same(pres, cres, names, tol=TOL):
    for name in names:
        pa, ca = getattr(pres, name, None), getattr(cres, name, None)
        assert (pa is None) == (ca is None), name
        if pa is None or len(pa) == 0:
            continue
        np.testing.assert_allclose(
            sorted_values(pa), sorted_values(ca), atol=tol, rtol=tol, err_msg=name
        )


FLOWS = ("voltages", "p_flow", "q_flow")


def load_with_thermal_limits(margin=1.2, tight_branches=2, tight_factor=1.0):
    """ieee13 with branch limits from a baseline solve.

    Most limits sit ``margin`` above the baseline apparent power. The most
    heavily loaded ``tight_branches`` phases use ``tight_factor`` instead;
    1.0 puts the limit exactly on the baseline flow (active for the circle).
    The inscribed octagon only reaches about 0.924 of the circle radius, so
    octagon tests need ``tight_factor`` above 1.082.
    """
    case = load("ieee13")
    base = cvx.create_lindist_model(case)
    cvx.set_objective(base, cvx_obj.loss_objective_rule)
    res = cvx.solve(base, duals=False)
    p, q = res.p_flow, res.q_flow
    apparent = {}
    for ph in ("a", "b", "c"):
        s_val = np.hypot(p[ph].to_numpy(float), q[ph].to_numpy(float))
        for fb, tb, value in zip(p["fb"], p["tb"], s_val):
            if not np.isnan(value):
                apparent[(int(fb), int(tb), ph)] = float(value)
    ranked = sorted(apparent, key=apparent.get, reverse=True)[:tight_branches]
    for ph in ("a", "b", "c"):
        case.branch_data[f"s_{ph}_max"] = np.nan
    for (fb, tb, ph), value in apparent.items():
        factor = tight_factor if (fb, tb, ph) in ranked else margin
        mask = (case.branch_data.fb == fb) & (case.branch_data.tb == tb)
        case.branch_data.loc[mask, f"s_{ph}_max"] = max(value * factor, 1e-3)
    return case


# ---------------------------------------------------------------- objectives
@pytest.mark.parametrize(
    "label,pyo_rule,cvx_rule",
    [
        ("cost", pyo_obj.cost_minimization_rule, cvx_obj.cost_minimization_rule),
        pytest.param(
            "gen_cost_substation_penalty",
            pyo_obj.generation_cost_with_substation_quadratic_penalty_objective_rule,
            cvx_obj.generation_cost_with_substation_quadratic_penalty_objective_rule,
            marks=pytest.mark.skip(
                reason="CLARABEL returns optimal_inaccurate for the 1e6-weighted "
                "objective; see Open analysis in cvxpy_models/todo.md"
            ),
        ),
        ("total_cost", pyo_obj.total_cost_rule, cvx_obj.total_cost_rule),
    ],
)
def test_cost_objectives_multiperiod(label, pyo_rule, cvx_rule):
    case = load("ieee123_30der", n_steps=3)
    case.gen_data["control_variable"] = "PQ"
    pm, cm = build_pair(
        case, {"circular_constraints": True, "thermal_constraints": True}
    )
    pres, cres = solve_pair(pm, cm, pyo_rule, cvx_rule)
    assert cres.objective_value == pytest.approx(
        pres.objective_value, abs=1e-5, rel=1e-4
    )


def test_multiperiod_batteries_schedules_loss():
    case = load("ieee123_30der", n_steps=4)
    pm, cm = build_pair(case)
    pres, cres = solve_pair(
        pm, cm, pyo_obj.cost_minimization_rule, cvx_obj.cost_minimization_rule
    )
    assert cres.objective_value == pytest.approx(
        pres.objective_value, abs=1e-5, rel=1e-4
    )
    assert sorted(cres.voltages.t.unique()) == [0, 1, 2, 3]
    assert_same(pres, cres, ("voltages",), tol=1e-3)


def test_penalized_objective_equality_only():
    case = load("ieee13")
    pm, cm = build_pair(case, {"equality_only": True})

    def pyo_rule(m):
        return (
            pyo_obj.loss_objective_rule(m)
            + pyo_obj.voltage_violation_penalty(m, 1e3)
            + pyo_obj.generator_violation_penalty(m, 1e3)
        )

    cvx_rule = cvx_obj.create_penalized_objective(
        cvx_obj.loss_objective_rule, voltage_weight=1e3, generator_weight=1e3
    )
    pres, cres = solve_pair(pm, cm, pyo_rule, cvx_rule)
    # Pyomo uses a smooth max(0, x) approximation; CVXPY uses the exact pos().
    assert cres.objective_value == pytest.approx(
        pres.objective_value, abs=5e-2, rel=5e-2
    )
    assert_same(pres, cres, ("voltages",), tol=5e-3)


def test_equality_only_constraint_sets_match():
    case = load("ieee13")
    pm, cm = build_pair(case, {"equality_only": True})
    pyo_names = {c.local_name for c in pm.component_objects(pyo.Constraint, active=True)}
    assert set(cm.constraint_names) == pyo_names
    for name in cm.constraint_names:
        assert all(
            piece.is_dcp() for pieces in getattr(cm, name).values() for piece in pieces
        )
    assert "voltage_limits" not in cm.constraint_names


# --------------------------------------------------------------------- slacks
@pytest.mark.parametrize("option", ["voltage_slacks", "free_swing_voltage"])
def test_slack_options_structure_and_value(option):
    case = load("ieee13")
    pm, cm = build_pair(case, {option: True})
    pyo_names = {c.local_name for c in pm.component_objects(pyo.Constraint, active=True)}
    assert set(cm.constraint_names) == pyo_names

    penalty = {
        "voltage_slacks": (
            pyo_obj.voltage_slack_penalty,
            cvx_obj.voltage_slack_penalty,
        ),
        "free_swing_voltage": (
            pyo_obj.swing_voltage_slack_penalty,
            cvx_obj.swing_voltage_slack_penalty,
        ),
    }[option]

    def pyo_rule(m):
        return pyo_obj.loss_objective_rule(m) + penalty[0](m, 1e3)

    def cvx_rule(m):
        return cvx_obj.loss_objective_rule(m) + penalty[1](m, 1e3)

    pres, cres = solve_pair(pm, cm, pyo_rule, cvx_rule)
    # IPOPT lets nonnegative slacks dip to about -1e-8 each (bound relaxation),
    # which lowers the penalized objective by a few 1e-4 at weight 1e3.
    assert cres.objective_value == pytest.approx(
        pres.objective_value, abs=2e-3, rel=1e-2
    )
    assert_same(pres, cres, ("voltages",), tol=1e-3)


def test_thermal_slacks_cvxpy_builds_and_solves():
    """Pyomo's thermal slack constraint is broken, so this checks CVXPY alone."""
    case = load_with_thermal_limits()
    cm = cvx.create_lindist_model(case, thermal_slacks=True, thermal_constraints=True)
    assert "thermal_slack_constraint" in cm.constraint_names
    assert "thermal_slack" in cm.var_names
    cvx.set_objective(
        cm,
        cvx_obj.loss_objective_rule(cm) + cvx_obj.thermal_slack_penalty(cm, 1e3),
    )
    res = cvx.solve(cm, duals=False)
    slack = res.thermal_slack[["a", "b", "c"]].to_numpy(dtype=float)
    assert np.nanmin(slack) >= -1e-8


# ------------------------------------------------------------ thermal forms
def test_thermal_circle_matches_pyomo():
    case = load_with_thermal_limits()
    kw = {"circular_constraints": True, "thermal_constraints": True}
    pm, cm = build_pair(case, kw)
    pyo_names = {c.local_name for c in pm.component_objects(pyo.Constraint, active=True)}
    assert set(cm.constraint_names) == pyo_names
    assert "thermal_limit_circle" in cm.constraint_names
    pres, cres = solve_pair(
        pm, cm, pyo_obj.loss_objective_rule, cvx_obj.loss_objective_rule
    )
    assert cres.objective_value == pytest.approx(
        pres.objective_value, abs=1e-5, rel=1e-4
    )
    assert_same(pres, cres, ("voltages",), tol=1e-3)


def test_thermal_octagon_is_inner_approximation_of_circle():
    """Pyomo cannot build the octagon form (it calls .get on an IndexedParam)."""
    case = load_with_thermal_limits(tight_factor=1.09)
    circle = cvx.create_lindist_model(
        case, circular_constraints=True, thermal_constraints=True
    )
    octagon = cvx.create_lindist_model(
        case, circular_constraints=False, thermal_constraints=True
    )
    assert "thermal_limit_circle" in circle.constraint_names
    assert [n for n in octagon.constraint_names if n.startswith("thermal_limit_")] == [
        f"thermal_limit_{i}" for i in range(1, 9)
    ]
    cvx.set_objective(circle, cvx_obj.loss_objective_rule)
    cvx.set_objective(octagon, cvx_obj.loss_objective_rule)
    obj_circle = cvx.solve(circle, duals=False).objective_value
    obj_octagon = cvx.solve(octagon, duals=False).objective_value
    # The octagon is inscribed in the circle, so it can only be more restrictive.
    assert obj_octagon >= obj_circle - 1e-8


def test_free_boundary_loads_structure():
    case = load("ieee13")
    case.bus_data.loc[case.bus_data.index[-1], "bus_type"] = "OUT"
    pm, cm = build_pair(case, {"free_boundary_loads": True})
    for name in ("cvr_p_load", "cvr_q_load"):
        pk = {k if isinstance(k, tuple) else (k,) for k in getattr(pm, name).keys()}
        ck = {k if isinstance(k, tuple) else (k,) for k in getattr(cm, name).keys()}
        assert pk == ck
        assert len(ck) < len(pm.bus_phase_set) * len(pm.time_set)


# --------------------------------------------------------------------- ADMM
def test_admm_penalties_match():
    case = load("ieee13")
    child = int(case.bus_data.id.iloc[-1])
    name = str(case.bus_data.set_index("id").loc[child, "name"]).strip()
    for ph in ("a", "b", "c"):
        for kind in ("p", "q", "v"):
            case.schedules[f"{name}.{ph}.{kind}"] = 0.01 if kind != "v" else 1.0
    kw = {}
    pm, cm = build_pair(case, kw)

    def pyo_rule(m):
        return (
            pyo_obj.loss_objective_rule(m)
            + pyo_obj.admm_v_up_penalty(m, 1.0)
            + pyo_obj.admm_s_up_penalty(m, 1.0)
            + pyo_obj.admm_v_down_penalty(m, 1.0)
            + pyo_obj.admm_s_down_penalty(m, 1.0)
        )

    def cvx_rule(m):
        return (
            cvx_obj.loss_objective_rule(m)
            + cvx_obj.admm_v_up_penalty(m, 1.0)
            + cvx_obj.admm_s_up_penalty(m, 1.0)
            + cvx_obj.admm_v_down_penalty(m, 1.0)
            + cvx_obj.admm_s_down_penalty(m, 1.0)
        )

    pres, cres = solve_pair(pm, cm, pyo_rule, cvx_rule)
    assert cres.objective_value == pytest.approx(
        pres.objective_value, abs=1e-5, rel=1e-4
    )
    assert_same(pres, cres, ("voltages",), tol=1e-3)


# ------------------------------------------------------- mixed-integer cap/reg
def _mip_solvers():
    out = []
    for name in ("appsi_highs", "highs"):
        if pyo.SolverFactory(name).available(False):
            out.append(name)
    return out


@pytest.mark.slow
@pytest.mark.skipif(not _mip_solvers(), reason="needs a Pyomo MIP solver (HiGHS)")
@pytest.mark.parametrize(
    "controls",
    [
        {"control_capacitors": True},
        {"control_regulators": True},
        {"control_capacitors": True, "control_regulators": True},
    ],
)
def test_mi_cap_reg_numeric_parity(controls):
    case = load("ieee13")
    pm, cm = build_pair(case, controls)
    pyo_names = {c.local_name for c in pm.component_objects(pyo.Constraint, active=True)}
    assert set(cm.constraint_names) == pyo_names
    pres, cres = solve_pair(
        pm,
        cm,
        pyo_obj.substation_power_objective_rule,
        cvx_obj.substation_power_objective_rule,
        pyo_solver=_mip_solvers()[0],
        cvx_solver="HIGHS",
    )
    assert cres.objective_value == pytest.approx(
        pres.objective_value, abs=1e-4, rel=1e-4
    )
    if controls.get("control_regulators"):
        assert cres.reg_taps is not None
        assert set(cres.reg_taps.columns) >= {"fb", "tb", "t"}
    if controls.get("control_capacitors"):
        u = cres.u_cap
        values = u[[c for c in u.columns if c in ("a", "b", "c", "value")]].to_numpy(float)
        values = values[~np.isnan(values)]
        assert values.size > 0
        assert np.all(np.isclose(values, np.round(values), atol=1e-6))


def test_reg_tap_change_limit_structure():
    case = load("ieee123_30der", n_steps=3)
    pm, cm = build_pair(
        case, {"control_regulators": True, "reg_tap_change_limit": 1}
    )
    for name in ("reg_tap_change_upper", "reg_tap_change_lower"):
        pk = {k if isinstance(k, tuple) else (k,) for k in getattr(pm, name).keys()}
        ck = {k if isinstance(k, tuple) else (k,) for k in getattr(cm, name).keys()}
        assert pk == ck and len(ck) > 0


def test_invalid_tap_change_limit_rejected():
    case = load("ieee13")
    with pytest.raises(ValueError, match="nonnegative integer"):
        cvx.create_lindist_model(
            case, control_regulators=True, reg_tap_change_limit=-1
        )
