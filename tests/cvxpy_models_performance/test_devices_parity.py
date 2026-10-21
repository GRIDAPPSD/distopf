"""Parity of vectorized device providers against ``distopf.cvxpy_models``."""

from __future__ import annotations

import numpy as np
import pytest

import distopf as opf
from distopf import cvxpy_models as scalar
from distopf import cvxpy_models_performance as perf
from distopf.cvxpy_models.common import objectives as sobj
from distopf.cvxpy_models_performance.common import objectives as pobj

from .test_core_network_parity import assert_frames_close, load

TOL = 1e-5


def pair(case, **kw):
    sm = scalar.create_lindist_model(case, **kw)
    pm = perf.create_lindist_model(case, **kw)
    return sm, pm


def run(
    case, objective, solver="CLARABEL", compare_vars=("p_flow", "q_flow"), tol=TOL, **kw
):
    sm, pm = pair(case, **kw)
    scalar.set_objective(sm, getattr(sobj, objective))
    perf.set_objective(pm, getattr(pobj, objective))
    rs = scalar.solve(sm, solver=solver)
    rp = perf.solve(pm, solver=solver)
    np.testing.assert_allclose(
        rs.objective_value, rp.objective_value, rtol=tol, atol=tol
    )
    assert_frames_close(rs.voltages, rp.voltages, tol=1e-4)
    for name in compare_vars:
        assert_frames_close(getattr(rs, name), getattr(rp, name), tol=1e-4)
    return sm, pm, rs, rp


def test_structure_with_all_devices():
    case = load("ieee123_30der")
    sm, pm = pair(case, circular_constraints=True, thermal_constraints=True)
    assert set(sm.constraint_names) == set(pm.constraint_names)
    assert set(sm.var_names) == set(pm.var_names)
    for name in sm.var_names:
        n_scalar = len(getattr(sm, name))
        assert getattr(pm, name).size == n_scalar, name
    assert len(pm.all_constraints()) * 10 < len(sm.all_constraints())


@pytest.mark.parametrize("circular", [True, False])
def test_generators_curtailment(circular):
    case = load("ieee123_30der")
    sm, pm, rs, rp = run(
        case,
        "generation_curtailment_objective_rule",
        circular_constraints=circular,
        compare_vars=(),
    )


def test_generators_loss_equality_only():
    run(
        load("ieee123_30der"),
        "loss_objective_rule",
        equality_only=True,
        compare_vars=(),
    )


def test_battery_multiperiod_matches_scalar():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der", n_steps=4)
    run(
        case,
        "loss_objective_rule",
        compare_vars=(),
        circular_constraints=True,
    )


def test_capacitor_and_regulator_mi_structure():
    case = load("ieee13")
    kw = dict(control_capacitors=True, control_regulators=True)
    sm, pm = pair(case, **kw)
    assert set(sm.constraint_names) == set(pm.constraint_names)
    assert set(sm.var_names) == set(pm.var_names)
    for name in sm.var_names:
        assert getattr(pm, name).size == len(getattr(sm, name)), name


@pytest.mark.parametrize(
    "kw",
    [
        {"control_capacitors": True},
        {"control_regulators": True},
        {"control_capacitors": True, "control_regulators": True},
    ],
)
def test_mi_numeric_parity(kw):
    sm, pm = pair(load("ieee13"), **kw)
    scalar.set_objective(sm, sobj.voltage_min_objective_rule)
    perf.set_objective(pm, pobj.voltage_min_objective_rule)
    rs = scalar.solve(sm, solver="HIGHS")
    rp = perf.solve(pm, solver="HIGHS")
    np.testing.assert_allclose(
        rs.objective_value, rp.objective_value, rtol=1e-4, atol=1e-5
    )
    if "control_regulators" in kw:
        assert_frames_close(rs.reg_taps, rp.reg_taps)
