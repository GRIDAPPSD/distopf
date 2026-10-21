"""Structure and numerical parity between pyomo_models and cvxpy_models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import distopf as opf
from distopf import cvxpy_models as cvx
from distopf import pyomo_models as pyo_models
from distopf.cvxpy_models.common import objectives as cvx_obj
from distopf.pyomo_models.common import objectives as pyo_obj
from distopf.pyomo_models.common.solvers import solve as pyo_solve
from distopf.pyomo_models.common.objectives import set_objective as pyo_set_objective

import pyomo.environ as pyo

TOL = 1e-5
CASES = ["ieee13", "ieee123_30der"]


def load(name):
    return opf.create_case(opf.CASES_DIR / "csv" / name)


def pyomo_constraint_names(model):
    return {c.local_name for c in model.component_objects(pyo.Constraint, active=True)}


def pyomo_var_names(model):
    return {v.local_name for v in model.component_objects(pyo.Var)}


def pyomo_constraint_keys(model, name):
    return {
        (k if isinstance(k, tuple) else (k,)) for k in getattr(model, name).keys()
    }


def assert_frames_close(a: pd.DataFrame, b: pd.DataFrame, tol=TOL):
    assert a.shape == b.shape, (a.shape, b.shape)
    num_cols = [c for c in a.columns if pd.api.types.is_numeric_dtype(a[c])]
    key_cols = [c for c in a.columns if c not in num_cols or c in ("t", "id", "fb", "tb")]
    a = a.sort_values(key_cols).reset_index(drop=True)
    b = b.sort_values(key_cols).reset_index(drop=True)
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]):
            va = a[col].to_numpy(dtype=float)
            vb = b[col].to_numpy(dtype=float)
            mask = ~(np.isnan(va) & np.isnan(vb))
            np.testing.assert_allclose(va[mask], vb[mask], atol=tol, rtol=tol, err_msg=col)
        else:
            assert (a[col].astype(str) == b[col].astype(str)).all(), col


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("kwargs", [{}, {"circular_constraints": True, "thermal_constraints": True}])
def test_structure_parity(case_name, kwargs):
    case = load(case_name)
    pm = pyo_models.create_lindist_model(case, **kwargs)
    cm = cvx.create_lindist_model(case, **kwargs)

    assert set(cm.constraint_names) == pyomo_constraint_names(pm)
    assert set(cm.var_names) == pyomo_var_names(pm)
    for name in cm.constraint_names:
        assert set(
            (k if isinstance(k, tuple) else (k,)) for k in getattr(cm, name)
        ) == pyomo_constraint_keys(pm, name), name
    for name in cm.var_names:
        pk = {(k if isinstance(k, tuple) else (k,)) for k in getattr(pm, name).keys()}
        ck = {(k if isinstance(k, tuple) else (k,)) for k in getattr(cm, name).keys()}
        assert pk == ck, name


def test_structure_parity_mi_cap_reg():
    case = load("ieee13")
    kw = dict(control_capacitors=True, control_regulators=True)
    pm = pyo_models.create_lindist_model(case, **kw)
    cm = cvx.create_lindist_model(case, **kw)
    assert set(cm.constraint_names) == pyomo_constraint_names(pm)
    assert set(cm.var_names) == pyomo_var_names(pm)
    for name in cm.var_names:
        assert len(getattr(cm, name)) == len(getattr(pm, name)), name


@pytest.mark.parametrize("socp", [True, False])
def test_structure_parity_branchflow(socp):
    case = load("ieee13")
    pm = pyo_models.create_nl_branchflow_model(case, socp_relaxation=socp)
    cm = cvx.create_nl_branchflow_model(case, socp_relaxation=socp)
    assert set(cm.constraint_names) == pyomo_constraint_names(pm)
    assert set(cm.var_names) == pyomo_var_names(pm)


OBJECTIVES = [
    ("loss", pyo_obj.loss_objective_rule, cvx_obj.loss_objective_rule),
    (
        "substation",
        pyo_obj.substation_power_objective_rule,
        cvx_obj.substation_power_objective_rule,
    ),
    (
        "voltage_deviation",
        pyo_obj.voltage_deviation_objective_rule,
        cvx_obj.voltage_deviation_objective_rule,
    ),
    (
        "voltage_min",
        pyo_obj.voltage_min_objective_rule,
        cvx_obj.voltage_min_objective_rule,
    ),
    (
        "curtail",
        pyo_obj.generation_curtailment_objective_rule,
        cvx_obj.generation_curtailment_objective_rule,
    ),
]


def solve_both(case, pyo_rule, cvx_rule, model_kwargs=None, builder="lindist", solver=None):
    model_kwargs = model_kwargs or {}
    if builder == "lindist":
        pm = pyo_models.create_lindist_model(case, **model_kwargs)
        cm = cvx.create_lindist_model(case, **model_kwargs)
    else:
        pm = pyo_models.create_nl_branchflow_model(case, **model_kwargs)
        cm = cvx.create_nl_branchflow_model(case, **model_kwargs)
    pyo_set_objective(pm, pyo.Objective(rule=pyo_rule, sense=pyo.minimize))
    cvx.set_objective(cm, cvx_rule)
    pres = pyo_solve(pm, solver="ipopt", duals=True)
    cres = cvx.solve(cm, solver=solver, duals=True)
    return pres, cres


def compare_results(pres, cres, tol=TOL):
    assert_frames_close(pres.voltages, cres.voltages, tol)
    for name in ("p_flow", "q_flow", "p_gen", "q_gen", "q_cap", "p_load", "q_load"):
        pa = getattr(pres, name, None)
        ca = getattr(cres, name, None)
        assert (pa is None) == (ca is None), name
        if pa is not None and len(pa) > 0:
            assert_frames_close(pa, ca, tol)


@pytest.mark.parametrize("case_name", CASES)
@pytest.mark.parametrize("label,pyo_rule,cvx_rule", OBJECTIVES)
def test_numerical_parity_lindist(case_name, label, pyo_rule, cvx_rule):
    case = load(case_name)
    pres, cres = solve_both(
        case,
        pyo_rule,
        cvx_rule,
        model_kwargs={"circular_constraints": True, "thermal_constraints": True},
    )
    assert cres.objective_value == pytest.approx(pres.objective_value, abs=1e-5, rel=1e-4)
    if label == "curtail":
        return  # linear objective with free Q: optimum is not unique
    compare_results(pres, cres, tol=1e-3 if label == "loss" else 1e-4)


def test_numerical_parity_battery_case():
    case = load("ieee13_battery")
    pres, cres = solve_both(
        case, pyo_obj.loss_objective_rule, cvx_obj.loss_objective_rule
    )
    assert cres.objective_value == pytest.approx(pres.objective_value, abs=1e-4, rel=1e-4)
    compare_results(pres, cres, tol=1e-4)
    for name in ("soc", "p_charge", "p_discharge", "p_bat", "q_bat"):
        assert_frames_close(getattr(pres, name), getattr(cres, name), 1e-4)


@pytest.mark.parametrize("case_name", ["minimal_triplex", "triplex_pv"])
def test_numerical_parity_triplex(case_name):
    case = load(case_name)
    pres, cres = solve_both(
        case, pyo_obj.loss_objective_rule, cvx_obj.loss_objective_rule
    )
    assert cres.objective_value == pytest.approx(pres.objective_value, abs=1e-5, rel=1e-4)
    assert_frames_close(pres.voltages, cres.voltages, 1e-4)


def test_numerical_parity_socp():
    case = load("ieee13")
    pres, cres = solve_both(
        case,
        pyo_obj.loss_objective_rule,
        cvx_obj.loss_objective_rule,
        model_kwargs={"socp_relaxation": True},
        builder="nl",
    )
    assert cres.objective_value == pytest.approx(pres.objective_value, abs=1e-4, rel=1e-3)


def _cyipopt_installed():
    import importlib.util

    return importlib.util.find_spec("cyipopt") is not None


@pytest.mark.skipif(_cyipopt_installed(), reason="cyipopt installed")
def test_exact_branchflow_needs_cyipopt():
    case = load("ieee13")
    cm = cvx.create_nl_branchflow_model(case)
    cvx.set_objective(cm, cvx_obj.loss_objective_rule)
    with pytest.raises(NotImplementedError, match="cyipopt"):
        cvx.solve(cm)


@pytest.mark.skipif(not _cyipopt_installed(), reason="needs cyipopt for CVXPY NLP")
@pytest.mark.slow
def test_numerical_parity_exact_branchflow():
    case = load("ieee13")
    pres = case.run_opf(objective="loss", formulation="branchflow", wrapper="pyomo")
    cres = case.run_opf(objective="loss", formulation="branchflow", wrapper="cvxpy")
    assert cres.objective_value == pytest.approx(pres.objective_value, abs=1e-6, rel=1e-4)
    assert_frames_close(pres.voltages, cres.voltages, 1e-4)


def test_exact_branchflow_is_dnlp():
    case = load("ieee13")
    cm = cvx.create_nl_branchflow_model(case)
    cvx.set_objective(cm, cvx_obj.loss_objective_rule)
    from distopf.cvxpy_models.common.solvers import build_problem

    problem = build_problem(cm)
    assert not problem.is_dcp()
    if hasattr(problem, "is_dnlp"):
        assert problem.is_dnlp()


def test_dual_extraction():
    case = load("ieee13")
    cm = cvx.create_lindist_model(case)
    cvx.set_objective(cm, cvx_obj.loss_objective_rule)
    res = cvx.solve(cm, duals=True)
    duals = res.get_dual("power_balance_p")
    assert not duals.empty
    assert {"fb", "tb", "phase", "dual"} <= set(duals.columns) or {
        "fb",
        "tb",
        "t",
        "phase",
        "dual",
    } <= set(duals.columns)
    assert hasattr(res, "dual_voltage_limits")


def test_duals_match_pyomo_up_to_sign():
    case = load("ieee13")
    pres, cres = solve_both(case, pyo_obj.loss_objective_rule, cvx_obj.loss_objective_rule)
    pd_ = pres.get_dual("power_balance_p")
    cd_ = cres.get_dual("power_balance_p")
    assert pd_.shape == cd_.shape
    keys = [c for c in pd_.columns if c != "dual"]
    merged = pd_.merge(cd_, on=keys, suffixes=("_pyo", "_cvx"))
    ratio = merged["dual_cvx"].to_numpy() * np.sign(
        np.sum(merged["dual_cvx"].to_numpy() * merged["dual_pyo"].to_numpy())
    )
    np.testing.assert_allclose(ratio, merged["dual_pyo"].to_numpy(), atol=1e-3, rtol=1e-3)


def test_mi_cap_reg_solves():
    case = load("ieee13")
    cm = cvx.create_lindist_model(
        case, control_capacitors=True, control_regulators=True
    )
    cvx.set_objective(cm, cvx_obj.substation_power_objective_rule)
    res = cvx.solve(cm, solver="HIGHS")
    assert res.reg_taps is not None
    assert res.u_cap is not None


def test_wrapper_lindist_matches_pyomo():
    case = load("ieee13")
    pres = case.run_opf(objective="loss", wrapper="pyomo")
    cres = case.run_opf(objective="loss", wrapper="cvxpy")
    assert_frames_close(pres.voltages, cres.voltages, 1e-4)
    assert cres.objective_value == pytest.approx(pres.objective_value, abs=1e-4, rel=1e-4)


def test_wrapper_socp_runs():
    case = load("ieee13")
    res = case.run_opf(objective="loss", formulation="socp", wrapper="cvxpy")
    assert res.voltages is not None


def test_formulation_registry_includes_cvxpy():
    from distopf.api import _resolve_wrapper
    from distopf.wrappers import CvxpyWrapper

    cls, extra = _resolve_wrapper("cvxpy", "lindist")
    assert cls is CvxpyWrapper
    assert extra == {"model_type": "lindist"}
    with pytest.raises(ValueError):
        _resolve_wrapper("cvxpy", "lindist_mp")


def test_case_to_cvxpy_model():
    case = load("ieee13")
    assert isinstance(case.to_cvxpy_model(), cvx.CvxModel)


def test_voltage_min_objective_string_alias():
    case = load("ieee13")
    pres = case.run_opf(objective="voltage_min", wrapper="pyomo")
    cres = case.run_opf(objective="voltage_min", wrapper="cvxpy")
    assert cres.objective_value == pytest.approx(pres.objective_value, abs=1e-5, rel=1e-5)
    assert_frames_close(pres.voltages, cres.voltages, 1e-4)
