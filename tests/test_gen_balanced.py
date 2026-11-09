"""Behavior of the gen_data `balanced` column across the model packages."""

from __future__ import annotations

import pytest

import distopf as opf
from distopf import cvxpy_models as scalar
from distopf import cvxpy_models_performance as perf
from distopf import pyomo_models
from distopf.pyomo_models.common import objectives as pyo_obj
from distopf.cvxpy_models.common import objectives as sobj
from distopf.cvxpy_models_performance.common import objectives as pobj

P_AVAILABLE = {"a": 0.01, "b": 0.005, "c": 0.01}


def make_case(balanced: bool):
    """ieee123_30der where the first generator has a smaller phase-b output."""
    case = opf.create_case(
        opf.CASES_DIR / "csv" / "ieee123_30der", ignore_schedule=True
    )
    case.modify(control_variable="P", v_min=0.95, v_max=1.05)
    case.gen_data["balanced"] = False
    case.gen_data.loc[case.gen_data.index[0], "balanced"] = balanced
    for phase, value in P_AVAILABLE.items():
        case.gen_data.loc[case.gen_data.index[0], f"p_{phase}"] = value
    return case


def first_device(model) -> str:
    return next(iter(model.gen_device_set))


def solved_pyomo(case):
    model = pyomo_models.create_lindist_model(case)
    pyomo_models.set_objective(model, pyo_obj.generation_curtailment_objective_rule)
    r = pyomo_models.solve(model)
    out = r.p_gen.loc[0, ["a", "b", "c"]].to_dict()
    return out


def solved_scalar(case):
    model = scalar.create_lindist_model(case)
    scalar.set_objective(model, sobj.generation_curtailment_objective_rule)
    scalar.solve(model, solver="CLARABEL")
    device = first_device(model)
    return {ph: model.p_gen[device, ph, 0].value for ph in P_AVAILABLE}


def solved_performance(case):
    model = perf.create_lindist_model(case)
    perf.set_objective(model, pobj.generation_curtailment_objective_rule)
    perf.solve(model, solver="CLARABEL")
    device = first_device(model)
    return {ph: float(model.at("p_gen", (device, ph), 0)) for ph in P_AVAILABLE}


@pytest.mark.parametrize("solve", [solved_scalar, solved_performance, solved_pyomo])
def test_balanced_forces_equal_phase_output(solve):
    p = solve(make_case(balanced=True))
    assert p["a"] == pytest.approx(P_AVAILABLE["b"], abs=1e-6)
    assert p["b"] == pytest.approx(P_AVAILABLE["b"], abs=1e-6)
    assert p["c"] == pytest.approx(P_AVAILABLE["b"], abs=1e-6)


@pytest.mark.parametrize("solve", [solved_scalar, solved_performance])
def test_unbalanced_keeps_phases_independent(solve):
    p = solve(make_case(balanced=False))
    assert p["a"] == pytest.approx(P_AVAILABLE["a"], abs=1e-6)
    assert p["b"] == pytest.approx(P_AVAILABLE["b"], abs=1e-6)


def test_pyomo_balanced_constraints_only_for_flagged_devices():
    model = pyomo_models.create_lindist_model(make_case(balanced=True))
    device = first_device(model)
    pairs = {(d, left, right) for d, left, right, _ in model.gen_balanced_p}
    assert pairs == {(device, "a", "b"), (device, "b", "c")}


def test_invalid_balanced_value_is_rejected():
    case = make_case(balanced=False)
    case.gen_data["balanced"] = case.gen_data["balanced"].astype(object)
    case.gen_data.loc[case.gen_data.index[0], "balanced"] = "maybe"
    with pytest.raises(ValueError, match="balanced"):
        scalar.create_lindist_model(case)


def make_triplex_case():
    """triplex_pv where pv4 (s1s2) has a smaller s2 output and is balanced."""
    case = opf.create_case(opf.CASES_DIR / "csv" / "triplex_pv", ignore_schedule=True)
    case.modify(control_variable="P", v_min=0.9, v_max=1.1)
    case.gen_data["balanced"] = False
    row = case.gen_data.index[case.gen_data["name"] == "pv4"][0]
    case.gen_data.loc[row, ["balanced", "p_s1", "p_s2"]] = [True, 0.0025, 0.0015]
    return case


def test_pyomo_balanced_pairs_s1_s2():
    model = pyomo_models.create_lindist_model(make_triplex_case())
    pairs = {(left, right) for _, left, right, _ in model.gen_balanced_p}
    assert pairs == {("s1", "s2")}


def test_balanced_forces_equal_s1_s2_output():
    model = perf.create_lindist_model(make_triplex_case())
    perf.set_objective(model, pobj.generation_curtailment_objective_rule)
    perf.solve(model, solver="CLARABEL")
    device = next(
        d
        for d, phases in model.gen_phases_by_device.items()
        if phases == ["s1", "s2"] and model.gen_balanced[d]
    )
    for phase in ("s1", "s2"):
        assert float(model.at("p_gen", (device, phase), 0)) == pytest.approx(
            0.0015, abs=1e-6
        )
