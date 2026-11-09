"""MPSSD and capacity-expansion extensions for cvxpy_models."""

from __future__ import annotations

import warnings

import pandas as pd
import pyomo.environ as pyo
import pytest

import distopf as opf
from distopf import cvxpy_models as cvx
from distopf import pyomo_models as pyo_models
from distopf.cvxpy_models.common import objectives as cvx_obj
from distopf.cvxpy_models.extensions.capacity_expansion.factory import (
    create_capacity_expansion_lindist_model,
)
from distopf.cvxpy_models.extensions.mpssd.factory import create_mpssd_lindist_model
from distopf.pyomo_models.common import objectives as pyo_obj
from distopf.pyomo_models.extensions.mpssd.factory import (
    create_mpssd_lindist_model as pyo_create_mpssd,
)

warnings.filterwarnings("ignore", category=FutureWarning)


def _mpssd_case():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
    buses = case.bus_data.loc[
        (case.bus_data.phases == "abc") & (case.bus_data.bus_type == "PQ")
    ]
    names = [str(n) for n in buses["name"].iloc[:2]]
    case.bus_data["name"] = case.bus_data["name"].astype(str)
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "device_name": f"port_{i}",
                "bus_name": name,
                "phases": "abc",
                "dc_bus": 1,
                "control_variable": "PQ",
                "s_a_max": 0.3,
                "s_b_max": 0.3,
                "s_c_max": 0.3,
            }
        )
    case.mpssd_data = pd.DataFrame(rows)
    return case


def test_mpssd_structure_and_numeric_parity():
    case = _mpssd_case()
    pm = pyo_create_mpssd(case)
    cm = create_mpssd_lindist_model(case)
    pyo_names = {c.local_name for c in pm.component_objects(pyo.Constraint, active=True)}
    assert set(cm.constraint_names) == pyo_names
    assert {v.local_name for v in pm.component_objects(pyo.Var)} == set(cm.var_names)

    pyo_models.set_objective(
        pm, pyo.Objective(rule=pyo_obj.loss_objective_rule, sense=pyo.minimize)
    )
    cvx.set_objective(cm, cvx_obj.loss_objective_rule)
    pres = pyo_models.solve(pm, solver="ipopt", duals=False)
    cres = cvx.solve(cm, duals=False)
    assert cres.objective_value == pytest.approx(pres.objective_value, abs=1e-6, rel=1e-5)
    assert len(cres.p_mpssd) == len(pres.p_mpssd)
    dc_sum = cres.p_mpssd[["a", "b", "c"]].sum().sum()
    assert dc_sum == pytest.approx(0.0, abs=1e-6)


def test_mpssd_ngon_variant_builds():
    case = _mpssd_case()
    cm = create_mpssd_lindist_model(case, circular_constraints=False)
    assert any(name.startswith("mpssd_ngon_limit_") for name in cm.constraint_names)
    assert "mpssd_circle" not in cm.constraint_names


def _expansion_case():
    case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
    edge = str(case.branch_data["name"].iloc[len(case.branch_data) // 2])
    return case, edge


def test_capacity_expansion_builds_and_solves():
    case, edge = _expansion_case()
    cm = create_capacity_expansion_lindist_model(
        case,
        zone_edges=[edge],
        new_pv=0.2,
        new_bess=0.1,
        pv_curtailment_max=0.5,
        pv_capacity_factor=0.8,
        bess_energy_capacity=1.0,
    )
    for name in ("der_budget", "der_inj_map", "pv_gen_min", "pv_gen_max", "bess_energy_1"):
        assert name in cm.constraint_names
    assert len(cm.zone_set) == 2
    cvx.set_objective(
        cm,
        cvx_obj.substation_power_objective_rule(cm)
        + cvx_obj.voltage_slack_penalty(cm)
        + cvx_obj.thermal_slack_penalty(cm),
    )
    res = cvx.solve(cm, duals=False)
    assert res.p_max["value"].sum() == pytest.approx(0.3, abs=1e-6)
    assert set(res.p_max["resource"]) == {"PV", "BESS"}


def test_capacity_expansion_rejects_unknown_edge():
    case, _ = _expansion_case()
    with pytest.raises(ValueError, match="not found"):
        create_capacity_expansion_lindist_model(case, zone_edges=["no_such_edge"])
