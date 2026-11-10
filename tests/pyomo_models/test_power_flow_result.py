"""to_power_flow_result on a directly built model matches the wrapper output."""

import pandas as pd
import pyomo.environ as pyo
import pytest

from distopf import CASES_DIR, create_case
from distopf.pyomo_models import (
    create_lindist_model,
    set_objective,
    solve,
    to_power_flow_result,
)
from distopf.pyomo_models.common.objectives import loss_objective_auto_rule

FIELDS = (
    "voltages",
    "active_power_flows",
    "reactive_power_flows",
    "active_power_generation",
    "reactive_power_generation",
    "capacitor_reactive_power",
)


def test_direct_solve_matches_wrapper():
    case = create_case(CASES_DIR / "csv" / "ieee123_30der")
    expected = case.run_opf("loss", wrapper="pyomo")

    model = create_lindist_model(case)
    set_objective(
        model, pyo.Objective(rule=loss_objective_auto_rule, sense=pyo.minimize)
    )
    result = to_power_flow_result(solve(model, solver="ipopt"), case, solver="ipopt")

    assert result.solver == "ipopt"
    assert result.model is model
    assert result.objective_value == pytest.approx(expected.objective_value)
    for name in FIELDS:
        pd.testing.assert_frame_equal(
            getattr(result, name), getattr(expected, name), atol=1e-6, rtol=0
        )
