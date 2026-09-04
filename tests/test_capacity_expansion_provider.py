"""Structural tests for capacity-expansion provider integration."""

import pandas as pd
import pyomo.environ as pyo

from distopf.pyomo_models.capacity_expansion_constraints import (
    add_capacity_expansion_as_fraction_of_load,
)
from distopf.pyomo_models.capacity_expansion_provider import CapacityExpansionProvider
from distopf.pyomo_models.injection_registry import InjectionRegistry


def test_capacity_budget_defaults_are_not_mutable():
    model = pyo.ConcreteModel()
    model.time_set = pyo.RangeSet(0, 0)
    case = type(
        "CaseLike",
        (),
        {
            "bus_data": pd.DataFrame(
                {"id": [1], "pl_a": [2.0], "pl_b": [0.0], "pl_c": [0.0]}
            ),
            "schedules": pd.DataFrame(index=[0]),
        },
    )()

    add_capacity_expansion_as_fraction_of_load(model, case)
    assert set(model.resource_set) == {"PV", "BESS"}
    assert pyo.value(model.total_capacity_expansion) == 0.2


def test_capacity_provider_registers_virtual_active_injection():
    model = pyo.ConcreteModel()
    model.bus_phase_set = pyo.Set(initialize=[(1, "a")], dimen=2)
    model.time_set = pyo.RangeSet(0, 0)
    model.p_der_inj = pyo.Var(model.bus_phase_set, model.time_set, initialize=2)
    registry = InjectionRegistry()
    provider = CapacityExpansionProvider(case=None, zones={}, enabled=True)
    provider.register_injections(model, registry, None)

    assert pyo.value(registry.expression(model, 1, "a", 0, reactive=False)) == 2
    assert pyo.value(registry.expression(model, 1, "a", 0, reactive=True)) == 0
