"""Structural tests for capacity-expansion provider integration."""

import pandas as pd
import pyomo.environ as pyo

from distopf.pyomo_models.extensions.capacity_expansion import (
    add_capacity_expansion_as_fraction_of_load,
    add_capacity_expansion_p_flow_constraints,
)
from distopf.pyomo_models.extensions.capacity_expansion_provider import (
    CapacityExpansionProvider,
)
from distopf.pyomo_models.common.registry import DeviceRegistry


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
    registry = DeviceRegistry()
    provider = CapacityExpansionProvider(case=None, zones={}, enabled=True)
    registry.add(provider)

    assert pyo.value(registry.active_power_injection(model, 1, "a", 0)) == 2
    assert pyo.value(registry.reactive_power_injection(model, 1, "a", 0)) == 0


def test_capacity_balance_sums_named_generators_at_receiving_bus():
    model = pyo.ConcreteModel()
    model.branch_phase_set = pyo.Set(initialize=[(1, 2, "a"), (2, 3, "a")], dimen=3)
    model.bus_phase_set = pyo.Set(initialize=[(2, "a"), (3, "a")], dimen=2)
    model.gen_device_phase_set = pyo.Set(
        initialize=[("pv1", "a"), ("pv2", "a")], dimen=2
    )
    model.bat_phase_set = pyo.Set(initialize=[], dimen=2)
    model.time_set = pyo.RangeSet(0, 0)
    model.to_bus_map = {2: [(2, 3)], 3: []}
    model.gen_devices_by_bus_phase = {(2, "a"): ["pv1", "pv2"]}
    model.p_flow = pyo.Var(
        model.branch_phase_set,
        model.time_set,
        initialize={(1, 2, "a", 0): 1.0, (2, 3, "a", 0): 0.7},
    )
    model.p_load = pyo.Var(
        model.bus_phase_set,
        model.time_set,
        initialize={(2, "a", 0): 1.0, (3, "a", 0): 0.7},
    )
    model.p_gen = pyo.Var(
        model.gen_device_phase_set,
        model.time_set,
        initialize={("pv1", "a", 0): 0.2, ("pv2", "a", 0): 0.3},
    )
    model.p_bat = pyo.Var(model.bat_phase_set, model.time_set)
    model.p_der_inj = pyo.Var(
        model.bus_phase_set,
        model.time_set,
        initialize={(2, "a", 0): 0.2, (3, "a", 0): 0.0},
    )
    model.power_balance_p = pyo.Constraint(
        model.branch_phase_set,
        model.time_set,
        rule=lambda model, fb, tb, phase, time: model.p_flow[fb, tb, phase, time] == 0,
    )
    original = model.power_balance_p

    add_capacity_expansion_p_flow_constraints(model)

    assert model.power_balance_p is not original
    assert len(model.power_balance_p) == 2
    assert all(
        abs(pyo.value(constraint.body)) < 1e-9
        for constraint in model.power_balance_p.values()
    )
