"""Tests for the composable Pyomo refactor foundation."""

import pandas as pd
import pyomo.environ as pyo

from distopf.pyomo_models.common.data import (
    create_bus_device_map,
    normalize_device_table,
    parse_phases,
)
from distopf.pyomo_models.common.registry import DeviceRegistry
from distopf.pyomo_models.common.injection_providers import MappedInjectionProvider
from distopf.pyomo_models.common.objectives import substation_cost_objective_rule


def test_parse_phases_preserves_order_and_repeated_triplex_phases():
    assert parse_phases("abc") == ["a", "b", "c"]
    assert parse_phases("ab") == ["a", "b"]
    assert parse_phases("ba") == ["b", "a"]
    assert parse_phases("s1") == ["s1"]
    assert parse_phases("s1s2") == ["s1", "s2"]
    assert parse_phases("s2s1") == ["s2", "s1"]
    assert parse_phases("s1s1") == ["s1", "s1"]
    assert parse_phases("s2s2") == ["s2", "s2"]


def test_normalize_legacy_rows_supports_duplicate_bus_devices():
    table = normalize_device_table(
        pd.DataFrame(
            {
                "id": [7, 7],
                "phases": ["a", "a"],
                "p_a": [1.0, 2.0],
            }
        ),
        kind="generator",
    )

    assert table.data.bus_id.tolist() == [7, 7]
    assert table.ids == ("generator_0", "generator_1")
    assert create_bus_device_map(table)[(7, "a")] == ["generator_0", "generator_1"]


def test_device_registry_combines_signed_pyomo_terms():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(initialize=2)
    model.y = pyo.Var(initialize=3)

    class LoadProvider:
        name = "load"

        def active_power_injection(self, model, bus, phase, time):
            return -model.x

        def reactive_power_injection(self, model, bus, phase, time):
            return 0

    class GeneratorProvider:
        name = "generator"

        def active_power_injection(self, model, bus, phase, time):
            return model.y

        def reactive_power_injection(self, model, bus, phase, time):
            return 0

    registry = DeviceRegistry()
    registry.add(LoadProvider())
    registry.add(GeneratorProvider())

    assert pyo.value(registry.active_power_injection(model, 1, "a", 0)) == 1


def test_mapped_provider_aggregates_multiple_entities_at_one_bus():
    model = pyo.ConcreteModel()
    model.entity_set = pyo.Set(initialize=[("g0", "a"), ("g1", "a")], dimen=2)
    model.entity_bus = pyo.Param(
        pyo.Set(initialize=["g0", "g1"]), initialize={"g0": 7, "g1": 7}
    )
    model.p = pyo.Var(model.entity_set, initialize={("g0", "a"): 1, ("g1", "a"): 2})
    registry = DeviceRegistry()
    registry.add(
        MappedInjectionProvider(
            "entities",
            entity_set="entity_set",
            bus_map="entity_bus",
            p_term=lambda m, device, ph, t: m.p[device, ph],
        )
    )

    assert pyo.value(registry.active_power_injection(model, 7, "a", 0)) == 3


def test_device_registry_defers_injection_terms_until_expression_build():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(initialize=4)
    calls = []

    class Provider:
        name = "deferred"

        def active_power_injection(self, model, bus, phase, time):
            calls.append(True)
            return model.x

        def reactive_power_injection(self, model, bus, phase, time):
            return 0

    registry = DeviceRegistry()
    registry.add(Provider())

    assert calls == []
    expression = registry.active_power_injection(model, 1, "a", 0)
    assert calls == [True]
    assert pyo.value(expression) == 4


def test_device_registry_sums_reactive_injections_separately():
    model = pyo.ConcreteModel()
    model.q = pyo.Var(initialize=6)

    class Provider:
        name = "test"

        def active_power_injection(self, model, bus, phase, time):
            return 0

        def reactive_power_injection(self, model, bus, phase, time):
            return model.q

    registry = DeviceRegistry()
    registry.add(Provider())

    assert pyo.value(registry.reactive_power_injection(model, 1, "a", 0)) == 6


def test_device_registry_rejects_duplicate_provider_names():
    provider = type(
        "Provider",
        (),
        {
            "name": "test",
        },
    )()

    try:
        DeviceRegistry([provider, provider])
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate providers should be rejected")


def test_substation_cost_objective_uses_branch_endpoints():
    model = pyo.ConcreteModel()
    model.branch_phase_set = pyo.Set(initialize=[(1, 2, "a"), (2, 3, "a")], dimen=3)
    model.time_set = pyo.RangeSet(0, 0)
    model.swing_bus_set = pyo.Set(initialize=[1])
    model.p_flow = pyo.Var(
        model.branch_phase_set,
        model.time_set,
        initialize={(1, 2, "a", 0): 2, (2, 3, "a", 0): 7},
    )
    model.schedule_price = pyo.Param(model.time_set, initialize={0: 3})

    assert pyo.value(substation_cost_objective_rule(model)) == 6
