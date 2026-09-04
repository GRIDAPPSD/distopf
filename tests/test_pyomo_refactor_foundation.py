"""Tests for the composable Pyomo refactor foundation."""

import pandas as pd
import pyomo.environ as pyo

from distopf.pyomo_models.device_data import (
    create_bus_device_map,
    normalize_device_table,
)
from distopf.pyomo_models.device_registry import DeviceRegistry
from distopf.pyomo_models.injection_providers import MappedInjectionProvider
from distopf.pyomo_models.injection_registry import InjectionRegistry


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


def test_injection_registry_combines_signed_pyomo_terms():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(initialize=2)
    model.y = pyo.Var(initialize=3)
    registry = InjectionRegistry()
    registry.add("load", p_term=lambda m, *_: -m.x)
    registry.add("generator", p_term=lambda m, *_: m.y)

    expr = registry.expression(model, 1, "a", 0, reactive=False)
    assert pyo.value(expr) == 1


def test_mapped_provider_aggregates_multiple_entities_at_one_bus():
    model = pyo.ConcreteModel()
    model.entity_set = pyo.Set(initialize=[("g0", "a"), ("g1", "a")], dimen=2)
    model.entity_bus = pyo.Param(
        pyo.Set(initialize=["g0", "g1"]), initialize={"g0": 7, "g1": 7}
    )
    model.p = pyo.Var(model.entity_set, initialize={("g0", "a"): 1, ("g1", "a"): 2})
    registry = InjectionRegistry()
    MappedInjectionProvider(
        "entities",
        entity_set="entity_set",
        bus_map="entity_bus",
        p_term=lambda m, device, ph, t: m.p[device, ph],
    ).register(model, registry)

    assert pyo.value(registry.expression(model, 7, "a", 0, reactive=False)) == 3


def test_device_registry_rejects_duplicate_provider_names():
    registry = DeviceRegistry()
    provider = type(
        "Provider",
        (),
        {
            "name": "test",
            "supported_formulations": frozenset({"lindist"}),
        },
    )()
    registry.add(provider)

    try:
        registry.add(provider)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("duplicate providers should be rejected")
