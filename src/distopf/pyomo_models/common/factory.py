"""Organized LinDistFlow composition root with legacy-compatible public API."""

from __future__ import annotations

import pyomo.environ as pyo  # type: ignore
from types import SimpleNamespace
from distopf.api import Case
from distopf.pyomo_models.devices.battery import BatteryProvider
from distopf.pyomo_models.devices.capacitor import CapacitorProvider
from distopf.pyomo_models.devices.generator import GeneratorProvider
from distopf.pyomo_models.devices.line_charging import LineChargingProvider
from distopf.pyomo_models.devices.load import LoadProvider
from distopf.pyomo_models.devices.regulator import RegulatorProvider
from distopf.pyomo_models.devices.transformer_shunt import TransformerShuntProvider
from distopf.pyomo_models.common.registry import DeviceRegistry
from distopf.pyomo_models.network.bfm import BFMProvider
from distopf.pyomo_models.common.protocol import LindistModelProtocol


# Options the providers read from the config; anything else is rejected.
BUILDER_OPTIONS = frozenset(
    {
        "linear",
        "circular_constraints",
        "thermal_constraints",
        "equality_only",
        "socp_relaxation",
        "free_swing_voltage",
        "free_boundary_loads",
        "voltage_slacks",
        "thermal_slacks",
        "reg_tap_change_limit",
    }
)


def create_model(
    case: Case,
    core=None,
    devices=(),
    control_capacitors: bool = False,
    control_regulators: bool = False,
    extra_options=(),
    **kwargs,
) -> LindistModelProtocol:
    """Build a LinDistFlow model from organized network and device modules.

    ``kwargs`` must be in ``BUILDER_OPTIONS`` or ``extra_options`` (options added
    by an extension's providers); unknown names raise ``TypeError``.
    """
    unknown = sorted(set(kwargs) - BUILDER_OPTIONS - set(extra_options))
    if unknown:
        raise TypeError(
            f"Unknown model option(s) {unknown}. "
            f"Valid options: {sorted(BUILDER_OPTIONS | set(extra_options))}"
        )
    model = pyo.ConcreteModel()
    model.cap_mi_enabled = control_capacitors
    model.reg_mi_enabled = control_regulators
    for key, value in kwargs.items():
        setattr(model, key, value)
    config = SimpleNamespace(
        control_capacitors=control_capacitors,
        control_regulators=control_regulators,
        **kwargs,
    )
    core.create_components(
        model,
        case,
        config=config,
    )

    registry = DeviceRegistry(devices)
    registry.create_components(model, case, config=config)
    model._device_registry = registry
    core.add_constraints(model, config=config)
    registry.add_constraints(model, config=config)
    return model


def create_lindist_model(
    case: Case,
    control_capacitors: bool = False,
    control_regulators: bool = False,
    **kwargs,
) -> LindistModelProtocol:
    core = BFMProvider()
    devices = [
        LoadProvider(),
        GeneratorProvider(),
        CapacitorProvider(),
        BatteryProvider(),
        RegulatorProvider(),
        LineChargingProvider(),
        TransformerShuntProvider(),
    ]
    return create_model(
        case=case,
        core=core,
        devices=devices,
        control_capacitors=control_capacitors,
        control_regulators=control_regulators,
        linear=True,
        **kwargs,
    )


def create_nl_branchflow_model(
    case: Case,
    control_capacitors: bool = False,
    control_regulators: bool = False,
    **kwargs,
) -> LindistModelProtocol:
    core = BFMProvider()
    devices = [
        LoadProvider(),
        GeneratorProvider(),
        CapacitorProvider(),
        BatteryProvider(),
        RegulatorProvider(),
        LineChargingProvider(),
        TransformerShuntProvider(),
    ]
    return create_model(
        case=case,
        core=core,
        devices=devices,
        control_capacitors=control_capacitors,
        control_regulators=control_regulators,
        linear=False,
        **kwargs,
    )
