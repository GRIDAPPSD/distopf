"""Organized LinDistFlow composition root with legacy-compatible public API."""

from __future__ import annotations

import pyomo.environ as pyo  # type: ignore
from types import SimpleNamespace
from distopf.api import Case
from distopf.pyomo_models.devices.battery import BatteryProvider
from distopf.pyomo_models.devices.capacitor import CapacitorProvider
from distopf.pyomo_models.devices.generator import GeneratorProvider
from distopf.pyomo_models.devices.load import LoadProvider
from distopf.pyomo_models.devices.regulator import RegulatorProvider
from distopf.pyomo_models.common.registry import DeviceRegistry
from distopf.pyomo_models.network.bfm import BFMProvider
from distopf.pyomo_models.common.protocol import LindistModelProtocol
from distopf.pyomo_models.devices.mpssd import MpssdProvider


def create_model(
    case: Case,
    core=None,
    devices=(),
    control_capacitors: bool = False,
    control_regulators: bool = False,
    **kwargs,
) -> LindistModelProtocol:
    """Build a LinDistFlow model from organized network and device modules."""
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

def create_mpssd_lindist_model(
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
        MpssdProvider(),
    ]
    return create_model(
        case=case,
        core=core,
        devices=devices,
        control_capacitors=control_capacitors,
        control_regulators=control_regulators,
        linear=True,
        mpssd=True,
        **kwargs,
    )