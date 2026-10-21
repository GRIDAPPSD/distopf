"""Composition root for CVXPY OPF models. Mirror of ``pyomo_models.common.factory``."""

from __future__ import annotations

from types import SimpleNamespace

from distopf.api import Case
from distopf.cvxpy_models.common.model import CvxModel
from distopf.cvxpy_models.common.registry import DeviceRegistry
from distopf.cvxpy_models.devices.battery import BatteryProvider
from distopf.cvxpy_models.devices.capacitor import CapacitorProvider
from distopf.cvxpy_models.devices.generator import GeneratorProvider
from distopf.cvxpy_models.devices.load import LoadProvider
from distopf.cvxpy_models.devices.regulator import RegulatorProvider
from distopf.cvxpy_models.network.bfm import BFMProvider


def create_model(
    case: Case,
    core=None,
    devices=(),
    control_capacitors: bool = False,
    control_regulators: bool = False,
    **kwargs,
) -> CvxModel:
    """Build a CVXPY model from organized network and device modules."""
    model = CvxModel()
    model.cap_mi_enabled = control_capacitors
    model.reg_mi_enabled = control_regulators
    for key, value in kwargs.items():
        setattr(model, key, value)
    config = SimpleNamespace(
        control_capacitors=control_capacitors,
        control_regulators=control_regulators,
        **kwargs,
    )
    core.create_components(model, case, config=config)

    registry = DeviceRegistry(devices)
    registry.create_components(model, case, config=config)
    model._device_registry = registry
    core.add_constraints(model, config=config)
    registry.add_constraints(model, config=config)
    return model


def _standard_devices() -> list:
    return [
        LoadProvider(),
        GeneratorProvider(),
        CapacitorProvider(),
        BatteryProvider(),
        RegulatorProvider(),
    ]


def create_lindist_model(
    case: Case,
    control_capacitors: bool = False,
    control_regulators: bool = False,
    **kwargs,
) -> CvxModel:
    return create_model(
        case=case,
        core=BFMProvider(),
        devices=_standard_devices(),
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
) -> CvxModel:
    """Build the BranchFlow model.

    With ``socp_relaxation=True`` the model is convex and solvable with CVXPY.
    Without it the current equalities are non-convex; the model can be built
    and inspected but ``solve`` raises ``NotImplementedError``.
    """
    return create_model(
        case=case,
        core=BFMProvider(),
        devices=_standard_devices(),
        control_capacitors=control_capacitors,
        control_regulators=control_regulators,
        linear=False,
        **kwargs,
    )
