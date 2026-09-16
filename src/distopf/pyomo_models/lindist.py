"""Organized LinDistFlow composition root with legacy-compatible public API."""

from __future__ import annotations

from types import SimpleNamespace

import pyomo.environ as pyo  # type: ignore

from distopf.api import Case
from distopf.pyomo_models.devices.battery import BatteryProvider
from distopf.pyomo_models.devices.capacitor import CapacitorProvider
from distopf.pyomo_models.devices.generator import GeneratorProvider
from distopf.pyomo_models.devices.load import LoadProvider
from distopf.pyomo_models.devices.regulator import RegulatorProvider
from distopf.pyomo_models.devices.registry import DeviceRegistry
from distopf.pyomo_models.devices.data import parse_phases, phase_tuples
from distopf.pyomo_models.devices.injections import InjectionRegistry
from distopf.pyomo_models.network.core_model import (
    create_network_components,
    create_network_operating_parameters,
)
from distopf.pyomo_models.protocol import LindistModelProtocol


def _create_core_variables(model: pyo.ConcreteModel) -> None:
    """Create variables owned by the core network formulation."""
    model.v2 = pyo.Var(model.bus_phase_set, model.time_set, domain=pyo.NonNegativeReals, initialize=1)
    model.p_flow = pyo.Var(model.branch_phase_set, model.time_set)
    model.q_flow = pyo.Var(model.branch_phase_set, model.time_set, initialize=0)
    model.v2_reg = pyo.Var(model.reg_phase_set, model.time_set, domain=pyo.NonNegativeReals, initialize=1)


def _create_device_sets(model: pyo.ConcreteModel, case: Case) -> None:
    """Create shared device indexes before device providers run."""
    model.gen_phase_set = pyo.Set(initialize=phase_tuples(case.gen_data), dimen=2)
    model.cap_phase_set = pyo.Set(initialize=phase_tuples(case.cap_data), dimen=2)
    model.reg_phase_set = pyo.Set(initialize=[(int(row.fb), int(row.tb), phase) for _, row in case.reg_data.iterrows() for phase in parse_phases(str(row.phases))], dimen=3)
    model.bat_phase_set = pyo.Set(initialize=phase_tuples(case.bat_data, "id"), dimen=2)
    model.bat_set = pyo.Set(initialize=case.bat_data.id.tolist())



def create_lindist_model(case: Case, control_capacitors: bool = False, control_regulators: bool = False, device_providers=None) -> LindistModelProtocol:
    """Build a LinDistFlow model from organized network and device modules."""
    model = pyo.ConcreteModel()
    model.cap_mi_enabled = control_capacitors
    model.reg_mi_enabled = control_regulators
    model.delta_t = pyo.Param(initialize=case.delta_t)
    model.start_step = pyo.Param(initialize=case.start_step)
    model.n_steps = pyo.Param(initialize=case.n_steps)
    create_network_components(model, case)
    _create_device_sets(model, case)
    create_network_operating_parameters(model, case)
    _create_core_variables(model)

    registry = DeviceRegistry([LoadProvider(), GeneratorProvider(), CapacitorProvider(), BatteryProvider(), RegulatorProvider()])
    if device_providers is not None:
        registry.extend(device_providers)
    registry.create_components(model, case, config=None)
    model._device_registry = registry
    return model


def add_constraints(model: pyo.ConcreteModel, circular_constraints: bool = True, thermal_constraints: bool = False, equality_only: bool = False, control_capacitors: bool = False, control_regulators: bool = False, reg_tap_change_limit: int | None = None, free_swing_voltage: bool = False, free_boundary_loads: bool = False, injection_registry=None, device_providers=None) -> None:
    """Add LinDistFlow constraints while preserving the established API."""
    from distopf.pyomo_models.lindist_constraints import add_constraints as add_lindist_constraints

    if injection_registry is not None:
        if not isinstance(injection_registry, InjectionRegistry):
            raise TypeError("injection_registry must be an InjectionRegistry")
        model._injection_registry = injection_registry
    injections = getattr(model, "_injection_registry", None)
    if injections is None:
        injections = InjectionRegistry()
        model._injection_registry = injections
    providers = getattr(model, "_device_registry", None)
    if device_providers is not None:
        providers = DeviceRegistry([LoadProvider(), GeneratorProvider(), CapacitorProvider(), BatteryProvider(), RegulatorProvider()])
        providers.extend(device_providers)
        model._device_registry = providers
    if providers is not None:
        providers.register_injections(model, injections, config=None)
        config = SimpleNamespace(
            circular_constraints=circular_constraints,
            equality_only=equality_only,
            reg_tap_change_limit=reg_tap_change_limit,
            free_boundary_loads=free_boundary_loads,
        )
        providers.add_constraints(model, config=config)
    add_lindist_constraints(model, circular_constraints=circular_constraints, thermal_constraints=thermal_constraints, equality_only=equality_only, control_capacitors=control_capacitors, control_regulators=control_regulators, reg_tap_change_limit=reg_tap_change_limit, free_swing_voltage=free_swing_voltage, free_boundary_loads=free_boundary_loads)


class LinDistModel:
    """Convenience wrapper around the organized factory and constraint builder."""

    def __init__(self, case: Case, circular_constraints: bool = True, thermal_constraints: bool = False, equality_only: bool = False, cap_mi: bool = False, reg_mi: bool = False, reg_tap_change_limit: int | None = None):
        self.case = case
        self.model = create_lindist_model(case, control_capacitors=cap_mi, control_regulators=reg_mi)
        add_constraints(self.model, circular_constraints=circular_constraints, thermal_constraints=thermal_constraints, equality_only=equality_only, control_capacitors=cap_mi, control_regulators=reg_mi, reg_tap_change_limit=reg_tap_change_limit)
