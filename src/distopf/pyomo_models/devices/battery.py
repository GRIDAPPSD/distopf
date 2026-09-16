"""Battery provider for the legacy-index migration path.

This provider owns battery-specific model components and constraints while
preserving the current one-battery-per-bus component names. It is the first
concrete provider for a built-in device family and is intentionally compatible
with the existing LinDistFlow and BranchFlow model factories.
"""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models import common_constraints
from distopf.pyomo_models.devices.data import parse_phases, phase_tuples
from distopf.pyomo_models.devices.injections import InjectionRegistry
from distopf.pyomo_models.model_types import CONTROL_VARIABLE_MAP


def create_battery_parameters(model: Any, case: Any) -> None:
    """Create every battery parameter consumed by common LinDistFlow constraints."""
    p_data, q_data, rating, q_min, q_max = {}, {}, {}, {}, {}
    energy, soc_min, soc_max, start_soc = {}, {}, {}, {}
    charge_eff, discharge_eff, cycles, control = {}, {}, {}, {}
    has_phase = {(device, phase): False for device in model.bat_set for phase in ("a", "b", "c")}
    has_a, has_b, has_c, n_phases = {}, {}, {}, {}
    for _, row in case.bat_data.iterrows():
        device = row.id
        phases = parse_phases(str(row.phases))
        count = len(phases)
        n_phases[device] = count
        has_a[device], has_b[device], has_c[device] = "a" in phases, "b" in phases, "c" in phases
        for phase in ("a", "b", "c"):
            has_phase[(device, phase)] = phase in phases
        energy[device] = getattr(row, "energy_capacity", 0)
        soc_min[device] = getattr(row, "min_soc", 0)
        soc_max[device] = getattr(row, "max_soc", 1)
        start_soc[device] = getattr(row, "start_soc", 0.5)
        charge_eff[device] = getattr(row, "charge_efficiency", 1)
        discharge_eff[device] = getattr(row, "discharge_efficiency", 1)
        cycles[device] = getattr(row, "annual_cycle_limit", 365)
        control[device] = CONTROL_VARIABLE_MAP[getattr(row, "control_variable", "P")]
        for phase in phases:
            key = (device, phase)
            if key not in model.bat_phase_set:
                continue
            s_max = getattr(row, "s_max", 1000.0) / count
            rating[key] = s_max
            q_min[key] = getattr(row, "q_min", -getattr(row, "s_max", 1000.0)) / count
            q_max[key] = getattr(row, "q_max", getattr(row, "s_max", 1000.0)) / count
            for time in model.time_set:
                p_data[(*key, time)] = getattr(row, "p", 0.0) / count
                q_data[(*key, time)] = getattr(row, "q", 0.0) / count
    model.p_bat_nom = pyo.Param(model.bat_phase_set, model.time_set, initialize=p_data, default=0.0)
    model.q_bat_nom = pyo.Param(model.bat_phase_set, model.time_set, initialize=q_data, default=0.0)
    model.s_bat_rated = pyo.Param(model.bat_phase_set, initialize=rating, default=1000.0)
    model.q_bat_min = pyo.Param(model.bat_phase_set, initialize=q_min, default=-1000.0)
    model.q_bat_max = pyo.Param(model.bat_phase_set, initialize=q_max, default=1000.0)
    model.bat_control_type = pyo.Param(model.bat_set, initialize=control, default=0)
    for name, values, default in (("energy_capacity", energy, 0), ("soc_min", soc_min, 0), ("soc_max", soc_max, 1), ("start_soc", start_soc, 0.5), ("charge_efficiency", charge_eff, 1.0), ("discharge_efficiency", discharge_eff, 1.0), ("annual_cycle_limit", cycles, 365), ("battery_has_a_phase", has_a, True), ("battery_has_b_phase", has_b, True), ("battery_has_c_phase", has_c, True), ("battery_n_phases", n_phases, 3)):
        setattr(model, name, pyo.Param(model.bat_set, initialize=values, default=default))
    model.battery_has_phase = pyo.Param(model.bat_set, ("a", "b", "c"), initialize=has_phase, default=True)


class BatteryProvider:
    """Create and constrain battery variables for a Pyomo model."""

    name = "batteries"
    supported_formulations = frozenset({"lindist", "nl_bfm"})

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        """Attach battery components only when the factory did not create them."""
        if not hasattr(model, "bat_phase_set"):
            model.bat_phase_set = pyo.Set(
                initialize=phase_tuples(case.bat_data, "id"), dimen=2
            )
        if not hasattr(model, "bat_set"):
            model.bat_set = pyo.Set(initialize=case.bat_data.id.tolist())

        if not hasattr(model, "p_charge"):
            model.p_charge = pyo.Var(model.bat_set, model.time_set, initialize=0)
        if not hasattr(model, "p_discharge"):
            model.p_discharge = pyo.Var(model.bat_set, model.time_set, initialize=0)
        if not hasattr(model, "p_bat"):
            model.p_bat = pyo.Var(model.bat_phase_set, model.time_set, initialize=0)
        if not hasattr(model, "q_bat"):
            model.q_bat = pyo.Var(model.bat_phase_set, model.time_set, initialize=0)
        if not hasattr(model, "soc"):
            model.soc = pyo.Var(model.bat_set, model.time_set, initialize=0.5)
        if not hasattr(model, "p_bat_nom"):
            create_battery_parameters(model, case)

    def register_injections(
        self, model: Any, injections: InjectionRegistry, config: Any
    ) -> None:
        """Register battery net active and reactive injection."""
        if any(provider.name == self.name for provider in injections.providers):
            return
        injections.add(
            self.name,
            p_term=lambda m, bus, phase, time: (
                m.p_bat[bus, phase, time] if (bus, phase, time) in m.p_bat else 0
            ),
            q_term=lambda m, bus, phase, time: (
                m.q_bat[bus, phase, time] if (bus, phase, time) in m.q_bat else 0
            ),
        )

    def add_constraints(self, model: Any, config: Any) -> None:
        """Attach shared battery operating constraints once."""
        if len(model.bat_set) == 0:
            return
        for name, builder in (
            (
                "battery_constant_q_bat",
                common_constraints.add_battery_constant_q_constraints_p_control,
            ),
            ("storage", common_constraints.add_battery_energy_constraints),
            (
                "net_discharge",
                common_constraints.add_battery_net_p_bat_equal_phase_constraints,
            ),
        ):
            if not hasattr(model, name):
                builder(model)

        equality_only = getattr(config, "equality_only", False) if config else False
        if equality_only:
            return
        if not hasattr(model, "battery_discharging_limits"):
            common_constraints.add_battery_power_limits(model)
        if not hasattr(model, "battery_soc_limits"):
            common_constraints.add_battery_soc_limits(model)
        circular = getattr(config, "circular_constraints", True) if config else True
        if circular and not hasattr(model, "bat_circle_constraint"):
            common_constraints.add_circular_battery_constraints(model)


__all__ = ["BatteryProvider", "create_battery_parameters"]
