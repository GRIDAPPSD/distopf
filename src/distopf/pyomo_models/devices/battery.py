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
from distopf.pyomo_models.devices.injections import InjectionRegistry
from distopf.pyomo_models.devices.data import phase_tuples


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
