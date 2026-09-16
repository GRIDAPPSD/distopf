"""Capacitor device provider for fixed and controlled shunt capacitors."""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models import common_constraints
from distopf.pyomo_models.devices.data import parse_phases, phase_tuples
from distopf.pyomo_models.devices.injections import InjectionRegistry


def create_capacitor_parameters(model: Any, case: Any) -> None:
    """Create capacitor parameter components from case data."""
    q_data = {(row.id, phase): getattr(row, f"q_{phase}", 0.0) for _, row in case.cap_data.iterrows() for phase in parse_phases(str(row.phases))}
    model.q_cap_nom = pyo.Param(model.cap_phase_set, initialize=q_data, default=0.0)


class CapacitorProvider:
    """Own capacitor components, voltage-dependent constraints, and Q injection."""

    name = "capacitors"
    supported_formulations = frozenset({"lindist", "nl_bfm"})

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        if not hasattr(model, "cap_phase_set"):
            model.cap_phase_set = pyo.Set(initialize=phase_tuples(case.cap_data), dimen=2)
        if not hasattr(model, "q_cap"):
            model.q_cap = pyo.Var(model.cap_phase_set, model.time_set)
        if not hasattr(model, "q_cap_nom"):
            create_capacitor_parameters(model, case)
        if getattr(model, "cap_mi_enabled", False) and not hasattr(model, "u_cap"):
            model.u_cap = pyo.Var(model.cap_phase_set, model.time_set, domain=pyo.Binary, initialize=1)
            model.z_cap = pyo.Var(model.cap_phase_set, model.time_set, domain=pyo.NonNegativeReals, initialize=1)

    def register_injections(
        self, model: Any, injections: InjectionRegistry, config: Any
    ) -> None:
        if any(provider.name == self.name for provider in injections.providers):
            return
        injections.add(
            self.name,
            q_term=lambda m, bus, phase, time: (
                m.q_cap[bus, phase, time]
                if (bus, phase, time) in m.q_cap
                else 0
            ),
        )

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.cap_phase_set) == 0:
            return
        if getattr(model, "cap_mi_enabled", False):
            if not hasattr(model, "capacitor_mi_injection"):
                common_constraints.add_capacitor_mi_constraints(model)
            if not hasattr(model, "cap_mccormick_u1"):
                common_constraints.add_capacitor_mccormick_constraints(model)
            if not hasattr(model, "z_cap_bounds"):
                common_constraints.add_capacitor_z_bounds(model)
        elif not hasattr(model, "capacitor_injection"):
            common_constraints.add_capacitor_constraints(model)


__all__ = ["CapacitorProvider", "create_capacitor_parameters"]
