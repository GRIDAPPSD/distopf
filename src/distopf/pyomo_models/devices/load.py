"""Load device provider and compatibility ownership for bus loads."""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models import common_constraints
from distopf.pyomo_models.devices.data import parse_phases
from distopf.pyomo_models.devices.injections import InjectionRegistry


def create_load_parameters(model: Any, case: Any) -> None:
    """Create load and CVR parameter components from case data."""
    p_data, q_data, cvr_p, cvr_q = {}, {}, {}, {}
    for _, row in case.bus_data.iterrows():
        for phase in parse_phases(str(row.phases)):
            if (row.id, phase) not in model.bus_phase_set:
                continue
            p_load = getattr(row, f"pl_{phase}", 0.0)
            q_load = getattr(row, f"ql_{phase}", 0.0)
            if phase in ("s1", "s2"):
                p_load += getattr(row, "pl_s1s2", 0.0) / 2
                q_load += getattr(row, "ql_s1s2", 0.0) / 2
            cvr_p[(row.id, phase)] = getattr(row, "cvr_p", 0.0)
            cvr_q[(row.id, phase)] = getattr(row, "cvr_q", 0.0)
            shape = getattr(row, "load_shape", "default")
            for time in model.time_set:
                multiplier_p = multiplier_q = 1.0
                if shape in case.schedules.columns:
                    multiplier_p = multiplier_q = case.schedules.at[time, shape]
                elif f"{shape}.{phase}.p" in case.schedules.columns:
                    multiplier_p = case.schedules.at[time, f"{shape}.{phase}.p"]
                    multiplier_q = case.schedules.at[time, f"{shape}.{phase}.q"]
                p_data[row.id, phase, time] = p_load * multiplier_p
                q_data[row.id, phase, time] = q_load * multiplier_q
    model.p_load_nom = pyo.Param(model.bus_phase_set, model.time_set, initialize=p_data, default=0.0)
    model.q_load_nom = pyo.Param(model.bus_phase_set, model.time_set, initialize=q_data, default=0.0)
    model.cvr_p = pyo.Param(model.bus_phase_set, initialize=cvr_p, default=0.0)
    model.cvr_q = pyo.Param(model.bus_phase_set, initialize=cvr_q, default=0.0)


class LoadProvider:
    """Own load parameters, variables, CVR constraints, and signed injections."""

    name = "loads"
    supported_formulations = frozenset({"lindist", "nl_bfm"})

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        """Create load components without replacing legacy factory components."""
        if not hasattr(model, "p_load"):
            model.p_load = pyo.Var(model.bus_phase_set, model.time_set)
        if not hasattr(model, "q_load"):
            model.q_load = pyo.Var(model.bus_phase_set, model.time_set)
        if not hasattr(model, "p_load_nom"):
            create_load_parameters(model, case)

    def register_injections(
        self, model: Any, injections: InjectionRegistry, config: Any
    ) -> None:
        """Register consumption as negative network injection."""
        if any(provider.name == self.name for provider in injections.providers):
            return
        injections.add(
            self.name,
            p_term=lambda m, bus, phase, time: (
                -m.p_load[bus, phase, time]
                if (bus, phase, time) in m.p_load
                else 0
            ),
            q_term=lambda m, bus, phase, time: (
                -m.q_load[bus, phase, time]
                if (bus, phase, time) in m.q_load
                else 0
            ),
        )

    def add_constraints(self, model: Any, config: Any) -> None:
        """Add voltage-dependent load equations exactly once."""
        if len(model.bus_phase_set) == 0 or hasattr(model, "cvr_p_load"):
            return
        free_boundary_loads = getattr(config, "free_boundary_loads", False)
        common_constraints.add_cvr_load_constraints(model, free_boundary_loads)


__all__ = ["LoadProvider", "create_load_parameters"]
