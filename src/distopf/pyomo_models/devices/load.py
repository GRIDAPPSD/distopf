"""Load device provider and compatibility ownership for bus loads."""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models.common.protocol import LindistModelProtocol
from distopf.pyomo_models.common.data import parse_phases


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
    model.p_load_nom = pyo.Param(
        model.bus_phase_set, model.time_set, initialize=p_data, default=0.0
    )
    model.q_load_nom = pyo.Param(
        model.bus_phase_set, model.time_set, initialize=q_data, default=0.0
    )
    model.cvr_p = pyo.Param(model.bus_phase_set, initialize=cvr_p, default=0.0)
    model.cvr_q = pyo.Param(model.bus_phase_set, initialize=cvr_q, default=0.0)


class LoadProvider:
    """Own load parameters, variables, CVR constraints, and signed injections."""

    name = "loads"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        if not hasattr(model, "p_load"):
            model.p_load = pyo.Var(model.bus_phase_set, model.time_set)
        if not hasattr(model, "q_load"):
            model.q_load = pyo.Var(model.bus_phase_set, model.time_set)
        if not hasattr(model, "p_load_nom"):
            create_load_parameters(model, case)

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        key = (bus, phase, time)
        return -model.p_load[key] if key in model.p_load else 0

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        key = (bus, phase, time)
        return -model.q_load[key] if key in model.q_load else 0

    def add_constraints(self, model: Any, config: Any) -> None:
        """Add voltage-dependent load equations exactly once."""
        if len(model.bus_phase_set) == 0 or hasattr(model, "cvr_p_load"):
            return
        free_boundary_loads = getattr(config, "free_boundary_loads", False)
        # add_cvr_load_constraints(model, free_boundary_loads)

        def cvr_p_rule(m: LindistModelProtocol, _id, ph, t):
            if free_boundary_loads and _id in m.boundary_out_set:
                return pyo.Constraint.Skip
            p_nom = m.p_load_nom[_id, ph, t]
            cvr_p = m.cvr_p[_id, ph]
            return m.p_load[_id, ph, t] == p_nom + cvr_p * p_nom / 2 * (
                m.v2[_id, ph, t] - 1
            )

        def cvr_q_rule(m: LindistModelProtocol, _id, ph, t):
            if free_boundary_loads and _id in m.boundary_out_set:
                return pyo.Constraint.Skip
            q_nom = m.q_load_nom[_id, ph, t]
            cvr_q = m.cvr_q[_id, ph]
            return m.q_load[_id, ph, t] == q_nom + cvr_q * q_nom / 2 * (
                m.v2[_id, ph, t] - 1
            )

        model.cvr_p_load = pyo.Constraint(
            model.bus_phase_set, model.time_set, rule=cvr_p_rule
        )
        model.cvr_q_load = pyo.Constraint(
            model.bus_phase_set, model.time_set, rule=cvr_q_rule
        )


__all__ = ["LoadProvider", "create_load_parameters"]
