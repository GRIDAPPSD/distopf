"""Capacitor device provider for fixed and controlled shunt capacitors."""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models.common.protocol import LindistModelProtocol
from distopf.pyomo_models.common.data import parse_phases, phase_tuples


def create_capacitor_parameters(model: Any, case: Any) -> None:
    """Create capacitor parameter components from case data."""
    q_data = {
        (row.id, phase): getattr(row, f"q_{phase}", 0.0)
        for _, row in case.cap_data.iterrows()
        for phase in parse_phases(str(row.phases))
    }
    model.q_cap_nom = pyo.Param(model.cap_phase_set, initialize=q_data, default=0.0)


class CapacitorProvider:
    """Own capacitor components, voltage-dependent constraints, and Q injection."""

    name = "capacitors"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        if not hasattr(model, "cap_phase_set"):
            model.cap_phase_set = pyo.Set(
                initialize=phase_tuples(case.cap_data), dimen=2
            )
        if not hasattr(model, "q_cap"):
            model.q_cap = pyo.Var(model.cap_phase_set, model.time_set)
        if not hasattr(model, "q_cap_nom"):
            create_capacitor_parameters(model, case)
        if getattr(model, "cap_mi_enabled", False) and not hasattr(model, "u_cap"):
            model.u_cap = pyo.Var(
                model.cap_phase_set, model.time_set, domain=pyo.Binary, initialize=1
            )
            model.z_cap = pyo.Var(
                model.cap_phase_set,
                model.time_set,
                domain=pyo.NonNegativeReals,
                initialize=1,
            )

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return 0

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        key = (bus, phase, time)
        return model.q_cap[key] if key in model.q_cap else 0

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.cap_phase_set) == 0:
            return
        if getattr(model, "cap_mi_enabled", False):
            if not hasattr(model, "capacitor_mi_injection"):
                add_capacitor_mi_constraints(model)
            if not hasattr(model, "cap_mccormick_u1"):
                add_capacitor_mccormick_constraints(model)
            if not hasattr(model, "z_cap_bounds"):
                add_capacitor_z_bounds(model)
        elif not hasattr(model, "capacitor_injection"):
            add_capacitor_constraints(model)


#  Capacitor Constraints (Standard and MI) ---------------------------------------------
def add_capacitor_constraints(m: LindistModelProtocol) -> None:
    """
    Add capacitor constraints.
    q_C = q_rated * v^2
    """

    def capacitor_rule(m: LindistModelProtocol, _id, ph, t):
        return m.q_cap[_id, ph, t] == m.q_cap_nom[_id, ph] * m.v2[_id, ph, t]

    m.capacitor_injection = pyo.Constraint(
        m.cap_phase_set, m.time_set, rule=capacitor_rule
    )


def add_capacitor_mi_constraints(m: LindistModelProtocol) -> None:
    """
    Add mixed-integer capacitor constraints using McCormick envelope.

    q_cap = q_cap_nom * z_cap

    where z_cap represents the product u_cap * v2.
    """

    def capacitor_q_rule(m: LindistModelProtocol, _id, ph, t):
        return m.q_cap[_id, ph, t] == m.q_cap_nom[_id, ph] * m.z_cap[_id, ph, t]

    m.capacitor_mi_injection = pyo.Constraint(
        m.cap_phase_set, m.time_set, rule=capacitor_q_rule
    )


def add_capacitor_mccormick_constraints(m: LindistModelProtocol) -> None:
    """
    Add McCormick envelope constraints to linearize z_cap = u_cap * v2.

    For binary u in {0,1} and continuous v2 in [v_min^2, v_max^2]:
        z <= v_max^2 * u           (when u=0, z=0)
        z <= v2                    (z bounded by v2)
        z >= v2 - v_max^2 * (1-u)  (when u=1, z=v2)
        z >= v_min^2 * u           (when u=1, z >= v_min^2)
    """

    def mccormick_upper_1(m: LindistModelProtocol, _id, ph, t):
        """z_cap <= v_max^2 * u_cap"""
        v2_max = m.v_max[_id, ph] ** 2
        return m.z_cap[_id, ph, t] <= v2_max * m.u_cap[_id, ph, t]

    def mccormick_upper_2(m: LindistModelProtocol, _id, ph, t):
        """z_cap <= v2"""
        return m.z_cap[_id, ph, t] <= m.v2[_id, ph, t]

    def mccormick_lower_1(m: LindistModelProtocol, _id, ph, t):
        """z_cap >= v2 - v_max^2 * (1 - u_cap)"""
        v2_max = m.v_max[_id, ph] ** 2
        return m.z_cap[_id, ph, t] >= m.v2[_id, ph, t] - v2_max * (
            1 - m.u_cap[_id, ph, t]
        )

    def mccormick_lower_2(m: LindistModelProtocol, _id, ph, t):
        """z_cap >= v_min^2 * u_cap"""
        v2_min = m.v_min[_id, ph] ** 2
        return m.z_cap[_id, ph, t] >= v2_min * m.u_cap[_id, ph, t]

    m.cap_mccormick_u1 = pyo.Constraint(
        m.cap_phase_set, m.time_set, rule=mccormick_upper_1
    )
    m.cap_mccormick_u2 = pyo.Constraint(
        m.cap_phase_set, m.time_set, rule=mccormick_upper_2
    )
    m.cap_mccormick_l1 = pyo.Constraint(
        m.cap_phase_set, m.time_set, rule=mccormick_lower_1
    )
    m.cap_mccormick_l2 = pyo.Constraint(
        m.cap_phase_set, m.time_set, rule=mccormick_lower_2
    )


def add_capacitor_z_bounds(m: LindistModelProtocol) -> None:
    """
    Add explicit bounds on z_cap auxiliary variable.

    0 <= z_cap <= v_max^2
    """

    def z_cap_bounds(m: LindistModelProtocol, _id, ph, t):
        v2_max = m.v_max[_id, ph] ** 2
        return (0, m.z_cap[_id, ph, t], v2_max)

    m.z_cap_bounds = pyo.Constraint(m.cap_phase_set, m.time_set, rule=z_cap_bounds)


__all__ = ["CapacitorProvider", "create_capacitor_parameters"]
