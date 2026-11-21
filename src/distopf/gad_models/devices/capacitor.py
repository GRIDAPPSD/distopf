"""Shunt capacitors (GAD schema v1), one ganged switch per device.

``cap_data`` columns: ``device_name``, ``bus_name``, ``phases``, ``q`` (total
reactive power at 1 pu voltage, >= 0, split equally over the phases),
``status`` (initial state 0/1, default 1) and ``controller`` (informational).
Each phase injects ``q / n_phases * v2`` of its own bus-phase.

Without capacitor control the state is fixed at ``status``. With control
(``model.cap_mi_enabled``) every capacitor is switchable: one binary
``u_cap[device, t]`` switches all of its phases, ``status`` is its initial value,
and ``z_cap[device, phase, t] = u_cap * v2`` is linearized with McCormick bounds.
Read CSV names with dtype={"device_name": str, "bus_name": str}.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyomo.environ as pyo  # type: ignore

from distopf.gad_models.common.protocol import LindistModelProtocol
from distopf.gad_models.common.data import parse_phases
from distopf.gad_models.common.device_data import (
    NetworkContext,
    cell_num,
    cell_opt_float,
    cell_text,
    cell_required_float,
    check_columns,
    check_connection,
    check_device_names,
    check_phases,
    finish_validation,
    row_label,
)

_PHASES = ("a", "b", "c", "s1", "s2")
_REQUIRED = {"device_name", "bus_name", "phases", "q"}
_LEGACY = {"id", "name", "bus_id", *(f"q_{phase}" for phase in _PHASES)}


def validate_cap_data(data: pd.DataFrame, ctx: NetworkContext) -> None:
    """Validate capacitor names, connections, total q and initial status."""
    if data.empty:
        return
    check_columns(
        data,
        "cap_data",
        _REQUIRED,
        _LEGACY,
        "capacitors use 'device_name', 'bus_name' and one total 'q' per device",
    )
    errors: list[str] = []
    check_device_names(data, errors)
    for index, row in data.iterrows():
        label = row_label(row, index)
        cell_required_float(row, "q", label, errors, ge=0)
        status = cell_opt_float(row, "status", label, errors)
        if status not in (None, 0, 1):
            errors.append(f"{label}: status={status} must be 0 or 1")
        phases = check_phases(label, row["phases"], _PHASES, errors)
        if phases is not None:
            check_connection(label, cell_text(row["bus_name"]), phases, ctx, errors)
    finish_validation("cap_data", errors, [], [])


class CapacitorProvider:
    """Own capacitor components, voltage-dependent constraints, and Q injection."""

    name = "capacitors"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        ctx = NetworkContext.from_model(model, "CapacitorProvider")
        data = getattr(case, "cap_data", None)
        if data is None:
            data = pd.DataFrame()
        validate_cap_data(data, ctx)
        bus_by_device: dict[str, int] = {}
        phases_by_device: dict[str, list[str]] = {}
        q_nom: dict[str, float] = {}
        status: dict[str, int] = {}
        devices_by_bus_phase: dict[tuple[int, str], list[str]] = {}
        for _, row in data.iterrows():
            device = cell_text(row["device_name"])
            bus = ctx.bus_name_to_id[cell_text(row["bus_name"])]
            phases = parse_phases(cell_text(row["phases"]))
            bus_by_device[device] = bus
            phases_by_device[device] = phases
            q_nom[device] = float(row["q"])
            status[device] = int(cell_num(row, "status", 1))
            for phase in phases:
                devices_by_bus_phase.setdefault((bus, phase), []).append(device)
        ports = [(device, ph) for device, phs in phases_by_device.items() for ph in phs]
        model.cap_device_set = pyo.Set(initialize=list(bus_by_device))
        model.cap_device_phase_set = pyo.Set(initialize=ports, dimen=2)
        model.cap_bus_by_device = bus_by_device
        model.cap_phases_by_device = phases_by_device
        model.cap_devices_by_bus_phase = devices_by_bus_phase
        model.cap_q_nom = pyo.Param(model.cap_device_set, initialize=q_nom)
        model.cap_status = pyo.Param(
            model.cap_device_set, initialize=status, within=pyo.Binary
        )
        model.q_cap = pyo.Var(model.cap_device_phase_set, model.time_set, initialize=0)
        if getattr(model, "cap_mi_enabled", False):
            model.u_cap = pyo.Var(
                model.cap_device_set,
                model.time_set,
                domain=pyo.Binary,
                initialize=lambda m, device, t: status[device],
            )
            model.z_cap = pyo.Var(
                model.cap_device_phase_set,
                model.time_set,
                domain=pyo.NonNegativeReals,
                initialize=lambda m, device, ph, t: status[device],
            )

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return 0

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return sum(
            model.q_cap[device, phase, time]
            for device in model.cap_devices_by_bus_phase.get((bus, phase), [])
        )

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.cap_device_phase_set) == 0:
            return
        if getattr(model, "cap_mi_enabled", False):
            add_capacitor_mi_constraints(model)
            add_capacitor_mccormick_constraints(model)
            add_capacitor_z_bounds(model)
        else:
            add_capacitor_constraints(model)


def _q_per_phase(m: LindistModelProtocol, device: str) -> Any:
    """Nominal Q of one phase: the device total split equally over its phases."""
    return m.cap_q_nom[device] / len(m.cap_phases_by_device[device])


#  Capacitor Constraints (Standard and MI) ---------------------------------------------
def add_capacitor_constraints(m: LindistModelProtocol) -> None:
    """Fixed state: Q = status * nominal Q per phase * squared bus voltage."""

    def capacitor_rule(m: LindistModelProtocol, device, ph, t):
        bus = m.cap_bus_by_device[device]
        return (
            m.q_cap[device, ph, t]
            == m.cap_status[device] * _q_per_phase(m, device) * m.v2[bus, ph, t]
        )

    m.capacitor_injection = pyo.Constraint(
        m.cap_device_phase_set, m.time_set, rule=capacitor_rule
    )


def add_capacitor_mi_constraints(m: LindistModelProtocol) -> None:
    """Set Q = nominal Q per phase * z, where z = switch state * squared voltage."""

    def capacitor_q_rule(m: LindistModelProtocol, device, ph, t):
        return (
            m.q_cap[device, ph, t] == _q_per_phase(m, device) * m.z_cap[device, ph, t]
        )

    m.capacitor_mi_injection = pyo.Constraint(
        m.cap_device_phase_set, m.time_set, rule=capacitor_q_rule
    )


def add_capacitor_mccormick_constraints(m: LindistModelProtocol) -> None:
    """Enforce z = u * v2 per phase, with the device's one binary u shared by all phases."""

    def mccormick_upper_1(m: LindistModelProtocol, device, ph, t):
        """z_cap <= v_max^2 * u_cap"""
        v2_max = m.v_max[m.cap_bus_by_device[device], ph] ** 2
        return m.z_cap[device, ph, t] <= v2_max * m.u_cap[device, t]

    def mccormick_upper_2(m: LindistModelProtocol, device, ph, t):
        """z_cap <= v2"""
        return m.z_cap[device, ph, t] <= m.v2[m.cap_bus_by_device[device], ph, t]

    def mccormick_lower_1(m: LindistModelProtocol, device, ph, t):
        """z_cap >= v2 - v_max^2 * (1 - u_cap)"""
        v2_max = m.v_max[m.cap_bus_by_device[device], ph] ** 2
        return m.z_cap[device, ph, t] >= m.v2[
            m.cap_bus_by_device[device], ph, t
        ] - v2_max * (1 - m.u_cap[device, t])

    def mccormick_lower_2(m: LindistModelProtocol, device, ph, t):
        """z_cap >= v_min^2 * u_cap"""
        v2_min = m.v_min[m.cap_bus_by_device[device], ph] ** 2
        return m.z_cap[device, ph, t] >= v2_min * m.u_cap[device, t]

    m.cap_mccormick_u1 = pyo.Constraint(
        m.cap_device_phase_set, m.time_set, rule=mccormick_upper_1
    )
    m.cap_mccormick_u2 = pyo.Constraint(
        m.cap_device_phase_set, m.time_set, rule=mccormick_upper_2
    )
    m.cap_mccormick_l1 = pyo.Constraint(
        m.cap_device_phase_set, m.time_set, rule=mccormick_lower_1
    )
    m.cap_mccormick_l2 = pyo.Constraint(
        m.cap_device_phase_set, m.time_set, rule=mccormick_lower_2
    )


def add_capacitor_z_bounds(m: LindistModelProtocol) -> None:
    """Bound each device's auxiliary variable between zero and its bus's v_max^2."""

    def z_cap_bounds(m: LindistModelProtocol, device, ph, t):
        v2_max = m.v_max[m.cap_bus_by_device[device], ph] ** 2
        return (0, m.z_cap[device, ph, t], v2_max)

    m.z_cap_bounds = pyo.Constraint(
        m.cap_device_phase_set, m.time_set, rule=z_cap_bounds
    )


__all__ = ["CapacitorProvider", "validate_cap_data"]
