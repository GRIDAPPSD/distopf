"""Fixed and switched shunt capacitors (vectorized CVXPY).

``cap_q_nom`` is an array ``(n_port,)`` aligned with ``cap_device_phase_set``;
``q_cap``, ``u_cap`` and ``z_cap`` have shape ``(n_port, n_steps)``. Switched
capacitors use binary ``u_cap`` with a McCormick envelope for
``z_cap = u_cap * v2``.
"""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np

from distopf.cvxpy_models.devices.capacitor import (
    _capacitor_data,
    validate_cap_data,
)
from distopf.cvxpy_models_performance.common.model import (
    CvxModel,
    IndexSet,
    add_constraint,
    add_ranged_constraint,
    column,
)
from distopf.cvxpy_models_performance.common.registry import port_to_bus_matrix
from distopf.pyomo_models.common.data import parse_phases
from distopf.pyomo_models.common.device_data import NetworkContext, cell_text


def create_capacitor_parameters(model: Any, case: Any) -> None:
    """Create ``cap_q_nom`` aligned with ``cap_device_phase_set``."""
    q_data = {
        (cell_text(row["device_name"]), phase): float(row[f"q_{phase}"])
        for _, row in _capacitor_data(model, case).iterrows()
        for phase in parse_phases(cell_text(row["phases"]))
    }
    model.cap_q_nom = np.array(
        [q_data[port] for port in model.cap_device_phase_set], dtype=float
    )


class CapacitorProvider:
    """Own capacitor components, voltage-dependent constraints, and Q injection."""

    name = "capacitors"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        ctx = NetworkContext.from_model(model, "CapacitorProvider")
        data = _capacitor_data(model, case)
        validate_cap_data(data, ctx)
        bus_by_device: dict[str, int] = {}
        ports: list[tuple[str, str]] = []
        devices_by_bus_phase: dict[tuple[int, str], list[str]] = {}
        for _, row in data.iterrows():
            device = cell_text(row["device_name"])
            bus = ctx.bus_name_to_id[cell_text(row["bus_name"])]
            bus_by_device[device] = bus
            for phase in parse_phases(cell_text(row["phases"])):
                ports.append((device, phase))
                devices_by_bus_phase.setdefault((bus, phase), []).append(device)
        model.cap_device_set = IndexSet(list(bus_by_device))
        model.cap_device_phase_set = IndexSet(ports)
        model.cap_bus_by_device = bus_by_device
        model.cap_devices_by_bus_phase = devices_by_bus_phase
        create_capacitor_parameters(model, case)
        model.add_var("q_cap", model.cap_device_phase_set, value=0.0)
        if getattr(model, "cap_mi_enabled", False):
            model.add_var("u_cap", model.cap_device_phase_set, boolean=True, value=1.0)
            model.add_var("z_cap", model.cap_device_phase_set, nonneg=True, value=1.0)
        model.cap_bus_matrix = port_to_bus_matrix(
            model, ports, lambda port: bus_by_device[port[0]], lambda port: port[1]
        )
        model.cap_v2_rows = np.array(
            [model.bus_phase_set.pos[(bus_by_device[d], ph)] for d, ph in ports],
            dtype=int,
        )

    def active_power_injection(self, model: Any) -> Any:
        return 0

    def reactive_power_injection(self, model: Any) -> Any:
        if len(model.cap_device_phase_set) == 0:
            return 0
        return model.cap_bus_matrix @ model.q_cap

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.cap_device_phase_set) == 0:
            return
        if getattr(model, "cap_mi_enabled", False):
            add_capacitor_mi_constraints(model)
            add_capacitor_mccormick_constraints(model)
            add_capacitor_z_bounds(model)
        else:
            add_capacitor_constraints(model)


def _v2_at_ports(m: CvxModel):
    return m.v2[m.cap_v2_rows, :]


def _port_v_max_sq(m: CvxModel):
    return column(m.v_max[m.cap_v2_rows] ** 2)


def _port_v_min_sq(m: CvxModel):
    return column(m.v_min[m.cap_v2_rows] ** 2)


def add_capacitor_constraints(m: CvxModel) -> None:
    """Scale each device's nominal Q by its bus's squared voltage."""
    add_constraint(
        m,
        "capacitor_injection",
        m.q_cap == cp.multiply(column(m.cap_q_nom), _v2_at_ports(m)),
        list(m.cap_device_phase_set),
    )


def add_capacitor_mi_constraints(m: CvxModel) -> None:
    """Set Q = nominal Q * z, where z represents switch state * squared voltage."""
    add_constraint(
        m,
        "capacitor_mi_injection",
        m.q_cap == cp.multiply(column(m.cap_q_nom), m.z_cap),
        list(m.cap_device_phase_set),
    )


def add_capacitor_mccormick_constraints(m: CvxModel) -> None:
    """Enforce z = u * v2 for binary u using the connected bus's voltage limits."""
    keys = list(m.cap_device_phase_set)
    v2 = _v2_at_ports(m)
    v2_max = _port_v_max_sq(m)
    v2_min = _port_v_min_sq(m)
    add_constraint(m, "cap_mccormick_u1", m.z_cap <= cp.multiply(v2_max, m.u_cap), keys)
    add_constraint(m, "cap_mccormick_u2", m.z_cap <= v2, keys)
    add_constraint(
        m, "cap_mccormick_l1", m.z_cap >= v2 - cp.multiply(v2_max, 1 - m.u_cap), keys
    )
    add_constraint(m, "cap_mccormick_l2", m.z_cap >= cp.multiply(v2_min, m.u_cap), keys)


def add_capacitor_z_bounds(m: CvxModel) -> None:
    """Bound each device's auxiliary variable between zero and its bus's v_max^2."""
    add_ranged_constraint(
        m,
        "z_cap_bounds",
        0,
        m.z_cap,
        np.broadcast_to(_port_v_max_sq(m), m.z_cap.shape),
        list(m.cap_device_phase_set),
    )


__all__ = ["CapacitorProvider", "create_capacitor_parameters", "validate_cap_data"]
