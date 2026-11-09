"""Named battery devices with balanced phase power and SOC dynamics (vectorized CVXPY).

Array layout: ``p_bat``, ``q_bat`` are ``(n_port, n_steps)`` (ports are
``bat_phase_set``); ``p_charge``, ``p_discharge``, ``soc`` are
``(n_device, n_steps)``. Per-port parameters (``s_bat_rated``, ``q_bat_min``,
``q_bat_max``) have shape ``(n_port,)``, ``p_bat_nom`` and ``q_bat_nom`` are
``(n_port, n_steps)``, and per-device parameters (``energy_capacity``,
``soc_min``, ``soc_max``, ``start_soc``, ``charge_efficiency``,
``discharge_efficiency``, ``annual_cycle_limit``, ``battery_n_phases``) have
shape ``(n_device,)``. ``bat_control_type`` is a list per device and
``bat_port_device`` maps each port to its device row.
"""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np

from distopf.cvxpy_models.devices.battery import (
    create_battery_parameters as _scalar_parameters,
)
from distopf.cvxpy_models_performance.common.model import (
    CvxModel,
    IndexSet,
    add_constraint,
    add_constraint_on_rows,
    add_ranged_constraint,
    column,
    select_rows,
)
from distopf.cvxpy_models_performance.common.registry import port_to_bus_matrix
from distopf.pyomo_models.common.data import parse_phases
from distopf.pyomo_models.common.device_data import (
    NetworkContext,
    cell_text,
)
from distopf.pyomo_models.common.model_types import ControlVariable
from distopf.pyomo_models.devices.battery import _battery_data, validate_bat_data


def create_battery_parameters(model: Any, case: Any) -> None:
    """Create battery parameters as arrays (see module docstring for shapes)."""
    _scalar_parameters(model, case)
    devices = list(model.bat_set)
    ports = list(model.bat_phase_set)
    times = list(model.time_set)
    n_t = len(times)
    model.bat_port_device = np.array(
        [model.bat_set.pos[device] for device, _ in ports], dtype=int
    )

    def per_device(name: str) -> np.ndarray:
        source = getattr(model, name)
        return np.array([source[device] for device in devices], dtype=float)

    def per_port(name: str) -> np.ndarray:
        source = getattr(model, name)
        return np.array([source[port] for port in ports], dtype=float)

    def per_port_time(name: str) -> np.ndarray:
        source = getattr(model, name)
        return np.array(
            [[source[(*port, t)] for t in times] for port in ports], dtype=float
        ).reshape(len(ports), n_t)

    model.bat_control_type = [model.bat_control_type[device] for device in devices]
    for name in (
        "energy_capacity",
        "soc_min",
        "soc_max",
        "start_soc",
        "charge_efficiency",
        "discharge_efficiency",
        "annual_cycle_limit",
        "battery_n_phases",
    ):
        setattr(model, name, per_device(name))
    for name in ("s_bat_rated", "q_bat_min", "q_bat_max"):
        setattr(model, name, per_port(name))
    for name in ("p_bat_nom", "q_bat_nom"):
        setattr(model, name, per_port_time(name))


class BatteryProvider:
    """Create and constrain battery variables for a vectorized CVXPY model."""

    name = "batteries"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        ctx = NetworkContext.from_model(model, "BatteryProvider")
        data = _battery_data(model, case)
        validate_bat_data(data, ctx)
        bus_by_device: dict[str, int] = {}
        ports: list[tuple[str, str]] = []
        devices_by_bus_phase: dict[tuple[int, str], list[str]] = {}
        phases_by_device: dict[str, list[str]] = {}
        for _, row in data.iterrows():
            device = cell_text(row["device_name"])
            bus = ctx.bus_name_to_id[cell_text(row["bus_name"])]
            phases_by_device[device] = parse_phases(cell_text(row["phases"]))
            bus_by_device[device] = bus
            for phase in phases_by_device[device]:
                ports.append((device, phase))
                devices_by_bus_phase.setdefault((bus, phase), []).append(device)
        model.bat_set = IndexSet(list(bus_by_device))
        model.bat_phase_set = IndexSet(ports)
        model.bat_bus_by_device = bus_by_device
        model.bat_devices_by_bus_phase = devices_by_bus_phase
        model.bat_phases_by_device = phases_by_device
        create_battery_parameters(model, case)
        model.add_var("p_charge", model.bat_set, value=0.0)
        model.add_var("p_discharge", model.bat_set, value=0.0)
        model.add_var("p_bat", model.bat_phase_set, value=0.0)
        model.add_var("q_bat", model.bat_phase_set, value=0.0)
        model.add_var("soc", model.bat_set, value=0.5)
        model.bat_bus_matrix = port_to_bus_matrix(
            model, ports, lambda port: bus_by_device[port[0]], lambda port: port[1]
        )

    def active_power_injection(self, model: Any) -> Any:
        if len(model.bat_phase_set) == 0:
            return 0
        return model.bat_bus_matrix @ model.p_bat

    def reactive_power_injection(self, model: Any) -> Any:
        if len(model.bat_phase_set) == 0:
            return 0
        return model.bat_bus_matrix @ model.q_bat

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.bat_set) == 0:
            return
        add_battery_constant_q_constraints_p_control(model)
        add_battery_energy_constraints(model)
        add_battery_net_p_bat_equal_phase_constraints(model)
        add_battery_constant_p_constraints(model)

        if config.equality_only:
            return
        add_battery_power_limits(model)
        add_battery_soc_limits(model)
        if config.circular_constraints:
            add_circular_battery_constraints_pq_control(model)


def _ports(m: CvxModel, predicate) -> tuple[np.ndarray, list]:
    """Rows (and keys) of battery ports whose device control mode satisfies ``predicate``."""
    return select_rows(
        m.bat_phase_set, (predicate(m.bat_control_type[d]) for d in m.bat_port_device)
    )


def add_battery_power_limits(m: CvxModel) -> None:
    device_rows = m.bat_port_device
    keys = list(m.bat_phase_set)
    upper = np.broadcast_to(column(m.s_bat_rated), (len(keys), len(m.time_set)))
    add_ranged_constraint(
        m, "battery_discharging_limits", 0, m.p_discharge[device_rows, :], upper, keys
    )
    add_ranged_constraint(
        m, "battery_charging_limits", 0, m.p_charge[device_rows, :], upper, keys
    )


def add_battery_soc_limits(m: CvxModel) -> None:
    n_t = len(m.time_set)
    shape = (len(m.bat_set), n_t)
    add_ranged_constraint(
        m,
        "battery_soc_limits",
        np.broadcast_to(column(m.soc_min), shape),
        m.soc,
        np.broadcast_to(column(m.soc_max), shape),
        list(m.bat_set),
    )


def add_battery_net_p_bat_constraints(m: CvxModel) -> None:
    n_dev = len(m.bat_set)
    rows = list(range(len(m.bat_phase_set)))
    summing = np.zeros((n_dev, len(rows)))
    summing[m.bat_port_device, rows] = 1.0
    add_constraint(
        m,
        "net_discharge",
        summing @ m.p_bat == m.p_discharge - m.p_charge,
        list(m.bat_set),
    )


def add_battery_net_p_bat_equal_phase_constraints(m: CvxModel) -> None:
    device_rows = m.bat_port_device
    net = m.p_discharge[device_rows, :] - m.p_charge[device_rows, :]
    add_constraint(
        m,
        "net_discharge",
        m.p_bat == cp.multiply(column(1.0 / m.battery_n_phases[device_rows]), net),
        list(m.bat_phase_set),
    )


def add_battery_energy_constraints(m: CvxModel) -> None:
    eta_d = column(m.discharge_efficiency)
    eta_c = column(m.charge_efficiency)
    start = column(m.start_soc)
    previous: Any = start
    if m.soc.shape[1] > 1:
        previous = cp.hstack([start, m.soc[:, :-1]])
    add_constraint(
        m,
        "storage",
        m.soc - previous
        == cp.multiply(eta_c * m.delta_t, m.p_charge)
        - cp.multiply(m.delta_t / eta_d, m.p_discharge),
        list(m.bat_set),
    )


def add_battery_constant_q_constraints_p_control(m: CvxModel) -> None:
    add_constraint_on_rows(
        m,
        "battery_constant_q_bat",
        _ports(m, lambda ctrl: ctrl in (ControlVariable.NONE, ControlVariable.P)),
        lambda r: m.q_bat[r, :] == m.q_bat_nom[r, :],
    )


def add_battery_constant_p_constraints(m: CvxModel) -> None:
    """Fix P for NONE and Q modes, retaining storage and balanced-phase equalities."""
    add_constraint_on_rows(
        m,
        "battery_constant_p_bat",
        _ports(m, lambda ctrl: ctrl in (ControlVariable.NONE, ControlVariable.Q)),
        lambda r: m.p_bat[r, :] == m.p_bat_nom[r, :],
    )


def add_circular_battery_constraints_pq_control(m: CvxModel) -> None:
    """``P_bat^2 + Q_bat^2 <= S_rated^2`` for PQ-controlled batteries."""
    add_constraint_on_rows(
        m,
        "bat_circle_constraint",
        _ports(m, lambda ctrl: ctrl == ControlVariable.PQ),
        lambda r: cp.norm(cp.stack([m.p_bat[r, :], m.q_bat[r, :]]), 2, axis=0)
        <= column(m.s_bat_rated[r]),
    )


__all__ = ["BatteryProvider", "create_battery_parameters", "validate_bat_data"]
