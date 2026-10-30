"""MPSSD model-component provider (vectorized CVXPY).

Port arrays follow ``mpssd_device_phase_set``: ``mpssd_s_max``, ``mpssd_q_min``,
``mpssd_q_max`` are ``(n_port,)``, ``mpssd_p_setpoint`` and ``mpssd_q_setpoint``
are ``(n_port, n_steps)``, and ``mpssd_control`` is a list per port.
"""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from distopf.cvxpy_models_performance.common.model import (
    CvxModel,
    IndexSet,
    add_constraint,
    add_constraint_on_rows,
    add_ranged_constraint,
    column,
    select_rows,
    sparse,
)
from distopf.cvxpy_models_performance.common.registry import port_to_bus_matrix
from distopf.pyomo_models.common.data import parse_phases
from distopf.pyomo_models.common.device_data import (
    InfeasibleCaseError as InfeasibleCaseError,
    NetworkContext,
    cell_control,
    cell_flag,
    cell_num,
    cell_text,
)
from distopf.pyomo_models.common.model_types import ControlVariable
from distopf.pyomo_models.extensions.mpssd.mpssd import validate_mpssd_data
from distopf.utils.ngon import ngon_line_equations


class MpssdProvider:
    """Own MPSSD sets, parameters, variables, injections, and constraints."""

    name = "mpssd"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        ctx = NetworkContext.from_model(model, "MpssdProvider")
        data = getattr(case, "mpssd_data", None)
        if data is None:
            data = pd.DataFrame()
        validate_mpssd_data(
            data,
            ctx.bus_name_to_id,
            set(ctx.injectable),
            set(ctx.swing_buses),
        )
        n_t = len(ctx.times)

        devices: list[str] = []
        ports: list[tuple[str, str]] = []
        phases_by_device: dict[str, list[str]] = {}
        bus_by_device: dict[str, int] = {}
        devices_by_bus_phase: dict[tuple[int, str], list[str]] = {}
        ports_by_dc_bus: dict[int, list[tuple[str, str]]] = {}

        s_max: list[float] = []
        q_min: list[float] = []
        q_max: list[float] = []
        control: list[ControlVariable] = []
        p_setpoint: list[float] = []
        q_setpoint: list[float] = []
        phase_balanced: dict[str, bool] = {}

        for _, row in data.iterrows():
            device = cell_text(row["device_name"])
            bus = int(ctx.bus_name_to_id[cell_text(row["bus_name"])])
            dc_bus = int(float(row["dc_bus"]))
            phases = list(parse_phases(cell_text(row["phases"])))
            ctrl = cell_control(row)

            devices.append(device)
            phases_by_device[device] = phases
            bus_by_device[device] = bus
            phase_balanced[device] = cell_flag(row, "balanced_phases", default=False)

            for phase in phases:
                port = (device, phase)
                rating = float(row[f"s_{phase}_max"])
                ports.append(port)
                devices_by_bus_phase.setdefault((bus, phase), []).append(device)
                ports_by_dc_bus.setdefault(dc_bus, []).append(port)
                s_max.append(rating)
                q_min.append(max(-rating, cell_num(row, f"q_{phase}_min", -rating)))
                q_max.append(min(rating, cell_num(row, f"q_{phase}_max", rating)))
                control.append(ctrl)
                p_setpoint.append(cell_num(row, f"p_{phase}", 0.0))
                q_setpoint.append(cell_num(row, f"q_{phase}", 0.0))

        model.mpssd_device_set = IndexSet(devices)
        model.mpssd_device_phase_set = IndexSet(ports)
        model.mpssd_dc_bus_set = IndexSet(sorted(ports_by_dc_bus))

        model.mpssd_phases_by_device = phases_by_device
        model.mpssd_bus_by_device = bus_by_device
        model.mpssd_devices_by_bus_phase = devices_by_bus_phase
        model.mpssd_ports_by_dc_bus = ports_by_dc_bus

        model.mpssd_s_max = np.array(s_max, dtype=float)
        model.mpssd_q_min = np.array(q_min, dtype=float)
        model.mpssd_q_max = np.array(q_max, dtype=float)
        model.mpssd_control = control
        model.mpssd_p_setpoint = np.repeat(
            np.array(p_setpoint, dtype=float).reshape(-1, 1), n_t, axis=1
        )
        model.mpssd_q_setpoint = np.repeat(
            np.array(q_setpoint, dtype=float).reshape(-1, 1), n_t, axis=1
        )
        model.mpssd_is_phase_balanced = phase_balanced

        model.add_var("p_mpssd", model.mpssd_device_phase_set, value=0.0)
        model.add_var("q_mpssd", model.mpssd_device_phase_set, value=0.0)
        model.mpssd_bus_matrix = port_to_bus_matrix(
            model, ports, lambda port: bus_by_device[port[0]], lambda port: port[1]
        )

    def active_power_injection(self, model: Any) -> Any:
        if len(model.mpssd_device_phase_set) == 0:
            return 0
        return model.mpssd_bus_matrix @ model.p_mpssd

    def reactive_power_injection(self, model: Any) -> Any:
        if len(model.mpssd_device_phase_set) == 0:
            return 0
        return model.mpssd_bus_matrix @ model.q_mpssd

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.mpssd_device_phase_set) == 0:
            return
        add_mpssd_constant_p_constraints_q_control(model)
        add_mpssd_constant_q_constraints_p_control(model)
        if not config.equality_only:
            add_mpssd_limits(model)
            if config.circular_constraints:
                add_circular_mpssd_constraints(model)
            else:
                add_ngon_constraints(model)
        add_dc_bus_balance_constraints(model)
        add_phase_balance_constraints(model)


def _ports(m: CvxModel, predicate) -> tuple[np.ndarray, list]:
    """Rows (and keys) of MPSSD ports whose control mode satisfies ``predicate``."""
    return select_rows(
        m.mpssd_device_phase_set, (predicate(c) for c in m.mpssd_control)
    )


def add_mpssd_constant_p_constraints_q_control(m: CvxModel) -> None:
    add_constraint_on_rows(
        m,
        "mpssd_constant_p",
        _ports(m, lambda c: c in (ControlVariable.NONE, ControlVariable.Q)),
        lambda r: m.p_mpssd[r, :] == m.mpssd_p_setpoint[r, :],
    )


def add_mpssd_constant_q_constraints_p_control(m: CvxModel) -> None:
    add_constraint_on_rows(
        m,
        "mpssd_constant_q",
        _ports(m, lambda c: c in (ControlVariable.NONE, ControlVariable.P)),
        lambda r: m.q_mpssd[r, :] == m.mpssd_q_setpoint[r, :],
    )


def add_mpssd_limits(m: CvxModel) -> None:
    keys = list(m.mpssd_device_phase_set)
    shape = m.p_mpssd.shape
    rating = np.broadcast_to(column(m.mpssd_s_max), shape)
    add_ranged_constraint(m, "mpssd_p_limits", -rating, m.p_mpssd, rating, keys)
    add_ranged_constraint(
        m,
        "mpssd_q_limits",
        np.broadcast_to(column(m.mpssd_q_min), shape),
        m.q_mpssd,
        np.broadcast_to(column(m.mpssd_q_max), shape),
        keys,
    )


def add_circular_mpssd_constraints(m: CvxModel) -> None:
    add_constraint_on_rows(
        m,
        "mpssd_circle",
        _ports(m, lambda c: c == ControlVariable.PQ),
        lambda r: cp.norm(cp.stack([m.p_mpssd[r, :], m.q_mpssd[r, :]]), 2, axis=0)
        <= column(m.mpssd_s_max[r]),
    )


def add_ngon_constraints(m: CvxModel, n: int = 8) -> None:
    selection = _ports(m, lambda c: c == ControlVariable.PQ)

    def half_plane(a: float, b: float):
        return lambda r: a * m.p_mpssd[r, :] + b * m.q_mpssd[r, :] <= column(
            m.mpssd_s_max[r]
        )

    for i, (a, b) in enumerate(ngon_line_equations(n), start=1):
        add_constraint_on_rows(
            m, f"mpssd_ngon_limit_{i}", selection, half_plane(float(a), float(b))
        )


def add_dc_bus_balance_constraints(m: CvxModel) -> None:
    port_pos = m.mpssd_device_phase_set.pos
    rows, cols = [], []
    for i, dc_bus in enumerate(m.mpssd_dc_bus_set):
        for port in m.mpssd_ports_by_dc_bus[dc_bus]:
            rows.append(i)
            cols.append(port_pos[port])
    select = sparse(
        rows,
        cols,
        [1.0] * len(rows),
        (len(m.mpssd_dc_bus_set), len(port_pos)),
    )
    add_constraint(
        m,
        "mpssd_dc_bus_balance",
        select @ m.p_mpssd == 0,
        list(m.mpssd_dc_bus_set),
    )


def add_phase_balance_constraints(m: CvxModel) -> None:
    port_pos = m.mpssd_device_phase_set.pos
    follower, leader = [], []
    for (device, phase), i in port_pos.items():
        if not m.mpssd_is_phase_balanced[device]:
            continue
        first = m.mpssd_phases_by_device[device][0]
        if phase == first:
            continue
        follower.append(i)
        leader.append(port_pos[(device, first)])
    keys = [m.mpssd_device_phase_set[i] for i in follower]
    for name, var in (("p", m.p_mpssd), ("q", m.q_mpssd)):
        add_constraint(
            m,
            f"mpssd_{name}_balanced_phases",
            var[follower, :] == var[leader, :] if follower else [],
            keys,
        )
