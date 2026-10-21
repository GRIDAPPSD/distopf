"""Generator device provider (vectorized CVXPY): dispatch, limits, and control policies.

Parameters are arrays aligned with ``gen_device_phase_set`` (ports): ``gen_s_max``,
``gen_q_min``, ``gen_q_max``, ``gen_cost`` have shape ``(n_port,)`` and
``gen_p_available``, ``gen_q_setpoint`` have shape ``(n_port, n_steps)``.
``gen_control`` is a list of ``ControlVariable`` per port. Variables ``p_gen`` and
``q_gen`` have shape ``(n_port, n_steps)``.
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
    add_ranged_constraint,
    column,
)
from distopf.cvxpy_models_performance.common.registry import port_to_bus_matrix
from distopf.pyomo_models.common.data import parse_phases
from distopf.pyomo_models.common.device_data import (
    NetworkContext,
    cell_control,
    cell_num,
    cell_text,
)
from distopf.pyomo_models.common.model_types import ControlVariable
from distopf.pyomo_models.devices.generator import (
    _OCTAGON_HALF,
    _P_FIXED,
    _Q_FIXED,
    _ShapeCache,
    _cfg,
    _q_bounds,
    _q_mode_range,
    _resolve_shape,
    translate_legacy_gen_data,
    validate_gen_data,
)


class GeneratorProvider:
    """Own generator sets, parameters, variables, injections, and constraints."""

    name = "generators"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        ctx = NetworkContext.from_model(model, "GeneratorProvider")
        data = getattr(case, "gen_data", None)
        if data is None:
            data = pd.DataFrame()
        data = translate_legacy_gen_data(data, getattr(model, "bus_id_to_name_map", {}))
        schedules = getattr(case, "schedules", None)
        if schedules is None:
            schedules = pd.DataFrame()
        validate_gen_data(
            data,
            ctx,
            schedules,
            limits_enforced=not _cfg(config, "equality_only", False),
        )
        shapes = _ShapeCache(schedules, ctx.times)
        times = list(ctx.times)

        devices: list[str] = []
        ports: list[tuple[str, str]] = []
        phases_by_device: dict[str, list[str]] = {}
        bus_by_device: dict[str, int] = {}
        devices_by_bus_phase: dict[tuple[int, str], list[str]] = {}

        s_max: list[float] = []
        q_min: list[float] = []
        q_max: list[float] = []
        control: list[ControlVariable] = []
        cost: list[float] = []
        p_available: list[list[float]] = []
        q_setpoint: list[list[float]] = []

        for _, row in data.iterrows():
            device = cell_text(row["device_name"])
            bus = ctx.bus_name_to_id[cell_text(row["bus_name"])]
            phases = list(parse_phases(cell_text(row["phases"])))
            ctrl = cell_control(row)
            device_cost = cell_num(row, "cost", 0.0)
            mult = shapes.get(_resolve_shape(row, schedules))

            devices.append(device)
            phases_by_device[device] = phases
            bus_by_device[device] = bus

            for ph in phases:
                rating = float(row[f"s_{ph}_max"])
                p_nom = float(row[f"p_{ph}"])
                q_nom = cell_num(row, f"q_{ph}", 0.0)
                lo, hi = _q_bounds(
                    rating,
                    cell_num(row, f"q_{ph}_min", None),
                    cell_num(row, f"q_{ph}_max", None),
                )
                ports.append((device, ph))
                devices_by_bus_phase.setdefault((bus, ph), []).append(device)
                s_max.append(rating)
                q_min.append(lo)
                q_max.append(hi)
                control.append(ctrl)
                cost.append(device_cost)
                p_available.append([p_nom * mult[t] for t in times])
                q_setpoint.append([q_nom] * len(times))

        phase_pairs: list[tuple[str, str, str]] = []
        for device, phases in phases_by_device.items():
            abc = [ph for ph in ("a", "b", "c") if ph in phases]
            phase_pairs.extend(
                (device, left, right) for left, right in zip(abc, abc[1:])
            )

        model.gen_device_set = IndexSet(devices, dimen=1)
        model.gen_device_phase_set = IndexSet(ports, dimen=2)
        model.gen_phase_pair_set = IndexSet(phase_pairs, dimen=3)

        model.gen_phases_by_device = phases_by_device
        model.gen_bus_by_device = bus_by_device
        model.gen_devices_by_bus_phase = devices_by_bus_phase

        n_ports, n_t = len(ports), len(times)
        model.gen_s_max = np.array(s_max, dtype=float)
        model.gen_q_min = np.array(q_min, dtype=float)
        model.gen_q_max = np.array(q_max, dtype=float)
        model.gen_control = control
        model.gen_cost = np.array(cost, dtype=float)
        model.gen_p_available = np.array(p_available, dtype=float).reshape(n_ports, n_t)
        model.gen_q_setpoint = np.array(q_setpoint, dtype=float).reshape(n_ports, n_t)
        model.gen_phase_lock = {device: False for device in devices}

        model.add_var("p_gen", model.gen_device_phase_set, nonneg=True)
        model.add_var("q_gen", model.gen_device_phase_set, value=0.0)
        model.gen_bus_matrix = port_to_bus_matrix(
            model, ports, lambda port: bus_by_device[port[0]], lambda port: port[1]
        )

    def active_power_injection(self, model: Any) -> Any:
        if len(model.gen_device_phase_set) == 0:
            return 0
        return model.gen_bus_matrix @ model.p_gen

    def reactive_power_injection(self, model: Any) -> Any:
        if len(model.gen_device_phase_set) == 0:
            return 0
        return model.gen_bus_matrix @ model.q_gen

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.gen_device_phase_set) == 0:
            return
        add_gen_constant_p_constraints(model)
        add_gen_constant_q_constraints(model)
        if _cfg(config, "equality_only", False):
            return
        add_gen_limits(model)
        if _cfg(config, "circular_constraints", True):
            add_gen_circle_constraints(model)
        else:
            add_gen_octagon_constraints(model)


def _ports_where(m: CvxModel, predicate) -> np.ndarray:
    return np.array(
        [i for i, ctrl in enumerate(m.gen_control) if predicate(ctrl)], dtype=int
    )


def _port_keys(m: CvxModel, rows: np.ndarray) -> list:
    return [m.gen_device_phase_set[i] for i in rows]


def add_gen_constant_p_constraints(m: CvxModel) -> None:
    """Fix P to available power for generators in control modes NONE and Q."""
    rows = _ports_where(m, lambda ctrl: ctrl in _P_FIXED)
    add_constraint(
        m,
        "gen_constant_p",
        m.p_gen[rows, :] == m.gen_p_available[rows, :] if len(rows) else [],
        _port_keys(m, rows),
    )


def add_gen_constant_q_constraints(m: CvxModel) -> None:
    """Fix Q to its setpoint for generators in control modes NONE and P."""
    rows = _ports_where(m, lambda ctrl: ctrl in _Q_FIXED)
    add_constraint(
        m,
        "gen_constant_q",
        m.q_gen[rows, :] == m.gen_q_setpoint[rows, :] if len(rows) else [],
        _port_keys(m, rows),
    )


def add_gen_limits(m: CvxModel) -> None:
    """P and Q bounds for every controlled generator (all modes except NONE)."""
    rows = _ports_where(m, lambda ctrl: ctrl != ControlVariable.NONE)
    if len(rows) == 0:
        add_constraint(m, "gen_p_limits", [], [])
        add_constraint(m, "gen_q_limits", [], [])
        return
    keys = _port_keys(m, rows)
    upper = np.minimum(m.gen_p_available[rows, :], column(m.gen_s_max[rows]))
    add_ranged_constraint(m, "gen_p_limits", 0, m.p_gen[rows, :], upper, keys)

    n_t = m.gen_p_available.shape[1]
    lo = np.tile(column(m.gen_q_min[rows]), (1, n_t))
    hi = np.tile(column(m.gen_q_max[rows]), (1, n_t))
    for k, port in enumerate(rows):
        if m.gen_control[port] == ControlVariable.Q:
            for j in range(n_t):
                lo[k, j], hi[k, j] = _q_mode_range(
                    m.gen_s_max[port],
                    m.gen_p_available[port, j],
                    lo[k, j],
                    hi[k, j],
                )
    add_ranged_constraint(m, "gen_q_limits", lo, m.q_gen[rows, :], hi, keys)


def add_gen_circle_constraints(m: CvxModel) -> None:
    """Exact apparent-power limit p^2 + q^2 <= s_max^2 for PQ-controlled generators."""
    rows = _ports_where(m, lambda ctrl: ctrl == ControlVariable.PQ)
    add_constraint(
        m,
        "gen_circle",
        cp.norm(cp.stack([m.p_gen[rows, :], m.q_gen[rows, :]]), 2, axis=0)
        <= column(m.gen_s_max[rows])
        if len(rows)
        else [],
        _port_keys(m, rows),
    )


def add_gen_octagon_constraints(m: CvxModel) -> None:
    """Inscribed-octagon approximation of the rating for PQ-controlled generators."""
    rows = _ports_where(m, lambda ctrl: ctrl == ControlVariable.PQ)
    keys = _port_keys(m, rows)
    for i, (a, b) in enumerate(_OCTAGON_HALF, start=1):
        add_constraint(
            m,
            f"gen_octagon_{i}",
            a * m.p_gen[rows, :] + b * m.q_gen[rows, :] <= column(m.gen_s_max[rows])
            if len(rows)
            else [],
            keys,
        )


__all__ = ["GeneratorProvider", "validate_gen_data"]
