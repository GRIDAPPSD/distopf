"""Generator device provider (CVXPY): dispatch, limits, and control policies.

Mirror of ``pyomo_models.devices.generator``. The CSV schema, validation, and
shape handling are shared with the Pyomo module; see its docstring for the
column definitions.
"""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import pandas as pd

from distopf.cvxpy_models.common.model import (
    SKIP,
    CvxModel,
    IndexSet,
    ParamDict,
    VarDict,
    add_constraint,
    index_product,
)
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

        devices: list[str] = []
        ports: list[tuple[str, str]] = []
        phases_by_device: dict[str, list[str]] = {}
        bus_by_device: dict[str, int] = {}
        devices_by_bus_phase: dict[tuple[int, str], list[str]] = {}

        s_max: dict[tuple[str, str], float] = {}
        q_min: dict[tuple[str, str], float] = {}
        q_max: dict[tuple[str, str], float] = {}
        control: dict[tuple[str, str], ControlVariable] = {}
        cost: dict[tuple[str, str], float] = {}
        p_available: dict[tuple[str, str, Any], float] = {}
        q_setpoint: dict[tuple[str, str, Any], float] = {}

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
                port = (device, ph)
                rating = float(row[f"s_{ph}_max"])
                p_nom = float(row[f"p_{ph}"])
                q_nom = cell_num(row, f"q_{ph}", 0.0)
                lo, hi = _q_bounds(
                    rating,
                    cell_num(row, f"q_{ph}_min", None),
                    cell_num(row, f"q_{ph}_max", None),
                )

                ports.append(port)
                devices_by_bus_phase.setdefault((bus, ph), []).append(device)

                s_max[port] = rating
                q_min[port] = lo
                q_max[port] = hi
                control[port] = ctrl
                cost[port] = device_cost
                for t in ctx.times:
                    p_available[(*port, t)] = p_nom * mult[t]
                    q_setpoint[(*port, t)] = q_nom

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

        model.gen_s_max = ParamDict(s_max)
        model.gen_q_min = ParamDict(q_min)
        model.gen_q_max = ParamDict(q_max)
        model.gen_control = ParamDict(control)
        model.gen_cost = ParamDict(cost)
        model.gen_p_available = ParamDict(p_available)
        model.gen_q_setpoint = ParamDict(q_setpoint)
        model.gen_phase_lock = ParamDict({device: False for device in devices})

        keys = index_product(model.gen_device_phase_set, model.time_set)
        model.p_gen = VarDict(keys, nonneg=True)
        model.q_gen = VarDict(keys, initialize=0.0)

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return sum(
            model.p_gen[device, phase, time]
            for device in model.gen_devices_by_bus_phase.get((bus, phase), [])
        )

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return sum(
            model.q_gen[device, phase, time]
            for device in model.gen_devices_by_bus_phase.get((bus, phase), [])
        )

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


def add_gen_constant_p_constraints(m: CvxModel) -> None:
    """Fix P to available power for generators in control modes NONE and Q."""

    def rule(m, device, phase, time):
        if m.gen_control[device, phase] in _P_FIXED:
            return (
                m.p_gen[device, phase, time] == m.gen_p_available[device, phase, time]
            )
        return SKIP

    add_constraint(m, "gen_constant_p", (m.gen_device_phase_set, m.time_set), rule)


def add_gen_constant_q_constraints(m: CvxModel) -> None:
    """Fix Q to its setpoint for generators in control modes NONE and P."""

    def rule(m, device, phase, time):
        if m.gen_control[device, phase] in _Q_FIXED:
            return m.q_gen[device, phase, time] == m.gen_q_setpoint[device, phase, time]
        return SKIP

    add_constraint(m, "gen_constant_q", (m.gen_device_phase_set, m.time_set), rule)


def add_gen_limits(m: CvxModel) -> None:
    """P and Q bounds for every controlled generator (all modes except NONE)."""

    def p_rule(m, device, phase, time):
        if m.gen_control[device, phase] == ControlVariable.NONE:
            return SKIP
        upper = min(m.gen_p_available[device, phase, time], m.gen_s_max[device, phase])
        return (0, m.p_gen[device, phase, time], upper)

    def q_rule(m, device, phase, time):
        ctrl = m.gen_control[device, phase]
        if ctrl == ControlVariable.NONE:
            return SKIP
        lo = m.gen_q_min[device, phase]
        hi = m.gen_q_max[device, phase]
        if ctrl == ControlVariable.Q:
            lo, hi = _q_mode_range(
                m.gen_s_max[device, phase],
                m.gen_p_available[device, phase, time],
                lo,
                hi,
            )
        return (lo, m.q_gen[device, phase, time], hi)

    add_constraint(m, "gen_p_limits", (m.gen_device_phase_set, m.time_set), p_rule)
    add_constraint(m, "gen_q_limits", (m.gen_device_phase_set, m.time_set), q_rule)


def add_gen_circle_constraints(m: CvxModel) -> None:
    """Exact apparent-power limit p^2 + q^2 <= s_max^2 for PQ-controlled generators."""

    def rule(m, device, phase, time):
        if m.gen_control[device, phase] != ControlVariable.PQ:
            return SKIP
        return (
            cp.norm(
                cp.hstack([m.p_gen[device, phase, time], m.q_gen[device, phase, time]]),
                2,
            )
            <= m.gen_s_max[device, phase]
        )

    add_constraint(m, "gen_circle", (m.gen_device_phase_set, m.time_set), rule)


def add_gen_octagon_constraints(m: CvxModel) -> None:
    """Inscribed-octagon approximation of the rating for PQ-controlled generators."""

    def make_rule(a: float, b: float):
        def rule(m, device, phase, time):
            if m.gen_control[device, phase] != ControlVariable.PQ:
                return SKIP
            return (
                a * m.p_gen[device, phase, time] + b * m.q_gen[device, phase, time]
                <= m.gen_s_max[device, phase]
            )

        return rule

    for i, (a, b) in enumerate(_OCTAGON_HALF, start=1):
        add_constraint(
            m,
            f"gen_octagon_{i}",
            (m.gen_device_phase_set, m.time_set),
            make_rule(a, b),
        )


__all__ = ["GeneratorProvider", "validate_gen_data"]
