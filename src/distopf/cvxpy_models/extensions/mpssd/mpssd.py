"""MPSSD model-component provider (CVXPY). Mirror of ``pyomo_models.extensions.mpssd``.

See the Pyomo module docstring for the CSV schema.
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
        times = ctx.times

        devices: list[str] = []
        ports: list[tuple[str, str]] = []
        phases_by_device: dict[str, list[str]] = {}
        bus_by_device: dict[str, int] = {}
        devices_by_bus_phase: dict[tuple[int, str], list[str]] = {}
        ports_by_dc_bus: dict[int, list[tuple[str, str]]] = {}

        s_max: dict[tuple[str, str], float] = {}
        q_min: dict[tuple[str, str], float] = {}
        q_max: dict[tuple[str, str], float] = {}
        control: dict[tuple[str, str], ControlVariable] = {}
        p_setpoint: dict[tuple[str, str, Any], float] = {}
        q_setpoint: dict[tuple[str, str, Any], float] = {}
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
                p_nom = cell_num(row, f"p_{phase}", 0.0)
                q_nom = cell_num(row, f"q_{phase}", 0.0)

                ports.append(port)
                devices_by_bus_phase.setdefault((bus, phase), []).append(device)
                ports_by_dc_bus.setdefault(dc_bus, []).append(port)

                s_max[port] = rating
                q_min[port] = max(-rating, cell_num(row, f"q_{phase}_min", -rating))
                q_max[port] = min(rating, cell_num(row, f"q_{phase}_max", rating))
                control[port] = ctrl
                for t in times:
                    p_setpoint[(*port, t)] = p_nom
                    q_setpoint[(*port, t)] = q_nom

        model.mpssd_device_set = IndexSet(devices, dimen=1)
        model.mpssd_device_phase_set = IndexSet(ports, dimen=2)
        model.mpssd_dc_bus_set = IndexSet(sorted(ports_by_dc_bus), dimen=1)

        model.mpssd_phases_by_device = phases_by_device
        model.mpssd_bus_by_device = bus_by_device
        model.mpssd_devices_by_bus_phase = devices_by_bus_phase
        model.mpssd_ports_by_dc_bus = ports_by_dc_bus

        model.mpssd_s_max = ParamDict(s_max)
        model.mpssd_q_min = ParamDict(q_min)
        model.mpssd_q_max = ParamDict(q_max)
        model.mpssd_control = ParamDict(control)
        model.mpssd_p_setpoint = ParamDict(p_setpoint)
        model.mpssd_q_setpoint = ParamDict(q_setpoint)
        model.mpssd_is_phase_balanced = ParamDict(phase_balanced)

        keys = index_product(model.mpssd_device_phase_set, model.time_set)
        model.p_mpssd = VarDict(keys, initialize=0.0)
        model.q_mpssd = VarDict(keys, initialize=0.0)

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return sum(
            model.p_mpssd[device, phase, time]
            for device in model.mpssd_devices_by_bus_phase.get((bus, phase), [])
        )

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return sum(
            model.q_mpssd[device, phase, time]
            for device in model.mpssd_devices_by_bus_phase.get((bus, phase), [])
        )

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.mpssd_device_phase_set) == 0:
            return
        circular = getattr(config, "circular_constraints", True) if config else True
        equality_only = getattr(config, "equality_only", False) if config else False
        add_mpssd_constant_p_constraints_q_control(model)
        add_mpssd_constant_q_constraints_p_control(model)
        if not equality_only:
            add_mpssd_limits(model)
            if circular:
                add_circular_mpssd_constraints(model)
            else:
                add_ngon_constraints(model)
        add_dc_bus_balance_constraints(model)
        add_phase_balance_constraints(model)


def add_mpssd_constant_p_constraints_q_control(m: CvxModel) -> None:
    def rule(m, device, phase, time):
        if m.mpssd_control[device, phase] in (ControlVariable.NONE, ControlVariable.Q):
            return (
                m.p_mpssd[device, phase, time]
                == m.mpssd_p_setpoint[device, phase, time]
            )
        return SKIP

    add_constraint(m, "mpssd_constant_p", (m.mpssd_device_phase_set, m.time_set), rule)


def add_mpssd_constant_q_constraints_p_control(m: CvxModel) -> None:
    def rule(m, device, phase, time):
        if m.mpssd_control[device, phase] in (ControlVariable.NONE, ControlVariable.P):
            return (
                m.q_mpssd[device, phase, time]
                == m.mpssd_q_setpoint[device, phase, time]
            )
        return SKIP

    add_constraint(m, "mpssd_constant_q", (m.mpssd_device_phase_set, m.time_set), rule)


def add_mpssd_limits(m: CvxModel) -> None:
    def p_bounds(m, device, phase, time):
        rating = m.mpssd_s_max[device, phase]
        return (-rating, m.p_mpssd[device, phase, time], rating)

    def q_bounds(m, device, phase, time):
        return (
            m.mpssd_q_min[device, phase],
            m.q_mpssd[device, phase, time],
            m.mpssd_q_max[device, phase],
        )

    index = (m.mpssd_device_phase_set, m.time_set)
    add_constraint(m, "mpssd_p_limits", index, p_bounds)
    add_constraint(m, "mpssd_q_limits", index, q_bounds)


def add_circular_mpssd_constraints(m: CvxModel) -> None:
    def rule(m, device, phase, time):
        if m.mpssd_control[device, phase] != ControlVariable.PQ:
            return SKIP
        return (
            cp.norm(
                cp.hstack(
                    [m.p_mpssd[device, phase, time], m.q_mpssd[device, phase, time]]
                ),
                2,
            )
            <= m.mpssd_s_max[device, phase]
        )

    add_constraint(m, "mpssd_circle", (m.mpssd_device_phase_set, m.time_set), rule)


def add_ngon_constraints(m: CvxModel, n: int = 8) -> None:
    def make_rule(a: float, b: float):
        def rule(m, device, phase, time):
            if m.mpssd_control[device, phase] != ControlVariable.PQ:
                return SKIP
            return (
                a * m.p_mpssd[device, phase, time] + b * m.q_mpssd[device, phase, time]
                <= m.mpssd_s_max[device, phase]
            )

        return rule

    for i, (a, b) in enumerate(ngon_line_equations(n), start=1):
        add_constraint(
            m,
            f"mpssd_ngon_limit_{i}",
            (m.mpssd_device_phase_set, m.time_set),
            make_rule(float(a), float(b)),
        )


def add_dc_bus_balance_constraints(m: CvxModel) -> None:
    def rule(m, dc_bus, time):
        return (
            sum(m.p_mpssd[d, ph, time] for d, ph in m.mpssd_ports_by_dc_bus[dc_bus])
            == 0
        )

    add_constraint(m, "mpssd_dc_bus_balance", (m.mpssd_dc_bus_set, m.time_set), rule)


def add_phase_balance_constraints(m: CvxModel) -> None:
    def make_rule(var: Any):
        def rule(m, device, phase, time):
            if not m.mpssd_is_phase_balanced[device]:
                return SKIP
            first = m.mpssd_phases_by_device[device][0]
            if phase == first:
                return SKIP
            return var[device, phase, time] == var[device, first, time]

        return rule

    for name, var in (("p", m.p_mpssd), ("q", m.q_mpssd)):
        add_constraint(
            m,
            f"mpssd_{name}_balanced_phases",
            (m.mpssd_device_phase_set, m.time_set),
            make_rule(var),
        )
