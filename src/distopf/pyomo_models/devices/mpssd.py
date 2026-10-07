"""MPSSD model-component provider for the LinDistFlow formulation."""

from __future__ import annotations

from typing import Any


from math import sqrt
import pandas as pd
import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models.common.registry import DeviceProvider
from distopf.pyomo_models.common.data import parse_phases
from distopf.pyomo_models.common.model_types import CONTROL_VARIABLE_MAP

from distopf.pyomo_models.common.model_types import ControlVariable
from distopf.pyomo_models.common.protocol import LindistModelProtocol


sqrt2 = sqrt(2)

"""
MPSSD CSV specs
columns:
id: unique identifier for the device
name: name of the device
bus_id: identifier of the bus the device is connected to
bus_name: name of the bus the device is connected to
phases: phases the device is connected to
dc_bus: identifier of the DC bus the device is connected to
control_variable: control variable type (e.g., "PQ", "P", "Q", "")
p_a, p_b, p_c: 
                Active power setpoints for phases a, b, c;
                Only active for control modes, "Q" and "", otherwise ignored; 
                Must satisfy the power balance for the DC bus. 
                Other wise it will result in an infeasible solution.
q_a, q_b, q_c: 
                Reactive power setpoints for phases a, b, c;
                Only active for control modes, "P" and "", otherwise ignored; 
s_a_max, s_b_max, s_c_max: maximum apparent power ratings for phases a, b, c

example:
id,name,bus_id,bus_name,phases,dc_bus,control_variable,p_a,p_b,p_c,q_a,q_b,q_c,s_a_max,s_b_max,s_c_max
1,mpssd_p1,61,151,abc,1,PQ,,,,,,,0.4,0.4,0.4
2,mpssd_p2,102,300,abc,1,PQ,,,,,,,0.4,0.4,0.4
"""


class MpssdProvider(DeviceProvider):
    """Own MPSSD sets, parameters, variables, injections, and constraints."""

    name = "mpssd"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        data = getattr(case, "mpssd_data", pd.DataFrame())
        model.mpssd_set = pyo.Set(
            initialize=[] if data.empty else data.id.astype(int).tolist()
        )
        device_phase_list = []
        device_phase_map = {}
        device_phase_dc_bus_list = []
        bus_map: dict[tuple[int, str], list[int]] = {}
        dc_bus_to_device_phase_map = {}
        if not data.empty:
            for _, row in data.iterrows():
                device = int(row.id)
                for phase in parse_phases(str(row.phases)):
                    bus_name = str(row.bus_name)
                    bus_id = int(model.bus_name_to_id_map.get(bus_name, 0))
                    dc_bus = int(row.dc_bus)
                    device_phase_list.append((device, phase))
                    device_phase_map.setdefault(device, []).append(phase)
                    device_phase_dc_bus_list.append((device, phase, dc_bus))
                    dc_bus_to_device_phase_map.setdefault(dc_bus, []).append(
                        (device, phase)
                    )
                    bus_map.setdefault((bus_id, phase), []).append(device)
        model.mpssd_phase_set = pyo.Set(initialize=device_phase_list, dimen=2)
        model.mpssd_phase_dc_bus_set = pyo.Set(
            initialize=device_phase_dc_bus_list, dimen=3
        )
        model.mpssd_phases_map = device_phase_map
        model.mpssd_bus_map = bus_map
        dc_labels = sorted(
            {int(value) for value in data.get("dc_bus", pd.Series(dtype=int)).tolist()}
        )
        model.dc_bus_set = pyo.Set(initialize=dc_labels)
        model.dc_bus_to_device_phase_map = dc_bus_to_device_phase_map
        model.p_mpssd = pyo.Var(model.mpssd_phase_set, model.time_set, initialize=0)
        model.q_mpssd = pyo.Var(model.mpssd_phase_set, model.time_set, initialize=0)

        if data.empty:
            return

        values: dict[str, dict[tuple[Any, ...], Any]] = {
            "s_rated": {},
            "q_min": {},
            "q_max": {},
            "dc_bus": {},
            "control": {},
            "p_nom": {},
            "q_nom": {},
            "balanced_phases": {},
        }
        for _, row in data.iterrows():
            device = int(row.id)
            dc_bus = int(row.dc_bus) if pd.notna(row.get("dc_bus", 0)) else 0
            control = CONTROL_VARIABLE_MAP.get(row.get("control_variable", "PQ"), 3)
            values["balanced_phases"][device] = bool(row.get("balanced_phases", False))
            for phase in parse_phases(str(row.phases)):
                key = (device, phase)
                rating = row.get(f"s_{phase}_max", 1000.0)
                values["s_rated"][key] = rating
                values["q_min"][key] = row.get(f"q_{phase}_min", -rating)
                values["q_max"][key] = row.get(f"q_{phase}_max", rating)
                values["dc_bus"][key] = dc_bus
                values["control"][key] = control
                for time in model.time_set:
                    values["p_nom"][(*key, time)] = row.get(f"p_{phase}", 0.0)
                    values["q_nom"][(*key, time)] = row.get(f"q_{phase}", 0.0)

        model.mpssd_s_rated = pyo.Param(
            model.mpssd_phase_set, initialize=values["s_rated"], default=1000.0
        )
        model.mpssd_q_min = pyo.Param(
            model.mpssd_phase_set, initialize=values["q_min"], default=-1000.0
        )
        model.mpssd_q_max = pyo.Param(
            model.mpssd_phase_set, initialize=values["q_max"], default=1000.0
        )
        model.mpssd_dc_bus = pyo.Param(
            model.mpssd_phase_set, initialize=values["dc_bus"], default=0
        )
        model.mpssd_control_type = pyo.Param(
            model.mpssd_phase_set, initialize=values["control"], default=3
        )
        model.p_mpssd_nom = pyo.Param(
            model.mpssd_phase_set, model.time_set, initialize=values["p_nom"], default=0
        )
        model.q_mpssd_nom = pyo.Param(
            model.mpssd_phase_set, model.time_set, initialize=values["q_nom"], default=0
        )
        model.mpssd_balanced_phases = pyo.Param(
            model.mpssd_set, initialize=values["balanced_phases"], default=False
        )

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return sum(
            model.p_mpssd[device, phase, time]
            for device in model.mpssd_bus_map.get((bus, phase), [])
        )

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return sum(
            model.q_mpssd[device, phase, time]
            for device in model.mpssd_bus_map.get((bus, phase), [])
        )

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.mpssd_phase_set) == 0:
            return
        circular = getattr(config, "circular_constraints", True) if config else True
        add_mpssd_constant_p_constraints_q_control(model)
        add_mpssd_constant_q_constraints_p_control(model)
        add_mpssd_limits(model)
        if circular:
            add_circular_mpssd_constraints(model)
        else:
            add_octagonal_mpssd_constraints(model)
        if len(model.dc_bus_set) > 0:
            add_dc_bus_balance_constraints(model)
        add_p_phase_balance_constraints(model)
        add_q_phase_balance_constraints(model)


def add_mpssd_constant_p_constraints_q_control(m: LindistModelProtocol) -> None:
    """Fix active MPSSD power for non-P-controlled ports."""

    def rule(m, device, phase, time):
        control = m.mpssd_control_type[device, phase]
        if control in (ControlVariable.NONE, ControlVariable.Q):
            return m.p_mpssd[device, phase, time] == m.p_mpssd_nom[device, phase, time]
        return pyo.Constraint.Skip

    m.mpssd_constant_p = pyo.Constraint(m.mpssd_phase_set, m.time_set, rule=rule)


def add_mpssd_constant_q_constraints_p_control(m: LindistModelProtocol) -> None:
    """Fix reactive MPSSD power for non-Q-controlled ports."""

    def rule(m, device, phase, time):
        control = m.mpssd_control_type[device, phase]
        if control in (ControlVariable.NONE, ControlVariable.P):
            return m.q_mpssd[device, phase, time] == m.q_mpssd_nom[device, phase, time]
        return pyo.Constraint.Skip

    m.mpssd_constant_q = pyo.Constraint(m.mpssd_phase_set, m.time_set, rule=rule)


def add_mpssd_limits(m: LindistModelProtocol) -> None:
    """Add rectangular P/Q operating limits for MPSSD ports."""

    def p_bounds(m, device, phase, time):
        rating = m.mpssd_s_rated[device, phase]
        return (-rating, m.p_mpssd[device, phase, time], rating)

    def q_bounds(m, device, phase, time):
        rating = m.mpssd_s_rated[device, phase]
        return (
            max(-rating, m.mpssd_q_min[device, phase]),
            m.q_mpssd[device, phase, time],
            min(rating, m.mpssd_q_max[device, phase]),
        )

    m.mpssd_p_limits = pyo.Constraint(m.mpssd_phase_set, m.time_set, rule=p_bounds)
    m.mpssd_q_limits = pyo.Constraint(m.mpssd_phase_set, m.time_set, rule=q_bounds)


def add_circular_mpssd_constraints(m: LindistModelProtocol) -> None:
    """Add exact apparent-power limits for MPSSD ports."""

    def rule(m, device, phase, time):
        return (
            m.p_mpssd[device, phase, time] ** 2 + m.q_mpssd[device, phase, time] ** 2
            <= m.mpssd_s_rated[device, phase] ** 2
        )

    m.mpssd_circle = pyo.Constraint(m.mpssd_phase_set, m.time_set, rule=rule)


def add_octagonal_mpssd_constraints(m: LindistModelProtocol) -> None:
    """Add an eight-sided linear approximation of apparent-power limits."""
    c = sqrt2 - 1

    def r1(m, device, phase, time):
        return (
            c * m.p_mpssd[device, phase, time] + m.q_mpssd[device, phase, time]
            <= m.mpssd_s_rated[device, phase]
        )

    def r2(m, device, phase, time):
        return (
            m.p_mpssd[device, phase, time] + c * m.q_mpssd[device, phase, time]
            <= m.mpssd_s_rated[device, phase]
        )

    def r3(m, device, phase, time):
        return (
            m.p_mpssd[device, phase, time] - c * m.q_mpssd[device, phase, time]
            <= m.mpssd_s_rated[device, phase]
        )

    def r4(m, device, phase, time):
        return (
            c * m.p_mpssd[device, phase, time] - m.q_mpssd[device, phase, time]
            <= m.mpssd_s_rated[device, phase]
        )

    def r5(m, device, phase, time):
        return (
            -c * m.p_mpssd[device, phase, time] - m.q_mpssd[device, phase, time]
            <= m.mpssd_s_rated[device, phase]
        )

    def r6(m, device, phase, time):
        return (
            -m.p_mpssd[device, phase, time] - c * m.q_mpssd[device, phase, time]
            <= m.mpssd_s_rated[device, phase]
        )

    def r7(m, device, phase, time):
        return (
            -m.p_mpssd[device, phase, time] + c * m.q_mpssd[device, phase, time]
            <= m.mpssd_s_rated[device, phase]
        )

    def r8(m, device, phase, time):
        return (
            -c * m.p_mpssd[device, phase, time] + m.q_mpssd[device, phase, time]
            <= m.mpssd_s_rated[device, phase]
        )

    for index, rule in enumerate((r1, r2, r3, r4, r5, r6, r7, r8), start=1):
        setattr(
            m,
            f"mpssd_oct_{index}",
            pyo.Constraint(m.mpssd_phase_set, m.time_set, rule=rule),
        )


def add_dc_bus_balance_constraints(m: LindistModelProtocol) -> None:
    """Enforce zero net active injection for each shared DC bus."""

    def rule(m, dc_bus, time):
        device_phase_list = m.dc_bus_to_device_phase_map.get(dc_bus, [])
        return (
            sum(m.p_mpssd[device, phase, time] for device, phase in device_phase_list)
            == 0
        )

    m.dc_bus_balance = pyo.Constraint(m.dc_bus_set, m.time_set, rule=rule)


def add_p_phase_balance_constraints(m: LindistModelProtocol) -> None:
    """Enforce phase balance for devices with the balanced_phases attribute set to True."""

    def rule(m, device, phase, time):
        if not m.mpssd_balanced_phases[device]:
            return pyo.Constraint.Skip
        phases = m.mpssd_phases_map.get(device, [])
        if phase == phases[0]:
            return pyo.Constraint.Skip
        return m.p_mpssd[device, phase, time] == m.p_mpssd[device, phases[0], time]

    m.mpssd_p_balanced_phases = pyo.Constraint(m.mpssd_phase_set, m.time_set, rule=rule)

def add_q_phase_balance_constraints(m: LindistModelProtocol) -> None:
    """Enforce phase balance for devices with the balanced_phases attribute set to True."""

    def rule(m, device, phase, time):
        if not m.mpssd_balanced_phases[device]:
            return pyo.Constraint.Skip
        phases = m.mpssd_phases_map.get(device, [])
        if phase == phases[0]:
            return pyo.Constraint.Skip
        return m.q_mpssd[device, phase, time] == m.q_mpssd[device, phases[0], time]

    m.mpssd_q_balanced_phases = pyo.Constraint(m.mpssd_phase_set, m.time_set, rule=rule)
