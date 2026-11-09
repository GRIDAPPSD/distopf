"""Named battery devices with balanced phase power and SOC dynamics (CVXPY).

Mirror of ``pyomo_models.devices.battery``.
"""

from __future__ import annotations

from typing import Any

import cvxpy as cp

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
from distopf.pyomo_models.devices.battery import _battery_data, validate_bat_data


def create_battery_parameters(model: Any, case: Any) -> None:
    """Create every battery parameter consumed by common LinDistFlow constraints."""
    p_data, q_data, rating, q_min, q_max = {}, {}, {}, {}, {}
    energy, soc_min, soc_max, start_soc = {}, {}, {}, {}
    charge_eff, discharge_eff, cycles, control = {}, {}, {}, {}
    has_phase = {
        (device, phase): False for device in model.bat_set for phase in ("a", "b", "c")
    }
    has_a, has_b, has_c, n_phases = {}, {}, {}, {}
    for _, row in _battery_data(model, case).iterrows():
        device = cell_text(row["device_name"])
        phases = parse_phases(cell_text(row["phases"]))
        count = len(phases)
        n_phases[device] = count
        has_a[device], has_b[device], has_c[device] = (
            "a" in phases,
            "b" in phases,
            "c" in phases,
        )
        for phase in ("a", "b", "c"):
            has_phase[(device, phase)] = phase in phases
        energy[device] = cell_num(row, "energy_capacity", 0)
        soc_min[device] = cell_num(row, "min_soc", 0)
        soc_max[device] = cell_num(row, "max_soc", 1)
        start_soc[device] = cell_num(row, "start_soc", 0.5)
        charge_eff[device] = cell_num(row, "charge_efficiency", 1)
        discharge_eff[device] = cell_num(row, "discharge_efficiency", 1)
        cycles[device] = cell_num(row, "annual_cycle_limit", 365)
        control[device] = cell_control(row)
        for phase in phases:
            key = (device, phase)
            if key not in model.bat_phase_set:
                continue
            s_max = cell_num(row, "s_max", 1000.0) / count
            rating[key] = s_max
            q_min[key] = max(-s_max, cell_num(row, "q_min", -s_max * count) / count)
            q_max[key] = min(s_max, cell_num(row, "q_max", s_max * count) / count)
            for time in model.time_set:
                p_data[(*key, time)] = cell_num(row, "p", 0.0) / count
                q_data[(*key, time)] = cell_num(row, "q", 0.0) / count
    model.p_bat_nom = ParamDict(p_data, default=0.0)
    model.q_bat_nom = ParamDict(q_data, default=0.0)
    model.s_bat_rated = ParamDict(rating, default=1000.0)
    model.q_bat_min = ParamDict(q_min, default=-1000.0)
    model.q_bat_max = ParamDict(q_max, default=1000.0)
    model.bat_control_type = ParamDict(control, default=ControlVariable.NONE)
    model.energy_capacity = ParamDict(energy, default=0)
    model.soc_min = ParamDict(soc_min, default=0)
    model.soc_max = ParamDict(soc_max, default=1)
    model.start_soc = ParamDict(start_soc, default=0.5)
    model.charge_efficiency = ParamDict(charge_eff, default=1.0)
    model.discharge_efficiency = ParamDict(discharge_eff, default=1.0)
    model.annual_cycle_limit = ParamDict(cycles, default=365)
    model.battery_has_a_phase = ParamDict(has_a, default=True)
    model.battery_has_b_phase = ParamDict(has_b, default=True)
    model.battery_has_c_phase = ParamDict(has_c, default=True)
    model.battery_n_phases = ParamDict(n_phases, default=3)
    model.battery_has_phase = ParamDict(has_phase, default=True)


class BatteryProvider:
    """Create and constrain battery variables for a CVXPY model."""

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
        model.bat_set = IndexSet(list(bus_by_device), dimen=1)
        model.bat_phase_set = IndexSet(ports, dimen=2)
        model.bat_bus_by_device = bus_by_device
        model.bat_devices_by_bus_phase = devices_by_bus_phase
        model.bat_phases_by_device = phases_by_device
        create_battery_parameters(model, case)
        device_keys = index_product(model.bat_set, model.time_set)
        port_keys = index_product(model.bat_phase_set, model.time_set)
        model.p_charge = VarDict(device_keys, initialize=0.0)
        model.p_discharge = VarDict(device_keys, initialize=0.0)
        model.p_bat = VarDict(port_keys, initialize=0.0)
        model.q_bat = VarDict(port_keys, initialize=0.0)
        model.soc = VarDict(device_keys, initialize=0.5)

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return sum(
            model.p_bat[device, phase, time]
            for device in model.bat_devices_by_bus_phase.get((bus, phase), [])
        )

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return sum(
            model.q_bat[device, phase, time]
            for device in model.bat_devices_by_bus_phase.get((bus, phase), [])
        )

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.bat_set) == 0:
            return
        add_battery_constant_q_constraints_p_control(model)
        add_battery_energy_constraints(model)
        add_battery_net_p_bat_equal_phase_constraints(model)
        add_battery_constant_p_constraints(model)

        equality_only = getattr(config, "equality_only", False) if config else False
        if equality_only:
            return
        add_battery_power_limits(model)
        add_battery_soc_limits(model)
        circular = getattr(config, "circular_constraints", True) if config else True
        if circular:
            add_circular_battery_constraints_pq_control(model)


def add_battery_power_limits(m: CvxModel) -> None:
    def _d(m, _id, ph, t):
        return (0, m.p_discharge[_id, t], m.s_bat_rated[_id, ph])

    def _c(m, _id, ph, t):
        return (0, m.p_charge[_id, t], m.s_bat_rated[_id, ph])

    index = (m.bat_phase_set, m.time_set)
    add_constraint(m, "battery_discharging_limits", index, _d)
    add_constraint(m, "battery_charging_limits", index, _c)


def add_battery_soc_limits(m: CvxModel) -> None:
    def battery_soc_limits(m, _id, t):
        return (m.soc_min[_id], m.soc[_id, t], m.soc_max[_id])

    add_constraint(m, "battery_soc_limits", (m.bat_set, m.time_set), battery_soc_limits)


def add_battery_net_p_bat_constraints(m: CvxModel) -> None:
    def net_discharge(m, _id, t):
        return (
            sum(m.p_bat[_id, phase, t] for phase in m.bat_phases_by_device[_id])
            == m.p_discharge[_id, t] - m.p_charge[_id, t]
        )

    add_constraint(m, "net_discharge", (m.bat_set, m.time_set), net_discharge)


def add_battery_net_p_bat_equal_phase_constraints(m: CvxModel) -> None:
    def net_discharge_equal_phases(m, _id, ph, t):
        n_phases = m.battery_n_phases[_id]
        return (
            m.p_bat[_id, ph, t]
            == (m.p_discharge[_id, t] - m.p_charge[_id, t]) / n_phases
        )

    add_constraint(
        m, "net_discharge", (m.bat_phase_set, m.time_set), net_discharge_equal_phases
    )


def add_battery_energy_constraints(m: CvxModel) -> None:
    def storage(m, _id, t):
        eta_d = m.discharge_efficiency[_id]
        eta_c = m.charge_efficiency[_id]
        if t == m.start_step:
            soc0 = m.start_soc[_id]
        else:
            soc0 = m.soc[_id, t - 1]
        return (
            m.soc[_id, t] - soc0
            == eta_c * m.delta_t * m.p_charge[_id, t]
            - (1 / eta_d) * m.delta_t * m.p_discharge[_id, t]
        )

    add_constraint(m, "storage", (m.bat_set, m.time_set), storage)


def add_battery_constant_q_constraints_p_control(m: CvxModel) -> None:
    def _rule(m, _id, ph, t):
        if m.bat_control_type[_id] not in (ControlVariable.NONE, ControlVariable.P):
            return SKIP
        return m.q_bat[_id, ph, t] == m.q_bat_nom[_id, ph, t]

    add_constraint(m, "battery_constant_q_bat", (m.bat_phase_set, m.time_set), _rule)


def add_battery_constant_p_constraints(model: CvxModel) -> None:
    """Fix P for NONE and Q modes, retaining storage and balanced-phase equalities."""

    def rule(model, device, phase, time):
        if model.bat_control_type[device] not in (
            ControlVariable.NONE,
            ControlVariable.Q,
        ):
            return SKIP
        return model.p_bat[device, phase, time] == model.p_bat_nom[device, phase, time]

    add_constraint(
        model, "battery_constant_p_bat", (model.bat_phase_set, model.time_set), rule
    )


def add_circular_battery_constraints_pq_control(m: CvxModel) -> None:
    """``P_bat^2 + Q_bat^2 <= S_rated^2`` for PQ-controlled batteries."""

    def bat_circle(m, _id, ph, t):
        if m.bat_control_type[_id] != ControlVariable.PQ:
            return SKIP
        return (
            cp.norm(cp.hstack([m.p_bat[_id, ph, t], m.q_bat[_id, ph, t]]), 2)
            <= m.s_bat_rated[_id, ph]
        )

    add_constraint(m, "bat_circle_constraint", (m.bat_phase_set, m.time_set), bat_circle)


def add_circular_battery_constraints(m: CvxModel) -> None:
    """``P_bat^2 + Q_bat^2 <= S_rated^2`` for all batteries."""

    def bat_circle(m, _id, ph, t):
        return (
            cp.norm(cp.hstack([m.p_bat[_id, ph, t], m.q_bat[_id, ph, t]]), 2)
            <= m.s_bat_rated[_id, ph]
        )

    add_constraint(m, "bat_circle_constraint", (m.bat_phase_set, m.time_set), bat_circle)


__all__ = ["BatteryProvider", "create_battery_parameters", "validate_bat_data"]
