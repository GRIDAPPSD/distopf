"""Named battery devices with balanced phase power and SOC dynamics.

Tables use device_name, bus_name, phases, s_max and energy_capacity; existing
bus-ID tables remain supported. Total P/Q and ratings are divided across phases.
Optional SOC/efficiency fields retain their legacy defaults. Missing control
mode defaults to P; blank fixes P and Q. Existing component names are retained.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models.common.model_types import ControlVariable
from distopf.pyomo_models.common.protocol import LindistModelProtocol
from distopf.pyomo_models.common.data import parse_phases
from distopf.pyomo_models.common.device_data import (
    NetworkContext,
    cell_control,
    cell_num,
    cell_opt_float,
    cell_required_float,
    cell_text,
    check_columns,
    check_connection,
    check_device_names,
    check_phases,
    finish_validation,
    row_label,
)

_PHASES = ("a", "b", "c", "s1", "s2")
_REQUIRED = {"device_name", "bus_name", "phases", "s_max", "energy_capacity"}
_KNOWN = _REQUIRED | {
    "p",
    "q",
    "q_min",
    "q_max",
    "min_soc",
    "max_soc",
    "start_soc",
    "charge_efficiency",
    "discharge_efficiency",
    "annual_cycle_limit",
    "control_variable",
    "s_base",
}


def _battery_data(model: Any, case: Any) -> pd.DataFrame:
    data = getattr(case, "bat_data", None)
    data = data.copy() if data is not None else pd.DataFrame()
    if not data.empty and "device_name" not in data and "id" in data:
        if "bus_name" in data:
            raise ValueError("bat_data mixes legacy 'id' with 'bus_name'")
        names = {bus: name for name, bus in model.bus_name_to_id_map.items()}
        data["bus_name"] = data["id"].map(names)
        data["device_name"] = [f"bat_{index}" for index in range(len(data))]
        data = data.drop(columns=[key for key in ("id", "name") if key in data])
    if "control_variable" not in data:
        data["control_variable"] = "P"
    return data


def validate_bat_data(data: pd.DataFrame, ctx: NetworkContext) -> None:
    """Validate battery connections, ratings, SOC ranges, and efficiencies."""
    if data.empty:
        return
    check_columns(
        data,
        "bat_data",
        _REQUIRED,
        _KNOWN,
        {"id", "name", "bus_id"},
        "batteries use 'device_name' and 'bus_name'",
    )
    errors: list[str] = []
    check_device_names(data, errors)
    for index, row in data.iterrows():
        label = row_label(row, index)
        phases = check_phases(label, row["phases"], _PHASES, errors)
        if phases is not None:
            check_connection(label, cell_text(row["bus_name"]), phases, ctx, errors)
        cell_required_float(row, "s_max", label, errors, gt=0)
        cell_required_float(row, "energy_capacity", label, errors, gt=0)
        values: dict[str, float] = {}
        for key, default in (
            ("min_soc", 0),
            ("max_soc", 1),
            ("start_soc", 0.5),
            ("charge_efficiency", 1),
            ("discharge_efficiency", 1),
            ("annual_cycle_limit", 365),
            ("p", 0),
            ("q", 0),
        ):
            value = cell_opt_float(row, key, label, errors)
            values[key] = default if value is None else value
        if not 0 <= values["min_soc"] <= values["start_soc"] <= values["max_soc"] <= 1:
            errors.append(f"{label}: require 0 <= min_soc <= start_soc <= max_soc <= 1")
        for key in ("charge_efficiency", "discharge_efficiency"):
            if not 0 < values[key] <= 1:
                errors.append(f"{label}: {key} must be in (0, 1]")
        if values["annual_cycle_limit"] < 0:
            errors.append(f"{label}: annual_cycle_limit must be >= 0")
        q_min = cell_opt_float(row, "q_min", label, errors)
        q_max = cell_opt_float(row, "q_max", label, errors)
        if q_min is not None and q_max is not None and q_min > q_max:
            errors.append(f"{label}: q_min must not exceed q_max")
        try:
            cell_control(row)
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
    finish_validation("bat_data", errors, [], [])


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
    model.p_bat_nom = pyo.Param(
        model.bat_phase_set, model.time_set, initialize=p_data, default=0.0
    )
    model.q_bat_nom = pyo.Param(
        model.bat_phase_set, model.time_set, initialize=q_data, default=0.0
    )
    model.s_bat_rated = pyo.Param(
        model.bat_phase_set, initialize=rating, default=1000.0
    )
    model.q_bat_min = pyo.Param(model.bat_phase_set, initialize=q_min, default=-1000.0)
    model.q_bat_max = pyo.Param(model.bat_phase_set, initialize=q_max, default=1000.0)
    model.bat_control_type = pyo.Param(
        model.bat_set, initialize=control, default=0, within=pyo.Any
    )
    model.energy_capacity = pyo.Param(model.bat_set, initialize=energy, default=0)
    model.soc_min = pyo.Param(model.bat_set, initialize=soc_min, default=0)
    model.soc_max = pyo.Param(model.bat_set, initialize=soc_max, default=1)
    model.start_soc = pyo.Param(model.bat_set, initialize=start_soc, default=0.5)
    model.charge_efficiency = pyo.Param(
        model.bat_set, initialize=charge_eff, default=1.0
    )
    model.discharge_efficiency = pyo.Param(
        model.bat_set, initialize=discharge_eff, default=1.0
    )
    model.annual_cycle_limit = pyo.Param(model.bat_set, initialize=cycles, default=365)
    model.battery_has_a_phase = pyo.Param(model.bat_set, initialize=has_a, default=True)
    model.battery_has_b_phase = pyo.Param(model.bat_set, initialize=has_b, default=True)
    model.battery_has_c_phase = pyo.Param(model.bat_set, initialize=has_c, default=True)
    model.battery_n_phases = pyo.Param(model.bat_set, initialize=n_phases, default=3)
    model.battery_has_phase = pyo.Param(
        model.bat_set, ("a", "b", "c"), initialize=has_phase, default=True
    )


class BatteryProvider:
    """Create and constrain battery variables for a Pyomo model."""

    name = "batteries"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        """Create battery components after the network provider."""
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
        model.bat_set = pyo.Set(initialize=list(bus_by_device))
        model.bat_phase_set = pyo.Set(initialize=ports, dimen=2)
        model.bat_bus_by_device = bus_by_device
        model.bat_devices_by_bus_phase = devices_by_bus_phase
        model.bat_phases_by_device = phases_by_device
        create_battery_parameters(model, case)
        model.p_charge = pyo.Var(model.bat_set, model.time_set, initialize=0)
        model.p_discharge = pyo.Var(model.bat_set, model.time_set, initialize=0)
        model.p_bat = pyo.Var(model.bat_phase_set, model.time_set, initialize=0)
        model.q_bat = pyo.Var(model.bat_phase_set, model.time_set, initialize=0)
        model.soc = pyo.Var(model.bat_set, model.time_set, initialize=0.5)

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
        """Attach battery operating constraints."""
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


# Constraints ----------------------------------------------------------------


def add_battery_power_limits(m: LindistModelProtocol) -> None:
    def _d(m: LindistModelProtocol, _id, ph, t):
        return (0, m.p_discharge[_id, t], m.s_bat_rated[_id, ph])

    def _c(m: LindistModelProtocol, _id, ph, t):
        return (0, m.p_charge[_id, t], m.s_bat_rated[_id, ph])

    m.battery_discharging_limits = pyo.Constraint(m.bat_phase_set, m.time_set, rule=_d)
    m.battery_charging_limits = pyo.Constraint(m.bat_phase_set, m.time_set, rule=_c)


def add_battery_soc_limits(m: LindistModelProtocol) -> None:
    def battery_soc_limits(m: LindistModelProtocol, _id, t):
        return (m.soc_min[_id], m.soc[_id, t], m.soc_max[_id])

    m.battery_soc_limits = pyo.Constraint(
        m.bat_set, m.time_set, rule=battery_soc_limits
    )


def add_battery_net_p_bat_constraints(m: LindistModelProtocol) -> None:
    def net_discharge(m: LindistModelProtocol, _id, t):
        return (
            sum(m.p_bat[_id, phase, t] for phase in m.bat_phases_by_device[_id])
            == m.p_discharge[_id, t] - m.p_charge[_id, t]
        )

    m.net_discharge = pyo.Constraint(m.bat_set, m.time_set, rule=net_discharge)


def add_battery_net_p_bat_equal_phase_constraints(m: LindistModelProtocol) -> None:
    def net_discharge_equal_phases(m: LindistModelProtocol, _id, ph, t):
        n_phases = m.battery_n_phases[_id]
        return (
            m.p_bat[_id, ph, t]
            == (m.p_discharge[_id, t] - m.p_charge[_id, t]) / n_phases
        )

    m.net_discharge = pyo.Constraint(
        m.bat_phase_set, m.time_set, rule=net_discharge_equal_phases
    )


def add_battery_energy_constraints(m: LindistModelProtocol) -> None:
    def storage(m: LindistModelProtocol, _id, t):
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

    m.storage = pyo.Constraint(m.bat_set, m.time_set, rule=storage)


def add_battery_constant_q_constraints_p_control(m: LindistModelProtocol) -> None:
    def _rule(m: LindistModelProtocol, _id, ph, t):
        if m.bat_control_type[_id] not in (ControlVariable.NONE, ControlVariable.P):
            return pyo.Constraint.Skip
        return m.q_bat[_id, ph, t] == m.q_bat_nom[_id, ph, t]

    m.battery_constant_q_bat = pyo.Constraint(m.bat_phase_set, m.time_set, rule=_rule)


def add_battery_constant_p_constraints(model: Any) -> None:
    """Fix P for NONE and Q modes, retaining storage and balanced-phase equalities."""

    def rule(model, device, phase, time):
        if model.bat_control_type[device] not in (
            ControlVariable.NONE,
            ControlVariable.Q,
        ):
            return pyo.Constraint.Skip
        return model.p_bat[device, phase, time] == model.p_bat_nom[device, phase, time]

    model.battery_constant_p_bat = pyo.Constraint(
        model.bat_phase_set, model.time_set, rule=rule
    )


def add_circular_battery_constraints_pq_control(m: LindistModelProtocol) -> None:
    """
    Add circular battery apparent power constraints.

    Enforces the exact quadratic constraint:
        P_bat^2 + Q_bat^2 <= S_rated^2

    This is a nonlinear (quadratic) constraint requiring a nonlinear solver
    (e.g., IPOPT) or a solver supporting second-order cone constraints.
    """

    def bat_circle(m: LindistModelProtocol, _id, ph, t):
        if m.bat_control_type[_id] != ControlVariable.PQ:
            return pyo.Constraint.Skip
        return (
            m.p_bat[_id, ph, t] ** 2 + m.q_bat[_id, ph, t] ** 2
            <= m.s_bat_rated[_id, ph] ** 2
        )

    m.bat_circle_constraint = pyo.Constraint(
        m.bat_phase_set, m.time_set, rule=bat_circle
    )


def add_circular_battery_constraints(m: LindistModelProtocol) -> None:
    """
    Add circular battery apparent power constraints.

    Enforces the exact quadratic constraint:
        P_bat^2 + Q_bat^2 <= S_rated^2

    This is a nonlinear (quadratic) constraint requiring a nonlinear solver
    (e.g., IPOPT) or a solver supporting second-order cone constraints.
    """

    def bat_circle(m: LindistModelProtocol, _id, ph, t):
        return (
            m.p_bat[_id, ph, t] ** 2 + m.q_bat[_id, ph, t] ** 2
            <= m.s_bat_rated[_id, ph] ** 2
        )

    m.bat_circle_constraint = pyo.Constraint(
        m.bat_phase_set, m.time_set, rule=bat_circle
    )


__all__ = ["BatteryProvider", "create_battery_parameters", "validate_bat_data"]
