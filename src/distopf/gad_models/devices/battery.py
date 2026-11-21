"""Named battery devices (GAD schema v1) with SOC dynamics.

One decision per device: `p_bat[device, t]` and `q_bat[device, t]` are device
totals (discharge positive). A device injects total / n_phases on each of its
phases. Powers, ratings and rate limits are per unit on `s_base`; `soc` is a
fraction of `energy_capacity`.

Optional columns default as in the data contract: `p_max` to `s_max`, the
charge/discharge rate limits to `p_max`, `q_min`/`q_max` to -/+ `s_max`.
A missing control mode defaults to P; blank fixes P and Q.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyomo.environ as pyo  # type: ignore

from distopf.gad_models.common.model_types import ControlVariable
from distopf.gad_models.common.protocol import LindistModelProtocol
from distopf.gad_models.common.data import parse_phases
from distopf.gad_models.common.device_data import (
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
# Columns of the old per-phase and bus-ID battery tables.
_LEGACY = {"id", "name", "bus_id"} | {
    f"{quantity}_{phase}{suffix}"
    for phase in _PHASES
    for quantity, suffix in (("p", ""), ("q", ""), ("s", "_max"))
}
_LEGACY_HINT = "batteries are device totals; use 'device_name' and 'bus_name'"
_RATE_COLUMNS = ("p_max", "p_charge_max", "p_discharge_max")


def _battery_data(case: Any) -> pd.DataFrame:
    data = getattr(case, "bat_data", None)
    data = data.copy() if data is not None else pd.DataFrame()
    if "control_variable" not in data:
        data["control_variable"] = "P"
    return data


def validate_bat_data(data: pd.DataFrame, ctx: NetworkContext) -> None:
    """Validate battery connections, ratings, rate limits, SOC ranges, efficiencies."""
    if data.empty:
        return
    check_columns(data, "bat_data", _REQUIRED, _LEGACY, _LEGACY_HINT)
    errors: list[str] = []
    check_device_names(data, errors)
    for index, row in data.iterrows():
        label = row_label(row, index)
        phases = check_phases(label, row["phases"], _PHASES, errors)
        if phases is not None:
            check_connection(label, cell_text(row["bus_name"]), phases, ctx, errors)
        cell_required_float(row, "s_max", label, errors, gt=0)
        cell_required_float(row, "energy_capacity", label, errors, gt=0)
        for key in _RATE_COLUMNS:
            value = cell_opt_float(row, key, label, errors)
            if value is not None and value < 0:
                errors.append(f"{label}: {key}={value} must be >= 0")
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
    """Create every battery parameter. Powers are device totals."""
    p_data, q_data = {}, {}
    rating, q_min, q_max, p_charge_max, p_discharge_max = {}, {}, {}, {}, {}
    energy, soc_min, soc_max, start_soc = {}, {}, {}, {}
    charge_eff, discharge_eff, cycles, control, n_phases = {}, {}, {}, {}, {}
    for _, row in _battery_data(case).iterrows():
        device = cell_text(row["device_name"])
        n_phases[device] = len(parse_phases(cell_text(row["phases"])))
        s_max = cell_num(row, "s_max", 0.0)
        p_max = cell_num(row, "p_max", s_max)
        rating[device] = s_max
        q_min[device] = max(-s_max, cell_num(row, "q_min", -s_max))
        q_max[device] = min(s_max, cell_num(row, "q_max", s_max))
        p_charge_max[device] = cell_num(row, "p_charge_max", p_max)
        p_discharge_max[device] = cell_num(row, "p_discharge_max", p_max)
        energy[device] = cell_num(row, "energy_capacity", 0)
        soc_min[device] = cell_num(row, "min_soc", 0)
        soc_max[device] = cell_num(row, "max_soc", 1)
        start_soc[device] = cell_num(row, "start_soc", 0.5)
        charge_eff[device] = cell_num(row, "charge_efficiency", 1)
        discharge_eff[device] = cell_num(row, "discharge_efficiency", 1)
        cycles[device] = cell_num(row, "annual_cycle_limit", 365)
        control[device] = cell_control(row)
        for time in model.time_set:
            p_data[device, time] = cell_num(row, "p", 0.0)
            q_data[device, time] = cell_num(row, "q", 0.0)
    model.p_bat_nom = pyo.Param(
        model.bat_set, model.time_set, initialize=p_data, default=0.0
    )
    model.q_bat_nom = pyo.Param(
        model.bat_set, model.time_set, initialize=q_data, default=0.0
    )
    model.s_bat_rated = pyo.Param(model.bat_set, initialize=rating, default=0.0)
    model.q_bat_min = pyo.Param(model.bat_set, initialize=q_min, default=0.0)
    model.q_bat_max = pyo.Param(model.bat_set, initialize=q_max, default=0.0)
    model.p_charge_max = pyo.Param(model.bat_set, initialize=p_charge_max, default=0.0)
    model.p_discharge_max = pyo.Param(
        model.bat_set, initialize=p_discharge_max, default=0.0
    )
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
    model.battery_n_phases = pyo.Param(model.bat_set, initialize=n_phases, default=3)


class BatteryProvider:
    """Create and constrain battery variables for a Pyomo model."""

    name = "batteries"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        """Create battery components after the network provider."""
        ctx = NetworkContext.from_model(model, "BatteryProvider")
        data = _battery_data(case)
        validate_bat_data(data, ctx)
        bus_by_device: dict[str, int] = {}
        devices_by_bus_phase: dict[tuple[int, str], list[str]] = {}
        phases_by_device: dict[str, list[str]] = {}
        for _, row in data.iterrows():
            device = cell_text(row["device_name"])
            bus = ctx.bus_name_to_id[cell_text(row["bus_name"])]
            phases_by_device[device] = parse_phases(cell_text(row["phases"]))
            bus_by_device[device] = bus
            for phase in phases_by_device[device]:
                devices_by_bus_phase.setdefault((bus, phase), []).append(device)
        model.bat_set = pyo.Set(initialize=list(bus_by_device))
        model.bat_bus_by_device = bus_by_device
        model.bat_devices_by_bus_phase = devices_by_bus_phase
        model.bat_phases_by_device = phases_by_device
        create_battery_parameters(model, case)
        model.p_charge = pyo.Var(model.bat_set, model.time_set, initialize=0)
        model.p_discharge = pyo.Var(model.bat_set, model.time_set, initialize=0)
        model.p_bat = pyo.Var(model.bat_set, model.time_set, initialize=0)
        model.q_bat = pyo.Var(model.bat_set, model.time_set, initialize=0)
        model.soc = pyo.Var(model.bat_set, model.time_set, initialize=0.5)

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return sum(
            model.p_bat[device, time] / model.battery_n_phases[device]
            for device in model.bat_devices_by_bus_phase.get((bus, phase), [])
        )

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return sum(
            model.q_bat[device, time] / model.battery_n_phases[device]
            for device in model.bat_devices_by_bus_phase.get((bus, phase), [])
        )

    def add_constraints(self, model: Any, config: Any) -> None:
        """Attach battery operating constraints."""
        if len(model.bat_set) == 0:
            return
        add_battery_constant_q_constraints_p_control(model)
        add_battery_energy_constraints(model)
        add_battery_net_p_bat_constraints(model)
        add_battery_constant_p_constraints(model)

        equality_only = getattr(config, "equality_only", False) if config else False
        if equality_only:
            return
        add_battery_power_limits(model)
        add_battery_q_limits(model)
        add_battery_soc_limits(model)
        circular = getattr(config, "circular_constraints", True) if config else True
        if circular:
            add_circular_battery_constraints_pq_control(model)


# Constraints ----------------------------------------------------------------


def add_battery_power_limits(m: LindistModelProtocol) -> None:
    """Charge and discharge rate limits on the device totals."""

    def _d(m: LindistModelProtocol, _id, t):
        return (0, m.p_discharge[_id, t], m.p_discharge_max[_id])

    def _c(m: LindistModelProtocol, _id, t):
        return (0, m.p_charge[_id, t], m.p_charge_max[_id])

    m.battery_discharging_limits = pyo.Constraint(m.bat_set, m.time_set, rule=_d)
    m.battery_charging_limits = pyo.Constraint(m.bat_set, m.time_set, rule=_c)


def add_battery_q_limits(m: LindistModelProtocol) -> None:
    """q_min <= q_bat <= q_max where Q is a decision (control Q or PQ)."""

    def _rule(m: LindistModelProtocol, _id, t):
        if m.bat_control_type[_id] not in (ControlVariable.Q, ControlVariable.PQ):
            return pyo.Constraint.Skip
        return (m.q_bat_min[_id], m.q_bat[_id, t], m.q_bat_max[_id])

    m.battery_q_limits = pyo.Constraint(m.bat_set, m.time_set, rule=_rule)


def add_battery_soc_limits(m: LindistModelProtocol) -> None:
    def battery_soc_limits(m: LindistModelProtocol, _id, t):
        return (m.soc_min[_id], m.soc[_id, t], m.soc_max[_id])

    m.battery_soc_limits = pyo.Constraint(
        m.bat_set, m.time_set, rule=battery_soc_limits
    )


def add_battery_net_p_bat_constraints(m: LindistModelProtocol) -> None:
    def net_discharge(m: LindistModelProtocol, _id, t):
        return m.p_bat[_id, t] == m.p_discharge[_id, t] - m.p_charge[_id, t]

    m.net_discharge = pyo.Constraint(m.bat_set, m.time_set, rule=net_discharge)


def add_battery_energy_constraints(m: LindistModelProtocol) -> None:
    """SOC (a fraction of energy_capacity) changes with the energy in and out."""

    def storage(m: LindistModelProtocol, _id, t):
        eta_d = m.discharge_efficiency[_id]
        eta_c = m.charge_efficiency[_id]
        if t == m.start_step:
            soc0 = m.start_soc[_id]
        else:
            soc0 = m.soc[_id, t - 1]
        energy_in = eta_c * m.delta_t * m.p_charge[_id, t]
        energy_out = (1 / eta_d) * m.delta_t * m.p_discharge[_id, t]
        return m.soc[_id, t] - soc0 == (energy_in - energy_out) / m.energy_capacity[_id]

    m.storage = pyo.Constraint(m.bat_set, m.time_set, rule=storage)


def add_battery_constant_q_constraints_p_control(m: LindistModelProtocol) -> None:
    def _rule(m: LindistModelProtocol, _id, t):
        if m.bat_control_type[_id] not in (ControlVariable.NONE, ControlVariable.P):
            return pyo.Constraint.Skip
        return m.q_bat[_id, t] == m.q_bat_nom[_id, t]

    m.battery_constant_q_bat = pyo.Constraint(m.bat_set, m.time_set, rule=_rule)


def add_battery_constant_p_constraints(model: Any) -> None:
    """Fix P for NONE and Q modes."""

    def rule(model, device, time):
        if model.bat_control_type[device] not in (
            ControlVariable.NONE,
            ControlVariable.Q,
        ):
            return pyo.Constraint.Skip
        return model.p_bat[device, time] == model.p_bat_nom[device, time]

    model.battery_constant_p_bat = pyo.Constraint(
        model.bat_set, model.time_set, rule=rule
    )


def add_circular_battery_constraints_pq_control(m: LindistModelProtocol) -> None:
    """
    Add circular battery apparent power constraints for PQ control.

    Enforces the exact quadratic constraint on the device totals:
        P_bat^2 + Q_bat^2 <= S_rated^2

    This is a nonlinear (quadratic) constraint requiring a nonlinear solver
    (e.g., IPOPT) or a solver supporting second-order cone constraints.
    """

    def bat_circle(m: LindistModelProtocol, _id, t):
        if m.bat_control_type[_id] != ControlVariable.PQ:
            return pyo.Constraint.Skip
        return m.p_bat[_id, t] ** 2 + m.q_bat[_id, t] ** 2 <= m.s_bat_rated[_id] ** 2

    m.bat_circle_constraint = pyo.Constraint(m.bat_set, m.time_set, rule=bat_circle)


__all__ = ["BatteryProvider", "create_battery_parameters", "validate_bat_data"]
