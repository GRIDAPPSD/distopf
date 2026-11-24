"""Loads as devices: one `load_data` row per load, with its own shape and CVR.

Each row lists its `phases` and a per-phase `p_{ph}`/`q_{ph}` (loads are not
balanced devices) in per-unit at 1 pu voltage. The nominal power of a phase at
time t is the phase value times the load's schedule multiplier:

* P uses the schedule column named `load_shape`, Q uses `<load_shape>.q` when
  that column exists and `<load_shape>` otherwise;
* a blank `load_shape` or `status == "fixed"` means no scaling.

The delivered power follows the load's own CVR exponents
(`p = p_nom * (1 + cvr_p / 2 * (v2 - 1))`). Loads must not be given in
`bus_data` (`pl_*`, `ql_*`, `cvr_*`, `load_shape`); that raises an error.
Read CSV names with dtype={"device_name": str, "bus_name": str}.

Model components: sets `load_device_set` and `load_device_phase_set`
(device, phase); parameters `p_load_nom[device, phase, t]`,
`q_load_nom[device, phase, t]`, `cvr_p[device]`, `cvr_q[device]`; variables
`p_load[device, phase, t]`, `q_load[device, phase, t]`; constraints
`cvr_p_load`, `cvr_q_load`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyomo.environ as pyo  # type: ignore

from distopf.gad_models.common.device_data import (
    NetworkContext,
    cell_opt_float,
    cell_text,
    check_columns,
    check_connection,
    check_device_names,
    check_phases,
    finish_validation,
    row_label,
)

_PHASES = ("a", "b", "c", "s1", "s2")
_REQUIRED = {"device_name", "bus_name", "phases"}
_STATUSES = ("", "variable", "fixed", "exempt")
_BUS_LOAD_COLUMNS = {"cvr_p", "cvr_q", "load_shape"}


def _check_bus_data_has_no_loads(bus_data: pd.DataFrame) -> None:
    old = sorted(
        c
        for c in bus_data.columns
        if c.startswith(("pl_", "ql_")) or c in _BUS_LOAD_COLUMNS
    )
    if old:
        raise ValueError(
            f"bus_data has load columns {old}; loads belong in load_data "
            "(one row per load)"
        )


def _load_data(case: Any) -> pd.DataFrame:
    data = getattr(case, "load_data", None)
    return pd.DataFrame() if data is None else data


def _device_phase_power(
    row: pd.Series, phases: list[str], label: str, errors: list[str]
) -> dict[str, tuple[float, float]]:
    """(p, q) of every listed phase; power on an unlisted phase is an error."""
    power = {}
    for phase in _PHASES:
        p = cell_opt_float(row, f"p_{phase}", label, errors) or 0.0
        q = cell_opt_float(row, f"q_{phase}", label, errors) or 0.0
        if phase in phases:
            power[phase] = (p, q)
        elif p or q:
            errors.append(f"{label}: has power on phase {phase!r} not in phases")
    return power


def _schedule_multipliers(
    case: Any, row: pd.Series, times: tuple, label: str, errors: list[str]
) -> tuple[dict[Any, float], dict[Any, float]]:
    """(P, Q) multiplier per time step of one load."""
    shape = cell_text(row.get("load_shape"))
    status = cell_text(row.get("status")).lower()
    if not shape or status == "fixed":
        return dict.fromkeys(times, 1.0), dict.fromkeys(times, 1.0)
    schedules = case.schedules
    if shape not in schedules.columns:
        errors.append(f"{label}: load_shape {shape!r} is not a schedules column")
        return dict.fromkeys(times, 1.0), dict.fromkeys(times, 1.0)
    q_shape = f"{shape}.q" if f"{shape}.q" in schedules.columns else shape
    multipliers = []
    for column in (shape, q_shape):
        values = {}
        for time in times:
            if time not in schedules.index:
                errors.append(f"{label}: schedule {column!r} has no row for time {time}")
                break
            values[time] = float(schedules.at[time, column])
        multipliers.append(values)
    return multipliers[0], multipliers[1]


def create_load_components(model: Any, case: Any) -> None:
    """Validate load_data and create the load sets and parameters."""
    _check_bus_data_has_no_loads(case.bus_data)
    data = _load_data(case)
    ctx = NetworkContext.from_model(model, "LoadProvider")
    errors: list[str] = []
    bus_by_device: dict[str, int] = {}
    ports: list[tuple[str, str]] = []
    devices_by_bus_phase: dict[tuple[int, str], list[str]] = {}
    p_nom: dict[tuple[str, str, Any], float] = {}
    q_nom: dict[tuple[str, str, Any], float] = {}
    cvr_p: dict[str, float] = {}
    cvr_q: dict[str, float] = {}
    if not data.empty:
        check_columns(
            data,
            "load_data",
            _REQUIRED,
            {"id", "name", "bus_id"},
            "loads use 'device_name' and 'bus_name'",
        )
        check_device_names(data, errors)
    for index, row in data.iterrows():
        label = row_label(row, index)
        device = cell_text(row["device_name"])
        bus_name = cell_text(row["bus_name"])
        phases = check_phases(label, row["phases"], _PHASES, errors)
        status = cell_text(row.get("status")).lower()
        if status not in _STATUSES:
            errors.append(f"{label}: status {status!r} must be one of {_STATUSES[1:]}")
        if phases is None:
            continue
        power = _device_phase_power(row, phases, label, errors)
        check_connection(
            label, bus_name, [ph for ph, pq in power.items() if any(pq)], ctx, errors
        )
        if bus_name in ctx.bus_name_to_id:
            bus_by_device[device] = ctx.bus_name_to_id[bus_name]
        cvr_p[device] = cell_opt_float(row, "cvr_p", label, errors) or 0.0
        cvr_q[device] = cell_opt_float(row, "cvr_q", label, errors) or 0.0
        p_mult, q_mult = _schedule_multipliers(case, row, ctx.times, label, errors)
        for phase, (p, q) in power.items():
            ports.append((device, phase))
            if device in bus_by_device:
                devices_by_bus_phase.setdefault(
                    (bus_by_device[device], phase), []
                ).append(device)
            for time in ctx.times:
                p_nom[device, phase, time] = p * p_mult.get(time, 1.0)
                q_nom[device, phase, time] = q * q_mult.get(time, 1.0)
    finish_validation("load_data", errors, [], [])

    model.load_device_set = pyo.Set(initialize=list(bus_by_device))
    model.load_device_phase_set = pyo.Set(initialize=ports, dimen=2)
    model.load_bus_by_device = bus_by_device
    model.load_devices_by_bus_phase = devices_by_bus_phase
    model.p_load_nom = pyo.Param(
        model.load_device_phase_set, model.time_set, initialize=p_nom, default=0.0
    )
    model.q_load_nom = pyo.Param(
        model.load_device_phase_set, model.time_set, initialize=q_nom, default=0.0
    )
    model.cvr_p = pyo.Param(model.load_device_set, initialize=cvr_p, default=0.0)
    model.cvr_q = pyo.Param(model.load_device_set, initialize=cvr_q, default=0.0)


class LoadProvider:
    """Own load parameters, variables, CVR constraints, and signed injections."""

    name = "loads"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        create_load_components(model, case)
        model.p_load = pyo.Var(
            model.load_device_phase_set, model.time_set, initialize=0
        )
        model.q_load = pyo.Var(
            model.load_device_phase_set, model.time_set, initialize=0
        )

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        devices = model.load_devices_by_bus_phase.get((bus, phase), [])
        return -sum(model.p_load[device, phase, time] for device in devices)

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        devices = model.load_devices_by_bus_phase.get((bus, phase), [])
        return -sum(model.q_load[device, phase, time] for device in devices)

    def add_constraints(self, model: Any, config: Any) -> None:
        """Add voltage-dependent load equations."""
        if len(model.load_device_phase_set) == 0:
            return
        free_boundary_loads = getattr(config, "free_boundary_loads", False)
        add_cvr_load_constraints(model, free_boundary_loads)


def add_cvr_load_constraints(model: Any, free_boundary_loads: bool = False) -> None:
    """Apply each load's CVR, leaving coordinated OUT-bus loads free."""

    def is_free(device):
        return free_boundary_loads and (
            model.load_bus_by_device[device] in model.boundary_out_set
        )

    def cvr_p_rule(model, device, phase, time):
        if is_free(device):
            return pyo.Constraint.Skip
        bus = model.load_bus_by_device[device]
        return model.p_load[device, phase, time] == model.p_load_nom[
            device, phase, time
        ] * (1 + model.cvr_p[device] / 2 * (model.v2[bus, phase, time] - 1))

    def cvr_q_rule(model, device, phase, time):
        if is_free(device):
            return pyo.Constraint.Skip
        bus = model.load_bus_by_device[device]
        return model.q_load[device, phase, time] == model.q_load_nom[
            device, phase, time
        ] * (1 + model.cvr_q[device] / 2 * (model.v2[bus, phase, time] - 1))

    model.cvr_p_load = pyo.Constraint(
        model.load_device_phase_set, model.time_set, rule=cvr_p_rule
    )
    model.cvr_q_load = pyo.Constraint(
        model.load_device_phase_set, model.time_set, rule=cvr_q_rule
    )


__all__ = ["LoadProvider", "add_cvr_load_constraints", "create_load_components"]
