"""Load device provider and compatibility ownership for bus loads."""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models.common.protocol import LindistModelProtocol
from distopf.pyomo_models.common.device_data import (
    NetworkContext,
    cell_opt_float,
    cell_text,
    check_connection,
    check_phases,
    finish_validation,
)


def create_load_parameters(model: Any, case: Any) -> None:
    """Create load and CVR parameter components from case data."""
    ctx = NetworkContext.from_model(model, "LoadProvider")
    errors: list[str] = []
    p_data: dict[tuple[int, str, Any], float] = {}
    q_data: dict[tuple[int, str, Any], float] = {}
    cvr_p: dict[tuple[int, str], float] = {}
    cvr_q: dict[tuple[int, str], float] = {}
    for _, row in case.bus_data.iterrows():
        bus = int(row["id"])
        bus_name = cell_text(row["name"])
        label = f"load at bus {bus_name!r}"
        phases = check_phases(label, row["phases"], ("a", "b", "c", "s1", "s2"), errors)
        if phases is None:
            continue
        for phase in phases:
            p_load = cell_opt_float(row, f"pl_{phase}", label, errors) or 0.0
            q_load = cell_opt_float(row, f"ql_{phase}", label, errors) or 0.0
            if phase in ("s1", "s2"):
                p_load += (cell_opt_float(row, "pl_s1s2", label, errors) or 0.0) / 2
                q_load += (cell_opt_float(row, "ql_s1s2", label, errors) or 0.0) / 2
            if p_load or q_load:
                check_connection(label, bus_name, [phase], ctx, errors)
            cvr_p[bus, phase] = cell_opt_float(row, "cvr_p", label, errors) or 0.0
            cvr_q[bus, phase] = cell_opt_float(row, "cvr_q", label, errors) or 0.0
            shape = cell_text(row.get("load_shape")) or "default"
            for time in ctx.times:
                for kind, nominal, values in (
                    ("p", p_load, p_data),
                    ("q", q_load, q_data),
                ):
                    column = (
                        shape
                        if shape in case.schedules.columns
                        else f"{shape}.{phase}.{kind}"
                    )
                    multiplier = 1.0
                    if column in case.schedules.columns:
                        if time not in case.schedules.index:
                            errors.append(
                                f"{label}: schedule {column!r} has no row for time {time}"
                            )
                        else:
                            scheduled = cell_opt_float(
                                case.schedules.loc[time], column, label, errors
                            )
                            if scheduled is not None:
                                multiplier = scheduled
                    values[bus, phase, time] = nominal * multiplier
    finish_validation("bus loads", errors, [], [])
    model.p_load_nom = pyo.Param(
        model.bus_phase_set, model.time_set, initialize=p_data, default=0.0
    )
    model.q_load_nom = pyo.Param(
        model.bus_phase_set, model.time_set, initialize=q_data, default=0.0
    )
    model.cvr_p = pyo.Param(model.bus_phase_set, initialize=cvr_p, default=0.0)
    model.cvr_q = pyo.Param(model.bus_phase_set, initialize=cvr_q, default=0.0)


class LoadProvider:
    """Own load parameters, variables, CVR constraints, and signed injections."""

    name = "loads"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        create_load_parameters(model, case)
        model.p_load = pyo.Var(model.bus_phase_set, model.time_set, initialize=0)
        model.q_load = pyo.Var(model.bus_phase_set, model.time_set, initialize=0)

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        key = (bus, phase, time)
        return -model.p_load[key] if key in model.p_load else 0

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        key = (bus, phase, time)
        return -model.q_load[key] if key in model.q_load else 0

    def add_constraints(self, model: Any, config: Any) -> None:
        """Add voltage-dependent load equations."""
        if len(model.bus_phase_set) == 0:
            return
        free_boundary_loads = getattr(config, "free_boundary_loads", False)
        add_cvr_load_constraints(model, free_boundary_loads)


def add_cvr_load_constraints(model: Any, free_boundary_loads: bool = False) -> None:
    """Apply CVR to bus loads, leaving coordinated OUT-bus loads free."""

    def cvr_p_rule(model, bus, phase, time):
        if free_boundary_loads and bus in model.boundary_out_set:
            return pyo.Constraint.Skip
        nominal = model.p_load_nom[bus, phase, time]
        return model.p_load[bus, phase, time] == nominal * (
            1 + model.cvr_p[bus, phase] / 2 * (model.v2[bus, phase, time] - 1)
        )

    def cvr_q_rule(model, bus, phase, time):
        if free_boundary_loads and bus in model.boundary_out_set:
            return pyo.Constraint.Skip
        nominal = model.q_load_nom[bus, phase, time]
        return model.q_load[bus, phase, time] == nominal * (
            1 + model.cvr_q[bus, phase] / 2 * (model.v2[bus, phase, time] - 1)
        )

    model.cvr_p_load = pyo.Constraint(
        model.bus_phase_set, model.time_set, rule=cvr_p_rule
    )
    model.cvr_q_load = pyo.Constraint(
        model.bus_phase_set, model.time_set, rule=cvr_q_rule
    )


__all__ = ["LoadProvider", "create_load_parameters"]
