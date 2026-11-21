"""Shared device-level math for generators and PV systems (GAD schema v1).

Both are "injectors": one total P and one total Q per device and time step,
split equally over the device's phases. The providers differ only in how they
compute the available power; everything else lives here, keyed by a prefix
(``"gen"`` or ``"pv"``):

    sets      {prefix}_device_set
    maps      {prefix}_phases_by_device, {prefix}_n_phases, {prefix}_bus_by_device,
              {prefix}_devices_by_bus_phase
    params    {prefix}_s_max[d], {prefix}_q_min[d], {prefix}_q_max[d],
              {prefix}_control[d], {prefix}_cost[d], {prefix}_p_min[d],
              {prefix}_p_available[d, t], {prefix}_q_setpoint[d, t]
    vars      p_{prefix}[d, t] >= 0, q_{prefix}[d, t]

Control modes: P is fixed to the available power in NONE and Q; Q is fixed to
its setpoint in NONE and P. Positive power is injection into the bus.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd
import pyomo.environ as pyo  # type: ignore

from distopf.gad_models.common.device_data import (
    TOL,
    NetworkContext,
    cell_control,
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
from distopf.gad_models.common.model_types import ControlVariable

PHASES = ("a", "b", "c", "s1", "s2")
_P_FIXED = (ControlVariable.NONE, ControlVariable.Q)
_Q_FIXED = (ControlVariable.NONE, ControlVariable.P)

# Right half (P >= 0) of an octagon inscribed in the rating circle:
# coefficients (a, b) of a*p + b*q <= s_max.
_C = math.sqrt(2) - 1
OCTAGON_HALF = ((_C, 1.0), (1.0, _C), (1.0, -_C), (_C, -1.0))

_LEGACY_PHASE_COLUMN = re.compile(r"^[pqs]_(a|b|c|s1|s2)(_(min|max))?$")
_LEGACY_COLUMNS = {"balanced", "id", "name", "bus_id", "primary_phase"}


def cfg(config: Any, key: str, default: Any) -> Any:
    return getattr(config, key, default) if config is not None else default


# --------------------------------------------------------------------------- #
# Validated device data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PowerTerms:
    """How a device turns its shape multiplier into available P (all totals).

    available(t) = min(scale * multiplier(t), cap); P never goes below ``floor``
    (itself limited to the available power).
    """

    scale: float
    cap: float
    floor: float


@dataclass(frozen=True)
class Injector:
    """One validated device row."""

    name: str
    bus: int
    phases: list[str]
    s_max: float
    q_min: float
    q_max: float
    control: ControlVariable
    cost: float
    p_min: float
    p_available: dict[Any, float]
    q_setpoint: dict[Any, float]


def reject_legacy_columns(data: pd.DataFrame, table: str) -> None:
    """Per-phase and legacy columns are not part of schema v1."""
    legacy = sorted(
        c for c in data.columns if _LEGACY_PHASE_COLUMN.match(c) or c in _LEGACY_COLUMNS
    )
    if legacy:
        raise ValueError(
            f"{table}: legacy columns {legacy} are not supported; devices carry totals "
            "(p, q, s_max, q_min, q_max) and are identified by device_name/bus_name"
        )


def shape_multipliers(
    schedules: pd.DataFrame, column: str, times: tuple[Any, ...]
) -> dict[Any, float]:
    """Multipliers of a schedule column at each time step (column names match exactly)."""
    if column not in schedules.columns:
        raise ValueError(f"shape {column!r} is not a column of case.schedules")
    missing = [t for t in times if t not in schedules.index]
    if missing:
        raise ValueError(f"schedule {column!r} has no rows for time steps {missing[:5]}")
    values = pd.to_numeric(schedules.loc[list(times), column], errors="coerce")
    if not values.map(math.isfinite).all():
        raise ValueError(f"schedule {column!r} has blank or non-numeric values")
    return dict(zip(times, map(float, values)))


def _resolve_multipliers(
    schedules: pd.DataFrame, shape: str, times: tuple[Any, ...]
) -> tuple[dict[Any, float], dict[Any, float]]:
    """(P multipliers, Q multipliers); a blank shape is constant 1.

    Q uses the ``<shape>.q`` column when it exists, else the P multipliers.
    """
    if not shape:
        constant = {t: 1.0 for t in times}
        return constant, constant
    p_mult = shape_multipliers(schedules, shape, times)
    if f"{shape}.q" in schedules.columns:
        return p_mult, shape_multipliers(schedules, f"{shape}.q", times)
    return p_mult, p_mult


def _q_bounds(
    s_max: float, q_min: float | None, q_max: float | None
) -> tuple[float, float]:
    """Static reactive limits, clamped to +-s_max."""
    lo = -s_max if q_min is None else max(-s_max, q_min)
    hi = s_max if q_max is None else min(s_max, q_max)
    return lo, hi


def q_range_with_fixed_p(s_max: float, p: float, lo: float, hi: float) -> tuple[float, float]:
    """Reactive range in Q mode, where the fixed P uses part of the rating."""
    cap = math.sqrt(max(0.0, s_max**2 - p**2))
    return max(-cap, lo), min(cap, hi)


def q_range_empty(s_max: float, p: float, lo: float, hi: float) -> bool:
    q_lo, q_hi = q_range_with_fixed_p(s_max, p, lo, hi)
    return q_lo > q_hi + TOL


def read_injectors(
    data: pd.DataFrame,
    table: str,
    shape_column: str,
    power_terms: Callable[[pd.Series, str, list[str]], PowerTerms | None],
    ctx: NetworkContext,
    schedules: pd.DataFrame,
    limits_enforced: bool,
) -> list[Injector]:
    """Validate a device table and return its devices.

    ``power_terms(row, label, errors)`` reads the device-specific P columns and
    returns None (after appending to ``errors``) if they are invalid.

    Raises:
        ValueError: malformed data (all row-level problems are listed together).
        InfeasibleCaseError: well-formed data that is provably infeasible.
    """
    if data.empty:
        return []
    check_columns(
        data, table, {"device_name", "bus_name", "phases", "s_max"}, set(), ""
    )
    reject_legacy_columns(data, table)

    errors: list[str] = []
    infeasible: list[str] = []
    check_device_names(data, errors)
    injectors: list[Injector] = []

    for idx, row in data.iterrows():
        label = row_label(row, idx)
        start = len(errors)
        phases = check_phases(label, row["phases"], PHASES, errors)
        if phases is None:
            continue
        bus_name = cell_text(row["bus_name"])
        check_connection(label, bus_name, phases, ctx, errors)
        try:
            control = cell_control(row)
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            continue
        cost = cell_opt_float(row, "cost", label, errors)
        s_max = cell_required_float(row, "s_max", label, errors, gt=0.0)
        q_set = cell_opt_float(row, "q", label, errors)
        q_lo = cell_opt_float(row, "q_min", label, errors)
        q_hi = cell_opt_float(row, "q_max", label, errors)
        terms = power_terms(row, label, errors)
        try:
            p_mult, q_mult = _resolve_multipliers(
                schedules, cell_text(row.get(shape_column)), ctx.times
            )
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            continue
        if len(errors) != start or s_max is None or terms is None:
            continue

        lo, hi = _q_bounds(s_max, q_lo, q_hi)
        if lo > hi + TOL:
            errors.append(
                f"{label}: q limits are empty after clamping to +-s_max ([{lo:g}, {hi:g}])"
            )
            continue
        q_set = 0.0 if q_set is None else q_set
        p_available = {t: min(terms.scale * p_mult[t], terms.cap) for t in ctx.times}
        q_setpoint = {t: q_set * q_mult[t] for t in ctx.times}
        if limits_enforced:
            infeasible += _infeasibility(
                label, control, s_max, lo, hi, p_available, q_setpoint
            )
        injectors.append(
            Injector(
                name=cell_text(row["device_name"]),
                bus=ctx.bus_name_to_id[bus_name],
                phases=phases,
                s_max=s_max,
                q_min=lo,
                q_max=hi,
                control=control,
                cost=0.0 if cost is None else cost,
                p_min=terms.floor,
                p_available=p_available,
                q_setpoint=q_setpoint,
            )
        )

    finish_validation(table, errors, infeasible, [])
    return injectors


def _infeasibility(
    label: str,
    control: ControlVariable,
    s_max: float,
    lo: float,
    hi: float,
    p_available: dict[Any, float],
    q_setpoint: dict[Any, float],
) -> list[str]:
    """Reasons a device cannot meet its rating whatever the other devices do."""
    problems = []
    for t, p in p_available.items():
        if control == ControlVariable.Q:
            if p > s_max + TOL:
                problems.append(f"{label}: fixed p={p:g} at t={t} exceeds s_max={s_max:g}")
            elif q_range_empty(s_max, p, lo, hi):
                problems.append(
                    f"{label}: q limits [{lo:g}, {hi:g}] do not fit in the rating "
                    f"left by p={p:g} at t={t}"
                )
        elif control == ControlVariable.P:
            q = q_setpoint[t]
            if not lo - TOL <= q <= hi + TOL:
                problems.append(f"{label}: fixed q={q:g} at t={t} is outside [{lo:g}, {hi:g}]")
    return problems[:1]  # one example is enough


# --------------------------------------------------------------------------- #
# Model components
# --------------------------------------------------------------------------- #


def add_injector_components(
    model: Any, prefix: str, devices: list[Injector], times: tuple[Any, ...]
) -> None:
    """Sets, parameters and the p_{prefix}/q_{prefix} variables of one device type."""
    names = [d.name for d in devices]
    by_bus_phase: dict[tuple[int, str], list[str]] = {}
    for d in devices:
        for phase in d.phases:
            by_bus_phase.setdefault((d.bus, phase), []).append(d.name)

    def per_time(attr: str) -> dict[tuple[str, Any], float]:
        return {(d.name, t): getattr(d, attr)[t] for d in devices for t in times}

    def per_device(attr: str) -> dict[str, Any]:
        return {d.name: getattr(d, attr) for d in devices}

    def add(name: str, component: Any) -> None:
        model.add_component(f"{prefix}_{name}", component)

    device_set = pyo.Set(initialize=names)
    add("device_set", device_set)
    setattr(model, f"{prefix}_phases_by_device", {d.name: list(d.phases) for d in devices})
    setattr(model, f"{prefix}_n_phases", {d.name: len(d.phases) for d in devices})
    setattr(model, f"{prefix}_bus_by_device", per_device("bus"))
    setattr(model, f"{prefix}_devices_by_bus_phase", by_bus_phase)

    add("s_max", pyo.Param(device_set, initialize=per_device("s_max")))
    add("q_min", pyo.Param(device_set, initialize=per_device("q_min")))
    add("q_max", pyo.Param(device_set, initialize=per_device("q_max")))
    add("control", pyo.Param(device_set, initialize=per_device("control"), within=pyo.Any))
    add("cost", pyo.Param(device_set, initialize=per_device("cost")))
    add("p_min", pyo.Param(device_set, initialize=per_device("p_min")))
    add(
        "p_available",
        pyo.Param(device_set, model.time_set, initialize=per_time("p_available")),
    )
    add(
        "q_setpoint",
        pyo.Param(device_set, model.time_set, initialize=per_time("q_setpoint")),
    )
    model.add_component(
        f"p_{prefix}", pyo.Var(device_set, model.time_set, domain=pyo.NonNegativeReals)
    )
    model.add_component(f"q_{prefix}", pyo.Var(device_set, model.time_set, initialize=0))


def injection(model: Any, prefix: str, kind: str, bus: int, phase: str, time: Any) -> Any:
    """Per-phase injection at a bus: each device's total divided by its phase count."""
    var = getattr(model, f"{kind}_{prefix}")
    n_phases = getattr(model, f"{prefix}_n_phases")
    devices = getattr(model, f"{prefix}_devices_by_bus_phase").get((bus, phase), [])
    return sum(var[device, time] / n_phases[device] for device in devices)


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #


def _get(model: Any, prefix: str, name: str) -> Any:
    return getattr(model, f"{prefix}_{name}")


def add_constant_p_constraints(m: Any, prefix: str) -> None:
    """Fix P to the available power in control modes NONE and Q."""
    p, available, control = f"p_{prefix}", _get(m, prefix, "p_available"), _get(m, prefix, "control")

    def rule(m, device, time):
        if control[device] in _P_FIXED:
            return getattr(m, p)[device, time] == available[device, time]
        return pyo.Constraint.Skip

    m.add_component(
        f"{prefix}_constant_p",
        pyo.Constraint(_get(m, prefix, "device_set"), m.time_set, rule=rule),
    )


def add_constant_q_constraints(m: Any, prefix: str) -> None:
    """Fix Q to its setpoint in control modes NONE and P."""
    q, setpoint, control = f"q_{prefix}", _get(m, prefix, "q_setpoint"), _get(m, prefix, "control")

    def rule(m, device, time):
        if control[device] in _Q_FIXED:
            return getattr(m, q)[device, time] == setpoint[device, time]
        return pyo.Constraint.Skip

    m.add_component(
        f"{prefix}_constant_q",
        pyo.Constraint(_get(m, prefix, "device_set"), m.time_set, rule=rule),
    )


def add_limits(m: Any, prefix: str) -> None:
    """P and Q bounds of controlled devices (every mode except NONE).

    P is bounded by the available power and by the rating left after the fixed Q;
    Q by its limits, and in mode Q by the rating left after the fixed P.
    """
    p, q = f"p_{prefix}", f"q_{prefix}"
    available, setpoint = _get(m, prefix, "p_available"), _get(m, prefix, "q_setpoint")
    control, s_max = _get(m, prefix, "control"), _get(m, prefix, "s_max")
    q_min, q_max, p_min = _get(m, prefix, "q_min"), _get(m, prefix, "q_max"), _get(m, prefix, "p_min")

    def p_rule(m, device, time):
        if control[device] not in (ControlVariable.P, ControlVariable.PQ):
            return pyo.Constraint.Skip
        upper = min(pyo.value(available[device, time]), pyo.value(s_max[device]))
        if control[device] == ControlVariable.P:  # Q is fixed and uses part of the rating
            upper = min(
                upper,
                math.sqrt(max(0.0, pyo.value(s_max[device]) ** 2 - pyo.value(setpoint[device, time]) ** 2)),
            )
        return (min(pyo.value(p_min[device]), upper), getattr(m, p)[device, time], upper)

    def q_rule(m, device, time):
        if control[device] == ControlVariable.NONE:
            return pyo.Constraint.Skip
        lo, hi = pyo.value(q_min[device]), pyo.value(q_max[device])
        if control[device] == ControlVariable.Q:
            lo, hi = q_range_with_fixed_p(
                pyo.value(s_max[device]), pyo.value(available[device, time]), lo, hi
            )
        return (lo, getattr(m, q)[device, time], hi)

    devices = _get(m, prefix, "device_set")
    m.add_component(f"{prefix}_p_limits", pyo.Constraint(devices, m.time_set, rule=p_rule))
    m.add_component(f"{prefix}_q_limits", pyo.Constraint(devices, m.time_set, rule=q_rule))


def add_circle_constraints(m: Any, prefix: str) -> None:
    """Exact rating p^2 + q^2 <= s_max^2 of PQ-controlled devices."""
    p, q = f"p_{prefix}", f"q_{prefix}"
    control, s_max = _get(m, prefix, "control"), _get(m, prefix, "s_max")

    def rule(m, device, time):
        if control[device] != ControlVariable.PQ:
            return pyo.Constraint.Skip
        return getattr(m, p)[device, time] ** 2 + getattr(m, q)[device, time] ** 2 <= s_max[device] ** 2

    m.add_component(
        f"{prefix}_circle",
        pyo.Constraint(_get(m, prefix, "device_set"), m.time_set, rule=rule),
    )


def add_octagon_constraints(m: Any, prefix: str) -> None:
    """Inscribed-octagon rating of PQ-controlled devices (linear).

    Only the P >= 0 half is needed because p_{prefix} is nonnegative.
    """
    p, q = f"p_{prefix}", f"q_{prefix}"
    control, s_max = _get(m, prefix, "control"), _get(m, prefix, "s_max")

    def make_rule(a: float, b: float):
        def rule(m, device, time):
            if control[device] != ControlVariable.PQ:
                return pyo.Constraint.Skip
            return a * getattr(m, p)[device, time] + b * getattr(m, q)[device, time] <= s_max[device]

        return rule

    for i, (a, b) in enumerate(OCTAGON_HALF, start=1):
        m.add_component(
            f"{prefix}_octagon_{i}",
            pyo.Constraint(_get(m, prefix, "device_set"), m.time_set, rule=make_rule(a, b)),
        )


def add_injector_constraints(model: Any, prefix: str, config: Any) -> None:
    """All constraints of one injector type, following the model options.

    ``equality_only`` skips the limits. The rating is the exact circle when
    ``circular_constraints`` is true (default: for nonlinear models only) and
    the octagon otherwise.
    """
    if len(_get(model, prefix, "device_set")) == 0:
        return
    add_constant_p_constraints(model, prefix)
    add_constant_q_constraints(model, prefix)
    if cfg(config, "equality_only", False):
        return
    add_limits(model, prefix)
    if cfg(config, "circular_constraints", not cfg(config, "linear", True)):
        add_circle_constraints(model, prefix)
    else:
        add_octagon_constraints(model, prefix)


def device_table(case: Any, table: str) -> pd.DataFrame:
    """A case's device table (empty if the case has none)."""
    data = getattr(case, table, None)
    return pd.DataFrame() if data is None else data


def case_schedules(case: Any) -> pd.DataFrame:
    schedules = getattr(case, "schedules", None)
    return pd.DataFrame() if schedules is None else schedules


__all__ = [
    "Injector",
    "PowerTerms",
    "add_injector_components",
    "add_injector_constraints",
    "injection",
    "read_injectors",
]
