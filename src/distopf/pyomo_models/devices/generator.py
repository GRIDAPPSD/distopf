"""Generator device provider: dispatch, limits, and control policies.

CSV schema (case.gen_data)
--------------------------
Required columns:
    device_name     Unique generator name (unique within this file).
    bus_name        AC bus the generator connects to. Several generators may share a bus.
    phases          e.g. "abc", "a", "s1s2".
    s_{ph}_max      Apparent-power rating for each listed phase (> 0).
    p_{ph}          Nominal active power for each listed phase (>= 0), before gen_shape
                    scaling. Fixed output in modes NONE and Q; available (maximum)
                    output in modes P and PQ.

Optional columns:
    control_variable  "PQ", "P", "Q", or blank. Blank means NONE: P and Q are fixed
                      and no rating limits are applied.
    q_{ph}            Reactive-power setpoint. Required in modes NONE and P; ignored
                      in modes Q and PQ (a warning is issued if nonzero).
    q_{ph}_min/_max   Reactive-power limits. Default ±s_max; clamped to ±s_max.
    gen_shape         Column of case.schedules that multiplies p_{ph} at each time step.
                      If case.schedules is empty, P is constant. Otherwise, a named
                      shape must exist, and blank means "PV" if that column exists,
                      else constant P.
    cost              Cost per unit of active power, default 0.
    s_base            Accepted for compatibility; not used by this provider.

Conventions:
    Positive p/q is injection into the AC bus. Generators cannot absorb P.
    Values use the same units as the network.

Read the file with dtype={"device_name": str, "bus_name": str}.

Example:
device_name,bus_name,phases,control_variable,gen_shape,cost,p_a,p_b,p_c,q_a,q_b,q_c,s_a_max,s_b_max,s_c_max,q_a_min,q_b_min,q_c_min,q_a_max,q_b_max,q_c_max
pv_1,1,abc,Q,PV,0,0.01,0.01,0.01,0,0,0,0.012,0.012,0.012,-100,-100,-100,100,100,100
"""

from __future__ import annotations

import math
from typing import Any, Mapping
import warnings
import pandas as pd
import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models.common.registry import DeviceProvider
from distopf.pyomo_models.common.data import parse_phases
from distopf.pyomo_models.common.model_types import ControlVariable
from distopf.pyomo_models.common.device_data import (
    TOL,
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
    preview,
    row_label,
    cell_blank,
    DeviceDataWarning,
)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

_TABLE = "gen_data"
_PHASES = ("a", "b", "c", "s1", "s2")
_DEFAULT_SHAPE = "PV"
_REQUIRED = {"device_name", "bus_name", "phases"}
_LEGACY = {"id", "name", "bus_id"}
_KNOWN = (
    _REQUIRED
    | {"control_variable", "gen_shape", "cost", "s_base"}
    | {f"{k}_{p}" for k in ("p", "q") for p in _PHASES}
    | {f"s_{p}_max" for p in _PHASES}
    | {f"q_{p}_{b}" for p in _PHASES for b in ("min", "max")}
)
_P_FIXED = (ControlVariable.NONE, ControlVariable.Q)
_Q_FIXED = (ControlVariable.NONE, ControlVariable.P)

# Right half (P >= 0) of an octagon inscribed in the rating circle:
# coefficients (a, b) of a*p + b*q <= s_max.
_C = math.sqrt(2) - 1
_OCTAGON_HALF = ((_C, 1.0), (1.0, _C), (1.0, -_C), (_C, -1.0))


def _cfg(config: Any, key: str, default: Any) -> Any:
    return getattr(config, key, default) if config is not None else default


def _legacy_bus_name(value: Any) -> str:
    """Normalize an old-schema bus name; float-parsed names like 151.0 become '151'."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return cell_text(value)


def translate_legacy_gen_data(
    data: pd.DataFrame, bus_id_to_name: Mapping[int, str]
) -> pd.DataFrame:
    """Convert the old gen_data schema (id/name = bus id/name) to device_name/bus_name.

    Returns `data` unchanged if it is empty or already uses the new schema.
    The old provider connected generators by bus `id`, so `id` takes precedence
    over `name` when both are present.
    """
    cols = set(data.columns)
    if data.empty or "device_name" in cols or not cols & {"id", "name"}:
        return data
    if "bus_name" in cols:
        raise ValueError(
            f"{_TABLE}: mixes old columns {sorted(cols & {'id', 'name'})} with "
            "'bus_name'; use either the old or the new schema"
        )

    errors: list[str] = []
    mismatches: list[str] = []
    bus_names: list[str] = []
    for idx, row in data.iterrows():
        bus_name = ""
        if "id" in cols and not cell_blank(row, "id"):
            try:
                bus_id = int(float(row["id"]))
            except (TypeError, ValueError):
                errors.append(f"row {idx}: id={row['id']!r} is not an integer bus id")
            else:
                if bus_id not in bus_id_to_name:
                    errors.append(f"row {idx}: bus id {bus_id} not found in network")
                else:
                    bus_name = bus_id_to_name[bus_id]
                    if "name" in cols and not cell_blank(row, "name"):
                        given = _legacy_bus_name(row["name"])
                        if given != bus_name:
                            mismatches.append(
                                f"row {idx}: id={bus_id} is bus {bus_name!r} but "
                                f"name={given!r}; using id"
                            )
        elif "name" in cols and not cell_blank(row, "name"):
            bus_name = _legacy_bus_name(row["name"])
        else:
            errors.append(f"row {idx}: needs a bus 'id' or 'name'")
        bus_names.append(bus_name)

    if errors:
        raise ValueError(
            f"Cannot translate legacy {_TABLE}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
    for message in mismatches:
        warnings.warn(f"{_TABLE}: {message}", DeviceDataWarning, stacklevel=2)
    warnings.warn(
        f"{_TABLE} uses the legacy 'id'/'name' schema; translated to "
        "'device_name'/'bus_name'. Update the file to the new schema.",
        FutureWarning,
        stacklevel=2,
    )

    out = data.drop(columns=[c for c in ("id", "name", "primary_phase") if c in cols])
    out.insert(0, "device_name", [f"gen_{b}" for b in bus_names])
    out.insert(1, "bus_name", bus_names)
    return out


# --------------------------------------------------------------------------- #
# Shared math (used by validation and constraints so they cannot drift)
# --------------------------------------------------------------------------- #


def _q_bounds(
    s_max: float, q_min: float | None, q_max: float | None
) -> tuple[float, float]:
    """Static reactive limits, clamped to ±s_max."""
    lo = -s_max if q_min is None else max(-s_max, q_min)
    hi = s_max if q_max is None else min(s_max, q_max)
    return lo, hi


def _q_mode_range(s_max: float, p: float, lo: float, hi: float) -> tuple[float, float]:
    """Reactive range in Q mode, where fixed P uses part of the rating."""
    cap = math.sqrt(max(0.0, s_max**2 - p**2))
    return max(-cap, lo), min(cap, hi)


# --------------------------------------------------------------------------- #
# Generation shapes
# --------------------------------------------------------------------------- #


def _has_schedules(schedules: pd.DataFrame) -> bool:
    return not schedules.empty and any(column != "time" for column in schedules.columns)


def _resolve_shape(row: pd.Series, schedules: pd.DataFrame) -> str | None:
    """Schedule column that scales this generator's P, or None for constant P."""
    if not _has_schedules(schedules):
        return None
    shape = cell_text(row.get("gen_shape"))
    if shape:
        if shape not in schedules.columns:
            raise ValueError(f"gen_shape {shape!r} is not a column of case.schedules")
        return shape
    return _DEFAULT_SHAPE if _DEFAULT_SHAPE in schedules.columns else None


def _shape_multipliers(
    schedules: pd.DataFrame, shape: str | None, times: tuple[Any, ...]
) -> dict[Any, float]:
    if shape is None:
        return {t: 1.0 for t in times}
    missing = [t for t in times if t not in schedules.index]
    if missing:
        raise ValueError(
            f"schedule {shape!r} has no rows for time steps {preview(missing)}"
        )
    values = pd.to_numeric(
        schedules.loc[list(times), shape], errors="coerce"
    ).to_numpy()
    bad = [t for t, v in zip(times, values) if not math.isfinite(v)]
    if bad:
        raise ValueError(
            f"schedule {shape!r} has blank or non-numeric values at time steps {preview(bad)}"
        )
    negative = [t for t, v in zip(times, values) if v < 0]
    if negative:
        raise ValueError(
            f"schedule {shape!r} has negative values at time steps {preview(negative)}"
        )
    return {t: float(v) for t, v in zip(times, values)}


class _ShapeCache:
    """Resolve each schedule column once, caching results and errors."""

    def __init__(self, schedules: pd.DataFrame, times: tuple[Any, ...]) -> None:
        self._schedules = schedules
        self._times = times
        self._cache: dict[str | None, dict[Any, float] | ValueError] = {}

    def get(self, shape: str | None) -> dict[Any, float]:
        if shape not in self._cache:
            try:
                self._cache[shape] = _shape_multipliers(
                    self._schedules, shape, self._times
                )
            except ValueError as exc:
                self._cache[shape] = exc
        result = self._cache[shape]
        if isinstance(result, ValueError):
            raise result
        return result


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_gen_data(
    data: pd.DataFrame,
    ctx: NetworkContext,
    schedules: pd.DataFrame,
    limits_enforced: bool = True,
) -> None:
    """Validate the generator table.

    Raises:
        ValueError: malformed data (all row-level problems are listed together).
        InfeasibleCaseError: well-formed data that is provably infeasible.
    """
    if data.empty:
        return
    check_columns(
        data,
        _TABLE,
        _REQUIRED,
        _KNOWN,
        _LEGACY,
        "generators are identified by 'device_name' and buses by 'bus_name'",
    )

    errors: list[str] = []
    infeasible: list[str] = []
    warns: list[str] = []
    check_device_names(data, errors)

    shapes = _ShapeCache(schedules, ctx.times)
    shape_errors: dict[str, list[str]] = {}  # message -> devices using that shape

    for idx, row in data.iterrows():
        label = row_label(row, idx)
        start = len(errors)

        phases = check_phases(label, row["phases"], _PHASES, errors)
        if phases is None:
            continue
        check_connection(label, cell_text(row["bus_name"]), phases, ctx, errors)

        ctrl: ControlVariable | None
        try:
            ctrl = cell_control(row)
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            ctrl = None
        cell_opt_float(row, "cost", label, errors)

        mult: dict[Any, float] | None = None
        try:
            mult = shapes.get(_resolve_shape(row, schedules))
        except ValueError as exc:
            shape_errors.setdefault(str(exc), []).append(label)

        # per-phase data checks
        ports: list[tuple[str, float, float, float | None, float, float]] = []
        for ph in phases:
            s_max = cell_required_float(row, f"s_{ph}_max", label, errors, gt=0.0)
            p_nom = cell_required_float(row, f"p_{ph}", label, errors, ge=0.0)
            q_set = cell_opt_float(row, f"q_{ph}", label, errors)
            q_lo = cell_opt_float(row, f"q_{ph}_min", label, errors)
            q_hi = cell_opt_float(row, f"q_{ph}_max", label, errors)
            if s_max is None or p_nom is None:
                continue

            lo, hi = _q_bounds(s_max, q_lo, q_hi)
            if lo > hi + TOL:
                errors.append(
                    f"{label}: q_{ph} limits are empty after clamping to "
                    f"±s_{ph}_max ([{lo:g}, {hi:g}])"
                )
            if ctrl in _Q_FIXED and q_set is None:
                errors.append(
                    f"{label}: q_{ph} is required in control mode {ctrl.name}"
                )
            elif ctrl is not None and ctrl not in _Q_FIXED and q_set not in (None, 0.0):
                warns.append(
                    f"{label}: q_{ph}={q_set:g} is ignored in control mode {ctrl.name}"
                )
            ports.append((ph, s_max, p_nom, q_set, lo, hi))

        # feasibility checks: only on rows with clean data
        if len(errors) != start or ctrl is None or mult is None:
            continue
        # The worst case is always the time step with the largest P multiplier.
        t_peak = max(mult, key=mult.get)
        for ph, s_max, p_nom, q_set, lo, hi in ports:
            p_peak = p_nom * mult[t_peak]
            if ctrl == ControlVariable.Q and limits_enforced:
                if p_peak > s_max + TOL:
                    infeasible.append(
                        f"{label}: fixed p_{ph}={p_peak:g} at t={t_peak} "
                        f"exceeds s_{ph}_max={s_max:g}"
                    )
                else:
                    q_lo_t, q_hi_t = _q_mode_range(s_max, p_peak, lo, hi)
                    if q_lo_t > q_hi_t + TOL:
                        infeasible.append(
                            f"{label}: q_{ph} limits [{lo:g}, {hi:g}] do not fit in the "
                            f"rating left by p_{ph}={p_peak:g} at t={t_peak}"
                        )
            elif ctrl == ControlVariable.P and limits_enforced:
                if not lo - TOL <= q_set <= hi + TOL:
                    infeasible.append(
                        f"{label}: fixed q_{ph}={q_set:g} is outside [{lo:g}, {hi:g}]"
                    )
            elif ctrl == ControlVariable.NONE:
                if p_peak**2 + q_set**2 > s_max**2 + TOL:
                    warns.append(
                        f"{label}: fixed (p_{ph}, q_{ph}) exceeds s_{ph}_max at "
                        f"t={t_peak}; ratings are not enforced in mode NONE"
                    )
            # PQ mode is always feasible: p = 0 with q in [lo, hi].

    for message, labels in shape_errors.items():
        errors.append(f"{message} (devices: {preview(labels)})")
    finish_validation(_TABLE, errors, infeasible, warns)


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #


class GeneratorProvider(DeviceProvider):
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
                q_nom = cell_num(row, f"q_{ph}", 0.0)  # blank only where ignored
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

        # Consecutive phase pairs among a, b, c (for phase-lock constraints elsewhere)
        phase_pairs: list[tuple[str, str, str]] = []
        for device, phases in phases_by_device.items():
            abc = [ph for ph in ("a", "b", "c") if ph in phases]
            phase_pairs.extend(
                (device, left, right) for left, right in zip(abc, abc[1:])
            )

        # Sets
        model.gen_device_set = pyo.Set(initialize=devices)
        model.gen_device_phase_set = pyo.Set(initialize=ports, dimen=2)
        model.gen_phase_pair_set = pyo.Set(initialize=phase_pairs, dimen=3)

        # Lookup maps (plain dicts)
        model.gen_phases_by_device = phases_by_device
        model.gen_bus_by_device = bus_by_device
        model.gen_devices_by_bus_phase = devices_by_bus_phase

        # Parameters
        ports_set, time_set = model.gen_device_phase_set, model.time_set
        model.gen_s_max = pyo.Param(ports_set, initialize=s_max)
        model.gen_q_min = pyo.Param(ports_set, initialize=q_min)  # clamped to ±s_max
        model.gen_q_max = pyo.Param(ports_set, initialize=q_max)
        model.gen_control = pyo.Param(ports_set, initialize=control, within=pyo.Any)
        model.gen_cost = pyo.Param(ports_set, initialize=cost)
        model.gen_p_available = pyo.Param(ports_set, time_set, initialize=p_available)
        model.gen_q_setpoint = pyo.Param(ports_set, time_set, initialize=q_setpoint)
        model.gen_phase_lock = pyo.Param(
            model.gen_device_set,
            initialize={device: False for device in devices},
            within=pyo.Boolean,
            mutable=True,
        )

        # Variables
        model.p_gen = pyo.Var(ports_set, time_set, domain=pyo.NonNegativeReals)
        model.q_gen = pyo.Var(ports_set, time_set, initialize=0)

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


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #


def add_gen_constant_p_constraints(m: Any) -> None:
    """Fix P to available power for generators in control modes NONE and Q."""

    def rule(m, device, phase, time):
        if m.gen_control[device, phase] in _P_FIXED:
            return (
                m.p_gen[device, phase, time] == m.gen_p_available[device, phase, time]
            )
        return pyo.Constraint.Skip

    m.gen_constant_p = pyo.Constraint(m.gen_device_phase_set, m.time_set, rule=rule)


def add_gen_constant_q_constraints(m: Any) -> None:
    """Fix Q to its setpoint for generators in control modes NONE and P."""

    def rule(m, device, phase, time):
        if m.gen_control[device, phase] in _Q_FIXED:
            return m.q_gen[device, phase, time] == m.gen_q_setpoint[device, phase, time]
        return pyo.Constraint.Skip

    m.gen_constant_q = pyo.Constraint(m.gen_device_phase_set, m.time_set, rule=rule)


def add_gen_limits(m: Any) -> None:
    """P and Q bounds for every controlled generator (all modes except NONE)."""

    def p_rule(m, device, phase, time):
        if m.gen_control[device, phase] == ControlVariable.NONE:
            return pyo.Constraint.Skip
        upper = min(
            pyo.value(m.gen_p_available[device, phase, time]),
            pyo.value(m.gen_s_max[device, phase]),
        )
        return (0, m.p_gen[device, phase, time], upper)

    def q_rule(m, device, phase, time):
        ctrl = m.gen_control[device, phase]
        if ctrl == ControlVariable.NONE:
            return pyo.Constraint.Skip
        lo = pyo.value(m.gen_q_min[device, phase])
        hi = pyo.value(m.gen_q_max[device, phase])
        if ctrl == ControlVariable.Q:
            lo, hi = _q_mode_range(
                pyo.value(m.gen_s_max[device, phase]),
                pyo.value(m.gen_p_available[device, phase, time]),
                lo,
                hi,
            )
        return (lo, m.q_gen[device, phase, time], hi)

    m.gen_p_limits = pyo.Constraint(m.gen_device_phase_set, m.time_set, rule=p_rule)
    m.gen_q_limits = pyo.Constraint(m.gen_device_phase_set, m.time_set, rule=q_rule)


def add_gen_circle_constraints(m: Any) -> None:
    """Exact apparent-power limit p² + q² <= s_max² for PQ-controlled generators."""

    def rule(m, device, phase, time):
        if m.gen_control[device, phase] != ControlVariable.PQ:
            return pyo.Constraint.Skip
        return (
            m.p_gen[device, phase, time] ** 2 + m.q_gen[device, phase, time] ** 2
            <= m.gen_s_max[device, phase] ** 2
        )

    m.gen_circle = pyo.Constraint(m.gen_device_phase_set, m.time_set, rule=rule)


def add_gen_octagon_constraints(m: Any) -> None:
    """Inscribed-octagon approximation of the rating for PQ-controlled generators.

    Only the P >= 0 half is needed because p_gen is nonnegative.
    """

    def make_rule(a: float, b: float):
        def rule(m, device, phase, time):
            if m.gen_control[device, phase] != ControlVariable.PQ:
                return pyo.Constraint.Skip
            return (
                a * m.p_gen[device, phase, time] + b * m.q_gen[device, phase, time]
                <= m.gen_s_max[device, phase]
            )

        return rule

    for i, (a, b) in enumerate(_OCTAGON_HALF, start=1):
        setattr(
            m,
            f"gen_octagon_{i}",
            pyo.Constraint(m.gen_device_phase_set, m.time_set, rule=make_rule(a, b)),
        )


__all__ = ["GeneratorProvider", "validate_gen_data"]
