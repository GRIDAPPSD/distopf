"""MPSSD model-component provider for the LinDistFlow formulation.

CSV schema
----------
Required columns:
    device_name     Unique, human-readable device name (unique within this file).
    bus_name        Name of the AC bus the device connects to. Must exist in the network.
    phases          Phases the device connects to, e.g. "abc", "ab", "c".
    dc_bus          Integer label. All ports sharing a label exchange power over a
                    common lossless DC link: the sum of P over those ports is 0.
    s_{a,b,c}_max   Apparent-power rating for each listed phase (> 0).

Optional columns:
    control_variable        "PQ", "P", "Q", or blank. Blank means NONE (P and Q fixed).
    p_{a,b,c}               Active-power setpoints. Required in modes NONE and Q,
                            ignored (with a warning) otherwise.
    q_{a,b,c}               Reactive-power setpoints. Required in modes NONE and P,
                            ignored (with a warning) otherwise.
    q_{a,b,c}_min/_max      Reactive-power limits. Default ±s_max; clamped to ±s_max.
    balanced_phases         True/False, default False. Forces equal P and equal Q
                            on all phases of the device.

Conventions:
    Positive p/q is injection into the AC bus.
    Values use the same units as the network.

Read the file with dtype={"device_name": str, "bus_name": str} so numeric-looking
names (e.g. bus "151") are not converted to floats.

Example:
device_name,bus_name,phases,dc_bus,control_variable,p_a,p_b,p_c,q_a,q_b,q_c,s_a_max,s_b_max,s_c_max
mpssd_p1,151,abc,1,PQ,,,,,,,0.4,0.4,0.4
mpssd_p2,300,abc,1,PQ,,,,,,,0.4,0.4,0.4
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import pandas as pd
import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models.common.registry import DeviceProvider
from distopf.pyomo_models.common.data import injectable_bus_phases, parse_phases
from distopf.pyomo_models.common.model_types import (
    ControlVariable,
    CONTROL_VARIABLE_MAP,
)
from distopf.utils.ngon import ngon_line_equations


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

_PHASES = "abc"
_REQUIRED = {"device_name", "bus_name", "phases", "dc_bus"}
_LEGACY = {"id", "name", "bus_id"}
_KNOWN = (
    _REQUIRED
    | {"control_variable", "balanced_phases"}
    | {f"{k}_{p}" for k in ("p", "q") for p in _PHASES}
    | {f"s_{p}_max" for p in _PHASES}
    | {f"q_{p}_{b}" for p in _PHASES for b in ("min", "max")}
)
_TOL = 1e-9


class InfeasibleCaseError(ValueError):
    """The input data is well-formed but provably infeasible."""


# --------------------------------------------------------------------------- #
# Cell parsing helpers
# --------------------------------------------------------------------------- #


def _text(value: Any) -> str:
    """Return a stripped string; missing or blank cells give ''."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _num(row: pd.Series, key: str, default: float) -> float:
    """Return row[key] as float, treating missing columns and blank cells as default."""
    value = row.get(key)
    return default if value is None or pd.isna(value) else float(value)


def _opt_float(row: pd.Series, key: str, label: str, errors: list[str]) -> float | None:
    """Parse an optional numeric cell. Blank gives None; junk records an error."""
    value = row.get(key)
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label}: {key}={value!r} is not a number")
        return None
    if not math.isfinite(number):
        errors.append(f"{label}: {key} must be finite")
        return None
    return number


def _flag(row: pd.Series, key: str, default: bool = False) -> bool:
    """Parse a True/False CSV cell. Missing columns and blank cells give default."""
    value = row.get(key)
    if value is None or pd.isna(value):
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "":
            return default
        if text in ("true", "1"):
            return True
        if text in ("false", "0"):
            return False
        raise ValueError(f"Cannot interpret {key}={value!r} as True/False")
    return bool(value)


def _control(row: pd.Series) -> ControlVariable:
    """Parse control_variable. Missing or blank means ControlVariable.NONE."""
    raw = row.get("control_variable")
    text = _text(raw).upper()
    if text == "":
        return ControlVariable.NONE
    try:
        return CONTROL_VARIABLE_MAP[text]
    except KeyError:
        raise ValueError(
            f"Unknown control_variable {raw!r}; expected one of "
            f"{sorted(k for k in CONTROL_VARIABLE_MAP if k)} or blank"
        ) from None


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def validate_mpssd_data(
    data: pd.DataFrame,
    bus_name_to_id_map: Any,
    injectable: set[tuple[int, str]],
    swing_buses: set[int],
) -> None:
    """Validate the MPSSD table.

    Raises:
        ValueError: malformed data (all row-level problems are listed together).
        InfeasibleCaseError: well-formed data that is provably infeasible.
    """
    if data.empty:
        return

    # ---- structural checks: fail immediately ----
    cols = set(data.columns)
    missing = _REQUIRED - cols
    if missing:
        raise ValueError(f"mpssd data missing columns: {sorted(missing)}")
    legacy = _LEGACY & cols
    if legacy:
        raise ValueError(
            f"mpssd data has removed columns {sorted(legacy)}; devices are "
            "identified by 'device_name' and buses by 'bus_name'"
        )
    unknown = cols - _KNOWN
    if unknown:
        raise ValueError(
            f"mpssd data has unrecognized columns (typo?): {sorted(unknown)}"
        )

    errors: list[str] = []
    warns: list[str] = []

    # ---- device names ----
    names = data["device_name"].map(_text)
    if (names == "").any():
        errors.append("blank device_name values")
    dupes = sorted(set(names[names.duplicated() & (names != "")]))
    if dupes:
        errors.append(f"duplicate device_name values: {dupes}")

    # dc_bus -> list of (label, p_lo, p_hi, n_ports)
    dc_intervals: dict[int, list[tuple[str, float, float, int]]] = {}

    for idx, row in data.iterrows():
        name = _text(row["device_name"])
        label = f"device {name!r}" if name else f"row {idx}"
        errors_before = len(errors)

        # bus
        bus_name = _text(row["bus_name"])
        if bus_name not in bus_name_to_id_map:
            errors.append(f"{label}: bus_name {bus_name!r} not found in network")

        # phases
        try:
            phases = list(parse_phases(str(row["phases"])))
        except Exception as exc:
            errors.append(f"{label}: cannot parse phases {row['phases']!r} ({exc})")
            continue
        if (
            not phases
            or len(set(phases)) != len(phases)
            or not set(phases) <= set(_PHASES)
        ):
            errors.append(
                f"{label}: MPSSD supports phases a, b, c only; "
                f"phases {row['phases']!r} must be unique letters from 'abc'"
            )
            continue

        if bus_name in bus_name_to_id_map:
            bus_id = bus_name_to_id_map[bus_name]
            if bus_id in swing_buses:
                errors.append(
                    f"{label}: bus {bus_name!r} is a swing/boundary bus; "
                    "device injections there are not modeled"
                )
            else:
                absent = [
                    phase for phase in phases if (bus_id, phase) not in injectable
                ]
                if absent:
                    errors.append(
                        f"{label}: phases {absent} have no incoming branch at bus "
                        f"{bus_name!r}; power on those ports would be unbalanced"
                    )

        # dc_bus
        dc_bus: int | None = None
        dc_value = _opt_float(row, "dc_bus", label, errors)
        if dc_value is None:
            if pd.isna(row.get("dc_bus")):
                errors.append(f"{label}: dc_bus is required")
        elif dc_value != round(dc_value):
            errors.append(f"{label}: dc_bus={dc_value} must be a whole number")
        else:
            dc_bus = int(dc_value)

        # control mode and phase-balance flag
        ctrl: ControlVariable | None
        try:
            ctrl = _control(row)
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            ctrl = None
        try:
            balanced = _flag(row, "balanced_phases", default=False)
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            balanced = False

        p_fixed = ctrl in (ControlVariable.NONE, ControlVariable.Q)
        q_fixed = ctrl in (ControlVariable.NONE, ControlVariable.P)

        s_list: list[float] = []
        p_vals: list[float] = []
        q_vals: list[float] = []

        # per-phase checks
        for phase in phases:
            s_max = _opt_float(row, f"s_{phase}_max", label, errors)
            if s_max is None:
                if pd.isna(row.get(f"s_{phase}_max")):
                    errors.append(f"{label}: s_{phase}_max is required")
            elif s_max <= 0:
                errors.append(f"{label}: s_{phase}_max must be > 0")
                s_max = None
            if s_max is not None:
                s_list.append(s_max)

            p_set = _opt_float(row, f"p_{phase}", label, errors)
            q_set = _opt_float(row, f"q_{phase}", label, errors)
            q_lo = _opt_float(row, f"q_{phase}_min", label, errors)
            q_hi = _opt_float(row, f"q_{phase}_max", label, errors)

            # effective Q bounds, same clamping as create_components
            lo = hi = None
            if s_max is not None:
                lo = max(-s_max, q_lo if q_lo is not None else -s_max)
                hi = min(s_max, q_hi if q_hi is not None else s_max)
                if lo > hi:
                    errors.append(
                        f"{label}: q_{phase} bounds are empty (min {lo} > max {hi})"
                    )

            if ctrl is None:
                continue

            # setpoints must be present and in range where they are used
            if p_fixed:
                if p_set is None:
                    errors.append(
                        f"{label}: p_{phase} is required in control mode {ctrl.name}"
                    )
                else:
                    p_vals.append(p_set)
                    if s_max is not None and abs(p_set) > s_max + _TOL:
                        errors.append(
                            f"{label}: p_{phase}={p_set} exceeds s_{phase}_max={s_max}"
                        )
            elif p_set is not None:
                warns.append(
                    f"{label}: p_{phase} is ignored in control mode {ctrl.name}"
                )

            if q_fixed:
                if q_set is None:
                    errors.append(
                        f"{label}: q_{phase} is required in control mode {ctrl.name}"
                    )
                else:
                    q_vals.append(q_set)
                    if (
                        lo is not None
                        and hi is not None
                        and not (lo - _TOL <= q_set <= hi + _TOL)
                    ):
                        errors.append(
                            f"{label}: q_{phase}={q_set} is outside [{lo}, {hi}]"
                        )
            elif q_set is not None:
                warns.append(
                    f"{label}: q_{phase} is ignored in control mode {ctrl.name}"
                )

            # fixed P and Q together must fit inside the rating circle
            if (
                p_fixed
                and q_fixed
                and s_max is not None
                and p_set is not None
                and q_set is not None
                and p_set**2 + q_set**2 > s_max**2 + _TOL
            ):
                errors.append(
                    f"{label}: fixed (p_{phase}, q_{phase}) = ({p_set}, {q_set}) "
                    f"exceeds s_{phase}_max={s_max}"
                )

        # phase balance vs. fixed setpoints
        if balanced and len(phases) > 1:
            for kind, vals in (("p", p_vals), ("q", q_vals)):
                if vals and max(vals) - min(vals) > _TOL:
                    errors.append(
                        f"{label}: balanced_phases=True but fixed {kind} setpoints "
                        "differ between phases"
                    )

        # achievable P range for the DC-bus check; only trust clean rows
        if dc_bus is not None and ctrl is not None and len(errors) == errors_before:
            n = len(phases)
            if p_fixed:
                p_lo = p_hi = sum(p_vals)
            elif balanced:
                p_lo, p_hi = -n * min(s_list), n * min(s_list)
            else:
                p_lo, p_hi = -sum(s_list), sum(s_list)
            dc_intervals.setdefault(dc_bus, []).append((label, p_lo, p_hi, n))

    # ---- stage 1: data errors ----
    for message in warns:
        warnings.warn(f"mpssd: {message}", stacklevel=2)
    if errors:
        raise ValueError(
            "Invalid mpssd data:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # ---- stage 2: provable infeasibility (data is well-formed here) ----
    infeasible: list[str] = []
    for dc_bus, items in sorted(dc_intervals.items()):
        lo = sum(item[1] for item in items)
        hi = sum(item[2] for item in items)
        if lo > _TOL or hi < -_TOL:
            who = ", ".join(item[0] for item in items)
            infeasible.append(
                f"dc_bus {dc_bus}: total P must be 0 but can only range over "
                f"[{lo:g}, {hi:g}] (devices: {who})"
            )
        if sum(item[3] for item in items) == 1:
            warnings.warn(
                f"mpssd: dc_bus {dc_bus} has only one port; its P will be forced to 0",
                stacklevel=2,
            )
    if infeasible:
        raise InfeasibleCaseError(
            "mpssd case is infeasible:\n" + "\n".join(f"  - {m}" for m in infeasible)
        )


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #


class MpssdProvider(DeviceProvider):
    """Own MPSSD sets, parameters, variables, injections, and constraints."""

    name = "mpssd"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        for attr in (
            "bus_name_to_id_map",
            "branch_phase_set",
            "swing_bus_set",
            "time_set",
        ):
            if not hasattr(model, attr):
                raise RuntimeError(
                    f"MpssdProvider needs model.{attr}; the network provider "
                    "must create components first"
                )
        data = getattr(case, "mpssd_data", None)
        if data is None:
            data = pd.DataFrame()
        validate_mpssd_data(
            data,
            model.bus_name_to_id_map,
            injectable_bus_phases(model),
            set(model.swing_bus_set),
        )
        times = list(model.time_set)

        devices: list[str] = []
        ports: list[tuple[str, str]] = []
        phases_by_device: dict[str, list[str]] = {}
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
            device = _text(row["device_name"])
            bus = int(model.bus_name_to_id_map[_text(row["bus_name"])])
            dc_bus = int(float(row["dc_bus"]))
            phases = list(parse_phases(str(row["phases"])))
            ctrl = _control(row)

            devices.append(device)
            phases_by_device[device] = phases
            phase_balanced[device] = _flag(row, "balanced_phases", default=False)

            for phase in phases:
                port = (device, phase)
                rating = float(row[f"s_{phase}_max"])
                p_nom = _num(row, f"p_{phase}", 0.0)  # blank only where ignored
                q_nom = _num(row, f"q_{phase}", 0.0)

                ports.append(port)
                devices_by_bus_phase.setdefault((bus, phase), []).append(device)
                ports_by_dc_bus.setdefault(dc_bus, []).append(port)

                s_max[port] = rating
                q_min[port] = max(-rating, _num(row, f"q_{phase}_min", -rating))
                q_max[port] = min(rating, _num(row, f"q_{phase}_max", rating))
                control[port] = ctrl

                # Constant for now. Replace with a schedule lookup later.
                for t in times:
                    p_setpoint[(*port, t)] = p_nom
                    q_setpoint[(*port, t)] = q_nom

        # Sets
        model.mpssd_device_set = pyo.Set(initialize=devices)
        model.mpssd_device_phase_set = pyo.Set(initialize=ports, dimen=2)
        model.mpssd_dc_bus_set = pyo.Set(initialize=sorted(ports_by_dc_bus))

        # Lookup maps (plain dicts)
        model.mpssd_phases_by_device = phases_by_device
        model.mpssd_devices_by_bus_phase = devices_by_bus_phase
        model.mpssd_ports_by_dc_bus = ports_by_dc_bus

        # Parameters
        ports_set, time_set = model.mpssd_device_phase_set, model.time_set
        model.mpssd_s_max = pyo.Param(ports_set, initialize=s_max)
        model.mpssd_q_min = pyo.Param(ports_set, initialize=q_min)  # clamped to ±s_max
        model.mpssd_q_max = pyo.Param(ports_set, initialize=q_max)
        model.mpssd_control = pyo.Param(ports_set, initialize=control, within=pyo.Any)
        model.mpssd_p_setpoint = pyo.Param(ports_set, time_set, initialize=p_setpoint)
        model.mpssd_q_setpoint = pyo.Param(ports_set, time_set, initialize=q_setpoint)
        model.mpssd_is_phase_balanced = pyo.Param(
            model.mpssd_device_set, initialize=phase_balanced, within=pyo.Boolean
        )

        # Variables
        model.p_mpssd = pyo.Var(ports_set, time_set, initialize=0)
        model.q_mpssd = pyo.Var(ports_set, time_set, initialize=0)

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
        """Add device equalities and, unless equality_only is set, operating limits."""
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


# --------------------------------------------------------------------------- #
# Constraints
# --------------------------------------------------------------------------- #


def add_mpssd_constant_p_constraints_q_control(m: Any) -> None:
    """Fix P to its setpoint for ports in control modes NONE and Q."""

    def rule(m, device, phase, time):
        if m.mpssd_control[device, phase] in (ControlVariable.NONE, ControlVariable.Q):
            return (
                m.p_mpssd[device, phase, time]
                == m.mpssd_p_setpoint[device, phase, time]
            )
        return pyo.Constraint.Skip

    m.mpssd_constant_p = pyo.Constraint(m.mpssd_device_phase_set, m.time_set, rule=rule)


def add_mpssd_constant_q_constraints_p_control(m: Any) -> None:
    """Fix Q to its setpoint for ports in control modes NONE and P."""

    def rule(m, device, phase, time):
        if m.mpssd_control[device, phase] in (ControlVariable.NONE, ControlVariable.P):
            return (
                m.q_mpssd[device, phase, time]
                == m.mpssd_q_setpoint[device, phase, time]
            )
        return pyo.Constraint.Skip

    m.mpssd_constant_q = pyo.Constraint(m.mpssd_device_phase_set, m.time_set, rule=rule)


def add_mpssd_limits(m: Any) -> None:
    """Rectangular P/Q operating limits for every port."""

    def p_bounds(m, device, phase, time):
        rating = m.mpssd_s_max[device, phase]
        return (-rating, m.p_mpssd[device, phase, time], rating)

    def q_bounds(m, device, phase, time):
        return (
            m.mpssd_q_min[device, phase],
            m.q_mpssd[device, phase, time],
            m.mpssd_q_max[device, phase],
        )

    m.mpssd_p_limits = pyo.Constraint(
        m.mpssd_device_phase_set, m.time_set, rule=p_bounds
    )
    m.mpssd_q_limits = pyo.Constraint(
        m.mpssd_device_phase_set, m.time_set, rule=q_bounds
    )


def add_circular_mpssd_constraints(m: Any) -> None:
    """Exact apparent-power limit for PQ-controlled ports."""

    def rule(m, device, phase, time):
        if m.mpssd_control[device, phase] != ControlVariable.PQ:
            return pyo.Constraint.Skip
        return (
            m.p_mpssd[device, phase, time] ** 2 + m.q_mpssd[device, phase, time] ** 2
            <= m.mpssd_s_max[device, phase] ** 2
        )

    m.mpssd_circle = pyo.Constraint(m.mpssd_device_phase_set, m.time_set, rule=rule)


def add_ngon_constraints(m: Any, n: int = 8) -> None:
    """Linear n-gon approximation of the apparent-power limit for PQ-controlled ports."""

    def make_rule(a: float, b: float):
        def rule(m, device, phase, time):
            if m.mpssd_control[device, phase] != ControlVariable.PQ:
                return pyo.Constraint.Skip
            return (
                a * m.p_mpssd[device, phase, time] + b * m.q_mpssd[device, phase, time]
                <= m.mpssd_s_max[device, phase]
            )

        return rule

    for i, (a, b) in enumerate(ngon_line_equations(n), start=1):
        if a < -1e-9:
            continue
        setattr(
            m,
            f"mpssd_ngon_limit_{i}",
            pyo.Constraint(
                m.mpssd_device_phase_set, m.time_set, rule=make_rule(float(a), float(b))
            ),
        )


def add_dc_bus_balance_constraints(m: Any) -> None:
    """Zero net active power over all ports sharing a DC bus."""

    def rule(m, dc_bus, time):
        return (
            sum(m.p_mpssd[d, ph, time] for d, ph in m.mpssd_ports_by_dc_bus[dc_bus])
            == 0
        )

    m.mpssd_dc_bus_balance = pyo.Constraint(m.mpssd_dc_bus_set, m.time_set, rule=rule)


def add_phase_balance_constraints(m: Any) -> None:
    """Equalize P and Q across phases for devices flagged as phase-balanced."""

    def make_rule(var: Any):
        def rule(m, device, phase, time):
            if not m.mpssd_is_phase_balanced[device]:
                return pyo.Constraint.Skip
            first = m.mpssd_phases_by_device[device][0]
            if phase == first:
                return pyo.Constraint.Skip
            return var[device, phase, time] == var[device, first, time]

        return rule

    for name, var in (("p", m.p_mpssd), ("q", m.q_mpssd)):
        setattr(
            m,
            f"mpssd_{name}_balanced_phases",
            pyo.Constraint(m.mpssd_device_phase_set, m.time_set, rule=make_rule(var)),
        )
