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

from typing import Any

import pandas as pd
import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models.common.registry import DeviceProvider
from distopf.pyomo_models.common.data import parse_phases
from distopf.pyomo_models.common.model_types import ControlVariable
from distopf.pyomo_models.common.device_data import (
    TOL,
    InfeasibleCaseError as InfeasibleCaseError,
    NetworkContext,
    cell_control,
    cell_flag,
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
    check_columns(
        data,
        "mpssd",
        _REQUIRED,
        _KNOWN,
        _LEGACY,
        "devices are identified by 'device_name' and buses by 'bus_name'",
    )
    ctx = NetworkContext(
        bus_name_to_id=bus_name_to_id_map,
        injectable=frozenset(injectable),
        swing_buses=frozenset(swing_buses),
        times=(),
    )

    errors: list[str] = []
    warns: list[str] = []

    # ---- device names ----
    check_device_names(data, errors)

    # dc_bus -> list of (label, p_lo, p_hi, n_ports)
    dc_intervals: dict[int, list[tuple[str, float, float, int]]] = {}

    for idx, row in data.iterrows():
        label = row_label(row, idx)
        errors_before = len(errors)

        # bus
        bus_name = cell_text(row["bus_name"])

        # phases
        phases = check_phases(label, row["phases"], _PHASES, errors)
        if phases is None:
            errors[-1] += "; MPSSD supports phases a, b, c only"
            continue

        check_connection(label, bus_name, phases, ctx, errors)

        # dc_bus
        dc_bus: int | None = None
        dc_value = cell_required_float(row, "dc_bus", label, errors)
        if dc_value is not None:
            if dc_value != round(dc_value):
                errors.append(f"{label}: dc_bus={dc_value} must be a whole number")
            else:
                dc_bus = int(dc_value)

        # control mode and phase-balance flag
        ctrl: ControlVariable | None
        try:
            ctrl = cell_control(row)
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            ctrl = None
        try:
            balanced = cell_flag(row, "balanced_phases", default=False)
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
            s_max = cell_required_float(row, f"s_{phase}_max", label, errors, gt=0)
            if s_max is not None:
                s_list.append(s_max)

            p_set = cell_opt_float(row, f"p_{phase}", label, errors)
            q_set = cell_opt_float(row, f"q_{phase}", label, errors)
            q_lo = cell_opt_float(row, f"q_{phase}_min", label, errors)
            q_hi = cell_opt_float(row, f"q_{phase}_max", label, errors)

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
                    if s_max is not None and abs(p_set) > s_max + TOL:
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
                        and not (lo - TOL <= q_set <= hi + TOL)
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
                and p_set**2 + q_set**2 > s_max**2 + TOL
            ):
                errors.append(
                    f"{label}: fixed (p_{phase}, q_{phase}) = ({p_set}, {q_set}) "
                    f"exceeds s_{phase}_max={s_max}"
                )

        # phase balance vs. fixed setpoints
        if balanced and len(phases) > 1:
            for kind, vals in (("p", p_vals), ("q", q_vals)):
                if vals and max(vals) - min(vals) > TOL:
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

    infeasible: list[str] = []
    for dc_bus, items in sorted(dc_intervals.items()):
        lo = sum(item[1] for item in items)
        hi = sum(item[2] for item in items)
        if lo > TOL or hi < -TOL:
            who = ", ".join(item[0] for item in items)
            infeasible.append(
                f"dc_bus {dc_bus}: total P must be 0 but can only range over "
                f"[{lo:g}, {hi:g}] (devices: {who})"
            )
        if not errors and sum(item[3] for item in items) == 1:
            warns.append(
                f"dc_bus {dc_bus} has only one port; its P will be forced to 0"
            )
    finish_validation("mpssd", errors, infeasible, warns)


# --------------------------------------------------------------------------- #
# Provider
# --------------------------------------------------------------------------- #


class MpssdProvider(DeviceProvider):
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
                p_nom = cell_num(row, f"p_{phase}", 0.0)  # blank only where ignored
                q_nom = cell_num(row, f"q_{phase}", 0.0)

                ports.append(port)
                devices_by_bus_phase.setdefault((bus, phase), []).append(device)
                ports_by_dc_bus.setdefault(dc_bus, []).append(port)

                s_max[port] = rating
                q_min[port] = max(-rating, cell_num(row, f"q_{phase}_min", -rating))
                q_max[port] = min(rating, cell_num(row, f"q_{phase}_max", rating))
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
        model.mpssd_bus_by_device = bus_by_device
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
