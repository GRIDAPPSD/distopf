"""Branch-attached regulator devices with fixed ratios or discrete tap selection.

One device (one `reg_data` row) has one tap shared by all of its phases. The
network sees regulators only through ``v2_reg[fb, tb, ph, t]`` on the regulated
branch-phases (``reg_phase_set``):

    v2_reg[fb, tb, ph, t] = ratio[d, t]^2 * v2[fb, ph, t]   for every phase ph of d
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyomo.environ as pyo  # type: ignore

from distopf.gad_models.common.device_data import (
    case_num_taps,
    cell_blank,
    cell_num,
    cell_opt_float,
    cell_text,
    check_columns,
    check_device_names,
    check_phases,
    finish_validation,
    reg_num_taps,
    reg_tap_step,
    row_label,
)
from distopf.gad_models.common.protocol import LindistModelProtocol
from distopf.gad_models.common.schema import LOAD_PHASES

REQUIRED_COLUMNS = {"device_name", "from_name", "to_name", "phases"}
LEGACY_COLUMNS = {f"{prefix}_{ph}" for prefix in ("tap", "ratio") for ph in LOAD_PHASES}
LEGACY_HINT = (
    "a regulator has one `tap` shared by its phases; model independent phases "
    "as one single-phase regulator device per phase"
)


def _device_branch(
    row: pd.Series, label: str, model: Any, errors: list[str]
) -> tuple[int, int] | None:
    """(fb, tb) of the branch named by from_name/to_name; legacy fb/tb must agree."""
    names = (cell_text(row.get("from_name")), cell_text(row.get("to_name")))
    unknown = [name for name in names if name not in model.bus_name_to_id_map]
    if unknown:
        errors.append(f"{label}: bus names {unknown} not found in network")
        return None
    fb, tb = (model.bus_name_to_id_map[name] for name in names)
    if (fb, tb) not in model.branch_set:
        reversed_hint = (
            " (the branch runs the other way)" if (tb, fb) in model.branch_set else ""
        )
        errors.append(
            f"{label}: {names[0]} -> {names[1]} is not a branch{reversed_hint}"
        )
        return None
    legacy = (
        cell_opt_float(row, "fb", label, errors),
        cell_opt_float(row, "tb", label, errors),
    )
    if legacy != (None, None) and legacy != (fb, tb):
        errors.append(
            f"{label}: fb/tb {legacy} do not match from_name/to_name {names} = {(fb, tb)}"
        )
    return fb, tb


def _check_tap(row: pd.Series, label: str, errors: list[str]) -> None:
    if cell_num(row, "ratio_max", 1.1) <= cell_num(row, "ratio_min", 0.9):
        errors.append(f"{label}: ratio_max must be greater than ratio_min")
    num_taps = reg_num_taps(row)
    if num_taps <= 0 or num_taps % 2:
        errors.append(f"{label}: num_taps={num_taps} must be a positive even integer")
    tap = cell_opt_float(row, "tap", label, errors)
    half = num_taps // 2
    if tap is not None and (tap != round(tap) or abs(tap) > half):
        errors.append(f"{label}: tap={tap} must be an integer in [{-half}, {half}]")
    limit = cell_opt_float(row, "max_tap_change", label, errors)
    if limit is not None and (limit != round(limit) or limit < 0):
        errors.append(f"{label}: max_tap_change={limit} must be a nonnegative integer")


def validate_reg_data(data: pd.DataFrame, model: Any) -> dict[str, tuple]:
    """Validate reg_data and return device -> (fb, tb, phases).

    Every device must sit on one branch (from_name -> to_name) with phases on
    that branch, and every branch-phase is regulated by at most one device.
    """
    if data.empty:
        return {}
    check_columns(data, "reg_data", REQUIRED_COLUMNS, LEGACY_COLUMNS, LEGACY_HINT)
    errors: list[str] = []
    check_device_names(data, errors)
    devices: dict[str, tuple] = {}
    owner: dict[tuple[int, int, str], str] = {}
    for index, row in data.iterrows():
        label = row_label(row, index)
        name = cell_text(row.get("device_name"))
        _check_tap(row, label, errors)
        phases = check_phases(label, row.get("phases"), LOAD_PHASES, errors)
        branch = _device_branch(row, label, model, errors)
        if phases is None or branch is None:
            continue
        fb, tb = branch
        for phase in phases:
            key = (fb, tb, phase)
            if key not in model.branch_phase_set:
                errors.append(f"{label}: branch {fb} -> {tb} has no phase {phase!r}")
            elif key in owner:
                errors.append(
                    f"{label}: branch phase {key} is already regulated by {owner[key]!r}"
                )
            else:
                owner[key] = name
        devices[name] = (fb, tb, phases)
    finish_validation("reg_data", errors, [], [])
    return devices


class RegulatorProvider:
    """Own regulator ratio, tap-selection, and tap-change constraints."""

    name = "regulators"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        """Create regulator sets, ratios, ``v2_reg`` and optional tap binaries."""
        data = getattr(case, "reg_data", None)
        if data is None:
            data = pd.DataFrame()
        devices = validate_reg_data(data, model)
        rows = {cell_text(row.device_name): row for _, row in data.iterrows()}

        model.reg_device_set = pyo.Set(initialize=list(devices), ordered=True)
        model.reg_branch_by_device = {d: (fb, tb) for d, (fb, tb, _) in devices.items()}
        model.reg_device_by_phase = {
            (fb, tb, ph): d for d, (fb, tb, phases) in devices.items() for ph in phases
        }
        model.reg_phase_set = pyo.Set(
            initialize=list(model.reg_device_by_phase), dimen=3
        )
        taps = {d: int(cell_num(row, "tap", 0)) for d, row in rows.items()}
        steps = {d: reg_tap_step(row) for d, row in rows.items()}
        model.reg_ratio = pyo.Param(
            model.reg_device_set,
            initialize={d: 1.0 + steps[d] * taps[d] for d in devices},
        )
        model.v2_reg = pyo.Var(
            model.reg_phase_set,
            model.time_set,
            domain=pyo.NonNegativeReals,
            initialize=1,
        )
        if not getattr(model, "reg_mi_enabled", False):
            return

        n_taps = case_num_taps(data)
        neutral = n_taps // 2
        model.tap_set = pyo.RangeSet(0, n_taps)
        ratios = {
            (d, k): 1.0 + steps[d] * (k - neutral)
            for d in devices
            for k in range(n_taps + 1)
        }
        model.tap_ratio = pyo.Param(
            model.reg_device_set, model.tap_set, initialize=ratios
        )
        model.tap_ratio_squared = pyo.Param(
            model.reg_device_set,
            model.tap_set,
            initialize={key: value**2 for key, value in ratios.items()},
        )
        model.reg_big_m = pyo.Param(initialize=1e3)
        model.reg_max_tap_change = {
            d: int(row.max_tap_change)
            for d, row in rows.items()
            if not cell_blank(row, "max_tap_change")
        }
        model.u_reg = pyo.Var(
            model.reg_device_set,
            model.tap_set,
            model.time_set,
            domain=pyo.Binary,
            initialize=lambda _m, d, k, _t: 1 if k == neutral + taps[d] else 0,
        )

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return 0

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return 0

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.reg_phase_set) == 0:
            return
        if not getattr(model, "reg_mi_enabled", False):
            add_regulator_constraints(model)
            return
        add_regulator_tap_sos1_constraints(model)
        default_limit = (
            getattr(config, "reg_tap_change_limit", None) if config else None
        )
        if default_limit is not None and (
            isinstance(default_limit, bool)
            or not isinstance(default_limit, int)
            or default_limit < 0
        ):
            raise ValueError("reg_tap_change_limit must be a nonnegative integer")
        limits = {
            d: model.reg_max_tap_change.get(d, default_limit)
            for d in model.reg_device_set
        }
        add_regulator_tap_change_limit_constraints(
            model, {d: limit for d, limit in limits.items() if limit is not None}
        )


# ============ Regulator Constraints (Standard and MI) =================================
# ======================================================================================


def add_regulator_constraints(m: LindistModelProtocol) -> None:
    """Fixed tap: v2_reg = v2[fb] * reg_ratio[device]^2."""

    def regulator_rule(m: LindistModelProtocol, fb, tb, ph, t):
        device = m.reg_device_by_phase[fb, tb, ph]
        return m.v2_reg[fb, tb, ph, t] == m.v2[fb, ph, t] * m.reg_ratio[device] ** 2

    m.regulator_ratio = pyo.Constraint(m.reg_phase_set, m.time_set, rule=regulator_rule)


def add_regulator_tap_sos1_constraints(m: LindistModelProtocol) -> None:
    """Exactly one tap per device and time; Big-M ties every phase to that tap.

    sum_k u_reg[d, k, t] == 1
    |v2_reg[fb, tb, ph, t] - tap_ratio[d, k]^2 * v2[fb, ph, t]| <= M (1 - u_reg[d, k, t])
    """

    def sos1_rule(m: LindistModelProtocol, device, t):
        return sum(m.u_reg[device, k, t] for k in m.tap_set) == 1

    def gap(m: LindistModelProtocol, fb, tb, ph, k, t):
        device = m.reg_device_by_phase[fb, tb, ph]
        slack = m.reg_big_m * (1 - m.u_reg[device, k, t])
        diff = (
            m.v2_reg[fb, tb, ph, t] - m.tap_ratio_squared[device, k] * m.v2[fb, ph, t]
        )
        return diff, slack

    def reg_tap_upper(m: LindistModelProtocol, fb, tb, ph, k, t):
        diff, slack = gap(m, fb, tb, ph, k, t)
        return diff <= slack

    def reg_tap_lower(m: LindistModelProtocol, fb, tb, ph, k, t):
        diff, slack = gap(m, fb, tb, ph, k, t)
        return diff >= -slack

    m.reg_tap_upper = pyo.Constraint(
        m.reg_phase_set, m.tap_set, m.time_set, rule=reg_tap_upper
    )
    m.reg_tap_lower = pyo.Constraint(
        m.reg_phase_set, m.tap_set, m.time_set, rule=reg_tap_lower
    )
    m.reg_tap_sos1 = pyo.Constraint(m.reg_device_set, m.time_set, rule=sos1_rule)


# ============ Regulator tap change limit ==============================================
# ======================================================================================


def add_regulator_tap_change_limit_constraints(
    m: LindistModelProtocol, max_tap_change: dict[str, int]
) -> None:
    """Limit each device's tap change between consecutive time steps.

    Devices missing from ``max_tap_change`` are unlimited.
    """

    def tap_change(m: LindistModelProtocol, device, t):
        tap_t = sum(k * m.u_reg[device, k, t] for k in m.tap_set)
        tap_prev = sum(k * m.u_reg[device, k, t - 1] for k in m.tap_set)
        return tap_t - tap_prev

    def limited(t, device) -> bool:
        return t != pyo.value(m.start_step) and device in max_tap_change

    def upper(m: LindistModelProtocol, device, t):
        if not limited(t, device):
            return pyo.Constraint.Skip
        return tap_change(m, device, t) <= max_tap_change[device]

    def lower(m: LindistModelProtocol, device, t):
        if not limited(t, device):
            return pyo.Constraint.Skip
        return tap_change(m, device, t) >= -max_tap_change[device]

    m.reg_tap_change_upper = pyo.Constraint(m.reg_device_set, m.time_set, rule=upper)
    m.reg_tap_change_lower = pyo.Constraint(m.reg_device_set, m.time_set, rule=lower)


__all__ = ["RegulatorProvider", "validate_reg_data"]
