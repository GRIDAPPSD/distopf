"""Branch-attached regulators with fixed ratios or discrete tap selection (CVXPY).

Mirror of ``pyomo_models.devices.regulator``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

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
    cell_blank,
    cell_num,
    cell_opt_float,
    cell_required_float,
    check_phases,
    finish_validation,
)


def validate_reg_data(data: pd.DataFrame, model: Any) -> None:
    """Validate branch membership, unique regulated phases, ratios, and taps."""
    errors: list[str] = []
    seen: set[tuple[int, int, str]] = set()
    for index, row in data.iterrows():
        label = f"regulator row {index}"
        from_bus = cell_required_float(row, "fb", label, errors)
        to_bus = cell_required_float(row, "tb", label, errors)
        if from_bus is None or to_bus is None:
            continue
        if from_bus != round(from_bus) or to_bus != round(to_bus):
            errors.append(f"{label}: fb and tb must be integer bus IDs")
            continue
        fb, tb = int(from_bus), int(to_bus)
        phases = check_phases(
            label, row.get("phases"), ("a", "b", "c", "s1", "s2"), errors
        )
        if phases is None:
            continue
        for phase in phases:
            key = (fb, tb, phase)
            if key not in model.branch_phase_set:
                errors.append(f"{label}: regulated branch phase {key} does not exist")
            if key in seen:
                errors.append(f"{label}: duplicate regulated branch phase {key}")
            seen.add(key)
            ratio_key = f"ratio_{phase}"
            tap = cell_opt_float(row, f"tap_{phase}", label, errors)
            if not cell_blank(row, ratio_key) and tap is None:
                errors.append(
                    f"{label}: tap_{phase} is required; {ratio_key} is derived from tap_{phase}"
                )
            if tap is not None and (tap != round(tap) or not -16 <= tap <= 16):
                errors.append(f"{label}: tap_{phase} must be an integer in [-16, 16]")
    finish_validation("reg_data", errors, [], [])


def create_regulator_parameters(model: Any, case: Any) -> None:
    """Create regulator ratio parameters from case data."""
    ratio = {
        (int(row.fb), int(row.tb), phase): 1.0
        + 0.00625 * cell_num(row, f"tap_{phase}", 0)
        for _, row in case.reg_data.iterrows()
        for phase in parse_phases(str(row.phases))
    }
    model.reg_ratio = ParamDict(ratio, default=1.0)


class RegulatorProvider:
    """Own regulator ratio, tap-selection, and tap-change constraints."""

    name = "regulators"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        data = getattr(case, "reg_data", None)
        if data is None:
            data = pd.DataFrame()
        validate_reg_data(data, model)
        model.reg_phase_set = IndexSet(
            [
                (int(row.fb), int(row.tb), phase)
                for _, row in data.iterrows()
                for phase in parse_phases(str(row.phases).strip())
            ],
            dimen=3,
        )
        if data.empty:
            model.reg_ratio = ParamDict({}, default=1.0)
        else:
            create_regulator_parameters(model, case)

        model.v2_reg = VarDict(
            index_product(model.reg_phase_set, model.time_set),
            nonneg=True,
            initialize=1.0,
        )
        if getattr(model, "reg_mi_enabled", False):
            model.tap_set = IndexSet(range(0, 33), dimen=1)
            ratios = {index: 0.9 + index * 0.00625 for index in range(33)}
            model.tap_ratio = ParamDict(ratios)
            model.tap_ratio_squared = ParamDict(
                {key: value**2 for key, value in ratios.items()}
            )
            model.reg_big_m = 1e6
            model.u_reg = VarDict(
                index_product(model.reg_phase_set, model.tap_set, model.time_set),
                boolean=True,
                initialize=lambda _fb, _tb, _ph, tap, _t: 1.0 if tap == 16 else 0.0,
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
        control = getattr(model, "reg_mi_enabled", False)
        if control:
            add_regulator_tap_sos1_constraints(model)
            tap_limit = (
                getattr(config, "reg_tap_change_limit", None) if config else None
            )
            if tap_limit is not None:
                if (
                    isinstance(tap_limit, bool)
                    or not isinstance(tap_limit, int)
                    or tap_limit < 0
                ):
                    raise ValueError(
                        "reg_tap_change_limit must be a nonnegative integer"
                    )
                add_regulator_tap_change_limit_constraints(
                    model, max_tap_change=tap_limit
                )
        else:
            add_regulator_constraints(model)


def add_regulator_constraints(m: CvxModel) -> None:
    """``v_reg = v_i * reg_ratio^2``."""

    def regulator_rule(m, fb, tb, ph, t):
        return m.v2_reg[fb, tb, ph, t] == m.v2[fb, ph, t] * m.reg_ratio[fb, tb, ph] ** 2

    add_constraint(
        m, "regulator_ratio", (m.reg_phase_set, m.time_set), regulator_rule
    )


def add_regulator_tap_sos1_constraints(m: CvxModel) -> None:
    """Exactly one tap per regulator, with Big-M links to ``v2_reg``.

    ``v2_reg = tap_ratio_squared[k] * v2[fb]`` when tap k is selected.
    """

    def sos1_rule(m, fb, tb, ph, t):
        return sum(m.u_reg[fb, tb, ph, k, t] for k in m.tap_set) == 1

    def reg_tap_upper(m, fb, tb, ph, k, t):
        return m.v2_reg[fb, tb, ph, t] - m.tap_ratio_squared[k] * m.v2[
            fb, ph, t
        ] <= m.reg_big_m * (1 - m.u_reg[fb, tb, ph, k, t])

    def reg_tap_lower(m, fb, tb, ph, k, t):
        return m.v2_reg[fb, tb, ph, t] - m.tap_ratio_squared[k] * m.v2[
            fb, ph, t
        ] >= -m.reg_big_m * (1 - m.u_reg[fb, tb, ph, k, t])

    add_constraint(
        m, "reg_tap_upper", (m.reg_phase_set, m.tap_set, m.time_set), reg_tap_upper
    )
    add_constraint(
        m, "reg_tap_lower", (m.reg_phase_set, m.tap_set, m.time_set), reg_tap_lower
    )
    add_constraint(m, "reg_tap_sos1", (m.reg_phase_set, m.time_set), sos1_rule)


def add_regulator_tap_change_limit_constraints(
    m: CvxModel, max_tap_change: int = 2
) -> None:
    """Limit regulator tap changes between consecutive time steps."""
    if not getattr(m, "reg_mi_enabled", False):
        return

    def _taps(m, fb, tb, ph, t):
        return sum(k * m.u_reg[fb, tb, ph, k, t] for k in m.tap_set)

    def tap_change_limit_upper(m, fb, tb, ph, t):
        if t == m.start_step:
            return SKIP
        return _taps(m, fb, tb, ph, t) - _taps(m, fb, tb, ph, t - 1) <= max_tap_change

    def tap_change_limit_lower(m, fb, tb, ph, t):
        if t == m.start_step:
            return SKIP
        return _taps(m, fb, tb, ph, t) - _taps(m, fb, tb, ph, t - 1) >= -max_tap_change

    add_constraint(
        m, "reg_tap_change_upper", (m.reg_phase_set, m.time_set), tap_change_limit_upper
    )
    add_constraint(
        m, "reg_tap_change_lower", (m.reg_phase_set, m.time_set), tap_change_limit_lower
    )


__all__ = ["RegulatorProvider", "create_regulator_parameters", "validate_reg_data"]
