"""Regulator provider for branch-attached legacy model components."""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models.common.protocol import LindistModelProtocol
from distopf.pyomo_models.common.data import parse_phases


def create_regulator_parameters(model: Any, case: Any) -> None:
    """Create regulator ratio parameter components from case data."""
    ratio = {
        (int(row.fb), int(row.tb), phase): getattr(row, f"ratio_{phase}", 1.0)
        for _, row in case.reg_data.iterrows()
        for phase in parse_phases(str(row.phases))
    }
    model.reg_ratio = pyo.Param(model.reg_phase_set, initialize=ratio, default=1.0)


class RegulatorProvider:
    """Own regulator ratio, tap-selection, and tap-change constraints."""

    name = "regulators"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        """Create regulator parameters and optional tap-control components."""
        if not hasattr(model, "reg_phase_set"):
            model.reg_phase_set = pyo.Set(
                initialize=[
                    (int(row.fb), int(row.tb), phase)
                    for _, row in case.reg_data.iterrows()
                    for phase in parse_phases(str(row.phases))
                ],
                dimen=3,
            )
        if not hasattr(model, "reg_ratio"):
            create_regulator_parameters(model, case)

        model.v2_reg = pyo.Var(
            model.reg_phase_set,
            model.time_set,
            domain=pyo.NonNegativeReals,
            initialize=1,
        )
        if getattr(model, "reg_mi_enabled", False) and not hasattr(model, "u_reg"):
            model.tap_set = pyo.RangeSet(0, 32)
            ratios = {index: 0.9 + index * 0.00625 for index in range(33)}
            model.tap_ratio = pyo.Param(model.tap_set, initialize=ratios)
            model.tap_ratio_squared = pyo.Param(
                model.tap_set,
                initialize={key: value**2 for key, value in ratios.items()},
            )
            model.reg_big_m = pyo.Param(initialize=1e6)
            model.u_reg = pyo.Var(
                model.reg_phase_set,
                model.tap_set,
                model.time_set,
                domain=pyo.Binary,
                initialize=lambda _m, _fb, _tb, _ph, tap, _t: 1 if tap == 16 else 0,
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
            if not hasattr(model, "reg_tap_sos1"):
                add_regulator_tap_sos1_constraints(model)
            tap_limit = (
                getattr(config, "reg_tap_change_limit", None) if config else None
            )
            if tap_limit is not None and not hasattr(model, "reg_tap_change_upper"):
                add_regulator_tap_change_limit_constraints(
                    model, max_tap_change=tap_limit
                )
        elif not hasattr(model, "regulator_ratio"):
            add_regulator_constraints(model)


# ============ Regulator Constraints (Standard and MI) =================================
# ======================================================================================


def add_regulator_constraints(m: LindistModelProtocol) -> None:
    """
    v_reg = vi*reg_ratio^2
    """

    def regulator_rule(m: LindistModelProtocol, fb, tb, ph, t):
        return m.v2_reg[fb, tb, ph, t] == m.v2[fb, ph, t] * m.reg_ratio[fb, tb, ph] ** 2

    m.regulator_ratio = pyo.Constraint(m.reg_phase_set, m.time_set, rule=regulator_rule)


def add_regulator_tap_sos1_constraints(m: LindistModelProtocol) -> None:
    """
    Add SOS1 (Special Ordered Set Type 1) constraint: exactly one tap position must be selected per regulator.

    sum_k(u_reg[id, ph, k, t]) == 1 for all (id, ph, t)

    Add Big-M regulator tap selection constraints for NL model.

    Uses Big-M to enforce: v2_reg = tap_ratio^2 * v_i when tap k is selected
    Then: v_j = v2_reg - 2*r*p_ij - 2*x*q_ij
    """

    def sos1_rule(m: LindistModelProtocol, fb, tb, ph, t):
        return sum(m.u_reg[fb, tb, ph, k, t] for k in m.tap_set) == 1

    def reg_tap_upper(m: LindistModelProtocol, fb, tb, ph, k, t):
        return m.v2_reg[fb, tb, ph, t] - m.tap_ratio_squared[k] * m.v2[
            fb, ph, t
        ] <= m.reg_big_m * (1 - m.u_reg[fb, tb, ph, k, t])

    def reg_tap_lower(m: LindistModelProtocol, fb, tb, ph, k, t):
        return m.v2_reg[fb, tb, ph, t] - m.tap_ratio_squared[k] * m.v2[
            fb, ph, t
        ] >= -m.reg_big_m * (1 - m.u_reg[fb, tb, ph, k, t])

    m.reg_tap_upper = pyo.Constraint(
        m.reg_phase_set, m.tap_set, m.time_set, rule=reg_tap_upper
    )
    m.reg_tap_lower = pyo.Constraint(
        m.reg_phase_set, m.tap_set, m.time_set, rule=reg_tap_lower
    )
    m.reg_tap_sos1 = pyo.Constraint(m.reg_phase_set, m.time_set, rule=sos1_rule)


# ============ Regulator tap change limit ==============================================
# ======================================================================================


def add_regulator_tap_change_limit_constraints(
    m: LindistModelProtocol, max_tap_change: int = 2
) -> None:
    """
    Limit regulator tap changes between time steps.

    Parameters
    ----------
    m : LindistModelProtocol
        Pyomo model
    max_tap_change : int
        Maximum tap position change allowed per time step (default: 2)
    """
    if not getattr(m, "reg_mi_enabled", False):
        return

    def tap_change_limit_upper(m: LindistModelProtocol, fb, tb, ph, t):
        if t == pyo.value(m.start_step):
            return pyo.Constraint.Skip
        tap_t = sum(k * m.u_reg[fb, tb, ph, k, t] for k in m.tap_set)
        tap_prev = sum(k * m.u_reg[fb, tb, ph, k, t - 1] for k in m.tap_set)
        return tap_t - tap_prev <= max_tap_change

    def tap_change_limit_lower(m: LindistModelProtocol, fb, tb, ph, t):
        if t == pyo.value(m.start_step):
            return pyo.Constraint.Skip
        tap_t = sum(k * m.u_reg[fb, tb, ph, k, t] for k in m.tap_set)
        tap_prev = sum(k * m.u_reg[fb, tb, ph, k, t - 1] for k in m.tap_set)
        return tap_t - tap_prev >= -max_tap_change

    m.reg_tap_change_upper = pyo.Constraint(
        m.reg_phase_set, m.time_set, rule=tap_change_limit_upper
    )
    m.reg_tap_change_lower = pyo.Constraint(
        m.reg_phase_set, m.time_set, rule=tap_change_limit_lower
    )


__all__ = ["RegulatorProvider", "create_regulator_parameters"]
