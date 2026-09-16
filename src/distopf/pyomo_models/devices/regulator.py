"""Regulator provider for branch-attached legacy model components."""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models import common_constraints
from distopf.pyomo_models.devices.data import parse_phases


def create_regulator_parameters(model: Any, case: Any) -> None:
    """Create regulator ratio parameter components from case data."""
    ratio = {(int(row.fb), int(row.tb), phase): getattr(row, f"ratio_{phase}", 1.0) for _, row in case.reg_data.iterrows() for phase in parse_phases(str(row.phases))}
    model.reg_ratio = pyo.Param(model.reg_phase_set, initialize=ratio, default=1.0)


class RegulatorProvider:
    """Own regulator ratio, tap-selection, and tap-change constraints."""

    name = "regulators"
    supported_formulations = frozenset({"lindist", "nl_bfm"})

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        """Create regulator parameters and optional tap-control components."""
        if not hasattr(model, "reg_phase_set"):
            model.reg_phase_set = pyo.Set(
                initialize=[
                    (int(row.fb), int(row.tb), phase)
                    for _, row in case.reg_data.iterrows()
                    for phase in parse_phases(str(row.phases))
                ], dimen=3,
            )
        if not hasattr(model, "reg_ratio"):
            create_regulator_parameters(model, case)
        if getattr(model, "reg_mi_enabled", False) and not hasattr(model, "u_reg"):
            model.tap_set = pyo.RangeSet(0, 32)
            ratios = {index: 0.9 + index * 0.00625 for index in range(33)}
            model.tap_ratio = pyo.Param(model.tap_set, initialize=ratios)
            model.tap_ratio_squared = pyo.Param(
                model.tap_set, initialize={key: value**2 for key, value in ratios.items()}
            )
            model.reg_big_m = pyo.Param(initialize=1e3)
            model.u_reg = pyo.Var(
                model.reg_phase_set, model.tap_set, model.time_set,
                domain=pyo.Binary,
                initialize=lambda _m, _fb, _tb, _ph, tap, _t: 1 if tap == 16 else 0,
            )

    def register_injections(self, model: Any, injections: Any, config: Any) -> None:
        """Regulators do not inject independent bus power."""

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.reg_phase_set) == 0:
            return
        control = getattr(model, "reg_mi_enabled", False)
        if control:
            if not hasattr(model, "reg_tap_sos1"):
                common_constraints.add_regulator_tap_sos1_constraints(model)
            tap_limit = (
                getattr(config, "reg_tap_change_limit", None) if config else None
            )
            if tap_limit is not None and not hasattr(model, "reg_tap_change_upper"):
                common_constraints.add_regulator_tap_change_limit_constraints(
                    model, max_tap_change=tap_limit
                )
        elif not hasattr(model, "regulator_ratio"):
            common_constraints.add_regulator_constraints(model)


__all__ = ["RegulatorProvider", "create_regulator_parameters"]
