"""Regulator provider for branch-attached legacy model components."""

from __future__ import annotations

from typing import Any

from distopf.pyomo_models import common_constraints


class RegulatorProvider:
    """Own regulator ratio, tap-selection, and tap-change constraints."""

    name = "regulators"
    supported_formulations = frozenset({"lindist", "nl_bfm"})

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        """Regulator variables are still created by each model factory."""

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
