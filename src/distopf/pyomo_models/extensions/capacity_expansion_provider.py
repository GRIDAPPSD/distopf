"""Provider adapter for capacity-expansion planning variables and constraints."""

from __future__ import annotations

from typing import Any

from distopf.pyomo_models.extensions.capacity_expansion import (
    add_bess_capacity_constraints,
    add_capacity_expansion_variables,
    add_der_capacity_injection_constraints,
    add_pv_capacity_constraints,
    add_zone_capacity_expansion_constraints,
)


class CapacityExpansionProvider:
    """Own capacity-expansion components without replacing power balance."""

    name = "capacity_expansion"

    def __init__(self, case: Any, zones: dict, *, enabled: bool = True):
        self.case = case
        self.zones = zones
        self.enabled = enabled

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        if self.enabled:
            add_capacity_expansion_variables(model, self.case, self.zones)

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        key = (bus, phase, time)
        if self.enabled and key in model.p_der_inj:
            return model.p_der_inj[key]
        return 0

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return 0

    def add_constraints(self, model: Any, config: Any) -> None:
        if not self.enabled:
            return
        add_zone_capacity_expansion_constraints(model)
        add_pv_capacity_constraints(model)
        add_bess_capacity_constraints(model)
        add_der_capacity_injection_constraints(model)
