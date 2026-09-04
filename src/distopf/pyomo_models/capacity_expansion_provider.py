"""Provider adapter for capacity-expansion planning variables and constraints."""

from __future__ import annotations

from typing import Any

from distopf.pyomo_models.capacity_expansion_constraints import (
    add_bess_capacity_constraints,
    add_capacity_expansion_variables,
    add_der_capacity_injection_constraints,
    add_pv_capacity_constraints,
    add_zone_capacity_expansion_constraints,
)
from distopf.pyomo_models.injection_registry import InjectionRegistry


class CapacityExpansionProvider:
    """Own capacity-expansion components without replacing power balance."""

    name = "capacity_expansion"
    supported_formulations = frozenset({"lindist"})

    def __init__(self, case: Any, zones: dict, *, enabled: bool = True):
        self.case = case
        self.zones = zones
        self.enabled = enabled

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        if self.enabled:
            add_capacity_expansion_variables(model, self.case, self.zones)

    def register_injections(
        self, model: Any, injections: InjectionRegistry, config: Any
    ) -> None:
        if not self.enabled:
            return
        injections.add(
            self.name,
            p_term=lambda m, bus, phase, time: (
                m.p_der_inj[bus, phase, time]
                if (bus, phase, time) in m.p_der_inj
                else 0
            ),
        )

    def add_constraints(self, model: Any, config: Any) -> None:
        if not self.enabled:
            return
        add_zone_capacity_expansion_constraints(model)
        add_pv_capacity_constraints(model)
        add_bess_capacity_constraints(model)
        add_der_capacity_injection_constraints(model)
