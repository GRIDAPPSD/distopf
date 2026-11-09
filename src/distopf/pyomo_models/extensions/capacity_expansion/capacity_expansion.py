"""Provider adapter for capacity-expansion planning variables and constraints."""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo  # type: ignore
from distopf.pyomo_models.extensions.capacity_expansion.capacity_expansion_constraints import (
    add_bess_capacity_constraints,
    add_capacity_expansion_variables,
    add_der_capacity_injection_constraints,
    add_pv_capacity_constraints,
    add_zone_capacity_expansion_constraints,
    create_zones_from_edge_names,
    add_pv_parameters,
    add_bess_parameters,
    add_capacity_expansion_from_absolute_capacity,
)


class CapacityExpansionProvider:
    """Own capacity-expansion components without replacing power balance."""

    name = "capacity_expansion"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        zones = create_zones_from_edge_names(
            case,
            config.zone_edges,
        )

        absolute_capacity = dict(PV=config.new_pv, BESS=config.new_bess)

        add_capacity_expansion_from_absolute_capacity(model, absolute_capacity)

        add_pv_parameters(
            model,
            curtailment_max=config.pv_curtailment_max,
            capacity_factor=config.pv_capacity_factor,
            case=case,
            pv_shape=config.pv_shape,
        )
        add_bess_parameters(
            model,
            e_max=config.bess_energy_capacity,
            soc=config.bess_soc,
            discharge_derate=config.bess_discharge_derate,
            charge_derate=config.bess_charge_derate,
        )
        add_capacity_expansion_variables(model, case, zones)

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        key = (bus, phase, time)
        if key in model.p_der_inj:
            return model.p_der_inj[key]
        return 0

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return 0

    def add_constraints(self, model: Any, config: Any) -> None:
        add_zone_capacity_expansion_constraints(model)
        add_pv_capacity_constraints(model)
        add_bess_capacity_constraints(model)
        add_der_capacity_injection_constraints(model)
