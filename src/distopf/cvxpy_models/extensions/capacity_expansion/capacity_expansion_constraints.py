"""Capacity-expansion setup and constraint helpers (CVXPY).

Mirror of ``pyomo_models.extensions.capacity_expansion.capacity_expansion_constraints``.
``add_der_capacity_injection_constraints`` uses the number of phases of the
bus (``phase_map``) where the Pyomo version references an undefined
``m.bus_phases``.
"""

from __future__ import annotations

from distopf.api import Case
from distopf.cvxpy_models.common.model import (
    CvxModel,
    IndexSet,
    ParamDict,
    VarDict,
    add_constraint,
    index_product,
)
from distopf.pyomo_models.extensions.capacity_expansion.capacity_expansion_constraints import (
    create_zones_from_edge_names as create_zones_from_edge_names,
)


def add_capacity_expansion_as_fraction_of_load(
    m,
    case: Case,
    fraction: float = 0.1,
    relative_capacity: dict[str, float] | None = None,
):
    if relative_capacity is None:
        relative_capacity = {"PV": 0.5, "BESS": 0.5}
    total_relative_capacity = sum(relative_capacity.values())
    if total_relative_capacity == 0:
        raise ValueError("Sum of relative capacities cannot be zero.")
    relative_capacity = {
        k: v / total_relative_capacity for k, v in relative_capacity.items()
    }
    m.resource_set = IndexSet(relative_capacity.keys(), dimen=1)
    total_load_p = case.bus_data[["pl_a", "pl_b", "pl_c"]].sum().sum()
    m.total_capacity_expansion = fraction * total_load_p
    m.relative_capacity = ParamDict(relative_capacity)


def add_capacity_expansion_from_absolute_capacity(
    m, absolute_capacity={"PV": 0.5, "BESS": 0.5}
):
    """Initialize capacity-expansion budget from absolute per-resource capacities (p.u.)."""
    if not absolute_capacity:
        raise ValueError("absolute_capacity cannot be empty.")
    total_capacity = sum(absolute_capacity.values())
    if any(v < 0 for v in absolute_capacity.values()):
        raise ValueError("Absolute capacities must be non-negative.")

    m.resource_set = IndexSet(absolute_capacity.keys(), dimen=1)
    m.total_capacity_expansion = total_capacity
    if total_capacity == 0:
        relative_capacity = {
            k: 1 / len(absolute_capacity) for k in absolute_capacity.keys()
        }
    else:
        relative_capacity = {
            k: v / total_capacity for k, v in absolute_capacity.items()
        }
    m.relative_capacity = ParamDict(relative_capacity)


def add_pv_parameters(
    m,
    curtailment_max=1.0,
    capacity_factor=0.8,
    case: Case | None = None,
    pv_shape: str = "PV",
):
    m.curtailment_max = curtailment_max
    if case is not None and pv_shape in case.schedules.columns:
        cf_by_time = {t: case.schedules.at[t, pv_shape] for t in m.time_set}
    else:
        cf_by_time = {t: capacity_factor for t in m.time_set}
    m.capacity_factor = ParamDict(cf_by_time)


def add_bess_parameters(
    m,
    e_max=4.0,
    soc=0.5,
    discharge_derate=1.0,
    charge_derate=1.0,
):
    m.e_max = e_max
    m.soc_new = soc
    m.discharge_derate = discharge_derate
    m.charging_derate = charge_derate


def _compute_load_proportional_allocation_factors(m, case: Case, zones):
    """Load-proportional DER allocation ``alpha[bus, resource]`` (Eq. 4 and 5)."""
    alpha = {(_id, r): 0.0 for _id in m.bus_set for r in m.resource_set}
    bus_data_by_id = case.bus_data.set_index("id", drop=False)
    for z, buses in zones.items():
        z_load = bus_data_by_id.loc[buses, ["pl_a", "pl_b", "pl_c"]].sum().sum()
        for _id in buses:
            frac = (
                bus_data_by_id.loc[_id, ["pl_a", "pl_b", "pl_c"]].sum() / z_load
                if z_load > 0
                else 1.0 / len(buses)
            )
            for r in m.resource_set:
                alpha[_id, r] = frac
    m.alpha = ParamDict(alpha, default=0.0)
    for z, buses in zones.items():
        for r in m.resource_set:
            alpha_sum = sum(m.alpha[_id, r] for _id in buses)
            assert abs(alpha_sum - 1) < 1e-6, (
                f"Zone {z} resource {r} alpha sum is {alpha_sum}"
            )


def add_capacity_expansion_variables(m, case: Case, zones):
    if not hasattr(m, "resource_set"):
        raise AttributeError(
            "Calling add_capacity_expansion_variables requires m.resource_set to be "
            "defined. Call add_capacity_expansion_as_fraction_of_load first. Then "
            "call add_pv_parameters and add_bess_parameters to set resource-specific "
            "parameters."
        )

    m.zone_set = IndexSet(list(zones.keys()), dimen=1)
    z_of_bus = {_id: None for _id in m.bus_set}
    for z, buses in zones.items():
        for _id in buses:
            z_of_bus[_id] = z
    m.z_of_bus = z_of_bus
    m.p_max = VarDict(
        index_product(m.zone_set, m.resource_set), nonneg=True, initialize=0.0
    )
    m.p_zr = VarDict(
        index_product(m.zone_set, m.resource_set, m.time_set), initialize=0.0
    )
    m.p_der_inj = VarDict(
        index_product(m.bus_phase_set, m.time_set), initialize=0.0
    )
    _compute_load_proportional_allocation_factors(m, case, zones)


def add_capacity_expansion_p_flow_constraints(m):
    """Custom power balance including ``p_der_inj`` (replaces ``power_balance_p``)."""

    def _p_balance(m: CvxModel, fb, tb, ph, t):
        load = m.p_load[tb, ph, t]
        gen = sum(
            m.p_gen[device, ph, t]
            for device in m.gen_devices_by_bus_phase.get((tb, ph), [])
        )
        bat = sum(
            m.p_bat[device, ph, t]
            for device in getattr(m, "bat_devices_by_bus_phase", {}).get((tb, ph), [])
        )
        out = sum(
            m.p_flow[to_fb, to_tb, ph, t]
            for to_fb, to_tb in m.to_bus_map[tb]
            if (to_fb, to_tb, ph) in m.branch_phase_set
        )
        return (
            m.p_flow[fb, tb, ph, t] == out + load - gen - bat - m.p_der_inj[tb, ph, t]
        )

    if "power_balance_p" in m.constraint_names:
        m.del_component("power_balance_p")
    add_constraint(m, "power_balance_p", (m.branch_phase_set, m.time_set), _p_balance)


def add_zone_capacity_expansion_constraints(m):
    r"""Eq. 3: ``sum_z p_max[z, r] == T_DER * c_r``."""
    add_constraint(
        m,
        "der_budget",
        (m.resource_set,),
        lambda m, r: sum(m.p_max[z, r] for z in m.zone_set)
        == m.total_capacity_expansion * m.relative_capacity[r],
    )


def add_pv_capacity_constraints(m):
    if "PV" not in m.resource_set:
        return
    m.pv_set = IndexSet(["PV"], dimen=1)
    index = (m.zone_set, m.pv_set, m.time_set)
    add_constraint(
        m,
        "pv_gen_min",
        index,
        lambda m, z, r, t: m.p_zr[z, r, t]
        >= (1 - m.curtailment_max) * m.capacity_factor[t] * m.p_max[z, r],
    )
    add_constraint(
        m,
        "pv_gen_max",
        index,
        lambda m, z, r, t: m.p_zr[z, r, t] <= m.capacity_factor[t] * m.p_max[z, r],
    )


def add_bess_capacity_constraints(m):
    if "BESS" not in m.resource_set:
        return
    m.bess_set = IndexSet(["BESS"], dimen=1)
    index = (m.zone_set, m.bess_set, m.time_set)
    add_constraint(
        m,
        "bess_charge_max",
        index,
        lambda m, z, r, t: -m.p_zr[z, r, t] <= m.charging_derate * m.p_max[z, r],
    )
    add_constraint(
        m,
        "bess_discharge_max",
        index,
        lambda m, z, r, t: m.p_zr[z, r, t] <= m.discharge_derate * m.p_max[z, r],
    )
    add_constraint(
        m,
        "bess_energy_1",
        index,
        lambda m, z, r, t: m.p_zr[z, r, t] <= m.e_max * m.soc_new / m.delta_t,
    )
    add_constraint(
        m,
        "bess_energy_2",
        index,
        lambda m, z, r, t: -m.p_zr[z, r, t]
        <= m.e_max * (1 - m.soc_new) / m.delta_t,
    )


def add_der_capacity_injection_constraints(m):
    """Each bus uses its assigned zone's dispatch, split equally across phases."""

    def _der_inj(m, _id, ph, t):
        z = m.z_of_bus[_id]
        if z is None:
            return m.p_der_inj[_id, ph, t] == 0
        inj = sum(m.alpha[_id, r] * m.p_zr[z, r, t] for r in m.resource_set)
        return m.p_der_inj[_id, ph, t] == inj / len(m.phase_map[_id])

    add_constraint(m, "der_inj_map", (m.bus_phase_set, m.time_set), _der_inj)
