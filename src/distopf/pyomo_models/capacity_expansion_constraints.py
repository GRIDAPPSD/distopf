import distopf as opf
import pyomo.environ as pyo
import networkx as nx

from distopf.pyomo_models.protocol import LindistModelProtocol
from distopf.pyomo_models.constraints import (
    add_battery_constant_q_constraints_p_control,
    add_battery_energy_constraints,
    add_battery_net_p_bat_equal_phase_constraints,
    add_battery_power_limits,
    add_battery_soc_limits,
    add_capacitor_constraints,
    add_octagonal_inverter_constraints_pq_control,
    add_cvr_load_constraints,
    add_generator_constant_p_constraints_q_control,
    add_generator_constant_q_constraints_p_control,
    add_generator_limits,
    add_q_flow_constraints,
    add_regulator_constraints,
    add_swing_bus_constraints,
    add_voltage_drop_constraints,
)
from distopf.pyomo_models.slack_constraints import (
    add_thermal_slack_constraints,
    add_voltage_slack_constraints,
)


def create_zones_from_edge_names(case: opf.Case, border_edges):
    # ---------------------------------------------------------------------------
    # Zone partitioning (two areas separated by edges)
    # ---------------------------------------------------------------------------

    topo = nx.Graph()
    for _, row in case.branch_data.iterrows():
        topo.add_edge(int(row["fb"]), int(row["tb"]))
    for edge_name in border_edges:
        if edge_name not in case.branch_data["name"].values:
            raise ValueError(f"Edge name '{edge_name}' not found in branch data.")

        sw_row = case.branch_data.loc[case.branch_data["name"] == edge_name].iloc[0]
        topo.remove_edge(int(sw_row["fb"]), int(sw_row["tb"]))
    zones = {
        i: [_id for _id in sorted(c)]
        for i, c in enumerate(nx.connected_components(topo))
    }
    return zones


def add_capacity_expansion_as_fraction_of_load(
    m, case: opf.Case, fraction=0.1, relative_capacity={"PV": 0.5, "BESS": 0.5}
):
    # ensure sum of relative capacities is 1
    # eq. 1 and eq. 2
    total_relative_capacity = sum(relative_capacity.values())
    if total_relative_capacity == 0:
        raise ValueError("Sum of relative capacities cannot be zero.")
    relative_capacity = {
        k: v / total_relative_capacity for k, v in relative_capacity.items()
    }
    m.resource_set = pyo.Set(initialize=relative_capacity.keys())
    total_load_p = case.bus_data[["pl_a", "pl_b", "pl_c"]].sum().sum()
    m.total_capacity_expansion = pyo.Param(initialize=fraction * total_load_p)
    m.relative_capacity = pyo.Param(m.resource_set, initialize=relative_capacity)


def add_pv_parameters(m, curtailment_max=1.0, capacity_factor=0.8):
    m.curtailment_max = pyo.Param(
        initialize=curtailment_max
    )  # max curtailment fraction (1 = fully curtailable)
    m.capacity_factor = pyo.Param(initialize=capacity_factor)


def add_bess_parameters(
    m,
    e_max=4.0,
    soc=0.5,
    discharge_derate=1.0,
    charge_derate=1.0,
):
    m.e_max = pyo.Param(initialize=e_max)
    m.soc_new = pyo.Param(initialize=soc)
    m.discharge_derate = pyo.Param(initialize=discharge_derate)
    m.charging_derate = pyo.Param(initialize=charge_derate)


def _compute_load_proportional_allocation_factors(m, case: opf.Case, zones):
    """
    Compute load-proportional allocation factors for DER injection based on bus loads.
    Returns alpha[_id, r] = fraction of load in zone z for bus _id (0 if not in any zone).
    Eq. 4 and Eq. 5
    """
    m.alpha = pyo.Param(m.bus_set, m.resource_set, initialize=0, mutable=True)
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
                m.alpha[_id, r] = frac
    # Assert eq. 5 Allocation normalization (per zone)
    for z, buses in zones.items():
        for r in m.resource_set:
            alpha_sum = sum(m.alpha[_id, r].value for _id in buses)
            assert abs(alpha_sum - 1) < 1e-6, (
                f"Zone {z} resource {r} alpha sum is {alpha_sum}"
            )


def add_capacity_expansion_variables(m, case: opf.Case, zones):
    # --- portfolio sets and variables (declared before power balance) ----------
    if not hasattr(m, "resource_set"):
        raise AttributeError(
            "Calling add_capacity_expansion_variables requires m.resource_set to be defined. Call add_capacity_expansion_as_fraction_of_load first. Then call add_pv_parameters and add_bess_parameters to set resource-specific parameters."
        )

    m.zone_set = pyo.Set(initialize=list(zones.keys()))
    # Map each bus to its zone (None if not in any zone)
    m.z_of_bus = pyo.Param(m.bus_set, initialize=None, mutable=True)
    for z, buses in zones.items():
        for _id in buses:
            m.z_of_bus[_id] = z
    m.p_max = pyo.Var(
        m.zone_set, m.resource_set, domain=pyo.NonNegativeReals, initialize=0
    )
    m.p_zr = pyo.Var(
        m.zone_set, m.resource_set, m.time_set, domain=pyo.Reals, initialize=0
    )

    m.p_der_inj = pyo.Var(m.bus_phase_set, m.time_set, domain=pyo.Reals, initialize=0)

    _compute_load_proportional_allocation_factors(m, case, zones)


def add_capacity_expansion_p_flow_constraints(m):
    # --- custom power balance (includes p_der_inj) ----------------------------
    def _p_balance(m: LindistModelProtocol, _id, ph, t):
        load = m.p_load[_id, ph, t]
        gen = m.p_gen[_id, ph, t] if (_id, ph, t) in m.p_gen else 0
        bat = m.p_bat[_id, ph, t] if (_id, ph, t) in m.p_bat else 0
        out = sum(
            m.p_flow[tb, ph, t]
            for tb in m.to_bus_map[_id]
            if (tb, ph) in m.branch_phase_set
        )
        return m.p_flow[_id, ph, t] == out + load - gen - bat - m.p_der_inj[_id, ph, t]

    m.power_balance_p = pyo.Constraint(m.branch_phase_set, m.time_set, rule=_p_balance)


def add_zone_capacity_expansion_constraints(m):
    r"""
    Eq. 3
    $$
    \sum_{z \in Z}{ p_{\max,z,r} } = T_{DER} \cdot c_r
    $$
    """
    m.der_budget = pyo.Constraint(
        m.resource_set,
        rule=lambda m, r: sum(m.p_max[z, r] for z in m.zone_set)
        == m.total_capacity_expansion * m.relative_capacity[r],
    )


def add_pv_capacity_constraints(m):
    if "PV" not in m.resource_set:
        return  # Skip if PV is not part of the resource set
    m.pv_set = pyo.Set(initialize=["PV"])
    m.pv_gen_min = pyo.Constraint(
        m.zone_set,
        m.pv_set,
        m.time_set,
        rule=lambda m, z, r, t: m.p_zr[z, r, t]
        >= (1 - m.curtailment_max) * m.capacity_factor * m.p_max[z, r],
    )
    m.pv_gen_max = pyo.Constraint(
        m.zone_set,
        m.pv_set,
        m.time_set,
        rule=lambda m, z, r, t: m.p_zr[z, r, t] <= m.capacity_factor * m.p_max[z, r],
    )


def add_bess_capacity_constraints(m):
    if "BESS" not in m.resource_set:
        return  # Skip if BESS is not part of the resource set
    m.bess_set = pyo.Set(initialize=["BESS"])
    m.bess_charge_max = pyo.Constraint(
        m.zone_set,
        m.bess_set,
        m.time_set,
        rule=lambda m, z, r, t: -m.p_zr[z, r, t] <= m.charging_derate * m.p_max[z, r],
    )
    m.bess_discharge_max = pyo.Constraint(
        m.zone_set,
        m.bess_set,
        m.time_set,
        rule=lambda m, z, r, t: m.p_zr[z, r, t] <= m.discharge_derate * m.p_max[z, r],
    )
    m.bess_energy_1 = pyo.Constraint(
        m.zone_set,
        m.bess_set,
        m.time_set,
        rule=lambda m, z, r, t: m.p_zr[z, r, t] <= m.e_max * m.soc_new / m.delta_t,
    )
    m.bess_energy_2 = pyo.Constraint(
        m.zone_set,
        m.bess_set,
        m.time_set,
        rule=lambda m, z, r, t: -m.p_zr[z, r, t]
        <= m.e_max * (1 - m.soc_new) / m.delta_t,
    )


def add_der_capacity_injection_constraints(m):
    # nodal injection mapping: each bus uses its assigned zone's dispatch
    def _der_inj(m, _id, ph, t):
        z = m.z_of_bus[_id].value
        if z is None:
            return m.p_der_inj[_id, ph, t] == 0
        inj = sum(m.alpha[_id, r] * m.p_zr[z, r, t] for r in m.resource_set)
        return m.p_der_inj[_id, ph, t] == inj / len(m.bus_phases[_id])

    m.der_inj_map = pyo.Constraint(m.bus_phase_set, m.time_set, rule=_der_inj)


def add_capacity_expansion_with_slack_constraints(m, case, zones):
    add_capacity_expansion_variables(m, case, zones)
    add_zone_capacity_expansion_constraints(m)
    add_capacity_expansion_p_flow_constraints(m)
    add_q_flow_constraints(m)
    add_voltage_drop_constraints(m)
    add_swing_bus_constraints(m)
    add_cvr_load_constraints(m)
    add_generator_constant_p_constraints_q_control(m)
    add_generator_constant_q_constraints_p_control(m)
    add_octagonal_inverter_constraints_pq_control(m)
    # add_circular_generator_constraints_pq_control(m)
    add_capacitor_constraints(m)
    add_regulator_constraints(m)
    # add_voltage_limits(m)
    add_generator_limits(m)
    add_battery_constant_q_constraints_p_control(m)
    add_battery_energy_constraints(m)
    add_battery_net_p_bat_equal_phase_constraints(m)
    add_battery_power_limits(m)
    add_battery_soc_limits(m)
    # slack constraints
    add_voltage_slack_constraints(m)
    add_thermal_slack_constraints(m)
    # capacity expansion constraints
    add_pv_capacity_constraints(m)
    add_bess_capacity_constraints(m)
    add_der_capacity_injection_constraints(m)
