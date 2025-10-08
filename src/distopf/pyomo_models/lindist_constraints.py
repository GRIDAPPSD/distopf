"""
Constraint functions for DistOPF Pyomo models.

Each function takes a Pyomo ConcreteModel and data, and adds constraints to the model.
Functions are designed to work with models created by create_lindist_model().
"""

import pyomo.environ as pyo
import pandas as pd
from typing import Optional
import numpy as np
from distopf.importer import Case


def add_power_flow_constraints(
    m: pyo.ConcreteModel,
    bus_data: pd.DataFrame,
    branch_data: pd.DataFrame,
    gen_data: Optional[pd.DataFrame] = None,
    cap_data: Optional[pd.DataFrame] = None,
) -> None:
    """
    Add LinDistFlow power balance constraints (equations 2.9a and 2.9b).

    Active power: P_ij = sum(P_jk) + p_L - p_D
    Reactive power: Q_ij = sum(Q_jk) + q_L - q_D - q_C
    """

    # Create lookup dictionaries for efficiency
    gen_lookup = {}
    if gen_data is not None and len(gen_data) > 0:
        for _, row in gen_data.iterrows():
            gen_lookup[row.id] = row

    cap_lookup = {}
    if cap_data is not None and len(cap_data) > 0:
        for _, row in cap_data.iterrows():
            cap_lookup[row.id] = row

    # Create from-to bus mapping
    branch_lookup = {}
    for _, row in branch_data.iterrows():
        if row.tb not in branch_lookup:
            branch_lookup[row.tb] = []
        branch_lookup[row.tb].append(row.fb)

    def active_power_balance_rule(m, bus_id, phase):
        """Active power balance at each bus-phase"""
        bus_row = bus_data[bus_data.id == bus_id].iloc[0]

        # Load at this bus-phase
        load = getattr(bus_row, f"pl_{phase}", 0.0)

        # Generation at this bus-phase
        generation = 0
        if bus_id in gen_lookup and (bus_id, phase) in m.gen_phase_set:
            generation = m.p_gen[bus_id, phase]
        elif bus_id in gen_lookup:
            # Use constant generation if not a control variable
            gen_row = gen_lookup[bus_id]
            generation = getattr(gen_row, f"p{phase}", 0.0)

        # Power flow balance
        if bus_id == 1:  # Swing bus
            return pyo.Constraint.Skip

        # Incoming power flow (from parent bus)
        incoming_flow = 0
        if bus_id in branch_lookup:
            for from_bus in branch_lookup[bus_id]:
                if (bus_id, phase) in m.branch_phase_set:
                    incoming_flow += m.p_flow[bus_id, phase]
                    break  # Should only be one parent in radial network

        # Outgoing power flows (to child buses)
        outgoing_flows = sum(
            m.p_flow[child_bus, phase]
            for child_bus, p in m.branch_phase_set
            if p == phase
            and child_bus in branch_data[branch_data.fb == bus_id].tb.values
        )

        return incoming_flow == outgoing_flows + load - generation

    def reactive_power_balance_rule(m, bus_id, phase):
        """Reactive power balance at each bus-phase"""
        bus_row = bus_data[bus_data.id == bus_id].iloc[0]

        # Load at this bus-phase
        load = getattr(bus_row, f"ql_{phase}", 0.0)

        # Generation at this bus-phase
        generation = 0
        if bus_id in gen_lookup and (bus_id, phase) in m.gen_phase_set:
            generation = m.q_gen[bus_id, phase]
        elif bus_id in gen_lookup:
            # Use constant generation if not a control variable
            gen_row = gen_lookup[bus_id]
            generation = getattr(gen_row, f"q{phase}", 0.0)

        # Capacitor injection
        capacitor = 0
        if bus_id in cap_lookup and (bus_id, phase) in m.cap_phase_set:
            capacitor = m.q_cap[bus_id, phase]
        elif bus_id in cap_lookup:
            # Use constant capacitor if not a control variable
            cap_row = cap_lookup[bus_id]
            capacitor = getattr(cap_row, f"q_{phase}", 0.0)

        # Power flow balance
        if bus_id == 1:  # Swing bus
            return pyo.Constraint.Skip

        # Incoming power flow (from parent bus)
        incoming_flow = 0
        if bus_id in branch_lookup:
            for from_bus in branch_lookup[bus_id]:
                if (bus_id, phase) in m.branch_phase_set:
                    incoming_flow += m.q_flow[bus_id, phase]
                    break

        # Outgoing power flows (to child buses)
        outgoing_flows = sum(
            m.q_flow[child_bus, phase]
            for child_bus, p in m.branch_phase_set
            if p == phase
            and child_bus in branch_data[branch_data.fb == bus_id].tb.values
        )

        return incoming_flow == outgoing_flows + load - generation - capacitor

    # Add constraints to model
    m.power_balance_p = pyo.Constraint(m.bus_phase_set, rule=active_power_balance_rule)
    m.power_balance_q = pyo.Constraint(
        m.bus_phase_set, rule=reactive_power_balance_rule
    )


def add_voltage_drop_constraints(m: pyo.ConcreteModel, case: Case) -> None:
    """
    Add voltage drop constraints (equation 2.9c).

    v_j = v_i - sum_q 2*Re[S_ij^pq * (z_ij^pq)*]
    Simplified for LinDistFlow: v_j = v_i - 2*(r*P + x*Q)
    """

    # Create from-bus lookup
    branch_lookup = {}
    for _, row in case.branch_data.iterrows():
        branch_lookup[row.tb] = row.fb

    def voltage_drop_rule(m, bus_id, phase):
        """Voltage drop constraint for each bus-phase"""
        if bus_id == 1:  # Skip swing bus
            return pyo.Constraint.Skip

        if bus_id not in branch_lookup:
            return pyo.Constraint.Skip

        if case.reg_data is not None:
            if bus_id in case.reg_data.tb:
                return pyo.Constraint.Skip

        from_bus = branch_lookup[bus_id]

        # Get branch data
        branch_row = case.branch_data[(case.branch_data.tb == bus_id)].iloc[0]

        # Calculate voltage drop for all phase combinations
        voltage_drop = 0

        for other_phase in ["a", "b", "c"]:
            if (bus_id, other_phase) in m.branch_phase_set:
                # Get impedance values
                phase_pair = [phase, other_phase].sort()
                phase_pair = "".join(phase_pair)
                r_val = getattr(branch_row, f"r{phase_pair}", 0.0)
                x_val = getattr(branch_row, f"x{phase_pair}", 0.0)

                # Add voltage drop contribution
                voltage_drop += 2 * (
                    r_val * m.p_flow[bus_id, other_phase]
                    + x_val * m.q_flow[bus_id, other_phase]
                )

        return m.v[bus_id, phase] == m.v[from_bus, phase] - voltage_drop

    m.voltage_drop = pyo.Constraint(m.branch_phase_set, rule=voltage_drop_rule)


# TODO finish writing this function
# TODO add a good way to lookup from_bus from to_bus
# def add_regulator_constraints(m: pyo.ConcreteModel, reg_data: pd.DataFrame) -> None:
#     """
#     vj - vx + 2r*pij + 2x*qij
#     vx - vi*reg_ratio^2
#     """

#     def regulator_voltage_drop_rule(m, bus_id, phase):
#         reg_row = reg_data[reg_data.tb == bus_id].iloc[0]
#         return m.v[bus_id, phase] - m.v_reg[bus_id, phase] + 2

#     def regulator_rule(m, bus_id, phase):
#         return


def add_cvr_load_constraints(m: pyo.ConcreteModel, bus_data: pd.DataFrame) -> None:
    """
    Add voltage-dependent load constraints (equation 2.10).

    p_L = p_0 + CVR_p * (p_0/2) * (v - 1)
    q_L = q_0 + CVR_q * (q_0/2) * (v - 1)

    Note: This modifies the power balance constraints, so should be used
    instead of simple constant loads in add_power_flow_constraints.
    """

    def cvr_active_load_rule(m, bus_id, phase):
        """CVR constraint for active power loads"""
        bus_row = bus_data[bus_data.id == bus_id].iloc[0]

        p_nom = getattr(bus_row, f"pl_{phase}", 0.0)
        cvr_p = getattr(bus_row, "cvr_p", 0.0)

        if p_nom == 0:
            return pyo.Constraint.Skip

        # p_L = p_0 + CVR_p * (p_0/2) * (v - 1)
        return m.p_load[bus_id, phase] == p_nom + cvr_p * (p_nom / 2) * (
            m.v[bus_id, phase] - 1
        )

    def cvr_reactive_load_rule(m, bus_id, phase):
        """CVR constraint for reactive power loads"""
        bus_row = bus_data[bus_data.id == bus_id].iloc[0]

        q_nom = getattr(bus_row, f"ql_{phase}", 0.0)
        cvr_q = getattr(bus_row, "cvr_q", 0.0)

        if q_nom == 0:
            return pyo.Constraint.Skip

        # q_L = q_0 + CVR_q * (q_0/2) * (v - 1)
        return m.q_load[bus_id, phase] == q_nom + cvr_q * (q_nom / 2) * (
            m.v[bus_id, phase] - 1
        )

    # Add load variables if they don't exist
    if not hasattr(m, "p_load"):
        m.p_load = pyo.Var(m.bus_phase_set)
    if not hasattr(m, "q_load"):
        m.q_load = pyo.Var(m.bus_phase_set)

    m.cvr_p_load = pyo.Constraint(m.bus_phase_set, rule=cvr_active_load_rule)
    m.cvr_q_load = pyo.Constraint(m.bus_phase_set, rule=cvr_reactive_load_rule)


def add_generator_capability_constraints(
    m: pyo.ConcreteModel, gen_data: pd.DataFrame
) -> None:
    """
    Add generator capability constraints (equations 2.12, 2.13, 2.14).

    Supports:
    - Simple P limits: 0 ≤ p ≤ P_max
    - Simple Q limits: Q_min ≤ q ≤ Q_max
    - Octagonal inverter approximation
    """

    if len(gen_data) == 0:
        return

    gen_lookup = {}
    for _, row in gen_data.iterrows():
        gen_lookup[row.id] = row

    # Simple active power limits (equation 2.13)
    def p_gen_lower_rule(m, bus_id, phase):
        return m.p_gen[bus_id, phase] >= 0

    def p_gen_upper_rule(m, bus_id, phase):
        gen_row = gen_lookup[bus_id]
        p_max = getattr(gen_row, f"p{phase}", 0.0)  # Use as limit if control variable
        return m.p_gen[bus_id, phase] <= p_max

    m.p_gen_lower = pyo.Constraint(m.gen_phase_set, rule=p_gen_lower_rule)
    m.p_gen_upper = pyo.Constraint(m.gen_phase_set, rule=p_gen_upper_rule)


def add_octagonal_inverter_constraints(
    m: pyo.ConcreteModel, gen_data: pd.DataFrame
) -> None:
    """
    Add octagonal inverter capability constraints (equation 2.14).

    Linear approximation of circular capability curve using 8 constraints.
    Only applied to generators with control_variable=="PQ".
    """

    if len(gen_data) == 0:
        return

    gen_lookup = {}
    for _, row in gen_data.iterrows():
        gen_lookup[row.id] = row

    sqrt2 = np.sqrt(2)
    coef = sqrt2 - 1  # ≈ 0.4142

    def octagon_constraint_1_rule(m, bus_id, phase):
        """√2 * p + (√2-1) * q ≤ √2 * s_rated"""
        gen_row = gen_lookup[bus_id]

        # Only apply to PQ controlled generators
        if getattr(gen_row, "control_variable", "") != "PQ":
            return pyo.Constraint.Skip

        s_rated = getattr(gen_row, f"s{phase}_max", 1000.0)
        return (
            sqrt2 * m.p_gen[bus_id, phase] + coef * m.q_gen[bus_id, phase]
            <= sqrt2 * s_rated
        )

    def octagon_constraint_2_rule(m, bus_id, phase):
        """√2 * p - (√2-1) * q ≤ √2 * s_rated"""
        gen_row = gen_lookup[bus_id]

        # Only apply to PQ controlled generators
        if getattr(gen_row, "control_variable", "") != "PQ":
            return pyo.Constraint.Skip

        s_rated = getattr(gen_row, f"s{phase}_max", 1000.0)
        return (
            sqrt2 * m.p_gen[bus_id, phase] - coef * m.q_gen[bus_id, phase]
            <= sqrt2 * s_rated
        )

    def octagon_constraint_3_rule(m, bus_id, phase):
        """(√2-1) * p + q ≤ s_rated"""
        gen_row = gen_lookup[bus_id]

        # Only apply to PQ controlled generators
        if getattr(gen_row, "control_variable", "") != "PQ":
            return pyo.Constraint.Skip

        s_rated = getattr(gen_row, f"s{phase}_max", 1000.0)
        return coef * m.p_gen[bus_id, phase] + m.q_gen[bus_id, phase] <= s_rated

    def octagon_constraint_4_rule(m, bus_id, phase):
        """(√2-1) * p - q ≤ s_rated"""
        gen_row = gen_lookup[bus_id]

        # Only apply to PQ controlled generators
        if getattr(gen_row, "control_variable", "") != "PQ":
            return pyo.Constraint.Skip

        s_rated = getattr(gen_row, f"s{phase}_max", 1000.0)
        return coef * m.p_gen[bus_id, phase] - m.q_gen[bus_id, phase] <= s_rated

    def octagon_constraint_5_rule(m, bus_id, phase):
        """p ≥ 0 (right half plane)"""
        gen_row = gen_lookup[bus_id]

        # Only apply to PQ controlled generators
        if getattr(gen_row, "control_variable", "") != "PQ":
            return pyo.Constraint.Skip

        return m.p_gen[bus_id, phase] >= 0

    # Add all octagonal constraints
    m.octagon_1 = pyo.Constraint(m.gen_phase_set, rule=octagon_constraint_1_rule)
    m.octagon_2 = pyo.Constraint(m.gen_phase_set, rule=octagon_constraint_2_rule)
    m.octagon_3 = pyo.Constraint(m.gen_phase_set, rule=octagon_constraint_3_rule)
    m.octagon_4 = pyo.Constraint(m.gen_phase_set, rule=octagon_constraint_4_rule)
    m.octagon_5 = pyo.Constraint(m.gen_phase_set, rule=octagon_constraint_5_rule)


def add_circular_generator_constraints(
    m: pyo.ConcreteModel, gen_data: pd.DataFrame
) -> None:
    """
    Add circular generator capability constraints.

    Uses the exact circular constraint: p_gen² + q_gen² ≤ s_rated²
    Only applied to generators with control_variable=="PQ".
    """

    if len(gen_data) == 0:
        return

    gen_lookup = {}
    for _, row in gen_data.iterrows():
        gen_lookup[row.id] = row

    def circular_constraint_rule(m, bus_id, phase):
        """
        Circular capability constraint: p² + q² ≤ s_rated²
        Only for PQ controlled generators.
        """
        gen_row = gen_lookup[bus_id]

        # Only apply to PQ controlled generators
        if getattr(gen_row, "control_variable", "") != "PQ":
            return pyo.Constraint.Skip

        s_rated = getattr(gen_row, f"s{phase}_max", 1000.0)
        return m.p_gen[bus_id, phase] ** 2 + m.q_gen[bus_id, phase] ** 2 <= s_rated**2

    def p_gen_non_negative_rule(m, bus_id, phase):
        """
        Ensure non-negative active power generation (right half-plane operation).
        Only for PQ controlled generators.
        """
        gen_row = gen_lookup[bus_id]

        # Only apply to PQ controlled generators
        if getattr(gen_row, "control_variable", "") != "PQ":
            return pyo.Constraint.Skip

        return m.p_gen[bus_id, phase] >= 0

    # Add circular capability constraints
    m.circular_capability = pyo.Constraint(
        m.gen_phase_set, rule=circular_constraint_rule
    )
    m.p_gen_non_negative = pyo.Constraint(m.gen_phase_set, rule=p_gen_non_negative_rule)


def add_capacitor_constraints(
    m: pyo.ConcreteModel, cap_data: pd.DataFrame, bus_data: pd.DataFrame
) -> None:
    """
    Add capacitor constraints (equation 2.17).

    For switched capacitors: q_C = u_cap * q_rated * v
    For fixed capacitors: q_C = q_rated * v
    """

    if len(cap_data) == 0:
        return

    cap_lookup = {}
    for _, row in cap_data.iterrows():
        cap_lookup[row.id] = row

    def capacitor_rule(m, bus_id, phase):
        """Capacitor reactive power injection"""
        cap_row = cap_lookup[bus_id]
        q_rated = getattr(cap_row, f"q_{phase}", 0.0)

        # For now, assume fixed capacitors (no switching)
        # q_C = q_rated * v
        return m.q_cap[bus_id, phase] == q_rated * m.v[bus_id, phase]

    m.capacitor_injection = pyo.Constraint(m.cap_phase_set, rule=capacitor_rule)


def add_thermal_limits(m: pyo.ConcreteModel, branch_data: pd.DataFrame) -> None:
    """
    Add thermal limit constraints (equation 2.18a).

    |I|² ≤ I_rated²
    Approximated as: P² + Q² ≤ (V * I_rated)²
    """

    def thermal_limit_rule(m, bus_id, phase):
        """Thermal limit for each branch"""
        # Find the branch data for this bus
        branch_row = None
        for _, row in branch_data.iterrows():
            if row.tb == bus_id:
                branch_row = row
                break

        if branch_row is None:
            return pyo.Constraint.Skip

        # Get current rating (if available)
        i_rated = getattr(
            branch_row, "i_rated", 1000.0
        )  # Default high value if not specified
        v_base = getattr(branch_row, "v_ln_base", 1.0)

        # Thermal limit: P² + Q² ≤ (V_base * I_rated)²
        return (
            m.p_flow[bus_id, phase] ** 2 + m.q_flow[bus_id, phase] ** 2
            <= (v_base * i_rated) ** 2
        )

    m.thermal_limits = pyo.Constraint(m.branch_phase_set, rule=thermal_limit_rule)


def add_swing_bus_constraints(m: pyo.ConcreteModel, bus_data: pd.DataFrame) -> None:
    """
    Add swing bus voltage constraints.

    Sets voltage at swing bus to specified values.
    """

    swing_buses = bus_data[bus_data.bus_type == "SWING"]

    def swing_voltage_rule(m, bus_id, phase):
        """Fix swing bus voltages"""
        if bus_id not in swing_buses.id.values:
            return pyo.Constraint.Skip

        bus_row = swing_buses[swing_buses.id == bus_id].iloc[0]
        v_specified = getattr(bus_row, f"v_{phase}", 1.0)

        # Voltage magnitude squared
        return m.v[bus_id, phase] == v_specified**2

    m.swing_voltage = pyo.Constraint(m.bus_phase_set, rule=swing_voltage_rule)
