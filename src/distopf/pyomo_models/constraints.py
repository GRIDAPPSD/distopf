"""
Constraint functions for DistOPF Pyomo models.

Each function takes a Pyomo ConcreteModel and data, and adds constraints to the model.
Functions are designed to work with models created by create_lindist_model().
"""

import pyomo.environ as pyo  # type: ignore
from distopf.pyomo_models.lindist import ControlVariable
from distopf.pyomo_models.protocol import LindistModelProtocol
from numpy import sqrt

sqrt2 = sqrt(2)
sqrt3 = sqrt(3)


def add_p_flow_constraints(m: LindistModelProtocol) -> None:
    """
    Add LinDistFlow power balance constraints.
    Active power: P_ij = sum(P_jk) + p_L - p_D
    """

    def p_balance_rule(m: LindistModelProtocol, _id, ph, t):
        load = m.p_load[_id, ph, t]
        generation = m.p_gen[_id, ph, t] if (_id, ph, t) in m.p_gen else 0
        p_bat = m.p_bat[_id, ph, t] if (_id, ph, t) in m.p_bat else 0
        incoming_flow = m.p_flow[_id, ph, t]
        outgoing_flows = sum(
            m.p_flow[to_bus, ph, t]
            for to_bus in m.to_bus_map[_id]
            if (to_bus, ph) in m.branch_phase_set
        )
        return incoming_flow == outgoing_flows + load - generation - p_bat

    m.power_balance_p = pyo.Constraint(
        m.branch_phase_set, m.time_set, rule=p_balance_rule
    )


def add_q_flow_constraints(m: LindistModelProtocol) -> None:
    """
    Add LinDistFlow power balance constraints.
    Reactive power: Q_ij = sum(Q_jk) + q_L - q_D - q_C
    """

    def q_balanced_rule(m: LindistModelProtocol, _id, ph, t):
        load = m.q_load[_id, ph, t]
        generation = m.q_gen[_id, ph, t] if (_id, ph, t) in m.q_gen else 0
        q_bat = m.q_bat[_id, ph, t] if (_id, ph, t) in m.q_bat else 0
        capacitor = m.q_cap[_id, ph, t] if (_id, ph, t) in m.q_cap else 0
        incoming_flow = m.q_flow[_id, ph, t]
        outgoing_flows = sum(
            m.q_flow[to_bus, ph, t]
            for to_bus in m.to_bus_map[_id]
            if (to_bus, ph) in m.branch_phase_set
        )
        return incoming_flow == outgoing_flows + load - generation - capacitor - q_bat

    m.power_balance_q = pyo.Constraint(
        m.branch_phase_set, m.time_set, rule=q_balanced_rule
    )


def add_voltage_drop_constraints(m: LindistModelProtocol) -> None:
    """
    Add voltage drop constraints.
    Excludes regulator branches so they can be handled by `add_regulator_constraints`.

    v_j = v_i - sum_q 2*Re[S_ij^pq * (z_ij^pq)*]
    Simplified for LinDistFlow: v_j = v_i - 2*(r*P + x*Q)
    """

    def voltage_drop_rule(m: LindistModelProtocol, _id, ph, t):
        if (_id, ph) in m.reg_phase_set:
            return pyo.Constraint.Skip
        # here, "a" represents the current phase,
        # b an c represent next and previous phase
        a = ph
        _i = "abc".index(a)
        b = "abc"[(_i + 1) % 3]  # next phase. If phase is "a" then "b"
        c = "abc"[(_i - 1) % 3]  # prev phase. If phase is "a" then "c"
        aa = "".join(sorted(a + a))

        voltage_drop = (
            2 * m.r[_id, aa] * m.p_flow[_id, ph, t]
            + 2 * m.x[_id, aa] * m.q_flow[_id, ph, t]
        )
        if (_id, b) in m.branch_phase_set:
            ab = "".join(sorted(a + b))
            voltage_drop += (-m.r[_id, ab] + sqrt3 * m.x[_id, ab]) * m.p_flow[_id, b, t]
            voltage_drop += (-m.x[_id, ab] - sqrt3 * m.r[_id, ab]) * m.q_flow[_id, b, t]
        if (_id, c) in m.branch_phase_set:
            ac = "".join(sorted(a + c))
            voltage_drop += (-m.r[_id, ac] - sqrt3 * m.x[_id, ac]) * m.p_flow[_id, c, t]
            voltage_drop += (-m.x[_id, ac] + sqrt3 * m.r[_id, ac]) * m.q_flow[_id, c, t]

        return m.v2[_id, ph, t] == m.v2[m.from_bus_map[_id], ph, t] - voltage_drop

    m.voltage_drop = pyo.Constraint(
        m.branch_phase_set, m.time_set, rule=voltage_drop_rule
    )


def add_regulator_constraints(m: LindistModelProtocol) -> None:
    """
    vj = v_reg - 2r*pij - 2x*qij
    v_reg = vi*reg_ratio^2
    """

    def regulator_v_drop(m: LindistModelProtocol, _id, ph, t):
        raa = m.r[_id, ph + ph]
        xaa = m.x[_id, ph + ph]
        voltage_drop = 2 * raa * m.p_flow[_id, ph, t] + 2 * xaa * m.q_flow[_id, ph, t]
        return m.v2[_id, ph, t] == m.v2_reg[_id, ph, t] - voltage_drop

    def regulator_rule(m: LindistModelProtocol, _id, ph, t):
        return (
            m.v2_reg[_id, ph, t]
            == m.v2[m.from_bus_map[_id], ph, t] * m.reg_ratio[_id, ph] ** 2
        )

    m.regulator_voltage_drop = pyo.Constraint(
        m.reg_phase_set, m.time_set, rule=regulator_v_drop
    )
    m.regulator_ratio = pyo.Constraint(m.reg_phase_set, m.time_set, rule=regulator_rule)


def add_cvr_load_constraints(m: LindistModelProtocol) -> None:
    """
    Add voltage-dependent load constraints.
    p_L = p_0 + CVR_p * (p_0/2) * (v - 1)
    q_L = q_0 + CVR_q * (q_0/2) * (v - 1)
    """

    def cvr_p_rule(m: LindistModelProtocol, _id, ph, t):
        p_nom = m.p_load_nom[_id, ph, t]
        cvr_p = m.cvr_p[_id, ph]
        return m.p_load[_id, ph, t] == p_nom + cvr_p * p_nom / 2 * (
            m.v2[_id, ph, t] - 1
        )

    def cvr_q_rule(m: LindistModelProtocol, _id, ph, t):
        q_nom = m.q_load_nom[_id, ph, t]
        cvr_q = m.cvr_q[_id, ph]
        return m.q_load[_id, ph, t] == q_nom + cvr_q * q_nom / 2 * (
            m.v2[_id, ph, t] - 1
        )

    m.cvr_p_load = pyo.Constraint(m.bus_phase_set, m.time_set, rule=cvr_p_rule)
    m.cvr_q_load = pyo.Constraint(m.bus_phase_set, m.time_set, rule=cvr_q_rule)


def add_generator_constant_p_constraints(m: LindistModelProtocol) -> None:
    m.constant_p_gen = pyo.Constraint(
        m.gen_phase_set,
        m.time_set,
        rule=lambda m, _id, ph, t: m.p_gen[_id, ph, t] == m.p_gen_nom[_id, ph, t],
    )


def add_generator_constant_q_constraints(m: LindistModelProtocol) -> None:
    m.constant_q_gen = pyo.Constraint(
        m.gen_phase_set,
        m.time_set,
        rule=lambda m, _id, ph, t: m.q_gen[_id, ph, t] == m.q_gen_nom[_id, ph, t],
    )


def add_generator_constant_p_constraints_q_control(m: LindistModelProtocol) -> None:
    def _rule(m: LindistModelProtocol, _id, ph, t):
        if m.gen_control_type[_id, ph] in [ControlVariable.P, ControlVariable.PQ]:
            return pyo.Constraint.Skip
        return m.p_gen[_id, ph, t] == m.p_gen_nom[_id, ph, t]

    m.constant_p_gen = pyo.Constraint(m.gen_phase_set, m.time_set, rule=_rule)


def add_generator_constant_q_constraints_p_control(m: LindistModelProtocol) -> None:
    def _rule(m: LindistModelProtocol, _id, ph, t):
        if m.gen_control_type[_id, ph] in [ControlVariable.Q, ControlVariable.PQ]:
            return pyo.Constraint.Skip
        return m.q_gen[_id, ph, t] == m.q_gen_nom[_id, ph, t]

    m.constant_q_gen = pyo.Constraint(m.gen_phase_set, m.time_set, rule=_rule)


def add_octagonal_inverter_constraints_pq_control(m: LindistModelProtocol) -> None:
    """
    Add octagonal inverter constraints (equation 2.14).

    Linear approximation of circular curve using 8 constraints.
    Only applied to generators with control_variable=="PQ".

    c = sqrt(2) - 1
    c * p_gen + 1 * q_gen <= s_rated
    1 * p_gen + c * q_gen <= s_rated
    1 * p_gen - c * q_gen <= s_rated
    c * p_gen - 1 * q_gen <= s_rated
    """
    c = sqrt2 - 1  # ≈ 0.4142

    # If the P-Q Plane was on a clock:
    # Line from 12:00 to 1:30. Or 90 to 45 deg.
    def _1(m: LindistModelProtocol, _id, ph, t):
        if m.gen_control_type[_id, ph] != ControlVariable.PQ:
            return pyo.Constraint.Skip
        return c * m.p_gen[_id, ph, t] + 1 * m.q_gen[_id, ph, t] <= m.s_rated[_id, ph]

    # Line from 1:30 to 3:00 on a clock. Or 45 to 0 deg.
    def _2(m: LindistModelProtocol, _id, ph, t):
        if m.gen_control_type[_id, ph] != ControlVariable.PQ:
            return pyo.Constraint.Skip
        return 1 * m.p_gen[_id, ph, t] + c * m.q_gen[_id, ph, t] <= m.s_rated[_id, ph]

    # Line from 3:00 to 4:30 on a clock. Or 0 to -45 deg.
    def _3(m: LindistModelProtocol, _id, ph, t):
        if m.gen_control_type[_id, ph] != ControlVariable.PQ:
            return pyo.Constraint.Skip
        return 1 * m.p_gen[_id, ph, t] - c * m.q_gen[_id, ph, t] <= m.s_rated[_id, ph]

    # Line from 4:30 to 6:00 on a clock. Or -45 to -90 deg.
    def _4(m: LindistModelProtocol, _id, ph, t):
        if m.gen_control_type[_id, ph] != ControlVariable.PQ:
            return pyo.Constraint.Skip
        return c * m.p_gen[_id, ph, t] - 1 * m.q_gen[_id, ph, t] <= m.s_rated[_id, ph]

    # Add all octagonal constraints
    m.gen_octagon_1 = pyo.Constraint(m.gen_phase_set, m.time_set, rule=_1)
    m.gen_octagon_2 = pyo.Constraint(m.gen_phase_set, m.time_set, rule=_2)
    m.gen_octagon_3 = pyo.Constraint(m.gen_phase_set, m.time_set, rule=_3)
    m.gen_octagon_4 = pyo.Constraint(m.gen_phase_set, m.time_set, rule=_4)


def add_circular_generator_constraints_pq_control(m: LindistModelProtocol) -> None:
    """
    Add circular generator constraints.

    Uses the exact circular constraint: p_gen² + q_gen² ≤ s_rated²
    Only applied to generators with control_variable=="PQ".
    """

    def _circle(m: LindistModelProtocol, _id, ph, t):
        if m.gen_control_type[_id, ph] != ControlVariable.PQ.value:
            return pyo.Constraint.Skip
        return (
            m.p_gen[_id, ph, t] ** 2 + m.q_gen[_id, ph, t] ** 2
            <= m.s_rated[_id, ph] ** 2
        )

    m.gen_circle_constraint = pyo.Constraint(m.gen_phase_set, m.time_set, rule=_circle)


def add_capacitor_constraints(m: LindistModelProtocol) -> None:
    """
    Add capacitor constraints.
    q_C = q_rated * v^2
    """

    def capacitor_rule(m: LindistModelProtocol, _id, ph, t):
        return m.q_cap[_id, ph, t] == m.q_cap_nom[_id, ph] * m.v2[_id, ph, t]

    m.capacitor_injection = pyo.Constraint(
        m.cap_phase_set, m.time_set, rule=capacitor_rule
    )


def add_swing_bus_constraints(m: LindistModelProtocol) -> None:
    """
    Add swing bus voltage constraints.

    Sets voltage at swing bus to specified values.
    """

    def swing_voltage_rule(m: LindistModelProtocol, _id, ph, t):
        """Fix swing bus voltages"""
        if _id not in m.swing_bus_set:
            return pyo.Constraint.Skip
        return m.v2[_id, ph, t] == m.v_swing[_id, ph, t] ** 2

    m.swing_voltage = pyo.Constraint(
        m.swing_phase_set, m.time_set, rule=swing_voltage_rule
    )


def add_voltage_limits(m: LindistModelProtocol) -> None:
    """Add voltage bounds (for voltage magnitude squared)"""

    def voltage_limits(m: LindistModelProtocol, _id, ph, t):
        return (m.v_min[_id, ph] ** 2, m.v2[_id, ph, t], m.v_max[_id, ph] ** 2)

    m.voltage_limits = pyo.Constraint(m.bus_phase_set, m.time_set, rule=voltage_limits)


def add_generator_limits(m: LindistModelProtocol) -> None:
    """Add generator bounds following the original base.py logic"""

    def p_gen_bounds(m: LindistModelProtocol, _id, ph, t):
        return (
            0,
            m.p_gen[_id, ph, t],
            min(m.p_gen_nom[_id, ph, t], m.s_rated[_id, ph]),
        )

    def q_gen_bounds(m: LindistModelProtocol, _id, ph, t):
        if m.gen_control_type[_id, ph] == ControlVariable.Q:
            q_max = sqrt(max(0, m.s_rated[_id, ph] ** 2 - m.p_gen_nom[_id, ph, t] ** 2))
            return (
                max(-q_max, m.q_gen_min[_id, ph]),
                m.q_gen[_id, ph, t],
                min(q_max, m.q_gen_max[_id, ph]),
            )
        return (
            max(-m.s_rated[_id, ph], m.q_gen_min[_id, ph]),
            m.q_gen[_id, ph, t],
            min(m.s_rated[_id, ph], m.q_gen_max[_id, ph]),
        )

    m.p_gen_limits = pyo.Constraint(m.gen_phase_set, m.time_set, rule=p_gen_bounds)
    m.q_gen_limits = pyo.Constraint(m.gen_phase_set, m.time_set, rule=q_gen_bounds)


# ============ Thermal Line Constraints ================================================
# ======================================================================================


def add_octagonal_thermal_constraints(m: LindistModelProtocol) -> None:
    """
    Add octagonal thermal limit constraints for branch power flows.

    Approximates the circular constraint |S_ij| <= S_max using 8 linear inequalities
    forming an octagon in the P-Q plane. This covers all four quadrants since
    power can flow in either direction.

    The octagon is defined by:
        +/- c*P +/- Q <= S_max
        +/- P +/- c*Q <= S_max

    where c = sqrt(2) - 1 ≈ 0.4142

    Requires branch_data to have columns 'sa_max', 'sb_max', 'sc_max' for
    per-phase apparent power limits. Branches without limits are skipped.
    """
    # Check if thermal limits exist in the model
    if not hasattr(m, "s_branch_max"):
        return

    c = sqrt2 - 1  # ≈ 0.4142

    def _has_thermal_limit(m, _id, ph):
        """Check if branch has a valid thermal limit."""
        limit = pyo.value(m.s_branch_max.get((_id, ph), None))
        return limit is not None and limit > 0

    # Quadrant 1: +P, +Q
    def thermal_1(m: LindistModelProtocol, _id, ph, t):
        if not _has_thermal_limit(m, _id, ph):
            return pyo.Constraint.Skip
        return (
            c * m.p_flow[_id, ph, t] + m.q_flow[_id, ph, t] <= m.s_branch_max[_id, ph]
        )

    def thermal_2(m: LindistModelProtocol, _id, ph, t):
        if not _has_thermal_limit(m, _id, ph):
            return pyo.Constraint.Skip
        return (
            m.p_flow[_id, ph, t] + c * m.q_flow[_id, ph, t] <= m.s_branch_max[_id, ph]
        )

    # Quadrant 4: +P, -Q
    def thermal_3(m: LindistModelProtocol, _id, ph, t):
        if not _has_thermal_limit(m, _id, ph):
            return pyo.Constraint.Skip
        return (
            m.p_flow[_id, ph, t] - c * m.q_flow[_id, ph, t] <= m.s_branch_max[_id, ph]
        )

    def thermal_4(m: LindistModelProtocol, _id, ph, t):
        if not _has_thermal_limit(m, _id, ph):
            return pyo.Constraint.Skip
        return (
            c * m.p_flow[_id, ph, t] - m.q_flow[_id, ph, t] <= m.s_branch_max[_id, ph]
        )

    # Quadrant 3: -P, -Q
    def thermal_5(m: LindistModelProtocol, _id, ph, t):
        if not _has_thermal_limit(m, _id, ph):
            return pyo.Constraint.Skip
        return (
            -c * m.p_flow[_id, ph, t] - m.q_flow[_id, ph, t] <= m.s_branch_max[_id, ph]
        )

    def thermal_6(m: LindistModelProtocol, _id, ph, t):
        if not _has_thermal_limit(m, _id, ph):
            return pyo.Constraint.Skip
        return (
            -m.p_flow[_id, ph, t] - c * m.q_flow[_id, ph, t] <= m.s_branch_max[_id, ph]
        )

    # Quadrant 2: -P, +Q
    def thermal_7(m: LindistModelProtocol, _id, ph, t):
        if not _has_thermal_limit(m, _id, ph):
            return pyo.Constraint.Skip
        return (
            -m.p_flow[_id, ph, t] + c * m.q_flow[_id, ph, t] <= m.s_branch_max[_id, ph]
        )

    def thermal_8(m: LindistModelProtocol, _id, ph, t):
        if not _has_thermal_limit(m, _id, ph):
            return pyo.Constraint.Skip
        return (
            -c * m.p_flow[_id, ph, t] + m.q_flow[_id, ph, t] <= m.s_branch_max[_id, ph]
        )

    m.thermal_limit_1 = pyo.Constraint(m.branch_phase_set, m.time_set, rule=thermal_1)
    m.thermal_limit_2 = pyo.Constraint(m.branch_phase_set, m.time_set, rule=thermal_2)
    m.thermal_limit_3 = pyo.Constraint(m.branch_phase_set, m.time_set, rule=thermal_3)
    m.thermal_limit_4 = pyo.Constraint(m.branch_phase_set, m.time_set, rule=thermal_4)
    m.thermal_limit_5 = pyo.Constraint(m.branch_phase_set, m.time_set, rule=thermal_5)
    m.thermal_limit_6 = pyo.Constraint(m.branch_phase_set, m.time_set, rule=thermal_6)
    m.thermal_limit_7 = pyo.Constraint(m.branch_phase_set, m.time_set, rule=thermal_7)
    m.thermal_limit_8 = pyo.Constraint(m.branch_phase_set, m.time_set, rule=thermal_8)


def add_circular_thermal_constraints(m: LindistModelProtocol) -> None:
    """
    Add circular thermal limit constraints for branch power flows.

    Enforces the exact quadratic constraint:
        P_ij^2 + Q_ij^2 <= S_max^2

    This is a nonlinear (quadratic) constraint requiring a nonlinear solver
    (e.g., IPOPT) or a solver supporting second-order cone constraints.

    Requires branch_data to have columns 'sa_max', 'sb_max', 'sc_max' for
    per-phase apparent power limits. Branches without limits are skipped.
    """
    if not hasattr(m, "s_branch_max"):
        return

    def _has_thermal_limit(m, _id, ph):
        """Check if branch has a valid thermal limit."""
        if (_id, ph) not in m.s_branch_max:
            return False
        limit = pyo.value(m.s_branch_max[_id, ph])
        return limit is not None and limit > 0

    def thermal_circle(m: LindistModelProtocol, _id, ph, t):
        if not _has_thermal_limit(m, _id, ph):
            return pyo.Constraint.Skip
        return (
            m.p_flow[_id, ph, t] ** 2 + m.q_flow[_id, ph, t] ** 2
            <= m.s_branch_max[_id, ph] ** 2
        )

    m.thermal_limit_circle = pyo.Constraint(
        m.branch_phase_set, m.time_set, rule=thermal_circle
    )


# ============ Battery Constraints =====================================================
# ======================================================================================


def add_battery_power_limits(m: LindistModelProtocol) -> None:
    def _d(m: LindistModelProtocol, _id, ph, t):
        return (0, m.p_discharge[_id, t], m.s_bat_rated[_id, ph])

    def _c(m: LindistModelProtocol, _id, ph, t):
        return (0, m.p_charge[_id, t], m.s_bat_rated[_id, ph])

    m.battery_discharging_limits = pyo.Constraint(m.bat_phase_set, m.time_set, rule=_d)
    m.battery_charging_limits = pyo.Constraint(m.bat_phase_set, m.time_set, rule=_c)


def add_battery_soc_limits(m: LindistModelProtocol) -> None:
    def battery_soc_limits(m: LindistModelProtocol, _id, t):
        return (m.soc_min[_id], m.soc[_id, t], m.soc_max[_id])

    m.battery_soc_limits = pyo.Constraint(
        m.bat_set, m.time_set, rule=battery_soc_limits
    )


def add_battery_net_p_bat_constraints(m: LindistModelProtocol) -> None:
    def net_discharge(m: LindistModelProtocol, _id, t):
        p_bat_a = m.p_bat[_id, "a", t] if m.battery_has_phase[_id, "a"] else 0
        p_bat_b = m.p_bat[_id, "b", t] if m.battery_has_phase[_id, "b"] else 0
        p_bat_c = m.p_bat[_id, "c", t] if m.battery_has_phase[_id, "c"] else 0
        return p_bat_a + p_bat_b + p_bat_c == m.p_discharge[_id, t] - m.p_charge[_id, t]

    m.net_discharge = pyo.Constraint(m.bat_phase_set, m.time_set, rule=net_discharge)


def add_battery_net_p_bat_equal_phase_constraints(m: LindistModelProtocol) -> None:
    def net_discharge_equal_phases(m: LindistModelProtocol, _id, ph, t):
        n_phases = m.battery_n_phases[_id]
        return (
            m.p_bat[_id, ph, t]
            == (m.p_discharge[_id, t] - m.p_charge[_id, t]) / n_phases
        )

    m.net_discharge = pyo.Constraint(
        m.bat_phase_set, m.time_set, rule=net_discharge_equal_phases
    )


def add_battery_energy_constraints(m: LindistModelProtocol) -> None:
    def storage(m: LindistModelProtocol, _id, t):
        eta_d = m.discharge_efficiency[_id]
        eta_c = m.charge_efficiency[_id]
        if t == m.start_step:
            soc0 = m.start_soc[_id]
        else:
            soc0 = m.soc[_id, t - 1]
        return (
            m.soc[_id, t] - soc0
            == eta_c * m.delta_t * m.p_charge[_id, t]
            - (1 / eta_d) * m.delta_t * m.p_discharge[_id, t]
        )

    m.storage = pyo.Constraint(m.bat_set, m.time_set, rule=storage)


# def add_battery_phase_equality_constraints(m: LindistModelProtocol) -> None:
#     def _p_equal(m: LindistModelProtocol, _id, ph, t):
#         a = ph
#         _i = "abc".index(a)
#         b = "abc"[(_i + 1) % 3]  # next phase. If phase is "a" then "b"
#         if (_id, b) not in m.bat_phase_set:
#             pyo.Constraint.Skip
#         return m.p_bat[_id, a, t] == m.p_bat[_id, b, t]

#     def _q_equal(m: LindistModelProtocol, _id, ph, t):
#         a = ph
#         _i = "abc".index(a)
#         b = "abc"[(_i + 1) % 3]  # next phase. If phase is "a" then "b"
#         if (_id, b) not in m.bat_phase_set:
#             pyo.Constraint.Skip
#         return m.q_bat[_id, a, t] == m.q_bat[_id, b, t]


def add_battery_constant_q_constraints_p_control(m: LindistModelProtocol) -> None:
    def _rule(m: LindistModelProtocol, _id, ph, t):
        if m.bat_control_type[_id] != ControlVariable.P:
            return pyo.Constraint.Skip
        return m.q_bat[_id, ph, t] == m.q_bat_nom[_id, ph, t]

    m.battery_constant_q_bat = pyo.Constraint(m.bat_phase_set, m.time_set, rule=_rule)


def add_circular_battery_constraints_pq_control(m: LindistModelProtocol) -> None:
    """
    Add circular battery apparent power constraints.

    Enforces the exact quadratic constraint:
        P_bat^2 + Q_bat^2 <= S_rated^2

    This is a nonlinear (quadratic) constraint requiring a nonlinear solver
    (e.g., IPOPT) or a solver supporting second-order cone constraints.
    """

    def bat_circle(m: LindistModelProtocol, _id, ph, t):
        if m.bat_control_type[_id] != ControlVariable.PQ:
            return pyo.Constraint.Skip
        return (
            m.p_bat[_id, ph, t] ** 2 + m.q_bat[_id, ph, t] ** 2
            <= m.s_bat_rated[_id, ph] ** 2
        )

    m.bat_circle_constraint = pyo.Constraint(
        m.bat_phase_set, m.time_set, rule=bat_circle
    )


def add_circular_battery_constraints(m: LindistModelProtocol) -> None:
    """
    Add circular battery apparent power constraints.

    Enforces the exact quadratic constraint:
        P_bat^2 + Q_bat^2 <= S_rated^2

    This is a nonlinear (quadratic) constraint requiring a nonlinear solver
    (e.g., IPOPT) or a solver supporting second-order cone constraints.
    """

    def bat_circle(m: LindistModelProtocol, _id, ph, t):
        return (
            m.p_bat[_id, ph, t] ** 2 + m.q_bat[_id, ph, t] ** 2
            <= m.s_bat_rated[_id, ph] ** 2
        )

    m.bat_circle_constraint = pyo.Constraint(
        m.bat_phase_set, m.time_set, rule=bat_circle
    )


# ============ Capacitor Constraints (Standard and MI) =================================
# ======================================================================================


def add_capacitor_constraints_auto(m: LindistModelProtocol) -> None:
    """
    Automatically add appropriate capacitor constraints based on model configuration.

    If cap_mi_enabled: adds McCormick envelope constraints
    Otherwise: adds standard voltage-dependent capacitor model
    """
    if getattr(m, "cap_mi_enabled", False):
        add_capacitor_mi_constraints(m)
        add_capacitor_mccormick_constraints(m)
        add_capacitor_z_bounds(m)
    else:
        add_capacitor_constraints(m)


def add_capacitor_mi_constraints(m: LindistModelProtocol) -> None:
    """
    Add mixed-integer capacitor constraints using McCormick envelope.

    q_cap = q_cap_nom * z_cap

    where z_cap represents the product u_cap * v2.
    """

    def capacitor_q_rule(m: LindistModelProtocol, _id, ph, t):
        return m.q_cap[_id, ph, t] == m.q_cap_nom[_id, ph] * m.z_cap[_id, ph, t]

    m.capacitor_mi_injection = pyo.Constraint(
        m.cap_phase_set, m.time_set, rule=capacitor_q_rule
    )


def add_capacitor_mccormick_constraints(m: LindistModelProtocol) -> None:
    """
    Add McCormick envelope constraints to linearize z_cap = u_cap * v2.

    For binary u in {0,1} and continuous v2 in [v_min^2, v_max^2]:
        z <= v_max^2 * u           (when u=0, z=0)
        z <= v2                    (z bounded by v2)
        z >= v2 - v_max^2 * (1-u)  (when u=1, z=v2)
        z >= v_min^2 * u           (when u=1, z >= v_min^2)
    """

    def mccormick_upper_1(m: LindistModelProtocol, _id, ph, t):
        """z_cap <= v_max^2 * u_cap"""
        v2_max = m.v_max[_id, ph] ** 2
        return m.z_cap[_id, ph, t] <= v2_max * m.u_cap[_id, ph, t]

    def mccormick_upper_2(m: LindistModelProtocol, _id, ph, t):
        """z_cap <= v2"""
        return m.z_cap[_id, ph, t] <= m.v2[_id, ph, t]

    def mccormick_lower_1(m: LindistModelProtocol, _id, ph, t):
        """z_cap >= v2 - v_max^2 * (1 - u_cap)"""
        v2_max = m.v_max[_id, ph] ** 2
        return m.z_cap[_id, ph, t] >= m.v2[_id, ph, t] - v2_max * (
            1 - m.u_cap[_id, ph, t]
        )

    def mccormick_lower_2(m: LindistModelProtocol, _id, ph, t):
        """z_cap >= v_min^2 * u_cap"""
        v2_min = m.v_min[_id, ph] ** 2
        return m.z_cap[_id, ph, t] >= v2_min * m.u_cap[_id, ph, t]

    m.cap_mccormick_u1 = pyo.Constraint(
        m.cap_phase_set, m.time_set, rule=mccormick_upper_1
    )
    m.cap_mccormick_u2 = pyo.Constraint(
        m.cap_phase_set, m.time_set, rule=mccormick_upper_2
    )
    m.cap_mccormick_l1 = pyo.Constraint(
        m.cap_phase_set, m.time_set, rule=mccormick_lower_1
    )
    m.cap_mccormick_l2 = pyo.Constraint(
        m.cap_phase_set, m.time_set, rule=mccormick_lower_2
    )


def add_capacitor_z_bounds(m: LindistModelProtocol) -> None:
    """
    Add explicit bounds on z_cap auxiliary variable.

    0 <= z_cap <= v_max^2
    """

    def z_cap_bounds(m: LindistModelProtocol, _id, ph, t):
        v2_max = m.v_max[_id, ph] ** 2
        return (0, m.z_cap[_id, ph, t], v2_max)

    m.z_cap_bounds = pyo.Constraint(m.cap_phase_set, m.time_set, rule=z_cap_bounds)


# ============ Regulator Constraints (Standard and MI) =================================
# ======================================================================================


def add_regulator_constraints_auto(m: LindistModelProtocol) -> None:
    """
    Automatically add appropriate regulator constraints based on model configuration.

    If reg_mi_enabled: adds Big-M tap selection constraints
    Otherwise: adds standard fixed-ratio regulator model
    """
    if getattr(m, "reg_mi_enabled", False):
        add_regulator_tap_sos1_constraints(m)
        add_regulator_mi_with_impedance_constraints(m)
    else:
        add_regulator_constraints(m)


def add_regulator_tap_sos1_constraints(m: LindistModelProtocol) -> None:
    """
    Add SOS1 (Special Ordered Set Type 1) constraint: exactly one tap position must be selected per regulator.

    sum_k(u_reg[id, ph, k, t]) == 1 for all (id, ph, t)
    """

    def sos1_rule(m: LindistModelProtocol, _id, ph, t):
        return sum(m.u_reg[_id, ph, k, t] for k in m.tap_set) == 1

    m.reg_tap_sos1 = pyo.Constraint(m.reg_phase_set, m.time_set, rule=sos1_rule)


def add_regulator_mi_with_impedance_constraints(m: LindistModelProtocol) -> None:
    """
    Full regulator model with tap selection and line impedance.

    Uses Big-M to enforce: v2_reg = tap_ratio^2 * v_i when tap k is selected
    Then: v_j = v2_reg - 2*r*p_ij - 2*x*q_ij
    """

    def reg_tap_upper(m: LindistModelProtocol, _id, ph, k, t):
        i = m.from_bus_map[_id]
        return m.v2_reg[_id, ph, t] - m.tap_ratio_squared[k] * m.v2[
            i, ph, t
        ] <= m.reg_big_m * (1 - m.u_reg[_id, ph, k, t])

    def reg_tap_lower(m: LindistModelProtocol, _id, ph, k, t):
        i = m.from_bus_map[_id]
        return m.v2_reg[_id, ph, t] - m.tap_ratio_squared[k] * m.v2[
            i, ph, t
        ] >= -m.reg_big_m * (1 - m.u_reg[_id, ph, k, t])

    def reg_v_drop_rule(m: LindistModelProtocol, _id, ph, t):
        raa = m.r[_id, ph + ph]
        xaa = m.x[_id, ph + ph]
        voltage_drop = 2 * raa * m.p_flow[_id, ph, t] + 2 * xaa * m.q_flow[_id, ph, t]
        return m.v2[_id, ph, t] == m.v2_reg[_id, ph, t] - voltage_drop

    m.reg_tap_upper = pyo.Constraint(
        m.reg_phase_set, m.tap_set, m.time_set, rule=reg_tap_upper
    )
    m.reg_tap_lower = pyo.Constraint(
        m.reg_phase_set, m.tap_set, m.time_set, rule=reg_tap_lower
    )
    m.reg_v_drop_mi = pyo.Constraint(m.reg_phase_set, m.time_set, rule=reg_v_drop_rule)


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

    def tap_change_limit_upper(m: LindistModelProtocol, _id, ph, t):
        if t == pyo.value(m.start_step):
            return pyo.Constraint.Skip
        tap_t = sum(k * m.u_reg[_id, ph, k, t] for k in m.tap_set)
        tap_prev = sum(k * m.u_reg[_id, ph, k, t - 1] for k in m.tap_set)
        return tap_t - tap_prev <= max_tap_change

    def tap_change_limit_lower(m: LindistModelProtocol, _id, ph, t):
        if t == pyo.value(m.start_step):
            return pyo.Constraint.Skip
        tap_t = sum(k * m.u_reg[_id, ph, k, t] for k in m.tap_set)
        tap_prev = sum(k * m.u_reg[_id, ph, k, t - 1] for k in m.tap_set)
        return tap_t - tap_prev >= -max_tap_change

    m.reg_tap_change_upper = pyo.Constraint(
        m.reg_phase_set, m.time_set, rule=tap_change_limit_upper
    )
    m.reg_tap_change_lower = pyo.Constraint(
        m.reg_phase_set, m.time_set, rule=tap_change_limit_lower
    )
