"""
Constraint functions for DistOPF Pyomo models.

Each function takes a Pyomo ConcreteModel and data, and adds constraints to the model.
"""

import pyomo.environ as pyo  # type: ignore
from distopf.pyomo_models.model_types import ControlVariable
from distopf.pyomo_models.protocol import LindistModelProtocol
from numpy import sqrt

sqrt2 = sqrt(2)
sqrt3 = sqrt(3)


def add_voltage_limits(m: LindistModelProtocol) -> None:
    """Add voltage bounds (for voltage magnitude squared)"""

    def voltage_limits(m: LindistModelProtocol, _id, ph, t):
        return (m.v_min[_id, ph] ** 2, m.v2[_id, ph, t], m.v_max[_id, ph] ** 2)

    m.voltage_limits = pyo.Constraint(m.bus_phase_set, m.time_set, rule=voltage_limits)


# ============ Bus Device Injection Models =============================================
# ======================================================================================


# Loads ------------------------------------------------------------------------


def add_cvr_load_constraints(
    m: LindistModelProtocol, free_boundary_loads: bool = False
) -> None:
    """
    Add voltage-dependent load constraints.

    OUT buses represent downstream-area aggregate loads. Their load variables
    must remain free when boundary loads are coordinated externally.
    """

    def cvr_p_rule(m: LindistModelProtocol, _id, ph, t):
        if free_boundary_loads and _id in m.boundary_out_set:
            return pyo.Constraint.Skip
        p_nom = m.p_load_nom[_id, ph, t]
        cvr_p = m.cvr_p[_id, ph]
        return m.p_load[_id, ph, t] == p_nom + cvr_p * p_nom / 2 * (
            m.v2[_id, ph, t] - 1
        )

    def cvr_q_rule(m: LindistModelProtocol, _id, ph, t):
        if free_boundary_loads and _id in m.boundary_out_set:
            return pyo.Constraint.Skip
        q_nom = m.q_load_nom[_id, ph, t]
        cvr_q = m.cvr_q[_id, ph]
        return m.q_load[_id, ph, t] == q_nom + cvr_q * q_nom / 2 * (
            m.v2[_id, ph, t] - 1
        )

    m.cvr_p_load = pyo.Constraint(m.bus_phase_set, m.time_set, rule=cvr_p_rule)
    m.cvr_q_load = pyo.Constraint(m.bus_phase_set, m.time_set, rule=cvr_q_rule)


# Generators ------------------------------------------------------------------------


def add_generator_limits(m: LindistModelProtocol) -> None:
    """Add generator bounds following the original base.py logic"""

    def p_gen_bounds(m: LindistModelProtocol, _id, ph, t):
        if m.gen_control_type[_id, ph] == ControlVariable.NONE:
            return pyo.Constraint.Skip
        return (
            0,
            m.p_gen[_id, ph, t],
            min(m.p_gen_nom[_id, ph, t], m.s_rated[_id, ph]),
        )

    def q_gen_bounds(m: LindistModelProtocol, _id, ph, t):
        if m.gen_control_type[_id, ph] == ControlVariable.NONE:
            return pyo.Constraint.Skip
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
        ct = m.gen_control_type[_id, ph]
        if ct in (ControlVariable.NONE, ControlVariable.Q):
            return m.p_gen[_id, ph, t] == m.p_gen_nom[_id, ph, t]
        return pyo.Constraint.Skip

    m.constant_p_gen = pyo.Constraint(m.gen_phase_set, m.time_set, rule=_rule)


def add_generator_constant_q_constraints_p_control(m: LindistModelProtocol) -> None:
    def _rule(m: LindistModelProtocol, _id, ph, t):
        ct = m.gen_control_type[_id, ph]
        if ct in (ControlVariable.NONE, ControlVariable.P):
            return m.q_gen[_id, ph, t] == m.q_gen_nom[_id, ph, t]
        return pyo.Constraint.Skip

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
        if m.gen_control_type[_id, ph] != ControlVariable.PQ:
            return pyo.Constraint.Skip
        return (
            m.p_gen[_id, ph, t] ** 2 + m.q_gen[_id, ph, t] ** 2
            <= m.s_rated[_id, ph] ** 2
        )

    m.gen_circle_constraint = pyo.Constraint(m.gen_phase_set, m.time_set, rule=_circle)


# Capacitors ------------------------------------------------------------------------
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
        """Fix swing bus voltages.

        `m.v_swing` is stored as voltage magnitude (p.u.), while `m.v2` is
        voltage magnitude squared.
        """
        if _id not in m.swing_bus_set:
            return pyo.Constraint.Skip
        return m.v2[_id, ph, t] == m.v_swing[_id, ph, t] ** 2

    m.swing_voltage = pyo.Constraint(
        m.swing_phase_set, m.time_set, rule=swing_voltage_rule
    )


# Battery Constraints ----------------------------------------------------------------


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


#  Capacitor Constraints (Standard and MI) ---------------------------------------------


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

    Requires branch_data to have columns for branch apparent power limits:
    primary phases: 's_a_max', 's_b_max', 's_c_max'
    triplex phases: 's_s1_max', 's_s2_max' (legacy: 's1_max', 's2_max')
    Branches without limits are skipped.
    """
    # Check if thermal limits exist in the model
    if not hasattr(m, "s_branch_max"):
        return

    c = sqrt2 - 1  # ≈ 0.4142

    def _has_thermal_limit(m, fb, tb, ph):
        """Check if branch has a valid thermal limit."""
        limit = pyo.value(m.s_branch_max.get((fb, tb, ph), None))
        return limit is not None and limit > 0

    # Quadrant 1: +P, +Q
    def thermal_1(m: LindistModelProtocol, fb, tb, ph, t):
        if not _has_thermal_limit(m, fb, tb, ph):
            return pyo.Constraint.Skip
        return (
            c * m.p_flow[fb, tb, ph, t] + m.q_flow[fb, tb, ph, t]
            <= m.s_branch_max[fb, tb, ph]
        )

    def thermal_2(m: LindistModelProtocol, fb, tb, ph, t):
        if not _has_thermal_limit(m, fb, tb, ph):
            return pyo.Constraint.Skip
        return (
            m.p_flow[fb, tb, ph, t] + c * m.q_flow[fb, tb, ph, t]
            <= m.s_branch_max[fb, tb, ph]
        )

    # Quadrant 4: +P, -Q
    def thermal_3(m: LindistModelProtocol, fb, tb, ph, t):
        if not _has_thermal_limit(m, fb, tb, ph):
            return pyo.Constraint.Skip
        return (
            m.p_flow[fb, tb, ph, t] - c * m.q_flow[fb, tb, ph, t]
            <= m.s_branch_max[fb, tb, ph]
        )

    def thermal_4(m: LindistModelProtocol, fb, tb, ph, t):
        if not _has_thermal_limit(m, fb, tb, ph):
            return pyo.Constraint.Skip
        return (
            c * m.p_flow[fb, tb, ph, t] - m.q_flow[fb, tb, ph, t]
            <= m.s_branch_max[fb, tb, ph]
        )

    # Quadrant 3: -P, -Q
    def thermal_5(m: LindistModelProtocol, fb, tb, ph, t):
        if not _has_thermal_limit(m, fb, tb, ph):
            return pyo.Constraint.Skip
        return (
            -c * m.p_flow[fb, tb, ph, t] - m.q_flow[fb, tb, ph, t]
            <= m.s_branch_max[fb, tb, ph]
        )

    def thermal_6(m: LindistModelProtocol, fb, tb, ph, t):
        if not _has_thermal_limit(m, fb, tb, ph):
            return pyo.Constraint.Skip
        return (
            -m.p_flow[fb, tb, ph, t] - c * m.q_flow[fb, tb, ph, t]
            <= m.s_branch_max[fb, tb, ph]
        )

    # Quadrant 2: -P, +Q
    def thermal_7(m: LindistModelProtocol, fb, tb, ph, t):
        if not _has_thermal_limit(m, fb, tb, ph):
            return pyo.Constraint.Skip
        return (
            -m.p_flow[fb, tb, ph, t] + c * m.q_flow[fb, tb, ph, t]
            <= m.s_branch_max[fb, tb, ph]
        )

    def thermal_8(m: LindistModelProtocol, fb, tb, ph, t):
        if not _has_thermal_limit(m, fb, tb, ph):
            return pyo.Constraint.Skip
        return (
            -c * m.p_flow[fb, tb, ph, t] + m.q_flow[fb, tb, ph, t]
            <= m.s_branch_max[fb, tb, ph]
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

    Requires branch_data to have columns for branch apparent power limits:
    primary phases: 's_a_max', 's_b_max', 's_c_max'
    triplex phases: 's_s1_max', 's_s2_max' (legacy: 's1_max', 's2_max')
    Branches without limits are skipped.
    """
    if not hasattr(m, "s_branch_max"):
        return

    def _has_thermal_limit(m, fb, tb, ph):
        """Check if branch has a valid thermal limit."""
        if (fb, tb, ph) not in m.s_branch_max:
            return False
        limit = pyo.value(m.s_branch_max[fb, tb, ph])
        return limit is not None and limit > 0

    def thermal_circle(m: LindistModelProtocol, fb, tb, ph, t):
        if not _has_thermal_limit(m, fb, tb, ph):
            return pyo.Constraint.Skip
        return (
            m.p_flow[fb, tb, ph, t] ** 2 + m.q_flow[fb, tb, ph, t] ** 2
            <= m.s_branch_max[fb, tb, ph] ** 2
        )

    m.thermal_limit_circle = pyo.Constraint(
        m.branch_phase_set, m.time_set, rule=thermal_circle
    )


# ============ Linear Slack Constraints ================================================
# ======================================================================================


def add_thermal_slack_constraints(m, derate_factor=1) -> None:
    """
    Add slack variable constraints for thermal limits.

    Converts hard thermal limit constraints to soft constraints using slack variables:
        P_flow ≤ S_max*derate_factor + s

    where s ≥ 0 is the slack variable representing thermal violations.
    """
    # Check if thermal limits exist in the model
    if not hasattr(m, "s_branch_max"):
        return

    # Add slack variable for thermal violations
    m.thermal_slack = pyo.Var(
        m.branch_phase_set,
        m.time_set,
        domain=pyo.NonNegativeReals,
        initialize=0,
        doc="Slack variable for thermal limit violations",
    )

    # Add constraint that allows violations via slack variables
    def thermal_slack_rule(m, _id, ph, t):
        """Allow apparent power to exceed limit by slack amount"""
        s_max = m.s_branch_max[_id, ph]
        if s_max is None or s_max <= 0:
            return pyo.Constraint.Skip
        # P <= S_max + slack
        return (
            m.p_flow[_id, ph, t]
            <= m.s_branch_max[_id, ph] * derate_factor + m.thermal_slack[_id, ph, t]
        )

    m.thermal_slack_constraint = pyo.Constraint(
        m.branch_phase_set,
        m.time_set,
        rule=thermal_slack_rule,
        doc="Slack constraint for thermal limits",
    )


def add_voltage_slack_constraints(m):
    """
    Add slack variable constraints for voltage bounds.

    Converts hard inequality constraints to soft constraints using slack variables:
        v_min² ≤ v² ≤ v_max²
    becomes:
        v² ≥ v_min² - s  (allows v² to go below v_min² by amount s)
        v² ≤ v_max² + s  (allows v² to go above v_max² by amount s)

    where s ≥ 0 is the slack variable representing voltage violations.
    """
    # Add single slack variable for voltage violations
    m.v2_slack = pyo.Var(
        m.bus_phase_set,
        m.time_set,
        domain=pyo.NonNegativeReals,
        initialize=0,
        doc="Slack variable for voltage bound violations",
    )

    # Add constraints that allow violations via slack variables
    def voltage_slack_under_rule(m, _id, ph, t):
        """Allow voltage to go below minimum by slack amount"""
        return m.v2[_id, ph, t] >= m.v_min[_id, ph] ** 2 - m.v2_slack[_id, ph, t]

    def voltage_slack_over_rule(m, _id, ph, t):
        """Allow voltage to go above maximum by slack amount"""
        return m.v2[_id, ph, t] <= m.v_max[_id, ph] ** 2 + m.v2_slack[_id, ph, t]

    m.voltage_slack_under = pyo.Constraint(
        m.bus_phase_set,
        m.time_set,
        rule=voltage_slack_under_rule,
        doc="Slack constraint for minimum voltage",
    )
    m.voltage_slack_over = pyo.Constraint(
        m.bus_phase_set,
        m.time_set,
        rule=voltage_slack_over_rule,
        doc="Slack constraint for maximum voltage",
    )


def add_swing_bus_voltage_slack_constraints(m):
    """
    Add slack variable constraints for swing bus voltage bounds.
    Converts hard equality constraints to soft constraints using slack variables:
        v² = v_swing²
    becomes:
        v² ≥ v_swing² - s  (allows v² to go below v_swing² by amount s)
        v² ≤ v_swing² + s  (allows v² to go above v_swing² by amount s)

    where s ≥ 0 is the slack variable representing swing bus voltage violations.
    """
    # Add single slack variable for swing bus voltage violations
    m.swing_v2_slack = pyo.Var(
        m.swing_phase_set,
        m.time_set,
        domain=pyo.NonNegativeReals,
        initialize=0,
        doc="Slack variable for swing bus voltage violations",
    )

    # Add constraints that allow violations via slack variables
    def swing_voltage_slack_under_rule(m, _id, ph, t):
        """Allow swing bus voltage to go below target by slack amount"""
        return (
            m.v2[_id, ph, t]
            >= m.v_swing[_id, ph, t] ** 2 - m.swing_v2_slack[_id, ph, t]
        )

    def swing_voltage_slack_over_rule(m, _id, ph, t):
        """Allow swing bus voltage to go above target by slack amount"""
        return (
            m.v2[_id, ph, t]
            <= m.v_swing[_id, ph, t] ** 2 + m.swing_v2_slack[_id, ph, t]
        )

    m.swing_voltage_slack_under = pyo.Constraint(
        m.swing_phase_set,
        m.time_set,
        rule=swing_voltage_slack_under_rule,
        doc="Slack constraint for minimum swing bus voltage",
    )
    m.swing_voltage_slack_over = pyo.Constraint(
        m.swing_phase_set,
        m.time_set,
        rule=swing_voltage_slack_over_rule,
        doc="Slack constraint for maximum swing bus voltage",
    )


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
