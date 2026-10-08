"""
Constraint functions for DistOPF Pyomo models.

Each function takes a Pyomo ConcreteModel and data, and adds constraints to the model.
"""

import pyomo.environ as pyo  # type: ignore
from distopf.pyomo_models.common.protocol import LindistModelProtocol
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
    """Forward CVR load equations to the owning provider."""
    from distopf.pyomo_models.devices.load import add_cvr_load_constraints as add

    add(m, free_boundary_loads)


# Generators ------------------------------------------------------------------------


def add_generator_limits(m: LindistModelProtocol) -> None:
    """Add limits using the generator provider's current formulation."""
    from distopf.pyomo_models.devices.generator import add_gen_limits

    add_gen_limits(m)


def add_generator_constant_p_constraints(m: LindistModelProtocol) -> None:
    m.gen_constant_p = pyo.Constraint(
        m.gen_device_phase_set,
        m.time_set,
        rule=lambda m, device, ph, t: m.p_gen[device, ph, t]
        == m.gen_p_available[device, ph, t],
    )


def add_generator_constant_q_constraints(m: LindistModelProtocol) -> None:
    m.gen_constant_q = pyo.Constraint(
        m.gen_device_phase_set,
        m.time_set,
        rule=lambda m, device, ph, t: m.q_gen[device, ph, t]
        == m.gen_q_setpoint[device, ph, t],
    )


def add_generator_constant_p_constraints_q_control(m: LindistModelProtocol) -> None:
    from distopf.pyomo_models.devices.generator import add_gen_constant_p_constraints

    add_gen_constant_p_constraints(m)


def add_generator_constant_q_constraints_p_control(m: LindistModelProtocol) -> None:
    from distopf.pyomo_models.devices.generator import add_gen_constant_q_constraints

    add_gen_constant_q_constraints(m)


def add_octagonal_inverter_constraints_pq_control(m: LindistModelProtocol) -> None:
    """Add the generator provider's octagonal rating constraints."""
    from distopf.pyomo_models.devices.generator import add_gen_octagon_constraints

    add_gen_octagon_constraints(m)


def add_circular_generator_constraints_pq_control(m: LindistModelProtocol) -> None:
    """Add the generator provider's circular rating constraints."""
    from distopf.pyomo_models.devices.generator import add_gen_circle_constraints

    add_gen_circle_constraints(m)


# Capacitors ------------------------------------------------------------------------
def add_capacitor_constraints(m: LindistModelProtocol) -> None:
    """Add the capacitor provider's voltage-dependent injection constraints."""
    from distopf.pyomo_models.devices.capacitor import add_capacitor_constraints as add

    add(m)


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
    """Add the capacitor provider's switched injection constraints."""
    from distopf.pyomo_models.devices.capacitor import (
        add_capacitor_mi_constraints as add,
    )

    add(m)


def add_capacitor_mccormick_constraints(m: LindistModelProtocol) -> None:
    """Add the capacitor provider's switching envelope."""
    from distopf.pyomo_models.devices.capacitor import (
        add_capacitor_mccormick_constraints as add,
    )

    add(m)


def add_capacitor_z_bounds(m: LindistModelProtocol) -> None:
    """Add the capacitor provider's auxiliary-variable bounds."""
    from distopf.pyomo_models.devices.capacitor import add_capacitor_z_bounds as add

    add(m)


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
