import distopf as opf
import pyomo.environ as pyo

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
            m.p_flow[_id, ph, t] <= m.s_branch_max[_id, ph] * derate_factor + m.thermal_slack[_id, ph, t]
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



def voltage_slack_penalty(m, weight=1e3):
    penalty = 0
    for _id, ph in m.bus_phase_set:
        for t in m.time_set:
            penalty += m.v2_slack[_id, ph, t]
    return weight * penalty


def thermal_slack_penalty(m, weight=1e3):
    if not hasattr(m, "thermal_slack"):
        return 0

    penalty = 0
    for _id, ph in m.branch_phase_set:
        for t in m.time_set:
            penalty += m.thermal_slack[_id, ph, t]
    return weight * penalty