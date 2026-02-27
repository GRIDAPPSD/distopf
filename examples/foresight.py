import distopf as opf
import pyomo.environ as pyo
from distopf.pyomo_models import create_lindist_model, add_constraints
from distopf.pyomo_models import objectives
from distopf.pyomo_models import constraints


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


def add_constraints_custom(m):
    constraints.add_p_flow_constraints(m)
    constraints.add_q_flow_constraints(m)
    constraints.add_voltage_drop_constraints(m)
    constraints.add_swing_bus_constraints(m)
    constraints.add_cvr_load_constraints(m)
    constraints.add_capacitor_constraints(m)
    constraints.add_regulator_constraints(m)
    # Generators
    constraints.add_generator_constant_p_constraints_q_control(m)
    constraints.add_generator_constant_q_constraints_p_control(m)
    # Batteries
    constraints.add_battery_constant_q_constraints_p_control(m)
    constraints.add_battery_energy_constraints(m)
    constraints.add_battery_net_p_bat_equal_phase_constraints(m)

    # Inequalities with slack variables:
    add_thermal_slack_constraints(m)
    add_voltage_slack_constraints(m)
    #     constraints.add_voltage_limits(model)

    # Generator Inequality
    constraints.add_generator_limits(m)
    constraints.add_octagonal_inverter_constraints_pq_control(m)
    # Battery Inequality
    constraints.add_battery_power_limits(m)
    constraints.add_battery_soc_limits(m)


# Load IEEE 123 bus case
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")

# Increase load to create voltage stress (makes the example more interesting)
case.bus_data.loc[:, ["pl_a", "pl_b", "pl_c"]] *= 3.5
case.bus_data.loc[:, ["ql_a", "ql_b", "ql_c"]] *= 3.5

case.branch_data.loc[case.branch_data.tb == 3, ["sa_max", "sb_max", "sc_max"]] = 3.0
# Build model with equality constraints only (no hard voltage/thermal limits)
# This uses equality_only=True to skip inequality constraints
model = create_lindist_model(case)

add_constraints_custom(model)


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


primary_objective_rule = objectives.voltage_deviation_objective_rule


def obj_rule(m):
    """Combined objective: minimize substation power + penalize voltage and thermal violations"""
    return (
        primary_objective_rule(m) + voltage_slack_penalty(m) + thermal_slack_penalty(m)
    )


# Create and set the objective
model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

# Solve with IPOPT
solver = pyo.SolverFactory("highs")
result = solver.solve(model, tee=False)

if result.solver.termination_condition == pyo.TerminationCondition.optimal:
    # Get results
    total_objective = pyo.value(model.objective)
    primary_objective_value = pyo.value(primary_objective_rule(model))

    # Extract voltage results
    voltages = {}
    for bus_id, ph in model.bus_phase_set:
        for t in model.time_set:
            v = pyo.value(model.v2[bus_id, ph, t]) ** 0.5
            voltages[(bus_id, ph)] = v

    v_min = min(voltages.values())
    v_max = max(voltages.values())
    undervoltage = sum(1 for v in voltages.values() if v < 0.95)
    overvoltage = sum(1 for v in voltages.values() if v > 1.05)

    # Extract p_flow results
    p_flows = {}
    for branch_id, ph in model.branch_phase_set:
        for t in model.time_set:
            p = pyo.value(model.p_flow[branch_id, ph, t])
            p_flows[(branch_id, ph)] = p

    p_min = min(p_flows.values())
    p_max = max(p_flows.values())
    
    # Count thermal violations (p_flow exceeding s_branch_max)
    thermal_violations = 0
    if hasattr(model, "s_branch_max"):
        for branch_id, ph in model.branch_phase_set:
            s_max = model.s_branch_max[branch_id, ph]
            if s_max is not None and s_max > 0:
                p = p_flows.get((branch_id, ph), 0)
                if p > s_max:
                    thermal_violations += 1

    # Extract voltage slack variables
    voltage_slack_violations = {}
    total_voltage_slack = 0
    for bus_id, ph in model.bus_phase_set:
        for t in model.time_set:
            slack_val = pyo.value(model.v2_slack[bus_id, ph, t])
            if slack_val > 1e-6:  # Only show non-zero slacks
                voltage_slack_violations[(bus_id, ph, t)] = slack_val
                total_voltage_slack += slack_val

    # Extract thermal slack variables
    thermal_slack_violations = {}
    total_thermal_slack = 0
    if hasattr(model, "thermal_slack"):
        for branch_id, ph in model.branch_phase_set:
            for t in model.time_set:
                slack_val = pyo.value(model.thermal_slack[branch_id, ph, t])
                if slack_val > 1e-6:  # Only show non-zero slacks
                    thermal_slack_violations[(branch_id, ph, t)] = slack_val
                    total_thermal_slack += slack_val

    # Print results
    print("=" * 60)
    print("Penalty-Based OPF Results - IEEE 123 Bus")
    print("=" * 60)
    print("\nObjective Breakdown:")
    print(f"  Primary objective:      {primary_objective_value:.6f}")
    print(f"  Total (with penalties): {total_objective:.6f}")
    print(f"  Penalty contribution:   {total_objective - primary_objective_value:.6f}")
    print("\nVoltage Summary:")
    print(f"  Min voltage: {v_min:.4f} p.u.")
    print(f"  Max voltage: {v_max:.4f} p.u.")
    print(f"  Undervoltage violations (<0.95): {undervoltage}")
    print(f"  Overvoltage violations (>1.05):  {overvoltage}")
    print("\nP_Flow Summary:")
    print(f"  Min p_flow: {p_min:.6f} p.u.")
    print(f"  Max p_flow: {p_max:.6f} p.u.")
    print(f"  Thermal violations (p_flow > s_max): {thermal_violations}")
    print("\nVoltage Slack Variables (Violations):")
    print(f"  Total slack (sum of violations): {total_voltage_slack:.6f}")
    print(f"  Number of buses with violations: {len(voltage_slack_violations)}")
    if voltage_slack_violations:
        print("\n  Detailed Voltage Slack Values:")
        print("  Bus  Phase  Time  Slack Value   Voltage")
        print("  " + "-" * 35)
        for (bus_id, ph, t), slack_val in sorted(voltage_slack_violations.items()):
            print(f"  {bus_id:3d}  {ph:5s}  {t:4d}  {slack_val:.8f}    {model.v2[bus_id, ph, t].value:.8f}")

    print("\nThermal Slack Variables (Violations):")
    print(f"  Total slack (sum of violations): {total_thermal_slack:.6f}")
    print(f"  Number of branches with violations: {len(thermal_slack_violations)}")
    if thermal_slack_violations:
        print("\n  Detailed Thermal Slack Values:")
        print("  Branch  Phase  Time  Slack Value   P_flow")
        print("  " + "-" * 37)
        for (branch_id, ph, t), slack_val in sorted(thermal_slack_violations.items()):
            print(f"  {branch_id:6d}  {ph:5s}  {t:4d}  {slack_val:.8f}    {model.p_flow[branch_id, ph, t].value:.8f}")
    print("=" * 60)
else:
    print(f"Solver failed: {result.solver.termination_condition}")
