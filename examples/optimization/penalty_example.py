"""
Simple example demonstrating penalty-based optimization with IEEE 123 bus.

This example shows how to use soft constraints (penalties) instead of hard
constraints for voltage and generator limits. This approach is useful when:
- Hard constraints may cause infeasibility
- You want to allow small violations with a cost
- You need a more robust optimization formulation
"""

import distopf as opf
import pyomo.environ as pyo
from distopf.pyomo_models import create_lindist_model, add_constraints
from distopf.pyomo_models import objectives

# Load IEEE 123 bus case
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123")

# Increase load to create voltage stress (makes the example more interesting)
case.bus_data.loc[:, ["pl_a", "pl_b", "pl_c"]] *= 1.5
case.bus_data.loc[:, ["ql_a", "ql_b", "ql_c"]] *= 1.5

# Build model with equality constraints only (no hard voltage/thermal limits)
# This uses equality_only=True to skip inequality constraints
model = create_lindist_model(case)
add_constraints(model, equality_only=True)

# Add penalized loss objective
# The optimizer will minimize: loss + voltage_penalty + thermal_penalty + ...
objectives.add_penalized_loss_objective(
    model,
    voltage_weight=1e4,  # High penalty for voltage violations
    thermal_weight=1e4,  # Penalty for thermal limit violations
    generator_weight=1e4,  # Penalty for generator limit violations
)

# Solve with IPOPT
solver = pyo.SolverFactory("ipopt")
result = solver.solve(model, tee=False)

if result.solver.termination_condition == pyo.TerminationCondition.optimal:
    # Get results
    total_objective = pyo.value(model.objective)
    pure_loss = pyo.value(objectives.loss_objective_rule(model))

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

    # Print results
    print("=" * 60)
    print("Penalty-Based OPF Results - IEEE 123 Bus")
    print("=" * 60)
    print("\nObjective Breakdown:")
    print(f"  Pure loss:              {pure_loss:.6f}")
    print(f"  Total (with penalties): {total_objective:.6f}")
    print(f"  Penalty contribution:   {total_objective - pure_loss:.6f}")
    print("\nVoltage Summary:")
    print(f"  Min voltage: {v_min:.4f} p.u.")
    print(f"  Max voltage: {v_max:.4f} p.u.")
    print(f"  Undervoltage violations (<0.95): {undervoltage}")
    print(f"  Overvoltage violations (>1.05):  {overvoltage}")
    print("=" * 60)
else:
    print(f"Solver failed: {result.solver.termination_condition}")
