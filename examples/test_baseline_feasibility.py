"""
Quick test: verify baseline OPF works on ieee13 before trying 2Bus.
"""

import distopf as opf
import pyomo.environ as pyo

from distopf.api import create_case
from distopf.pyomo_models.lindist import create_lindist_model
from distopf.pyomo_models.constraints import (
    add_p_flow_constraints,
    add_q_flow_constraints,
    add_voltage_drop_constraints,
    add_swing_bus_constraints,
    add_generator_constant_p_constraints_q_control,
    add_generator_constant_q_constraints_p_control,
    add_generator_limits,
    add_voltage_limits,
)
from distopf.pyomo_models.objectives import substation_power_objective_rule

# Test on 2Bus-1ph-batt FIRST (simpler)
print(f"Testing on 2Bus-1ph-batt:")
case2 = create_case(data_path=opf.CASES_DIR / "csv" / "2Bus-1ph-batt")
print(f"  Buses: {case2.bus_data.shape[0]}")
print(f"  Branches: {case2.branch_data.shape[0]}")
print(f"  Generators: {case2.gen_data.shape[0]}")

model2 = create_lindist_model(case2)
add_p_flow_constraints(model2)
add_q_flow_constraints(model2)
add_voltage_drop_constraints(model2)
add_swing_bus_constraints(model2)
add_generator_constant_p_constraints_q_control(model2)
add_generator_constant_q_constraints_p_control(model2)
add_generator_limits(model2)
add_voltage_limits(model2)

model2.objective = pyo.Objective(
    rule=substation_power_objective_rule, sense=pyo.minimize
)

opt = pyo.SolverFactory("ipopt")
results2 = opt.solve(model2, tee=False)

if results2.solver.status == pyo.SolverStatus.ok:
    print(f"✓ 2Bus-1ph-batt baseline OPF successful")
    print(f"  Objective: {pyo.value(model2.objective):.6f}")
else:
    print(f"✗ 2Bus-1ph-batt baseline OPF failed: {results2.solver.status}")

# Test on ieee13
print(f"\n{'=' * 60}")
print(f"Testing on ieee13:")
case = create_case(data_path=opf.CASES_DIR / "csv" / "ieee13")
print(f"  Buses: {case.bus_data.shape[0]}")
print(f"  Branches: {case.branch_data.shape[0]}")

model = create_lindist_model(case)
add_p_flow_constraints(model)
add_q_flow_constraints(model)
add_voltage_drop_constraints(model)
add_swing_bus_constraints(model)
add_generator_constant_p_constraints_q_control(model)
add_generator_constant_q_constraints_p_control(model)
add_generator_limits(model)
add_voltage_limits(model)

model.objective = pyo.Objective(
    rule=substation_power_objective_rule, sense=pyo.minimize
)

results = opt.solve(model, tee=False)

if results.solver.status == pyo.SolverStatus.ok:
    print(f"✓ ieee13 baseline OPF successful")
    print(f"  Objective: {pyo.value(model.objective):.6f}")
else:
    print(f"✗ ieee13 baseline OPF failed: {results.solver.status}")
