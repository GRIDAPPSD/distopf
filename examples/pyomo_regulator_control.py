"""
Example: Pyomo OPF with Regulator Control over 24 Hours

This example demonstrates using the Pyomo backend to optimize
regulator tap positions over a 24-hour period on the IEEE 123 bus
system with 30 DERs.

Uses IPOPT for NLP solving with continuous regulator tap control.
"""

import pandas as pd
import distopf as opf
import pyomo.environ as pyo
from distopf.pyomo_models import create_lindist_model, add_constraints
from distopf.pyomo_models.objectives import (
    loss_objective_rule,
    substation_power_objective_rule,
)
from distopf.pyomo_models.results import PyoResult

# Load IEEE 123 bus with 30 DERs for 24 hours
case = opf.create_case(
    opf.CASES_DIR / "csv" / "ieee123_30der",
    n_steps=5,
    start_step=0,
)

# Configure case for optimization
# For this example, we use fixed schedules (no DER optimization)
# to focus on regulator control
case.gen_data["control_variable"] = ""  # Fixed generators
case.bat_data = case.bat_data.head(0)  # Remove batteries

print("=" * 60)
print("Pyomo OPF with Regulator Control - 24 Hour Simulation")
print("=" * 60)
print(f"Buses: {len(case.bus_data)}")
print(f"Branches: {len(case.branch_data)}")
print(f"Generators: {len(case.gen_data)}")
print(f"Regulators: {len(case.reg_data)}")
print(f"Time steps: {case.n_steps}")

# Create Pyomo model with regulator control enabled
print("\nCreating Pyomo model with regulator control...")
model = create_lindist_model(
    case,
    control_regulators=True,  # Enable regulator tap control variables
    control_capacitors=False,
)

# Add constraints with continuous regulator control (NLP)
# Using octagonal (linear) constraints - less restrictive
add_constraints(
    model,
    circular_constraints=True,  # Use linear constraints
    control_regulators=True,  # mixed integer regulator tap control
)

# Set loss minimization objective
model.objective = pyo.Objective(
    rule=substation_power_objective_rule, sense=pyo.minimize
)

# HiGHS is a fast MILP or QP solver. But not MIQP.
# However it is still able to solve when circle constraints are included, which makes the
# problem a Mixed-integer SOCP.
# As long as the objective is linear then HiGHs or CBC will work.
solver_name = "highs"
print(f"\nSolving with {solver_name}...")
solver = pyo.SolverFactory(solver_name)
result = solver.solve(model, tee=False)

print(f"Solver status: {result.solver.termination_condition}")

if result.solver.termination_condition == pyo.TerminationCondition.optimal:
    # Extract results
    pyo_result = PyoResult(model)
    u_reg = pd.DataFrame(
        data=[
            [model.from_bus_map[_id], _id, ph, k, t, val]
            for ((_id, ph, k, t), val) in model.u_reg.extract_values().items()
        ],
        columns=["fb", "tb", "phase", "k", "t", "value"],
    )
    taps = u_reg.loc[u_reg.value == 1].reset_index(drop=True)
    taps["tap"] = taps.k
    taps["ratio"] = taps["tap"].map(model.tap_ratio)
    taps = taps.loc[:, ["fb", "tb", "phase", "t", "tap", "ratio"]]
    taps = taps.sort_values(["t", "tb", "phase"])
    print(taps)
    # Get key results
#     voltages = pyo_result.voltages
#     p_flows = pyo_result.p_flow
#     q_flows = pyo_result.q_flow
#     p_gens = pyo_result.p_gen
#     q_gens = pyo_result.q_gen

#     # Get regulator results if available
#     if hasattr(pyo_result, "reg_ratio"):
#         reg_ratios = pyo_result.reg_ratio
#         print("\nRegulator tap ratios over time:")
#         print(reg_ratios.head(10))

#     # Voltage statistics
#     print("\nVoltage Statistics:")
#     print("-" * 40)
#     v_min = voltages[["a", "b", "c"]].min().min()
#     v_max = voltages[["a", "b", "c"]].max().max()
#     print(f"  Min voltage: {v_min:.4f} p.u.")
#     print(f"  Max voltage: {v_max:.4f} p.u.")

#     # Per-timestep voltage range
#     print("\nVoltage range by timestep:")
#     for t in range(min(6, case.n_steps)):  # Show first 6 timesteps
#         v_t = voltages[voltages["t"] == t][["a", "b", "c"]]
#         print(f"  t={t}: {v_t.min().min():.4f} - {v_t.max().max():.4f}")
#     if case.n_steps > 6:
#         print("  ...")

#     # Objective value
#     obj_value = pyo.value(model.objective)
#     print(f"\nTotal loss objective: {obj_value:.6f}")

#     # Plot results
#     print("\nGenerating plots...")

#     # Plot voltages over time
#     opf.plot_voltages(voltages).show(renderer="browser")

#     # Plot power flows
#     opf.plot_power_flows(p_flows, q_flows).show(renderer="browser")

#     # Plot generator output
#     opf.plot_gens(p_gens, q_gens).show(renderer="browser")

#     # Plot network with voltage coloring (for first timestep)
#     v_t0 = voltages[voltages["t"] == 0].copy()
#     p_t0 = p_flows[p_flows["t"] == 0].copy()
#     q_t0 = q_flows[q_flows["t"] == 0].copy()
#     pg_t0 = p_gens[p_gens["t"] == 0].copy()
#     qg_t0 = q_gens[q_gens["t"] == 0].copy()

#     opf.plot_network(
#         case,
#         v=v_t0,
#         p_flow=p_t0,
#         q_flow=q_t0,
#         p_gen=pg_t0,
#         q_gen=qg_t0,
#     ).show(renderer="browser")

#     print("\nPlots opened in browser.")

# else:
#     print(f"Optimization failed: {result.solver.termination_condition}")
#     print("Try adjusting constraints or using a different solver.")

print("\n" + "=" * 60)
print("Example complete")
print("=" * 60)
