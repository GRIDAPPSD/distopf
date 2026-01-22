"""
Example 4: Comparing Different Backends

Run the same OPF problem using different solvers (backends) and compare results.

Shows that the API is backend-agnostic - you choose the solver that works best
for your problem.
"""

import distopf as opf

# Load the small IEEE 13-bus test case (fast to solve)
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

print("=" * 60)
print("Comparing Different Optimization Backends")
print("=" * 60)
print("Network: IEEE 13-bus")
print("Objective: Minimize loss")
print()

# Backend 1: Matrix-based convex solver (CVXPY/CLARABEL)
print("Running Matrix Backend (CVXPY + CLARABEL)...")
result_matrix = case.run_opf("loss_min", backend="matrix")
v_matrix = result_matrix.voltages[["a", "b", "c"]]
s_base = case.bus_data["s_base"].iloc[0]
print("  Status: Success")
print(f"  Voltage min: {v_matrix.min().min():.4f} p.u.")
print(f"  Voltage max: {v_matrix.max().max():.4f} p.u.")
print(
    f"  Objective: {result_matrix.objective_value:.6f} pu ({result_matrix.objective_value * s_base / 1e6:.6f} MW)"
)

# Backend 2: Pyomo-based nonlinear solver (IPOPT)
print("\nRunning Pyomo Backend (Pyomo + IPOPT)...")
result_pyomo = case.run_opf("loss_min", backend="pyomo")
v_pyomo = result_pyomo.voltages[["a", "b", "c"]]
print("  Status: Success")
print(f"  Voltage min: {v_pyomo.min().min():.4f} p.u.")
print(f"  Voltage max: {v_pyomo.max().max():.4f} p.u.")
if result_pyomo.objective_value is not None:
    print(
        f"  Objective: {result_pyomo.objective_value:.6f} pu ({result_pyomo.objective_value * s_base / 1e6:.6f} MW)"
    )
else:
    print("  Objective: Not available")
