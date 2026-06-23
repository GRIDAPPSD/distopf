"""
Example 4: Comparing Different Wrappers

Run the same OPF problem using different wrappers and compare results.

Shows that the API is wrapper-agnostic while still letting you choose the
solver stack that fits your problem.
"""

import distopf as opf

# Load the small IEEE 13-bus test case (fast to solve)
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

print("=" * 60)
print("Comparing Different Optimization Wrappers")
print("=" * 60)
print("Network: IEEE 13-bus")
print("Objective: Minimize loss")
print()

# Wrapper 1: Matrix-based convex solver (CVXPY/CLARABEL)
print("Running Matrix Wrapper (CVXPY + CLARABEL)...")
result_matrix = case.run_opf("loss_min", wrapper="matrix")
v_matrix = result_matrix.voltages[["a", "b", "c"]]
s_base = case.bus_data["s_base"].iloc[0]
print("  Status: Success")
print(f"  Voltage min: {v_matrix.min().min():.4f} p.u.")
print(f"  Voltage max: {v_matrix.max().max():.4f} p.u.")
print(
    f"  Objective: {result_matrix.objective_value:.6f} pu ({result_matrix.objective_value * s_base / 1e6:.6f} MW)"
)

# Wrapper 2: Pyomo-based solver (IPOPT)
print("\nRunning Pyomo Wrapper (Pyomo + IPOPT)...")
result_pyomo = case.run_opf("loss_min", wrapper="pyomo")
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
