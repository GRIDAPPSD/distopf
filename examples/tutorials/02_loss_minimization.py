"""
Example 2: Simple Optimization - Loss Minimization

Run optimal power flow (OPF) to minimize real power losses in the network.
This is the most common OPF objective.

Shows how easy it is to optimize a network with a single function call.
"""

import distopf as opf

# Load the IEEE 123-bus network with DERs
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")

# Run OPF with loss minimization
# DERs can control both active (P) and reactive (Q) power
result = case.run_opf("loss_min", control_variable="PQ", backend="matrix")

# Get base power for unit conversions
s_base = case.bus_data["s_base"].iloc[0]

# Get results
voltages_df = result.voltages
# Extract numeric voltage data (phases a, b, c)
voltages = voltages_df[["a", "b", "c"]]
p_gens_df = result.p_gens
q_gens_df = result.q_gens
# Extract numeric generator power data (phases a, b, c)
p_gens = p_gens_df[["a", "b", "c"]]
q_gens = q_gens_df[["a", "b", "c"]]

# Print summary
print("=" * 60)
print("Loss Minimization OPF Results")
print("=" * 60)
print("\nNetwork: IEEE 123-bus with 30 DERs")
print("Objective: Minimize real power losses")
print("Control: DERs control active and reactive power (PQ)")

print("\n--- Generator Outputs ---")
if len(p_gens) > 0:
    print(f"Active power (MW): {(p_gens.sum(axis=0) * s_base / 1e6).values}")
    print(f"Reactive power (MVAr): {(q_gens.sum(axis=0) * s_base / 1e6).values}")

print("\n--- Voltage Profile ---")
print(f"Min voltage (p.u.): {voltages.min().min():.4f}")
print(f"Max voltage (p.u.): {voltages.max().max():.4f}")
print(f"Mean voltage (p.u.): {voltages.mean().mean():.4f}")

print("\n--- Optimization Results ---")
print(f"Objective function value (pu): {result.objective_value:.6f}")
print(f"Total losses (MW): {result.objective_value * s_base / 1e6:.6f}")

# Note: Results stored in result object; see example 10 for visualization
print("\nResults available as DataFrames for analysis and export.")
