"""
Example 3: Voltage Optimization

Run OPF to minimize voltage deviations (keep voltages close to 1.0 p.u.).
This is useful for grid stability and equipment longevity.

Shows different optimization objectives and how to choose them.
"""

import distopf as opf

# Load the IEEE 123-bus network with DERs
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")

# Get base power for unit conversions
s_base = case.bus_data["s_base"].iloc[0]

# Run OPF with loss minimization while controlling Q for voltage support
# Note: matrix backend only supports "loss_min"; use pyomo backend for voltage_dev
result = case.run_opf("loss_min", control_variable="Q", wrapper="matrix")

# Extract numeric voltage data (phases a, b, c)
voltages = result.voltages[["a", "b", "c"]]
q_gens_df = result.q_gens
# Extract numeric generator reactive power data (phases a, b, c)
q_gens = q_gens_df[["a", "b", "c"]]

print("=" * 60)
print("Voltage Support via Reactive Power Control")
print("=" * 60)
print("Network: IEEE 123-bus with 30 DERs")
print("Objective: Minimize losses (while Q control improves voltage)")
print("Control: DERs control reactive power only (Q) for voltage support")

print("\n--- Voltage Profile ---")
print(f"Min voltage (p.u.): {voltages.min().min():.4f}")
print(f"Max voltage (p.u.): {voltages.max().max():.4f}")
print(f"Mean voltage (p.u.): {voltages.mean().mean():.4f}")
print(f"Std dev (p.u.): {voltages.std().mean():.4f}")

print("\n--- Reactive Power Support ---")
if len(q_gens) > 0:
    total_q = (q_gens.sum(axis=0) * s_base / 1e6).values
    print(f"Total reactive power supplied (MVAr): {total_q}")

print("\n--- Optimization Results ---")
print(f"Objective function value (pu): {result.objective_value:.6f}")
print(f"Total losses (MW): {result.objective_value * s_base / 1e6:.6f}")

# Note: Results stored in result object; see example 10 for visualization
print("\nResults available as DataFrames for analysis and export.")
