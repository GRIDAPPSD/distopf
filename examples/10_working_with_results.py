"""
Example 10: Working with Results

After running OPF, you have access to all results.
This example shows how to extract and use the results.
"""

import distopf as opf
import pandas as pd

# Run OPF
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
result = case.run_opf("loss_min", control_variable="PQ", backend="matrix")

# Get base power for unit conversions
s_base = case.bus_data["s_base"].iloc[0]

# Access individual result DataFrames
voltages_df = result.voltages
# Extract numeric voltage data (phases a, b, c)
voltages = voltages_df[["a", "b", "c"]].apply(pd.to_numeric, errors="coerce")
p_flows = result.p_flows
q_flows = result.q_flows
p_gens_df = result.p_gens
q_gens_df = result.q_gens
# Extract numeric generator power data (phases a, b, c)
p_gens = p_gens_df[["a", "b", "c"]].apply(pd.to_numeric, errors="coerce")
q_gens = q_gens_df[["a", "b", "c"]].apply(pd.to_numeric, errors="coerce")

print("=" * 60)
print("Working with OPF Results")
print("=" * 60)

print("\n--- Optimization Results ---")
print(f"Objective function value (pu): {result.objective_value:.6f}")
print(f"Total losses (MW): {result.objective_value * s_base / 1e6:.6f}")

print(f"\nVoltages DataFrame shape: {voltages.shape}")
print("  - Rows: buses")
print("  - Columns: phases")
print("Sample voltages (first 5 buses):")
print(voltages.head())

print(f"\nActive Power Flows DataFrame shape: {p_flows.shape}")
print(f"Reactive Power Flows DataFrame shape: {q_flows.shape}")
print("  - Rows: lines")
print("  - Columns: phases a, b, c")
print("Sample active power flows (first 3 lines):")
print(p_flows.head(3))

print(f"\nGenerator Active Power shape: {p_gens.shape}")
if len(p_gens) > 0:
    print(f"Total active power (MW): {(p_gens.sum().sum() * s_base / 1e6):.3f}")

print(f"\nGenerator Reactive Power shape: {q_gens.shape}")
if len(q_gens) > 0:
    print(f"Total reactive power (MVAr): {(q_gens.sum().sum() * s_base / 1e6):.3f}")

# Perform analysis
print("\n" + "=" * 60)
print("Result Analysis Examples:")
print("=" * 60)

# Find maximum voltage deviation
nominal_voltage = 1.0
max_dev = (abs(voltages - nominal_voltage)).max().max()
print(f"\nMax voltage deviation from 1.0 p.u.: {max_dev:.4f} p.u.")

# Find most heavily loaded line
if len(p_flows) > 0 and len(q_flows) > 0:
    # Combine p_flows and q_flows to get apparent power (S = sqrt(P^2 + Q^2))
    phase_cols = [col for col in p_flows.columns if col in ["a", "b", "c"]]
    if phase_cols:
        # Calculate apparent power magnitude for each phase
        s_mag = ((p_flows[phase_cols] ** 2 + q_flows[phase_cols] ** 2) ** 0.5).sum(
            axis=1
        ) ** 0.5
        max_line_idx = s_mag.idxmax()
        max_line_loading = s_mag.max()
        print(
            f"Most loaded line: {max_line_idx} with apparent power: {max_line_loading * s_base / 1e6:.3f} MVA"
        )

# Export results to CSV
print("\nResults can be easily exported to CSV for further analysis:")
# voltages.to_csv("voltages.csv")
# power_flows.to_csv("power_flows.csv")
print("  voltages.to_csv('voltages.csv')")
print("  power_flows.to_csv('power_flows.csv')")

# Note: Plotting requires browser renderer; commented out for scripts
# To visualize, uncomment the line below:
# result.plot_network().show(renderer="browser")
