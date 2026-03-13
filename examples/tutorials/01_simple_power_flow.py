"""
Example 1: Simple Power Flow Analysis

This is the simplest example. Just load a network and run power flow to see
what happens under the current operating conditions.

No optimization - just physics-based analysis.
"""

import distopf as opf

# Load the IEEE 13-bus test network (smallest, fastest)
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

# Run power flow analysis
result = case.run_pf()

# Convert phase columns to numeric
voltages = result.voltages[["a", "b", "c"]]
p_flows = result.p_flows[["a", "b", "c"]]
q_flows = result.q_flows[["a", "b", "c"]]

# Get base power for unit conversions
s_base = case.bus_data["s_base"].iloc[0]

# Print summary
print("=" * 60)
print("Power Flow Analysis Results")
print("=" * 60)
print("\nNetwork: IEEE 13-bus")
print(f"Buses: {len(voltages)}")
print(f"Lines: {len(p_flows)}")

print("\n--- Voltage Magnitudes (p.u.) ---")
print(voltages.describe())

print("\n--- Active Power Flows (MW) ---")
print((p_flows * s_base / 1e6).describe())

print("\n--- Reactive Power Flows (MVAr) ---")
print((q_flows * s_base / 1e6).describe())

print("\nNo optimization run yet, so plotting skipped.")
print("See examples 02-10 for visualization after OPF/control objectives.")
