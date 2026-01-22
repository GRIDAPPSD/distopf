"""
Example 11: Forward-Backward Sweep Power Flow Analysis

This example demonstrates how to use the Forward-Backward Sweep (FBS) method,
a traditional power flow solver specifically designed for 3-phase unbalanced
radial distribution networks.

FBS is particularly useful for:
- Fast convergence on radial networks
- 3-phase unbalanced load flow analysis
- Networks with voltage regulators
- Comparison with optimization-based methods
"""

import distopf as opf
import pandas as pd
import numpy as np

# Load the IEEE 13-bus test network
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")

print("=" * 70)
print("Forward-Backward Sweep (FBS) Power Flow Analysis")
print("=" * 70)

print("\n--- Network Configuration ---")
print(f"Buses: {len(case.bus_data)}")
print(f"Branches: {len(case.branch_data)}")
print(f"Generators: {len(case.gen_data) if case.gen_data is not None else 0}")
print(f"Voltage Regulators: {len(case.reg_data) if case.reg_data is not None else 0}")


print("\n--- Running FBS via Case.run_fbs() ---")
result = case.run_fbs(max_iterations=100, tolerance=1e-6, verbose=False)

# PowerFlowResult provides direct attribute access
print(f"\n--- Result Summary ---")
print(result.summary())

# Access results directly from PowerFlowResult
voltages = result.voltages
voltage_angles = result.voltage_angles
p_flows = result.p_flows
q_flows = result.q_flows
currents = result.currents
current_angles = result.current_angles


print("\n--- Voltage Results (p.u.) ---")
print(voltages.head(10))

print("\n--- Voltage Magnitudes Summary ---")
v_phases = voltages[["a", "b", "c"]].apply(pd.to_numeric, errors="coerce")
print(v_phases.describe())

print("\n--- Voltage Angles (degrees) ---")
va_phases = voltage_angles[["a", "b", "c"]].apply(pd.to_numeric, errors="coerce")
print(va_phases.describe())

# Get power flow results from Case.run_fbs() results
print("\n--- Active Power Flows (p.u.) ---")
p_phases = p_flows[["a", "b", "c"]].apply(pd.to_numeric, errors="coerce")
s_base = case.bus_data["s_base"].iloc[0]
print((p_phases * s_base / 1e6).describe())

print("\n--- Reactive Power Flows (p.u.) ---")
q_phases = q_flows[["a", "b", "c"]].apply(pd.to_numeric, errors="coerce")
print((q_phases * s_base / 1e6).describe())

print("\n--- Branch Currents (p.u.) ---")
i_phases = currents[["a", "b", "c"]].apply(pd.to_numeric, errors="coerce")
print(i_phases.describe())

print("\n--- Current Angles (degrees) ---")
ia_phases = current_angles[["a", "b", "c"]].apply(pd.to_numeric, errors="coerce")
print(ia_phases.describe())

# Compare with Case.run_pf() results
print("\n--- Comparison with Case.run_pf() ---")
print("Note: FBS and run_pf use different solution methods.")
print("Results should agree well on this balanced IEEE 13-bus network.")

result = case.run_pf()
case_v_phases = result.voltages[["a", "b", "c"]].apply(pd.to_numeric, errors="coerce")

# Align data by bus ID for proper comparison (important!)
fbs_v_idx = voltages["id"].values
case_v_idx = result.voltages["id"].values

# Create aligned series for comparison
fbs_voltages_aligned = v_phases.set_index(fbs_v_idx)
case_voltages_aligned = case_v_phases.set_index(case_v_idx)

# Find common buses and compare
common_buses = fbs_voltages_aligned.index.intersection(case_voltages_aligned.index)
voltage_diff = (
    fbs_voltages_aligned.loc[common_buses] - case_voltages_aligned.loc[common_buses]
).abs()
voltage_diff_valid = voltage_diff.dropna()

print("\nVoltage magnitude differences (aligned by bus ID):")
if len(voltage_diff_valid) > 0:
    print(
        f"  Max: {voltage_diff_valid.max().max():.4f} p.u. ({voltage_diff_valid.max().max() * 100:.2f}%)"
    )
    print(f"  Mean: {voltage_diff_valid.mean().mean():.4e} p.u.")
    print(f"  RMS: {np.sqrt((voltage_diff_valid**2).mean().mean()):.4e} p.u.")
    print(
        f"  Valid comparisons: {voltage_diff_valid.count().sum()} out of {len(voltage_diff_valid) * 3}"
    )
    if voltage_diff_valid.max().max() < 0.01:
        print("  ✓ Excellent agreement (< 1% max difference)")
    elif voltage_diff_valid.max().max() < 0.05:
        print("  ✓ Good agreement (< 5% max difference)")

print("\nNote: Power flow differences may be larger due to different formulations")
print("and solution methods (FBS vs CVXPY matrix-based optimization).")

print("\n" + "=" * 70)
print("FBS solver completed successfully!")
print("FBS is ideal for quick radial network analysis and validation.")
print("=" * 70)
