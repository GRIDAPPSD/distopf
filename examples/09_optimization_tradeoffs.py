"""
Example 9: Network Optimization Trade-offs

Shows the trade-off between different objectives:
- Minimize losses (cheap to operate)
- Minimize voltage deviations (safe and stable)
- Minimize control action (less switching)

In practice, you might combine these with weights.
"""

import distopf as opf
import pandas as pd

print("=" * 60)
print("Optimization Objectives - Finding the Right Balance")
print("=" * 60)

case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
s_base = case.bus_data["s_base"].iloc[0]

# Objective 1: Minimize losses
print("\n1. Loss Minimization Objective")
case1 = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
result1 = case1.run_opf("loss_min", control_variable="PQ", backend="matrix")
# Extract numeric voltage data (phases a, b, c)
v1 = result1.voltages[["a", "b", "c"]].apply(pd.to_numeric, errors="coerce")
print(f"   Voltage min: {v1.min().min():.4f} p.u.")
print(f"   Voltage max: {v1.max().max():.4f} p.u.")
print(f"   Losses: {result1.objective_value * s_base / 1e6:.6f} MW")
print("   Focus: Efficiency - minimize energy waste")

# Objective 2: Q-only control for voltage support (less aggressive than PQ)
print("\n2. Q-Control Only (Voltage Support Focused)")
case2 = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
result2 = case2.run_opf("loss_min", control_variable="Q", backend="matrix")
# Extract numeric voltage data (phases a, b, c)
v2 = result2.voltages[["a", "b", "c"]].apply(pd.to_numeric, errors="coerce")
print(f"   Voltage min: {v2.min().min():.4f} p.u.")
print(f"   Voltage max: {v2.max().max():.4f} p.u.")
print(f"   Losses: {result2.objective_value * s_base / 1e6:.6f} MW")
print("   Focus: Safety via reactive power - tighter voltage control")

print("\n" + "=" * 60)
print("Trade-off Analysis:")
volt_range_1 = v1.max().max() - v1.min().min()
volt_range_2 = v2.max().max() - v2.min().min()
print(
    f"  PQ control voltage range: {volt_range_1:.4f} p.u. (Losses: {result1.objective_value * s_base / 1e6:.6f} MW)"
)
print(
    f"  Q-only voltage range: {volt_range_2:.4f} p.u. (Losses: {result2.objective_value * s_base / 1e6:.6f} MW)"
)
print()
print("Different control strategies suit different scenarios:")
print("- PQ control: Flexible (both P and Q), best for loss minimization")
print("- Q-only: More conservative, prioritizes voltage stability")
print("- P-only: Limited control, fewer switching actions")
print("=" * 60)
