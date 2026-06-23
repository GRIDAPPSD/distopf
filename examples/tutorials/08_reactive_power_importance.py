"""
Example 8: Reactive Power - When and Why

Demonstrates the importance of reactive power in distribution networks.
Compare scenarios with and without reactive power optimization.
"""

import distopf as opf
import pandas as pd

print("=" * 60)
print("The Role of Reactive Power in Distribution Networks")
print("=" * 60)

case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
s_base = case.bus_data["s_base"].iloc[0]

# Scenario A: No reactive power control
print("\nScenario A: DERs provide only active power (P only)")
case_a = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
case_a.gen_data["control_variable"] = "P"
result_a = case_a.run_opf("loss_min", wrapper="matrix")
v_a = result_a.voltages[["a", "b", "c"]].apply(pd.to_numeric, errors="coerce")
vmin_a = v_a.min().min()
vmax_a = v_a.max().max()
print(f"  Voltage range: {vmin_a:.4f} - {vmax_a:.4f} p.u.")
print(f"  Voltage spread: {vmax_a - vmin_a:.4f} p.u.")
print(f"  Losses: {result_a.objective_value * s_base / 1e6:.6f} MW")

# Scenario B: With reactive power control
print("\nScenario B: DERs provide active and reactive power (PQ)")
case_b = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
case_b.gen_data["control_variable"] = "PQ"
result_b = case_b.run_opf("loss_min", wrapper="matrix")
v_b = result_b.voltages[["a", "b", "c"]].apply(pd.to_numeric, errors="coerce")
vmin_b = v_b.min().min()
vmax_b = v_b.max().max()
print(f"  Voltage range: {vmin_b:.4f} - {vmax_b:.4f} p.u.")
print(f"  Voltage spread: {vmax_b - vmin_b:.4f} p.u.")
print(f"  Losses: {result_b.objective_value * s_base / 1e6:.6f} MW")

# Calculate improvement
improvement = ((vmax_a - vmin_a) - (vmax_b - vmin_b)) / (vmax_a - vmin_a) * 100
loss_reduction = (
    (result_a.objective_value - result_b.objective_value)
    / result_a.objective_value
    * 100
)
print(f"\n  Improvement in voltage regulation: {improvement:.1f}%")
print(f"  Loss reduction with Q control: {loss_reduction:.1f}%")
