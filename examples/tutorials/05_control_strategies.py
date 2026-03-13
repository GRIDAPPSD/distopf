"""
Example 5: Generator Control Strategies

Shows how different generator control strategies affect the network.
Compare fixed output vs. controlled output.

Demonstrates the flexibility of specifying different control modes per generator.
"""

import distopf as opf

# Load the IEEE 123-bus network with DERs
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
s_base = case.bus_data["s_base"].iloc[0]

print("=" * 60)
print("Generator Control Strategy Comparison -- Loss Minimization Objective")
print("=" * 60)

# Scenario 1: Only substation controls active power (P)
print("\nScenario 1: Only substation active power control")
case1 = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
# Set all DERs to fixed output (no control)
case1.gen_data["control_variable"] = ""
result1 = case1.run_opf("loss_min", wrapper="matrix")
v1 = result1.voltages[["a", "b", "c"]]
print(f"  Min voltage: {v1.min().min():.4f} p.u.")
print(f"  Max voltage: {v1.max().max():.4f} p.u.")
print(f"  Losses: {result1.objective_value * s_base / 1e6:.6f} MW")

# Scenario 2: All generators control active and reactive power
print("\nScenario 2: All generators control P and Q")
case2 = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
case2.gen_data["control_variable"] = "PQ"
result2 = case2.run_opf("loss_min", wrapper="pyomo")
v2 = result2.voltages[["a", "b", "c"]]
print(f"  Min voltage: {v2.min().min():.4f} p.u.")
print(f"  Max voltage: {v2.max().max():.4f} p.u.")
print(f"  Losses: {result2.objective_value * s_base / 1e6:.6f} MW")

# Scenario 3: DERs control Q only
print("\nScenario 3: DERs control Q only")
case3 = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
case3.gen_data["control_variable"] = "Q"
result3 = case3.run_opf("loss_min", wrapper="matrix")
v3 = result3.voltages[["a", "b", "c"]]
print(f"  Min voltage: {v3.min().min():.4f} p.u.")
print(f"  Max voltage: {v3.max().max():.4f} p.u.")
print(f"  Losses: {result3.objective_value * s_base / 1e6:.6f} MW")
