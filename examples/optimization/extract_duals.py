"""
Example: Extracting Dual Variables from OPF Solutions

Dual variables (Lagrange multipliers) represent the marginal cost of relaxing
each constraint. They're useful for sensitivity analysis and identifying
binding constraints.
"""

import distopf as opf

# Create and solve case with duals
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee13")
result = case.run_opf(objective="loss", wrapper="pyomo", duals=True)

# Access pre-extracted duals
print("Power Balance (Active) Duals:")
print(result.raw_result.dual_power_balance_p.head())

# Generic extraction for any constraint
print("\nAll constraints with duals:")
all_duals = result.raw_result.get_all_duals()
for name in all_duals:
    print(f"  {name}")

# Identify binding constraints (high dual magnitude)
pb_duals = result.raw_result.dual_power_balance_p.copy()
pb_duals["abs_dual"] = pb_duals["dual"].abs()
print("\nTop 5 binding constraints:")
print(pb_duals.nlargest(5, "abs_dual")[["id", "name", "dual"]])
