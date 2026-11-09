"""
Basic optimal power flow example using the new Case API.

This example demonstrates running optimal power flow on a distribution network
with DERs (distributed energy resources). The new API uses:
- create_case() to load data
- run_opf() with objective function and control variables
"""

import distopf as opf
from distopf import plot_gens, plot_polar

# Load the case using the new API
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der", ignore_schedule=True)

# Run optimal power flow with loss minimization objective
# control_variable="PQ" allows both active and reactive power control
result = case.run_opf("loss_min", control_variable="PQ", circular_constraints=True)

# To get the raw optimization result, use raw_result=True:
# result = case.run_opf("loss_min", control_variable="PQ", raw_result=True)
# print(f"Objective value: {result.fun}")

# Plot results
result.plot_voltages().show(renderer="browser")
result.plot_power_flows().show(renderer="browser")
plot_gens(result.active_power_generation, result.reactive_power_generation).show(
    renderer="browser"
)
plot_polar(result.active_power_generation, result.reactive_power_generation).show(
    renderer="browser"
)
result.plot_network(show_reactive_power=True).show(renderer="browser")
