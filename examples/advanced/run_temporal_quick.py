"""Quick example: Temporal decomposition for multi-period battery scheduling.

Demonstrates basic usage of TENAPP algorithms.
"""

import numpy as np
import distopf as opf
from distopf.distributed.temporal import solve_tenapp_1o, energy_cost_min

# Create a 6-hour multi-period case for the IEEE 123-bus network
case = opf.create_case(
    opf.CASES_DIR / "csv" / "ieee123_30der_bat",
    n_steps=6,  # 6 hours
    start_step=0,
)

# Set battery initial conditions
case.bat_data["start_soc"] = 0.5  # 50% initial state of charge
case.bat_data["max_soc"] = 0.9

# Realistic hourly electricity prices ($/MWh)
cost_curve = np.array([40, 50, 80, 90, 70, 45])  # Peak at hours 3-4

# Solve using TENAPP-1O algorithm
result = solve_tenapp_1o(
    case,
    objective=energy_cost_min,
    max_iterations=15,
    tolerance=1e-3,
    cost_curve=cost_curve,
    solver="CLARABEL",
)

# Print results  (result is a PowerFlowResult)
print(f"✓ Converged: {result.converged}")
print(f"✓ Total Cost: ${result.objective_value:.2f}")
print(f"✓ Iterations: {result.iterations}")
print(f"✓ Time: {result.solve_time:.3f}s")
print(f"\nIteration History:")
summaries = result.raw_result["iteration_summaries"]
print(summaries[["iteration", "t", "objective"]].to_string())
