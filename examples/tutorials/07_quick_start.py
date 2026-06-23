"""
Example 7: Quick Start - 30 Seconds to Your First OPF

The absolute minimal example to get OPF working.
Copy this, change the network name, and you're done!
"""

import distopf as opf
import pandas as pd

# Load network (choose one):
# - ieee13 (smallest, fastest)
# - ieee123 (medium)
# - ieee123_30der (has distributed energy resources)
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")

# Run optimization in 1 line:
result = case.run_opf("loss_min", control_variable="PQ")

# That's it! Results are in the result object
v = result.voltages[["a", "b", "c"]].apply(pd.to_numeric, errors="coerce")
s_base = case.bus_data["s_base"].iloc[0]
print(f"Network voltage range: {v.min().min():.4f} - {v.max().max():.4f} p.u.")
print(f"Objective (losses): {result.objective_value * s_base / 1e6:.6f} MW")
