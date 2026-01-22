"""
Basic power flow example using the new Case API.

This example demonstrates running a power flow analysis on a distribution network.
The new API uses create_case() to load data and run_pf() for power flow analysis.
"""

import distopf as opf

# Load the case using the new API
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")

# Run power flow (no optimization, just solve the base case)
voltages, power_flows = case.run_pf()

# Display results
print("Voltage magnitudes (first 10 buses):")
print(voltages.head(10))

# Plot the network with voltage results
case.plot_network().show(renderer="browser")
