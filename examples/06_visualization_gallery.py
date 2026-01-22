"""
Example 6: Visualization Gallery

Shows all the visualization options available in DistOPF.
Each plot tells a different story about the grid state.
"""

import distopf as opf

# Load the IEEE 123-bus network with DERs
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")

# Run OPF
result = case.run_opf("loss_min", control_variable="PQ", backend="matrix")

print("=" * 60)
print("DistOPF Visualization Suite")
print("=" * 60)

# Plot 1: Network diagram with voltage coloring
print("\n1. Network Plot (voltage heatmap)")
print("   Shows the entire network topology with voltage levels")
print("   Green = nominal voltage, Red = low, Blue = high")
result.plot_network(v_min=0.95, v_max=1.05).show(renderer="browser")

# Plot 2: Voltage profile
print("\n2. Voltage Profile Plot")
print("   Shows voltage magnitude at each bus")
result.plot_voltages().show(renderer="browser")

# Plot 3: Power flows
print("\n3. Power Flow Plot")
print("   Shows active and reactive power flows on each line")
result.plot_power_flows().show(renderer="browser")

# Plot 4: Generator outputs
print("\n4. Generator Output Plot")
print("   Shows active and reactive power from each DER")
result.plot_gens().show(renderer="browser")
