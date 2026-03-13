"""
Example: Comparison of Linear (Pyomo) vs Nonlinear (BranchFlow) OPF.

This example demonstrates how to compare results from the linear wrapper
(wrapper='pyomo') with the nonlinear branchflow formulation (formulation='branchflow')
using the unified Case.run_opf() API.

Key features:
- Compares linear and nonlinear optimization results
- Uses FBS initialization for the branchflow formulation
- Demonstrates result extraction and comparison
- Shows how to handle solver availability gracefully
- Visualizes voltage and power flow comparisons across algorithms
"""

import distopf as opf
import pandas as pd
import numpy as np
from distopf.api import create_case
from distopf.fbs import fbs_solve
from distopf.dss_importer import DSSToCSVConverter
from distopf.plot import plot_voltage_vs_distance, plot_line_flow_vs_distance

# Configuration
start_step = 12
case_path = opf.CASES_DIR / "dss/ieee123_dss/Run_IEEE123Bus.DSS"

# Create case
case = create_case(case_path, start_step=start_step)
print("Case loaded:")
print(case.bus_data)

# Configure case
case.gen_data.control_variable = "P"
case.bus_data.v_max = 2.0
case.bus_data.v_min = 0.0
case.gen_data = case.gen_data.iloc[0:0]  # Remove generators
case.bat_data = case.bat_data.iloc[0:0]  # Remove batteries

# Get DSS reference solution
print("\nExtracting DSS reference solution...")
try:
    dss_parser = DSSToCSVConverter(case_path)
    v_dss = dss_parser.v_solved
    v_dss = v_dss.reset_index(drop=True)
    v_dss = pd.merge(case.bus_data.loc[:, ["id", "name"]], v_dss, on=["name"])
    v_dss["algorithm"] = "dss"
    print("DSS solution extracted")
except Exception as e:
    print(f"Could not extract DSS solution: {e}")
    v_dss = None

# Run FBS power flow
print("\nRunning FBS power flow...")
fbs_results = fbs_solve(case)
v_fbs = fbs_results.voltages
v_fbs["algorithm"] = "fbs"
print("FBS solution completed")

# Run Linear OPF (wrapper='pyomo')
print("\n" + "=" * 60)
print("Running Linear OPF (wrapper='pyomo')")
print("=" * 60)
try:
    result_linear = case.run_opf(
        wrapper="pyomo",
        objective="loss",
        raw_result=False,
    )
    print("Linear OPF completed successfully!")
    v_linear = result_linear.voltages
    v_linear["algorithm"] = "linear"
    print("\nLinear OPF Voltages:")
    print(v_linear.head())
except Exception as e:
    print(f"Linear OPF failed: {e}")
    v_linear = None

# Run Nonlinear OPF (formulation='branchflow')
print("\n" + "=" * 60)
print("Running Nonlinear OPF (formulation='branchflow')")
print("=" * 60)
try:
    result_nlp = case.run_opf(
        formulation="branchflow",
        objective="loss",
        initialize="fbs",  # Initialize from FBS results
        solver="ipopt",
        raw_result=False,
    )
    print("Nonlinear OPF completed successfully!")
    v_nlp = result_nlp.voltages
    v_nlp["algorithm"] = "nlp"
    print("\nNonlinear OPF Voltages:")
    print(v_nlp.head())
except Exception as e:
    print(f"Nonlinear OPF failed: {e}")
    print("This may be due to solver unavailability or model infeasibility.")
    v_nlp = None

# Compare results
print("\n" + "=" * 60)
print("Comparison of Results")
print("=" * 60)

# Collect all voltage results
v_list = [v_fbs]
if v_dss is not None:
    v_list.insert(0, v_dss)
if v_linear is not None:
    v_list.append(v_linear)
if v_nlp is not None:
    v_list.append(v_nlp)

if len(v_list) > 1:
    v_combined = pd.concat(v_list, ignore_index=True)
    print("\nVoltage comparison (first 10 rows):")
    print(v_combined.head(10))

    # Calculate voltage differences
    if v_linear is not None and v_nlp is not None:
        print("\n" + "-" * 60)
        print("Voltage difference between Linear and Nonlinear OPF:")
        print("-" * 60)
        v_linear_sorted = v_linear.sort_values(["id", "t"]).reset_index(drop=True)
        v_nlp_sorted = v_nlp.sort_values(["id", "t"]).reset_index(drop=True)

        for phase in ["a", "b", "c"]:
            if phase in v_linear_sorted.columns and phase in v_nlp_sorted.columns:
                diff = (v_nlp_sorted[phase] - v_linear_sorted[phase]).abs()
                print(f"\nPhase {phase}:")
                print(f"  Max difference: {diff.max():.6f} p.u.")
                print(f"  Mean difference: {diff.mean():.6f} p.u.")
                print(f"  Std deviation: {diff.std():.6f} p.u.")

# Compare power flows
print("\n" + "=" * 60)
print("Power Flow Comparison")
print("=" * 60)

p_dss = None
p_fbs = None
p_linear = None
p_nlp = None

# Extract DSS power flows
print("\nExtracting DSS power flows...")
try:
    dss_parser = DSSToCSVConverter(case_path)
    p_dss = dss_parser.get_p_flows()
    p_dss["algorithm"] = "dss"
    print("DSS Power Flows (first 5 rows):")
    print(p_dss.head())
except Exception as e:
    print(f"Could not extract DSS power flows: {e}")

# Extract FBS power flows
try:
    p_fbs = fbs_results.p_flows
    p_fbs["algorithm"] = "fbs"
    print("\nFBS Power Flows (first 5 rows):")
    print(p_fbs.head())
except Exception as e:
    print(f"Could not extract FBS power flows: {e}")

# Extract Linear OPF power flows
if v_linear is not None:
    try:
        p_linear = result_linear.p_flows
        p_linear["algorithm"] = "linear"
        print("\nLinear OPF Power Flows (first 5 rows):")
        print(p_linear.head())
    except Exception as e:
        print(f"Could not extract Linear OPF power flows: {e}")

# Extract Nonlinear OPF power flows
if v_nlp is not None:
    try:
        p_nlp = result_nlp.p_flows
        p_nlp["algorithm"] = "nlp"
        print("\nNonlinear OPF Power Flows (first 5 rows):")
        print(p_nlp.head())
    except Exception as e:
        print(f"Could not extract Nonlinear OPF power flows: {e}")

# Calculate power flow differences with respect to FBS
if p_fbs is not None and p_linear is not None:
    try:
        print("\n" + "-" * 60)
        print("Power flow difference between Linear OPF and FBS:")
        print("-" * 60)
        # FBS uses tb column, Linear uses id column - they should correspond
        p_fbs_sorted = p_fbs.sort_values(["tb", "t"]).reset_index(drop=True)
        p_linear_sorted = p_linear.sort_values(["id", "t"]).reset_index(drop=True)

        for phase in ["a", "b", "c"]:
            if phase in p_fbs_sorted.columns and phase in p_linear_sorted.columns:
                diff = (p_linear_sorted[phase] - p_fbs_sorted[phase]).abs()
                print(f"\nPhase {phase}:")
                print(f"  Max difference: {diff.max():.6f} kW")
                print(f"  Mean difference: {diff.mean():.6f} kW")
    except Exception as e:
        print(f"Could not compare Linear OPF with FBS: {e}")

if p_fbs is not None and p_nlp is not None:
    try:
        print("\n" + "-" * 60)
        print("Power flow difference between Nonlinear OPF and FBS:")
        print("-" * 60)
        # FBS uses tb column, NLP uses id column - they should correspond
        p_fbs_sorted = p_fbs.sort_values(["tb", "t"]).reset_index(drop=True)
        p_nlp_sorted = p_nlp.sort_values(["id", "t"]).reset_index(drop=True)

        for phase in ["a", "b", "c"]:
            if phase in p_fbs_sorted.columns and phase in p_nlp_sorted.columns:
                diff = (p_nlp_sorted[phase] - p_fbs_sorted[phase]).abs()
                print(f"\nPhase {phase}:")
                print(f"  Max difference: {diff.max():.6f} kW")
                print(f"  Mean difference: {diff.mean():.6f} kW")
    except Exception as e:
        print(f"Could not compare Nonlinear OPF with FBS: {e}")

# Generate visualization plots
print("\n" + "=" * 60)
print("Generating Visualization Plots")
print("=" * 60)

# Plot voltage comparison
if len(v_list) > 1:
    try:
        print("\nGenerating voltage vs distance plot...")

        # Transform voltage data from wide format (a, b, c columns) to tidy format (phase, value columns)
        v_tidy_list = []
        for v_df in v_list:
            if v_df is not None:
                # Melt the dataframe to convert phase columns to rows
                v_melted = v_df.melt(
                    id_vars=["id", "name", "algorithm"],
                    value_vars=["a", "b", "c"],
                    var_name="phase",
                    value_name="value",
                )
                v_tidy_list.append(v_melted)

        if v_tidy_list:
            v_tidy_combined = pd.concat(v_tidy_list, ignore_index=True)
            fig_voltage = plot_voltage_vs_distance(
                case,
                v_tidy_combined,
                title="Voltage Comparison: Linear vs Nonlinear OPF",
                color_by="algorithm",
            )
            fig_voltage.show(renderer="browser")
            print("Voltage plot generated successfully!")
    except Exception as e:
        print(f"Could not generate voltage plot: {e}")

# Plot power flow comparison
if p_fbs is not None or p_dss is not None:
    try:
        print("\nGenerating power flow vs distance plot...")

        # Prepare power flow data for plotting
        # DSS and FBS already have fb, tb, from_name, to_name columns
        # Linear and NLP need to be joined with branch data
        p_plot_list = []

        # Add DSS data (already has the required columns)
        if p_dss is not None:
            p_dss_plot = p_dss.copy()
            p_plot_list.append(p_dss_plot)

        # Add FBS data (already has the required columns)
        if p_fbs is not None:
            p_fbs_plot = p_fbs.copy()
            p_plot_list.append(p_fbs_plot)

        # Add Linear OPF data (needs to be joined with branch data)
        if p_linear is not None:
            branch_data_for_merge = case.branch_data.reset_index()[
                ["fb", "tb", "from_name", "to_name"]
            ]
            p_linear_with_branches = p_linear.merge(
                branch_data_for_merge,
                left_on="id",
                right_on="tb",
                how="left",
            )
            p_plot_list.append(p_linear_with_branches)

        # Add NLP data (needs to be joined with branch data)
        if p_nlp is not None:
            branch_data_for_merge = case.branch_data.reset_index()[
                ["fb", "tb", "from_name", "to_name"]
            ]
            p_nlp_with_branches = p_nlp.merge(
                branch_data_for_merge,
                left_on="id",
                right_on="tb",
                how="left",
            )
            p_plot_list.append(p_nlp_with_branches)

        if len(p_plot_list) > 1:
            # Combine all power flow data
            p_combined = pd.concat(p_plot_list, ignore_index=True)

            # Melt the dataframe to convert phase columns to rows
            # Need columns: fb, from_name, tb, to_name, algorithm, phase, value
            p_tidy = p_combined.melt(
                id_vars=["fb", "from_name", "tb", "to_name", "algorithm"],
                value_vars=["a", "b", "c"],
                var_name="phase",
                value_name="value",
            )

            fig_flow = plot_line_flow_vs_distance(
                case,
                p_tidy,
                flow_name="Active Power",
                title="Power Flow Comparison: DSS vs FBS vs Linear vs Nonlinear OPF",
                color_by="algorithm",
            )
            fig_flow.show(renderer="browser")
            print("Power flow plot generated successfully!")
    except Exception as e:
        print(f"Could not generate power flow plot: {e}")

print("\n" + "=" * 60)
print("Comparison completed!")
print("=" * 60)
