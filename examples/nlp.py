 """
Example: Nonlinear OPF using the new backend='nlp' API.

This example demonstrates how to use the new NLP backend for nonlinear optimal
power flow optimization. The NLP backend uses the BranchFlow model with IPOPT
(for continuous optimization) or MINLP solvers (for discrete controls).

Key features:
- Uses case.run_opf(backend='nlp', ...) for simple API
- Supports optional initialization from FBS results
- Supports discrete control variables (regulators, capacitors)
- Returns standard PowerFlowResult objects
"""

import distopf as opf
from distopf.api import create_case
from distopf.fbs import fbs_solve

# Configuration
start_step = 12
case_path = opf.CASES_DIR / "dss/ieee123_dss/Run_IEEE123Bus.DSS"

# Create case
case = create_case(case_path, start_step=start_step)
print("Case loaded:")
print(case.bus_data)

# Configure case
case.gen_data.control_variable = "P"
case.bus_data.v_max = 2
case.bus_data.v_min = 0
case.gen_data = case.gen_data.iloc[0:0]  # Remove generators
case.bat_data = case.bat_data.iloc[0:0]  # Remove batteries

# Run FBS for initialization
print("\nRunning FBS power flow...")
fbs_results = fbs_solve(case)

# Run NLP OPF with FBS initialization
print("\nRunning NLP OPF with FBS initialization...")
try:
    result = case.run_opf(
        backend="nlp",
        objective="loss",
        initialize="fbs",  # Initialize from FBS results
        solver="ipopt",
        raw_result=False,
    )
    print("NLP OPF completed successfully!")
    print("\nVoltages:")
    print(result.voltages)
    print("\nPower flows:")
    print(result.p_flows)
except Exception as e:
    print(f"NLP OPF failed: {e}")
    print("This may be due to solver unavailability or model infeasibility.")

# Example: Run with discrete controls (requires MINLP solver)
print("\n" + "=" * 60)
print("Example: NLP OPF with discrete controls (requires MINLP solver)")
print("=" * 60)
try:
    result_discrete = case.run_opf(
        backend="nlp",
        objective="loss",
        control_regulators=True,
        control_capacitors=True,
        initialize="fbs",
        solver="bonmin",  # MINLP solver
        raw_result=False,
    )
    print("NLP OPF with discrete controls completed successfully!")
    print("\nVoltages:")
    print(result_discrete.voltages)
except Exception as e:
    print(f"NLP OPF with discrete controls failed: {e}")
    print("This may be due to MINLP solver unavailability or model infeasibility.")

print("\n" + "=" * 60)
print("Example completed!")
print("=" * 60)
