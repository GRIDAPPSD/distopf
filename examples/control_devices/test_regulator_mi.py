"""
Simple test script for LinDistModelRegulatorMI.

This script tests the mixed-integer regulator tap control model
to identify any issues.
"""

import distopf as opf
import numpy as np
from distopf.matrix_models.lindist_regulator_mi import LinDistModelRegulatorMI

# Load a case with regulators
case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123")

print("=" * 60)
print("Testing LinDistModelRegulatorMI")
print("=" * 60)

# Check regulator data
print(f"\nRegulator data shape: {case.reg_data.shape}")
print(f"Regulator data columns: {list(case.reg_data.columns)}")
print(f"\nRegulator data:\n{case.reg_data}")

# Create the MI model
print("\n" + "-" * 60)
print("Creating LinDistModelRegulatorMI...")
try:
    model = LinDistModelRegulatorMI(
        branch_data=case.branch_data,
        bus_data=case.bus_data,
        gen_data=case.gen_data,
        cap_data=case.cap_data,
        reg_data=case.reg_data,
    )
    print(f"Model created successfully")
    print(f"  n_x (number of variables): {model.n_x}")
    print(f"  reg_buses: {model.reg_buses}")
    print(f"  Number of regulators per phase:")
    print(f"    a: {len(model.reg_buses.get('a', []))}")
    print(f"    b: {len(model.reg_buses.get('b', []))}")
    print(f"    c: {len(model.reg_buses.get('c', []))}")
except Exception as e:
    print(f"Error creating model: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

# Try to solve
print("\n" + "-" * 60)
print("Attempting to solve...")


def loss_objective(model, x, **kwargs):
    """Simple loss minimization objective."""
    loss = 0
    for j in model.branch.index:
        for ph in "abc":
            if not model.phase_exists(ph, j):
                continue
            p_idx = model.idx("pij", j, ph)
            q_idx = model.idx("qij", j, ph)
            r = model.branch.loc[j, f"r{ph}{ph}"]
            loss += r * (x[p_idx] ** 2 + x[q_idx] ** 2)
    return loss


try:
    # First try with CLARABEL (convex solver - won't work for MI but shows if setup is correct)
    print("\nTrying solve with SCIP...")
    result = model.solve(loss_objective, solver="SCIP")

    print(f"\nSolve result:")
    print(f"  Success: {result.success}")
    print(f"  Status: {result.message}")
    print(f"  Objective: {result.fun}")
    print(f"  Iterations: {result.nit}")
    print(f"  Runtime: {result.runtime:.3f}s")

    if result.success:
        # Get regulator taps
        print("\n" + "-" * 60)
        print("Regulator tap results:")
        reg_taps = model.get_regulator_taps()
        print(reg_taps)

        # Get voltages
        voltages = model.get_voltages(result.x)
        print(
            f"\nVoltage range: {voltages[['a', 'b', 'c']].min().min():.4f} - {voltages[['a', 'b', 'c']].max().max():.4f}"
        )

except Exception as e:
    print(f"Error during solve: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete")
print("=" * 60)
