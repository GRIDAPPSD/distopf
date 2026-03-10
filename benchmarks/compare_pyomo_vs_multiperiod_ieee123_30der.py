#!/usr/bin/env python
"""
Benchmark: Compare Pyomo (IPOPT) vs Multiperiod (CLARABEL) on ieee123_30der.

This script compares the linear OPF results from two different backends:
- Pyomo backend with IPOPT solver
- Multiperiod matrix backend with CLARABEL solver

Both solve the same case with the same objective (loss minimization).
The script records solver status, termination conditions, and voltage deltas.

Usage:
    uv run python benchmarks/compare_pyomo_vs_multiperiod_ieee123_30der.py

Output:
    - Prints summary table to stdout
    - Writes detailed report to benchmarks/results/<timestamp>_comparison.md
    - Writes JSON results to benchmarks/results/<timestamp>_comparison.json
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path

import distopf as opf
import distopf.matrix_models.multiperiod as mpopf
from distopf.api import create_case
from distopf import CASES_DIR

from distopf.benchmarking import (
    SolverResult,
    compare_results,
    format_comparison_table,
    format_detailed_report,
)


def run_pyomo_backend(case) -> SolverResult:
    """Run OPF using Pyomo backend with IPOPT."""
    print("\n" + "=" * 60)
    print("Running Pyomo backend (IPOPT)...")
    print("=" * 60)

    start_time = time.time()
    try:
        result = case.run_opf(
            objective="loss",
            control_variable="",
            backend="pyomo",
            raw_result=True,
        )
        solve_time = time.time() - start_time

        # Extract solver status from raw result
        status = "ok"
        termination_condition = None
        error_message = None

        if hasattr(result, "solver") and hasattr(result.solver, "status"):
            status = str(result.solver.status)
            if hasattr(result.solver, "termination_condition"):
                termination_condition = str(result.solver.termination_condition)

        # Extract results
        voltages = result.voltages if hasattr(result, "voltages") else None
        p_flows = result.p_flows if hasattr(result, "p_flows") else None
        q_flows = result.q_flows if hasattr(result, "q_flows") else None
        p_gens = result.p_gens if hasattr(result, "p_gens") else None
        q_gens = result.q_gens if hasattr(result, "q_gens") else None

        # Compute objective value (sum of losses)
        objective_value = None
        if p_flows is not None:
            # Losses are typically in the p_flows dataframe
            if "loss" in p_flows.columns:
                objective_value = p_flows["loss"].sum()

        success = status.lower() == "ok"

        print(f"✓ Pyomo solve completed in {solve_time:.2f}s")
        print(f"  Status: {status}")
        if termination_condition:
            print(f"  Termination: {termination_condition}")

        return SolverResult(
            backend="pyomo",
            solver="ipopt",
            case_name="ieee123_30der",
            converged=success,
            solver_status=status,
            termination_condition=termination_condition,
            error_message=error_message,
            objective_value=objective_value,
            voltages=voltages,
            p_flows=p_flows,
            q_flows=q_flows,
            p_gens=p_gens,
            q_gens=q_gens,
            solve_time=solve_time,
        )

    except Exception as e:
        solve_time = time.time() - start_time
        error_msg = str(e)
        print(f"✗ Pyomo solve failed: {error_msg}")

        return SolverResult(
            backend="pyomo",
            solver="ipopt",
            case_name="ieee123_30der",
            converged=False,
            solver_status="error",
            termination_condition=None,
            error_message=error_msg,
            objective_value=None,
            voltages=None,
            p_flows=None,
            q_flows=None,
            p_gens=None,
            q_gens=None,
            solve_time=solve_time,
        )


def run_multiperiod_backend(case) -> SolverResult:
    """Run OPF using multiperiod matrix backend with CLARABEL."""
    print("\n" + "=" * 60)
    print("Running Multiperiod backend (CLARABEL)...")
    print("=" * 60)

    start_time = time.time()
    try:
        # Create multiperiod model
        m = mpopf.LinDistMPL(case=case)

        # Solve with CLARABEL
        result = mpopf.cvxpy_solve(m, mpopf.cp_obj_loss, solver="CLARABEL")
        solve_time = time.time() - start_time

        # Extract results
        voltages = m.get_voltages(result.x)
        p_flows = m.get_p_flows(result.x)
        q_flows = m.get_q_flows(result.x)
        p_gens = m.get_p_gens(result.x)
        q_gens = m.get_q_gens(result.x)

        # Solver status
        status = "ok" if result.success else "failed"
        termination_condition = result.message if hasattr(result, "message") else None

        print(f"✓ Multiperiod solve completed in {solve_time:.2f}s")
        print(f"  Status: {status}")
        if termination_condition:
            print(f"  Message: {termination_condition}")
        print(f"  Objective value: {result.fun:.6f}")

        return SolverResult(
            backend="multiperiod",
            solver="clarabel",
            case_name="ieee123_30der",
            converged=result.success,
            solver_status=status,
            termination_condition=termination_condition,
            error_message=None,
            objective_value=result.fun,
            voltages=voltages,
            p_flows=p_flows,
            q_flows=q_flows,
            p_gens=p_gens,
            q_gens=q_gens,
            solve_time=solve_time,
        )

    except Exception as e:
        solve_time = time.time() - start_time
        error_msg = str(e)
        print(f"✗ Multiperiod solve failed: {error_msg}")

        return SolverResult(
            backend="multiperiod",
            solver="clarabel",
            case_name="ieee123_30der",
            converged=False,
            solver_status="error",
            termination_condition=None,
            error_message=error_msg,
            objective_value=None,
            voltages=None,
            p_flows=None,
            q_flows=None,
            p_gens=None,
            q_gens=None,
            solve_time=solve_time,
        )


def main():
    """Run the benchmark."""
    print("\n" + "=" * 60)
    print("distOPF Backend Comparison Benchmark")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Create case
    print("\nLoading case: csv/ieee123_30der")
    case = create_case(
        data_path=CASES_DIR / "csv" / "ieee123_30der",
        n_steps=1,
        start_step=0,
    )

    # Configure case for comparison
    case.gen_data.control_variable = ""
    case.schedules.default = 1
    case.schedules.PV = 1
    case.bat_data = case.bat_data.head(0)  # Remove batteries

    print(f"  Buses: {len(case.bus_data)}")
    print(f"  Branches: {len(case.branch_data)}")
    print(f"  Generators: {len(case.gen_data)}")
    print(f"  Capacitors: {len(case.cap_data)}")

    # Run both backends
    result_pyomo = run_pyomo_backend(case)
    result_multiperiod = run_multiperiod_backend(case)

    # Compare results
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)

    comparison = compare_results(result_pyomo, result_multiperiod)

    # Print summary table
    print("\n" + format_comparison_table([comparison]))

    # Print detailed report
    detailed_report = format_detailed_report([comparison])
    print("\n" + detailed_report)

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_file = results_dir / f"{timestamp}_comparison.json"
    with open(json_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "case": "ieee123_30der",
                "objective": "loss",
                "pyomo": result_pyomo.to_dict(),
                "multiperiod": result_multiperiod.to_dict(),
                "comparison": comparison.to_dict(),
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\n✓ JSON results saved to: {json_file}")

    # Save markdown report
    md_file = results_dir / f"{timestamp}_comparison.md"
    with open(md_file, "w") as f:
        f.write(detailed_report)
    print(f"✓ Markdown report saved to: {md_file}")

    # Exit with appropriate code
    if not result_pyomo.converged or not result_multiperiod.converged:
        print("\n⚠️  One or both backends failed to solve.")
        return 1

    print("\n✓ Benchmark completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
