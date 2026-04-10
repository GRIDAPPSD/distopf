"""Example demonstrating temporal decomposition algorithms (TENAPP).

This example shows how to use three temporal decomposition algorithms:
- TENAPP-1O: First-order temporal decomposition
- TENAPP-APRX: Approximate dual method
- TENAPP-ADMM: Alternating Direction Method of Multipliers

These algorithms solve multi-period optimal power flow problems by decomposing
them into per-time-step subproblems, coordinating through battery SOC constraints.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import distopf as opf
from distopf.distributed.temporal import (
    solve_tenapp_1o,
    solve_tenapp_aprx,
    solve_tenapp_admm,
    energy_cost_min,
)

OUTPUT_DIR = Path("scratch/temporal_results")


def main():
    """Run all three temporal decomposition algorithms on IEEE 123-bus test case."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Load multi-period test case
    # ========================================================================
    print("Loading test case...")
    case = opf.create_case(
        opf.CASES_DIR / "csv" / "ieee123_30der_bat",
        n_steps=4,  # 4-hour dispatch with 1-hour steps (can increase for longer horizon)
        start_step=0,
        ignore_gen=False,
    )

    # Configure case parameters
    case.bus_data.v_max = 1.05
    case.bus_data.v_min = 0.95
    case.gen_data.control_variable = "PQ"
    case.bat_data.control_variable = "P"

    # Set battery initial conditions
    if len(case.bat_data) > 0:
        case.bat_data["start_soc"] = 0.5  # 50% initial SOC
        case.bat_data["max_soc"] = 0.9  # 90% max SOC
        case.bat_data["min_soc"] = 0.1  # 10% min SOC

    print(f"  Test case: {len(case.bus_data)} buses, {len(case.branch_data)} lines")
    print(f"  Time periods: {case.n_steps}, Battery units: {len(case.bat_data)}")
    print(f"  Forecast horizon: {case.n_steps} hours\n")

    # ========================================================================
    # Prepare common parameters
    # ========================================================================
    # Create hourly cost curve ($/MWh) - can modify for realistic pricing
    cost_curve = np.ones(case.n_steps) * 50.0  # Flat $50/MWh
    # Add peak hour pricing
    if case.n_steps >= 4:
        cost_curve[1:3] = 100.0  # Peak hours

    kwargs = {
        "cost_curve": cost_curve,
        "solver": "CLARABEL",  # or "SCS", "MOSEK", etc.
    }

    print(f"Hourly electricity costs: {cost_curve} $/MWh\n")

    # ========================================================================
    # Algorithm 1: TENAPP-1O (First-Order)
    # ========================================================================
    print("=" * 70)
    print("RUNNING TENAPP-1O (First-Order Temporal Decomposition)")
    print("=" * 70)

    result_1o = solve_tenapp_1o(
        case,
        objective=energy_cost_min,
        max_iterations=20,
        tolerance=1e-3,
        **kwargs,
    )

    print_results("TENAPP-1O", result_1o)

    # ========================================================================
    # Algorithm 2: TENAPP-APRX (Approximate Dual)
    # ========================================================================
    print("\n" + "=" * 70)
    print("RUNNING TENAPP-APRX (Approximate Dual Temporal Decomposition)")
    print("=" * 70)

    result_aprx = solve_tenapp_aprx(
        case,
        objective=energy_cost_min,
        max_iterations=20,
        tolerance=1e-3,
        **kwargs,
    )

    print_results("TENAPP-APRX", result_aprx)

    # ========================================================================
    # Algorithm 3: TENAPP-ADMM (ADMM-based)
    # ========================================================================
    print("\n" + "=" * 70)
    print("RUNNING TENAPP-ADMM (ADMM-based Temporal Decomposition)")
    print("=" * 70)

    result_admm = solve_tenapp_admm(
        case,
        objective=energy_cost_min,
        max_iterations=20,
        tolerance=1e-3,
        weight=1e2,
        weight_scale=1.0,
        **kwargs,
    )

    print_results("TENAPP-ADMM", result_admm)

    # ========================================================================
    # Extract and display results
    # ========================================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY & COMPARISON")
    print("=" * 70)

    summary_df = pd.DataFrame(
        {
            "Algorithm": ["TENAPP-1O", "TENAPP-APRX", "TENAPP-ADMM"],
            "Total Cost ($)": [
                result_1o.objective_value,
                result_aprx.objective_value,
                result_admm.objective_value,
            ],
            "Iterations": [
                result_1o.iterations,
                result_aprx.iterations,
                result_admm.iterations,
            ],
            "Converged": [
                result_1o.converged,
                result_aprx.converged,
                result_admm.converged,
            ],
            "Solve Time (s)": [
                result_1o.solve_time,
                result_aprx.solve_time,
                result_admm.solve_time,
            ],
        }
    )

    print("\n" + summary_df.to_string(index=False))
    summary_df.to_csv(OUTPUT_DIR / "temporal_results_summary.csv", index=False)
    print(f"\nSummary saved to: {OUTPUT_DIR / 'temporal_results_summary.csv'}")

    # ========================================================================
    # Save detailed iteration histories
    # ========================================================================
    print("\n" + "=" * 70)
    print("SAVING DETAILED RESULTS")
    print("=" * 70)

    result_1o.raw_result["iteration_summaries"].to_csv(
        OUTPUT_DIR / "tenapp_1o_iterations.csv", index=False
    )
    result_aprx.raw_result["iteration_summaries"].to_csv(
        OUTPUT_DIR / "tenapp_aprx_iterations.csv", index=False
    )
    result_admm.raw_result["iteration_summaries"].to_csv(
        OUTPUT_DIR / "tenapp_admm_iterations.csv", index=False
    )

    print(f"✓ TENAPP-1O iterations: {OUTPUT_DIR / 'tenapp_1o_iterations.csv'}")
    print(f"✓ TENAPP-APRX iterations: {OUTPUT_DIR / 'tenapp_aprx_iterations.csv'}")
    print(f"✓ TENAPP-ADMM iterations: {OUTPUT_DIR / 'tenapp_admm_iterations.csv'}")

    # ========================================================================
    # Extract battery dispatch from results
    # ========================================================================
    print("\n" + "=" * 70)
    print("BATTERY DISPATCH ANALYSIS")
    print("=" * 70)

    for algo_name, result in [
        ("TENAPP-1O", result_1o),
        ("TENAPP-APRX", result_aprx),
        ("TENAPP-ADMM", result_admm),
    ]:
        print(f"\n{algo_name}:")
        # SOC is available directly on the PowerFlowResult
        soc = result.soc
        if soc is not None and not soc.empty:
            soc_t0 = soc.loc[soc["t"] == 0]
            avg_soc = soc_t0["value"].mean() if not soc_t0.empty else float("nan")
            print(f"  Battery SOC (t=0): {avg_soc:.2%}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print(f"Number of algorithms compared: 3")
    print(f"Time periods solved: {case.n_steps}")
    print(f"Total buses: {len(case.bus_data)}")
    print(f"Total lines: {len(case.branch_data)}")
    print(f"Battery units: {len(case.bat_data)}")


def print_results(algo_name, result):
    """Print formatted results for an algorithm (result is PowerFlowResult)."""
    print(f"\n{algo_name} Results:")
    print(f"  Total Cost: ${result.objective_value:.2f}")
    print(f"  Iterations: {result.iterations}")
    print(f"  Converged: {'✓ Yes' if result.converged else '✗ No'}")
    print(f"  Solve Time: {result.solve_time:.3f}s")

    summaries = result.raw_result.get("iteration_summaries", None)
    if summaries is not None and not summaries.empty:
        print(f"  Final Objective: ${summaries['objective'].iloc[-1]:.2f}")
        print(
            f"  Time-per-iteration: {result.solve_time / max(result.iterations, 1):.3f}s"
        )


if __name__ == "__main__":
    main()
