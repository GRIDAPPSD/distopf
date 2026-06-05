"""Comprehensive example comparing all three temporal decomposition algorithms.

This example:
1. Creates a multi-period OPF case
2. Solves it with all three TENAPP algorithms
3. Compares solution cost, convergence, and computation time
4. Visualizes results
"""

from pathlib import Path
import numpy as np
import pandas as pd
import distopf as opf
from distopf.distributed.temporal import (
    solve_tenapp_1o,
    solve_tenapp_aprx,
    solve_tenapp_admm,
    energy_cost_min,
)


def create_multi_period_case(n_steps=8, case_name="ieee123_30der_bat"):
    """Create a multi-period case for comparison."""
    print(f"Creating {n_steps}-step case from {case_name}...")
    case = opf.create_case(
        opf.CASES_DIR / "csv" / case_name,
        n_steps=n_steps,
        start_step=0,
    )

    # Configure voltage limits
    case.bus_data.v_max = 1.05
    case.bus_data.v_min = 0.95

    # Configure generator control
    case.gen_data.control_variable = "PQ"  # Both active and reactive power

    # Configure battery control and initial conditions
    if len(case.bat_data) > 0:
        case.bat_data.control_variable = "P"  # Active power control only
        case.bat_data["start_soc"] = 0.5  # 50% initial SOC
        case.bat_data["max_soc"] = 0.9  # 90% max SOC
        case.bat_data["min_soc"] = 0.1  # 10% min SOC

    print(
        f"  Buses: {len(case.bus_data)}, Lines: {len(case.branch_data)}, "
        f"Batteries: {len(case.bat_data)}\n"
    )
    return case


def create_pricing_signals(n_steps, peak_hours=(2, 3)):
    """Create hourly electricity pricing with peak hours."""
    price = np.ones(n_steps) * 50.0  # Base price: $50/MWh
    for hour in peak_hours:
        if hour < n_steps:
            price[hour] = 120.0  # Peak price: $120/MWh
    return price


def run_all_algorithms(case, cost_curve, output_dir="scratch/temporal_comparison"):
    """Run all three TENAPP algorithms and return results."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Common solver parameters
    common_kwargs = {
        "cost_curve": cost_curve,
        "solver": "CLARABEL",
    }

    algorithms = {
        "TENAPP-1O": lambda: solve_tenapp_1o(
            case,
            objective=energy_cost_min,
            max_iterations=30,
            tolerance=1e-3,
            **common_kwargs,
        ),
        "TENAPP-APRX": lambda: solve_tenapp_aprx(
            case,
            objective=energy_cost_min,
            max_iterations=30,
            tolerance=1e-3,
            **common_kwargs,
        ),
        "TENAPP-ADMM": lambda: solve_tenapp_admm(
            case,
            objective=energy_cost_min,
            max_iterations=30,
            tolerance=1e-3,
            weight=1e2,
            weight_scale=1.0,
            **common_kwargs,
        ),
    }

    results = {}
    for algo_name, solve_func in algorithms.items():
        print(f"Running {algo_name}...", end=" ", flush=True)
        try:
            result = solve_func()
            results[algo_name] = result
            status = "✓ OK" if result.converged else "⚠ Not converged"
            cost = result.objective_value
            iters = result.iterations
            print(f"{status} [Cost: ${cost:.2f}, Iterations: {iters}]")
        except Exception as e:
            print(f"✗ FAILED: {str(e)}")
            results[algo_name] = None

    return results, output_dir


def compare_results(results, output_dir):
    """Compare results across all algorithms."""

    print("\n" + "=" * 80)
    print("COMPARISON OF TEMPORAL DECOMPOSITION ALGORITHMS")
    print("=" * 80)

    # Build comparison table
    comparison = []
    for algo_name, result in results.items():
        if result is not None:
            comparison.append(
                {
                    "Algorithm": algo_name,
                    "Total Cost ($)": result.objective_value,
                    "Iterations": result.iterations,
                    "Converged": "✓" if result.converged else "✗",
                    "Time (s)": f"{result.solve_time:.3f}",
                    "Time/Iter (ms)": f"{1000 * result.solve_time / max(result.iterations, 1):.1f}",
                }
            )

    comparison_df = pd.DataFrame(comparison)
    print("\n" + comparison_df.to_string(index=False))

    # Save to CSV
    comparison_df.to_csv(output_dir / "algorithm_comparison.csv", index=False)
    print(f"\n✓ Comparison saved to: {output_dir / 'algorithm_comparison.csv'}")

    # Show cost difference
    print("\n" + "=" * 80)
    print("COST ANALYSIS")
    print("=" * 80)
    valid_costs = [r.objective_value for r in results.values() if r is not None]
    if valid_costs:
        min_cost = min(valid_costs)
        max_cost = max(valid_costs)
        print(f"Minimum cost: ${min_cost:.2f}")
        print(f"Maximum cost: ${max_cost:.2f}")
        print(
            f"Cost spread: ${max_cost - min_cost:.2f} ({100 * (max_cost - min_cost) / min_cost:.1f}%)"
        )

    return comparison_df


def extract_battery_schedule(result, time_step=0):
    """Extract battery scheduling summary from a PowerFlowResult."""
    if result is None or result.soc is None:
        return None

    soc_t = result.soc.loc[result.soc["t"] == time_step]
    p_batt_t = (
        result.battery_active_power.loc[result.battery_active_power["t"] == time_step]
        if result.battery_active_power is not None
        else None
    )

    schedule = {"time_step": time_step}
    schedule["avg_soc"] = float(soc_t["value"].mean()) if not soc_t.empty else None
    if p_batt_t is not None and not p_batt_t.empty:
        phase_cols = [c for c in ["a", "b", "c"] if c in p_batt_t.columns]
        schedule["total_power_pu"] = float(p_batt_t[phase_cols].sum(axis=1).sum())
    return schedule


def summarize_dispatch(results):
    """Summarize battery dispatch for each algorithm."""

    print("\n" + "=" * 80)
    print("BATTERY DISPATCH SUMMARY (First Hour)")
    print("=" * 80)

    for algo_name, result in results.items():
        if result is not None:
            schedule = extract_battery_schedule(result, time_step=0)
            if schedule:
                power = schedule.get("total_power_pu", 0)
                soc = schedule.get("avg_soc", 0)
                print(
                    f"{algo_name:12} | Power: {power:8.4f} p.u. | Avg SOC: {soc:6.1%}"
                )


def main():
    """Run complete comparison example."""

    print("=" * 80)
    print("TEMPORAL DECOMPOSITION ALGORITHM COMPARISON")
    print("=" * 80 + "\n")

    # Setup
    case = create_multi_period_case(n_steps=6)
    cost_curve = create_pricing_signals(6, peak_hours=(2, 3))

    print("Electricity pricing ($/MWh):")
    for t, price in enumerate(cost_curve):
        print(f"  Hour {t}: ${price:.0f}/MWh")
    print()

    # Run algorithms
    results, output_dir = run_all_algorithms(case, cost_curve)
    print()

    # Compare and analyze
    comparison_df = compare_results(results, output_dir)
    summarize_dispatch(results)

    # Save iteration histories
    print("\n" + "=" * 80)
    print("SAVING ITERATION HISTORIES")
    print("=" * 80)

    for algo_name, result in results.items():
        if result is not None:
            filename = f"{algo_name.lower().replace('-', '_')}_iterations.csv"
            result.raw_result["iteration_summaries"].to_csv(
                output_dir / filename, index=False
            )
            print(f"✓ {algo_name}: {output_dir / filename}")

    print(f"\n✓ All results saved to: {output_dir}")


if __name__ == "__main__":
    main()
