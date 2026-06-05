"""Example: ENAPP (spatial) + TENAPP-APRX (temporal) nested decomposition.

Combines two levels of decomposition:
- ENAPP: coordinates voltage/power boundary conditions between network areas (spatial)
- TENAPP-APRX: optimizes multi-period battery scheduling within each area (temporal)

In each ENAPP coordination iteration, areas with batteries are solved by
TENAPP-APRX (approximate-dual temporal decomposition) rather than a single
monolithic multiperiod OPF.  Areas without batteries fall back to a direct
multiperiod ``case.run_opf()`` call.

Network: IEEE 33-bus feeder split into 3 areas.
Horizon: 6 hours (morning peak, start_step=8) with realistic load/price profiles.

The two results should match within solver tolerance, showing that the nested
decomposition reproduces the same cost as the monolithic per-area baseline.
"""

from pathlib import Path
import numpy as np
import distopf as opf
from distopf.distributed.spatial.enapp import solve_enapp
from distopf.distributed.temporal import solve_tenapp_aprx, energy_cost_min

OUTPUT_DIR = Path("scratch/enapp_tenapp")

# ---------------------------------------------------------------------------
# Area definitions for IEEE 33-bus (matches decomposition in run_enapp.py)
# ---------------------------------------------------------------------------
AREA_INFO = {
    "area1": {
        "up_areas": [],
        "down_areas": ["area2", "area3"],
        "up_buses": ["1"],
        "down_buses": [5, 19],
    },
    "area2": {
        "up_areas": ["area1"],
        "down_areas": [],
        "up_buses": ["5"],
        "down_buses": [],
    },
    "area3": {
        "up_areas": ["area1"],
        "down_areas": [],
        "up_buses": ["19"],
        "down_buses": [],
    },
}


# ---------------------------------------------------------------------------
# Callback factory
# ---------------------------------------------------------------------------
def make_tenapp_aprx_solver(
    wrapper: str = "matrix_bess",
    max_iterations: int = 20,
    tolerance: float = 1e-3,
    min_converged_iterations: int = 2,
):
    """Return a solve_callback for solve_enapp that uses TENAPP-APRX per area.

    Areas that contain batteries are solved with TENAPP-APRX temporal
    decomposition.  Areas without batteries fall back to a direct multiperiod
    ``case.run_opf()`` call using the same wrapper.

    Parameters
    ----------
    wrapper : str
        Solver backend passed to each area solve (``'matrix_bess'`` or
        ``'pyomo'``).
    max_iterations : int
        Maximum TENAPP-APRX iterations per ENAPP coordination step.
    tolerance : float
        TENAPP-APRX convergence tolerance on relative objective change.
    min_converged_iterations : int
        Consecutive iterations below tolerance before TENAPP-APRX stops.

    Returns
    -------
    callable
        ``callback(cases, objective, **kwargs) -> dict[str, PowerFlowResult]``
        Compatible with the ``solve_callback`` parameter of ``solve_enapp``.
    """

    def callback(cases: dict, objective, **kwargs):
        all_results = {}
        for area_name, area_case in cases.items():
            has_bats = area_case.bat_data is not None and len(area_case.bat_data) > 0
            if has_bats:
                result = solve_tenapp_aprx(
                    area_case,
                    objective=objective,
                    wrapper=wrapper,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    min_converged_iterations=min_converged_iterations,
                    **kwargs,
                )
            else:
                # No intertemporal coupling: solve the full horizon directly.
                result = area_case.run_opf(
                    objective=objective, wrapper=wrapper, **kwargs
                )
            all_results[area_name] = result
        return all_results

    return callback


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build 6-hour case (morning peak: hours 8–13)
    # ieee33 has tie switches; drop them first to get a radial network.
    # ------------------------------------------------------------------
    case = opf.create_case(
        opf.CASES_DIR / "csv" / "ieee33",
        n_steps=6,  # 6-hour horizon
        start_step=8,  # start at hour 8 (morning peak)
    )
    case.branch_data.drop(
        index=case.branch_data.loc[case.branch_data.status == "open"].index,
        inplace=True,
    )
    case.bus_data.v_max = 1.05
    case.bus_data.v_min = 0.95
    case.gen_data.control_variable = "PQ"
    case.bat_data.control_variable = "P"
    case.bat_data["start_soc"] = 0.5
    case.bat_data["max_soc"] = 0.9
    case.bat_data["min_soc"] = 0.1

    n_buses = len(case.bus_data)
    n_lines = len(case.branch_data)
    n_bats = len(case.bat_data)
    print(f"Case: {n_buses} buses, {n_lines} lines, {n_bats} battery unit(s)")
    print(
        f"Horizon: {case.n_steps} steps (hours {case.start_step}–"
        f"{case.start_step + case.n_steps - 1})"
    )

    # Full 24-hr price schedule; objective accesses absolute time indices 8–13.
    cost_curve = case.schedules.price.to_numpy()
    print(
        f"Peak-hour prices ($/MWh): {cost_curve[case.start_step : case.start_step + case.n_steps]}\n"
    )

    common_kwargs = dict(
        cost_curve=cost_curve,
        solver="CLARABEL",
    )

    # ------------------------------------------------------------------
    # Run 1: ENAPP with TENAPP-APRX per-area temporal solve
    # ------------------------------------------------------------------
    print("=" * 65)
    print("ENAPP + TENAPP-APRX  (nested spatial + temporal decomposition)")
    print("=" * 65)

    callback = make_tenapp_aprx_solver(
        wrapper="matrix_bess",
        max_iterations=15,
        tolerance=1e-3,
        min_converged_iterations=2,
    )

    result_nested = solve_enapp(
        case,
        area_info=AREA_INFO,
        objective=energy_cost_min,
        tol=1e-4,
        max_iterations=15,
        parallel=False,
        solve_callback=callback,
        **common_kwargs,
    )

    obj_nested = result_nested.objective_value
    conv_nested = result_nested.converged
    enapp_iters = (
        result_nested.raw_result.get("enapp_iterations", "?")
        if result_nested.raw_result
        else "?"
    )
    print(f"  Converged:   {conv_nested}")
    print(f"  Objective:   ${obj_nested:.4f}")
    print(f"  ENAPP iters: {enapp_iters}")

    # ------------------------------------------------------------------
    # Run 2: Baseline ENAPP (monolithic multiperiod solve per area)
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("ENAPP baseline  (monolithic multiperiod solve per area)")
    print("=" * 65)

    result_base = solve_enapp(
        case,
        area_info=AREA_INFO,
        objective=energy_cost_min,
        tol=1e-4,
        max_iterations=15,
        parallel=False,
        wrapper="matrix_bess",
        **common_kwargs,
    )

    obj_base = result_base.objective_value
    conv_base = result_base.converged
    enapp_iters_base = (
        result_base.raw_result.get("enapp_iterations", "?")
        if result_base.raw_result
        else "?"
    )
    print(f"  Converged:   {conv_base}")
    print(f"  Objective:   ${obj_base:.4f}")
    print(f"  ENAPP iters: {enapp_iters_base}")

    # ------------------------------------------------------------------
    # Comparison summary
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("COMPARISON SUMMARY")
    print("=" * 65)
    obj_diff = abs(obj_nested - obj_base) if (obj_nested and obj_base) else None
    print(f"  {'Method':<30} {'Objective':>12}  {'Converged':>10}")
    print(f"  {'-' * 54}")
    print(f"  {'ENAPP + TENAPP-APRX':<30} ${obj_nested:>10.4f}  {str(conv_nested):>10}")
    print(f"  {'ENAPP baseline':<30} ${obj_base:>10.4f}  {str(conv_base):>10}")
    if obj_diff is not None:
        print(f"\n  Objective difference: {obj_diff:.6f} p.u.")

    # ------------------------------------------------------------------
    # Battery SOC results for the nested run
    # ------------------------------------------------------------------
    if result_nested.soc is not None:
        print()
        print("Battery SOC schedule (ENAPP + TENAPP-APRX):")
        soc_df = result_nested.soc.copy()
        e_cap = float(case.bat_data["energy_capacity"].iloc[0])
        soc_df["soc_pct"] = soc_df["value"].astype(float) / e_cap * 100
        for bat_id in soc_df["id"].unique():
            bat_soc = soc_df.loc[soc_df["id"] == bat_id].sort_values("t")
            vals = bat_soc["soc_pct"].to_numpy(dtype=float).round(1)
            print(f"  Battery {bat_id}: {vals}")
        soc_df.to_csv(OUTPUT_DIR / "soc_nested.csv", index=False)

    if result_base.soc is not None:
        result_base.soc.to_csv(OUTPUT_DIR / "soc_baseline.csv", index=False)

    if result_nested.voltages is not None:
        result_nested.voltages.to_csv(OUTPUT_DIR / "voltages_nested.csv", index=False)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
