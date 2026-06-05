"""Approximate dual temporal decomposition algorithm (TENAPP-APRX).

Uses approximate dual updates based on SOC gradient information to coordinate
battery scheduling.  Each subproblem is solved via ``Case.run_opf()`` so the
algorithm is independent of the underlying solver backend.
"""

from time import perf_counter
from typing import Callable
import pandas as pd
import numpy as np
import logging

from distopf import Case
from distopf.results import PowerFlowResult

from .objectives import energy_cost_min, tenapp_aprx_augmentation
from .utils import (
    build_timestep_cases,
    update_bat_start_soc,
    combine_temporal_results,
    compile_iteration_summary,
)

logger = logging.getLogger(__name__)


def solve_tenapp_aprx(
    case: Case,
    objective: Callable = energy_cost_min,
    wrapper: str = "matrix_bess",
    max_iterations: int = 100,
    tolerance: float = 1e-3,
    min_converged_iterations: int = 1,
    **kwargs,
) -> PowerFlowResult:
    """Solve multi-period OPF using approximate dual temporal decomposition (TENAPP-APRX).

    Decomposes the multi-period problem into per-time-step OPF subproblems,
    coordinating via approximate dual estimates on battery SOC constraints.
    Each subproblem is solved with ``case_t.run_opf()``, keeping the algorithm
    independent of the underlying solver backend.

    Parameters
    ----------
    case : Case
        Multi-period case with ``n_steps > 1``, schedules, and battery data.
    objective : Callable
        Per-step objective accepted by ``Case.run_opf()``.
    wrapper : str, default ``"matrix_bess"``
        Solver wrapper passed to ``Case.run_opf()``.
    max_iterations : int
        Maximum number of coordination iterations.
    tolerance : float
        Convergence tolerance on relative objective change.
    min_converged_iterations : int
        Consecutive iterations below tolerance required before declaring convergence.
    **kwargs
        Extra arguments forwarded to ``Case.run_opf()``.

    Returns
    -------
    PowerFlowResult
        Aggregated result across all time steps.
    """
    if case.n_steps <= 1:
        raise ValueError("TENAPP-APRX requires multi-period case (n_steps > 1)")
    if case.bat_data is None or len(case.bat_data) == 0:
        raise ValueError("TENAPP-APRX requires battery_data in case")

    tic = perf_counter()
    n_steps = case.n_steps
    cases = build_timestep_cases(case)

    approx_duals: list = [None] * n_steps
    all_results: list = []
    iteration_summaries = []
    converged = False
    converged_count = 0
    iteration = 0

    for iteration in range(max_iterations):
        logger.info(f"TENAPP-APRX iteration {iteration + 1}/{max_iterations}")
        new_results = []

        for t, case_t in enumerate(cases):
            approx_dual = approx_duals[t]

            # Capture approx_dual by value via default argument
            def obj_with_aprx_dual(m, xk, _ad=approx_dual, **kw):
                return objective(m, xk, **kw) + tenapp_aprx_augmentation(
                    m, xk, approx_dual=_ad
                )

            result_t = case_t.run_opf(
                objective=obj_with_aprx_dual, wrapper=wrapper, **kwargs
            )
            new_results.append(result_t)

        # Convergence: relative objective change
        if all_results:
            changes = [
                abs(
                    (n.objective_value - o.objective_value)
                    / (abs(o.objective_value) + 1e-6)
                )
                for o, n in zip(all_results, new_results)
                if o.objective_value is not None and n.objective_value is not None
            ]
            max_change = max(changes) if changes else 0.0
            if max_change < tolerance:
                converged_count += 1
                if converged_count >= min_converged_iterations:
                    logger.info(
                        f"TENAPP-APRX converged after {iteration + 1} iterations "
                        f"(Δobj={max_change:.2e})"
                    )
                    converged = True
            else:
                converged_count = 0

        all_results = new_results

        # Update approximate duals from current SOC values
        for t in range(n_steps - 1):
            soc_df = new_results[t].soc
            if soc_df is not None and not soc_df.empty:
                approx_duals[t] = -0.1 * float(soc_df["value"].sum())

        cases = update_bat_start_soc(cases, all_results)
        iteration_summaries.append(
            compile_iteration_summary(iteration, cases, all_results)
        )

        if converged:
            break

    summaries_df = (
        pd.concat(iteration_summaries, ignore_index=True)
        if iteration_summaries
        else pd.DataFrame()
    )
    return combine_temporal_results(
        all_results,
        case,
        iterations=iteration + 1,
        converged=converged,
        solve_time=perf_counter() - tic,
        iteration_summaries=summaries_df,
    )
