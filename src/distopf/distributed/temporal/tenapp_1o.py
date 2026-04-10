"""First-order temporal decomposition algorithm (TENAPP-1O).

Uses dual variable information from future time periods to coordinate battery
scheduling across sequential time steps.  Each subproblem is solved via
``Case.run_opf()`` so the algorithm is independent of the underlying solver
backend.
"""

from time import perf_counter
from typing import Callable
import pandas as pd
import numpy as np
import logging

from distopf import Case
from distopf.results import PowerFlowResult

from .objectives import energy_cost_min, tenapp_1o_augmentation
from .utils import (
    build_timestep_cases,
    update_bat_start_soc,
    combine_temporal_results,
    compile_iteration_summary,
)

logger = logging.getLogger(__name__)


def solve_tenapp_1o(
    case: Case,
    objective: Callable = energy_cost_min,
    wrapper: str = "matrix_bess",
    max_iterations: int = 100,
    tolerance: float = 1e-3,
    **kwargs,
) -> PowerFlowResult:
    """Solve multi-period OPF using first-order temporal decomposition (TENAPP-1O).

    Decomposes the multi-period problem into per-time-step OPF subproblems,
    coordinating battery scheduling via dual variables on SOC constraints.
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
        Use ``"pyomo"`` with a pyomo-compatible objective for IPOPT.
    max_iterations : int
        Maximum number of coordination iterations.
    tolerance : float
        Convergence tolerance on inter-iteration SOC change.
    **kwargs
        Extra arguments forwarded to ``Case.run_opf()``
        (e.g. ``cost_curve``, ``solver``).

    Returns
    -------
    PowerFlowResult
        Aggregated result across all time steps.  Per-step results and
        iteration summaries are in ``result.raw_result``.
    """
    if case.n_steps <= 1:
        raise ValueError("TENAPP-1O requires multi-period case (n_steps > 1)")
    if case.bat_data is None or len(case.bat_data) == 0:
        raise ValueError("TENAPP-1O requires battery_data in case")

    tic = perf_counter()
    n_steps = case.n_steps
    cases = build_timestep_cases(case)

    # Zero duals to start — updated each iteration (currently approximate)
    all_duals: list = [
        pd.DataFrame(columns=["id", "t", "dual"]) for _ in range(n_steps)
    ]
    all_results: list = []
    prev_socs: list = []
    iteration_summaries = []
    converged = False
    iteration = 0

    for iteration in range(max_iterations):
        logger.info(f"TENAPP-1O iteration {iteration + 1}/{max_iterations}")
        new_results = []

        for t, case_t in enumerate(cases):
            future_duals = all_duals[t + 1] if t < n_steps - 1 else None

            # Capture future_duals by value via default argument
            def obj_with_dual(m, xk, _fd=future_duals, **kw):
                return objective(m, xk, **kw) + tenapp_1o_augmentation(
                    m, xk, future_duals=_fd
                )

            result_t = case_t.run_opf(
                objective=obj_with_dual, wrapper=wrapper, **kwargs
            )
            new_results.append(result_t)

        # Convergence: max SOC change across all time steps
        cur_socs = [r.soc for r in new_results]
        if prev_socs:
            max_change = 0.0
            for prev, curr in zip(prev_socs, cur_socs):
                if (
                    prev is not None
                    and curr is not None
                    and not prev.empty
                    and not curr.empty
                ):
                    max_change = max(
                        max_change,
                        float(
                            np.max(np.abs(curr["value"].values - prev["value"].values))
                        ),
                    )
            if max_change < tolerance:
                logger.info(
                    f"TENAPP-1O converged after {iteration + 1} iterations "
                    f"(max SOC \u0394={max_change:.2e})"
                )
                converged = True

        all_results = new_results
        prev_socs = cur_socs
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
