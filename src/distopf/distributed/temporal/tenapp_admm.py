"""ADMM-based temporal decomposition algorithm (TENAPP-ADMM).

Uses alternating direction method of multipliers to coordinate battery
scheduling across time periods.  Each subproblem is solved via
``Case.run_opf()`` so the algorithm is independent of the underlying
solver backend.
"""

from time import perf_counter
from typing import Callable
import pandas as pd
import numpy as np
import logging

from distopf import Case
from distopf.results import PowerFlowResult

from .objectives import energy_cost_min, tenapp_admm_augmentation
from .utils import (
    build_timestep_cases,
    update_bat_start_soc,
    combine_temporal_results,
    compile_iteration_summary,
)

logger = logging.getLogger(__name__)


def solve_tenapp_admm(
    case: Case,
    objective: Callable = energy_cost_min,
    wrapper: str = "matrix_bess",
    max_iterations: int = 100,
    tolerance: float = 1e-3,
    weight: float = 1e2,
    weight_scale: float = 1.0,
    **kwargs,
) -> PowerFlowResult:
    """Solve multi-period OPF using ADMM temporal decomposition (TENAPP-ADMM).

    Decomposes the multi-period problem into per-time-step OPF subproblems,
    coordinating via ADMM with quadratic penalties on SOC boundary conditions.
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
        Maximum number of ADMM iterations.
    tolerance : float
        Convergence tolerance for ADMM primal residual.
    weight : float
        ADMM penalty weight (rho).
    weight_scale : float
        Multiplicative scaling applied to ``weight`` each iteration.
    **kwargs
        Extra arguments forwarded to ``Case.run_opf()``.

    Returns
    -------
    PowerFlowResult
        Aggregated result across all time steps.
    """
    if case.n_steps <= 1:
        raise ValueError("TENAPP-ADMM requires multi-period case (n_steps > 1)")
    if case.bat_data is None or len(case.bat_data) == 0:
        raise ValueError("TENAPP-ADMM requires battery_data in case")

    tic = perf_counter()
    n_steps = case.n_steps
    cases = build_timestep_cases(case)

    soc_targets: list = [None] * n_steps
    all_results: list = []
    iteration_summaries = []
    converged = False
    current_weight = weight
    iteration = 0

    for iteration in range(max_iterations):
        logger.info(f"TENAPP-ADMM iteration {iteration + 1}/{max_iterations}")
        new_results = []
        new_soc_vals = []  # scalar average SOC per time step

        for t, case_t in enumerate(cases):
            soc0 = soc_targets[t] if t > 0 else None
            soc_end = soc_targets[t + 1] if t < n_steps - 1 else None

            # Capture values by default argument
            def obj_with_admm(m, xk, _s0=soc0, _se=soc_end, _w=current_weight, **kw):
                return objective(m, xk, **kw) + tenapp_admm_augmentation(
                    m, xk, soc0=_s0, soc_end=_se, weight=_w
                )

            result_t = case_t.run_opf(
                objective=obj_with_admm, wrapper=wrapper, **kwargs
            )
            new_results.append(result_t)

            soc_df = result_t.soc
            soc_val = (
                float(soc_df["value"].mean())
                if soc_df is not None and not soc_df.empty
                else 0.0
            )
            new_soc_vals.append(soc_val)

        # Z-update: compute new SOC targets as averages of neighbouring periods
        new_soc_targets = list(soc_targets)
        new_soc_targets[0] = (
            float(case.bat_data["start_soc"].mean())
            if "start_soc" in case.bat_data.columns
            else 0.5
        )
        for t in range(1, n_steps - 1):
            new_soc_targets[t] = 0.5 * (new_soc_vals[t - 1] + new_soc_vals[t])
        if n_steps > 1:
            new_soc_targets[-1] = float(
                case.bat_data["max_soc"].mean()
                if "max_soc" in case.bat_data.columns
                else 0.5
            )

        # Primal residual: change in SOC targets
        prev_valid = [v for v in soc_targets if v is not None]
        new_valid = [v for v in new_soc_targets if v is not None]
        if prev_valid and len(prev_valid) == len(new_valid):
            residual_primal = float(
                np.sqrt(sum((a - b) ** 2 for a, b in zip(new_valid, prev_valid)))
            )
        else:
            residual_primal = float("inf")

        soc_targets = new_soc_targets
        all_results = new_results

        if residual_primal < tolerance:
            logger.info(
                f"TENAPP-ADMM converged after {iteration + 1} iterations "
                f"(primal residual={residual_primal:.2e})"
            )
            converged = True

        current_weight *= weight_scale
        cases = update_bat_start_soc(cases, all_results)
        iteration_summaries.append(
            compile_iteration_summary(
                iteration, cases, all_results, residual_primal=residual_primal
            )
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
