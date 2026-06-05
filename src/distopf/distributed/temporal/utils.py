"""Utility functions for temporal decomposition algorithms."""

import pandas as pd
import numpy as np
from typing import Optional
from distopf import Case
from distopf.results import PowerFlowResult


def build_timestep_cases(case: Case) -> list:
    """Create one single-step Case per time period.

    Each case shares ``branch_data`` with the original but gets independent
    copies of mutable DataFrames (``bus_data``, ``bat_data``, etc.) so that
    per-iteration updates (e.g. ``start_soc``) are isolated.

    Parameters
    ----------
    case : Case
        Multi-period case with ``n_steps > 1``.

    Returns
    -------
    list[Case]
        One Case per time step, each with ``n_steps=1``.
    """
    cases = []
    for t in range(case.n_steps):
        abs_t = case.start_step + t
        if case.schedules is not None:
            # Select the schedule row matching this absolute time step.
            mask = case.schedules.time == abs_t
            if mask.any():
                schedules_t = case.schedules.loc[mask]
            else:
                schedules_t = case.schedules.iloc[[min(t, len(case.schedules) - 1)]]
        else:
            schedules_t = case.schedules
        case_t = Case(
            branch_data=case.branch_data,
            bus_data=case.bus_data.copy(),
            gen_data=case.gen_data.copy(),
            cap_data=case.cap_data.copy(),
            reg_data=case.reg_data.copy(),
            bat_data=case.bat_data.copy(),
            schedules=schedules_t,
            start_step=abs_t,
            n_steps=1,
            delta_t=case.delta_t,
        )
        cases.append(case_t)
    return cases


def update_bat_start_soc(cases: list, step_results: list) -> list:
    """Chain battery SOC across time steps.

    Updates each subsequent case's ``bat_data["start_soc"]`` from the SOC
    values in the preceding step's ``PowerFlowResult.soc``.  Operates
    in-place so the next ``case_t.run_opf()`` call picks up the updated
    initial conditions.

    Parameters
    ----------
    cases : list[Case]
        Per-time-step cases produced by :func:`build_timestep_cases`.
    step_results : list[PowerFlowResult]
        Per-time-step results from the latest solve iteration.

    Returns
    -------
    list[Case]
        The same list, mutated in-place.
    """
    for i, result_t in enumerate(step_results[:-1]):
        soc_df = result_t.soc
        if soc_df is None or soc_df.empty:
            continue
        next_case = cases[i + 1]
        if next_case.bat_data is None or len(next_case.bat_data) == 0:
            continue
        for _, soc_row in soc_df.iterrows():
            mask = next_case.bat_data["id"] == soc_row["id"]
            if mask.any():
                # result.soc["value"] is in energy units (same as energy_capacity);
                # bat_data["start_soc"] is a fraction in [0, 1].
                e_cap = float(next_case.bat_data.loc[mask, "energy_capacity"].iloc[0])
                soc_fraction = float(soc_row["value"]) / e_cap if e_cap > 0 else 0.0
                next_case.bat_data.loc[mask, "start_soc"] = soc_fraction
    return cases


def combine_temporal_results(
    step_results: list,
    case: Case,
    *,
    iterations: int,
    converged: bool,
    solve_time: float,
    iteration_summaries: Optional[pd.DataFrame] = None,
) -> PowerFlowResult:
    """Aggregate per-time-step PowerFlowResults into a single result.

    Concatenates all DataFrame fields across time steps and sums scalar
    objective values.  The per-step results and iteration summaries are
    stored in ``raw_result`` for diagnostics.

    Parameters
    ----------
    step_results : list[PowerFlowResult]
        One result per time step, in temporal order.
    case : Case
        The original multi-period case (stored as reference).
    iterations : int
        Number of coordination iterations completed.
    converged : bool
        Whether the coordination algorithm converged.
    solve_time : float
        Total wall-clock time in seconds.
    iteration_summaries : pd.DataFrame, optional
        Per-iteration summary table produced by :func:`compile_iteration_summary`.

    Returns
    -------
    PowerFlowResult
        Aggregated result.  ``raw_result`` contains
        ``{"step_results": ..., "iteration_summaries": ...}``.
    """

    def _concat(field: str) -> Optional[pd.DataFrame]:
        frames = [
            getattr(r, field)
            for r in step_results
            if getattr(r, field, None) is not None
        ]
        return pd.concat(frames, ignore_index=True) if frames else None

    total_obj = sum(
        r.objective_value for r in step_results if r.objective_value is not None
    )
    all_steps_converged = all(r.converged for r in step_results)

    return PowerFlowResult(
        voltages=_concat("voltages"),
        active_power_flows=_concat("active_power_flows"),
        reactive_power_flows=_concat("reactive_power_flows"),
        active_power_generation=_concat("active_power_generation"),
        reactive_power_generation=_concat("reactive_power_generation"),
        active_power_loads=_concat("active_power_loads"),
        reactive_power_loads=_concat("reactive_power_loads"),
        battery_active_power=_concat("battery_active_power"),
        battery_reactive_power=_concat("battery_reactive_power"),
        soc=_concat("soc"),
        p_charge=_concat("p_charge"),
        p_discharge=_concat("p_discharge"),
        capacitor_reactive_power=_concat("capacitor_reactive_power"),
        objective_value=total_obj,
        converged=converged and all_steps_converged,
        iterations=iterations,
        solve_time=solve_time,
        solver="tenapp",
        result_type="opf",
        solver_status="optimal" if converged else "max_iterations",
        termination_condition="converged" if converged else "max_iterations",
        case=case,
        raw_result={
            "step_results": step_results,
            "iteration_summaries": iteration_summaries,
        },
    )


def compile_iteration_summary(
    iteration: int,
    cases: list,
    results: list,
    residual_primal: Optional[float] = None,
    residual_dual: Optional[float] = None,
) -> pd.DataFrame:
    """Build a one-row-per-time-step summary DataFrame for a single iteration.

    Parameters
    ----------
    iteration : int
        Iteration index (0-based).
    cases : list[Case]
        Per-time-step cases (used for ``start_step`` labelling).
    results : list[PowerFlowResult]
        Per-time-step results from this iteration.
    residual_primal : float, optional
        ADMM primal residual (if applicable).
    residual_dual : float, optional
        ADMM dual residual (if applicable).

    Returns
    -------
    pd.DataFrame
        Columns: ``iteration``, ``t``, ``objective``, ``converged``,
        ``solve_time`` (plus residuals when provided).
    """
    rows = [
        {
            "iteration": iteration,
            "t": c.start_step,
            "objective": r.objective_value,
            "converged": r.converged,
            "solve_time": r.solve_time,
        }
        for c, r in zip(cases, results)
    ]
    df = pd.DataFrame(rows)
    if residual_primal is not None:
        df["residual_primal"] = residual_primal
    if residual_dual is not None:
        df["residual_dual"] = residual_dual
    return df
