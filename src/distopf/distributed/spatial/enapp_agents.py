import logging
from copy import deepcopy
from time import perf_counter
from typing import Callable, Optional

import numpy as np
import pandas as pd
import distopf as opf
from distopf.distributed.spatial.decompose import decompose
from distopf.results import PowerFlowResult
from .boundary import BoundaryVars, parse_v_up
from .execution import (
    _agent_boundaries,
    _agent_results,
    _aggregate_root_objective,
    _get_root_areas,
    _record_iteration_results,
    _remap_area_results_to_global_ids,
    _solve_iteration,
    combine_powerflow_results,
    create_area_agents,
    finalize_result,
)
from .messaging import AreaAgent, send_enapp_messages

logger = logging.getLogger(__name__)


def _configure_logging(verbose_enapp: bool) -> None:
    """Enable ENAPP diagnostics without changing an application's log policy."""
    if not verbose_enapp or logger.level != logging.NOTSET:
        return

    logger.setLevel(logging.DEBUG)

    # A library normally leaves handler configuration to its caller.  When
    # verbose ENAPP logging is requested without an application-level setup,
    # provide a private console handler so the request is still useful and
    # does not enable DEBUG output from unrelated libraries.
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(name)s | %(levelname)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False


def _safe_frame_max(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0

    values = frame.to_numpy(dtype=float, na_value=np.nan).ravel()
    values = values[~np.isnan(values)]

    if values.size == 0:
        return 0.0

    return float(np.max(values))


def calculate_boundary_deviation(
    boundaries: dict[str, BoundaryVars],
    boundaries_prev: dict[str, BoundaryVars],
) -> float:
    diff_maxes = []

    for area_name, boundary in boundaries.items():
        if area_name not in boundaries_prev:
            raise ValueError(f"Previous boundary values are missing area {area_name}")

        diff = abs(boundary - boundaries_prev[area_name])

        p = diff.s_up.loc[:, ["a", "b", "c"]].apply(np.real)
        q = diff.s_up.loc[:, ["a", "b", "c"]].apply(np.imag)
        v = diff.v_down.loc[:, ["a", "b", "c"]]

        p_max = _safe_frame_max(p)
        q_max = _safe_frame_max(q)
        v_max = _safe_frame_max(v)

        diff_maxes.append(float(np.nanmax([v_max, p_max, q_max])))

    return max(diff_maxes, default=0.0)


def _calculate_swing_voltage_errors(
    agents: dict[str, AreaAgent],
) -> dict[str, dict[str, float]]:
    swing_voltage_errors = {}

    for area_name, agent in agents.items():
        result = agent.result
        area_errors = {phase: 0.0 for phase in "abc"}

        if result is None:
            swing_voltage_errors[area_name] = area_errors
            continue

        v_result = parse_v_up(agent.case, result)
        schedules = agent.case.schedules

        if v_result.empty or not {"time", "v_a", "v_b", "v_c"}.issubset(
            schedules.columns
        ):
            swing_voltage_errors[area_name] = area_errors
            continue

        v_compare = v_result.merge(
            schedules[["time", "v_a", "v_b", "v_c"]],
            left_on="t",
            right_on="time",
        )

        for phase in "abc":
            error = (v_compare[phase] - v_compare[f"v_{phase}"]).abs()
            area_errors[phase] = _safe_frame_max(error.to_frame())

        swing_voltage_errors[area_name] = area_errors

    return swing_voltage_errors


def dampen_boundaries(
    boundaries: dict[str, BoundaryVars],
    boundaries_last: dict[str, BoundaryVars],
    alpha: float = 0.5,
) -> dict[str, BoundaryVars]:
    boundaries_damped = {}
    value_cols = ["a", "b", "c"]

    for area_name, boundary in boundaries.items():
        boundary_last = boundaries_last.get(area_name)

        if boundary_last is None:
            boundaries_damped[area_name] = boundary
            continue

        s_up = boundary.s_up.copy()
        v_down = boundary.v_down.copy()
        s_down = boundary.s_down.copy()
        v_up = boundary.v_up.copy()

        s_up.loc[:, value_cols] = boundary.s_up.loc[
            :, value_cols
        ] * alpha + boundary_last.s_up.loc[:, value_cols] * (1 - alpha)
        v_down.loc[:, value_cols] = boundary.v_down.loc[
            :, value_cols
        ] * alpha + boundary_last.v_down.loc[:, value_cols] * (1 - alpha)
        s_down.loc[:, value_cols] = boundary.s_down.loc[
            :, value_cols
        ] * alpha + boundary_last.s_down.loc[:, value_cols] * (1 - alpha)
        v_up.loc[:, value_cols] = boundary.v_up.loc[
            :, value_cols
        ] * alpha + boundary_last.v_up.loc[:, value_cols] * (1 - alpha)

        boundaries_damped[area_name] = BoundaryVars(
            s_up=s_up,
            v_down=v_down,
            v_up=v_up,
            s_down=s_down,
        )

    return boundaries_damped
def solve_enapp(
    case: opf.Case,
    area_info: dict[str, dict[str, list]],
    objective: Callable | str,
    swing_voltage_slack_penalty: float = 1e6,
    damping_factor: float = 1.0,
    tol: float = 1e-6,
    max_iterations: int = 100,
    parallel: bool = True,
    solve_callback: Optional = None,
    iteration_callback: Optional[
        Callable[
            [
                int,
                dict[str, "opf.Case"],
                dict[str, "PowerFlowResult"],
                dict[str, "BoundaryVars"],
            ],
            None,
        ]
    ] = None,
    verbose_enapp: bool = False,
    **kwargs,
) -> PowerFlowResult:
    """Solve a decomposed OPF/PF problem with ENAPP."""
    if not 0.0 <= damping_factor <= 1.0:
        raise ValueError("damping_factor must be between 0.0 and 1.0")

    _configure_logging(verbose_enapp)
    case._distributed_area_info = area_info

    tic = perf_counter()

    sources = {area_name: info["up_buses"][0] for area_name, info in area_info.items()}

    _cases = decompose(case, sources)
    agents = create_area_agents(_cases, area_info)

    if "free_swing_voltage" in kwargs:
        raise TypeError("free_swing_voltage is controlled by solve_enapp")
    free_swing_voltage = True
    if swing_voltage_slack_penalty == 0:
        free_swing_voltage = False
    solve_kwargs = {
        **kwargs,
        "free_swing_voltage": free_swing_voltage,
        "swing_voltage_slack_penalty": swing_voltage_slack_penalty,
    }

    root_areas = _get_root_areas(area_info)
    boundaries: dict[str, BoundaryVars] = {}
    boundary_error_per_iter: list[float] = []
    iteration_summaries: list[dict[str, float | int]] = []
    area_iteration_summaries: list[dict[str, object]] = []

    failed_areas: set[str] = set()
    any_area_solve_failed = False
    converged = False

    parallel_used = parallel and solve_callback is None

    for iterations in range(1, max_iterations + 1):
        iteration_results = _solve_iteration(
            agents=agents,
            objective=objective,
            solve_callback=solve_callback,
            parallel=parallel,
            solve_kwargs=solve_kwargs,
            uniform_solve_kwargs=True,
        )

        iteration_solve_failed = _record_iteration_results(
            agents=agents,
            iteration_results=iteration_results,
            iteration=iterations,
            failed_areas=failed_areas,
        )

        any_area_solve_failed |= iteration_solve_failed
        all_results = _agent_results(agents)
        area_iteration_summaries.extend(
            {
                "iteration": iterations,
                "area": area_name,
                "objective": result.objective_value,
                "solve_time": result.solve_time,
                "converged": result.converged,
                "solve_failed": area_name in failed_areas,
                "solver_status": result.solver_status,
                "termination_condition": result.termination_condition,
                "has_result": True,
            }
            for area_name, result in all_results.items()
        )
        area_iteration_summaries.extend(
            {
                "iteration": iterations,
                "area": area_name,
                "objective": float("nan"),
                "solve_time": float("nan"),
                "converged": False,
                "solve_failed": True,
                "solver_status": None,
                "termination_condition": None,
                "has_result": False,
            }
            for area_name in agents
            if area_name not in all_results
        )

        objective_value = _aggregate_root_objective(all_results, root_areas)
        solve_times = [
            result.solve_time
            for result in all_results.values()
            if result.solve_time is not None
        ]
        iteration_summary = {
            "iteration": iterations,
            "objective": objective_value,
            "solve_time": sum(solve_times) if solve_times else float("nan"),
            "solve_time_max": max(solve_times) if solve_times else float("nan"),
            "boundary_error": float("nan"),
        }
        iteration_summaries.append(iteration_summary)

        # Preserve current behavior: terminate if an area has never solved.
        if len(all_results) < len(agents):
            _remap_area_results_to_global_ids(all_results, case, area_info)

            objective_value = _aggregate_root_objective(
                all_results,
                root_areas,
            )

            partial_result = combine_powerflow_results(
                all_results.values(),
                case_ref=case,
                objective_value=objective_value,
            )

            runtime = perf_counter() - tic

            logger.warning(
                "Only %d/%d ENAPP areas solved; returning a partial result.",
                len(all_results),
                len(agents),
            )

            return finalize_result(
                partial_result,
                case=case,
                objective_value=objective_value,
                converged=False,
                iterations=iterations,
                runtime=runtime,
                solver_status="failure",
                termination_condition="incomplete_solve",
                area_results=all_results,
                boundary_errors=boundary_error_per_iter,
                iteration_summaries=pd.DataFrame(iteration_summaries),
                area_iteration_summaries=pd.DataFrame(area_iteration_summaries),
                parallel_used=parallel_used,
                area_solve_failed=True,
                failed_areas=failed_areas,
            )

        previous_boundaries = deepcopy(boundaries)
        boundaries = _agent_boundaries(agents)

        boundaries = dampen_boundaries(
            boundaries,
            previous_boundaries,
            alpha=damping_factor,
        )

        # Agents send the final, potentially damped, boundary values.
        for area_name, boundary in boundaries.items():
            agents[area_name].boundary = boundary

        swing_voltage_errors = _calculate_swing_voltage_errors(agents)

        # Updates schedules for the next iteration.
        send_enapp_messages(agents)

        if iteration_callback is not None:
            iteration_callback(
                iterations,
                {agent.name: agent.case for agent in agents},
                all_results,
                boundaries,
            )

        # Preserve existing behavior: exchange first, then begin convergence
        # checks on iteration two.
        if iterations == 1:
            continue

        boundary_error = calculate_boundary_deviation(
            boundaries,
            previous_boundaries,
        )

        boundary_error_per_iter.append(boundary_error)
        iteration_summaries[-1]["boundary_error"] = boundary_error

        logger.info(
            "ENAPP iteration %d boundary error: %.6e",
            iterations,
            boundary_error,
        )

        swing_voltage_error_text = "".join(
            f"\n{area}: a={errors['a']:.6e}, b={errors['b']:.6e}, c={errors['c']:.6e}"
            for area, errors in swing_voltage_errors.items()
        )

        logger.debug(
            "swing voltage errors: %s",
            swing_voltage_error_text,
        )

        if boundary_error < tol and not iteration_solve_failed:
            logger.info(
                "ENAPP converged after %d iterations with boundary error %.6e",
                iterations,
                boundary_error,
            )
            converged = True
            break

    all_results = _agent_results(agents)

    _remap_area_results_to_global_ids(
        all_results,
        case,
        area_info,
    )

    objective_value = _aggregate_root_objective(
        all_results,
        root_areas,
    )

    aggregated_result = combine_powerflow_results(
        all_results.values(),
        case_ref=case,
        objective_value=objective_value,
    )

    runtime = perf_counter() - tic

    if converged:
        solver_status = "optimal"
        termination_condition = "converged"
    else:
        solver_status = "max_iterations"
        termination_condition = "max_iterations"

    return finalize_result(
        aggregated_result,
        case=case,
        objective_value=objective_value,
        converged=converged,
        iterations=iterations,
        runtime=runtime,
        solver_status=solver_status,
        termination_condition=termination_condition,
        area_results=all_results,
        boundary_errors=boundary_error_per_iter,
        iteration_summaries=pd.DataFrame(iteration_summaries),
        area_iteration_summaries=pd.DataFrame(area_iteration_summaries),
        parallel_used=parallel_used,
        area_solve_failed=any_area_solve_failed,
        failed_areas=failed_areas,
    )
