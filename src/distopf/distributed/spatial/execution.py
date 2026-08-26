from typing import Any, Callable, Optional, Iterable
import multiprocessing as mp
import logging
import pandas as pd

import distopf as opf
from distopf.api import Case
from distopf.results import PowerFlowResult
from .boundary import BoundaryVars
from .messaging import AreaAgent, safe_area_solve

logger = logging.getLogger(__name__)


def create_area_agents(
    cases: dict[str, Case],
    area_info: dict[str, dict[str, list]],
) -> dict[str, AreaAgent]:
    """Create agents while preserving the prior upstream-routing behavior."""
    agents = {
        area_name: AreaAgent(
            name=area_name,
            case=area_case,
            down_areas=area_info[area_name]["down_areas"],
        )
        for area_name, area_case in cases.items()
    }

    # Preserve prior send_s_up/send_v_up behavior.
    for sending_area, sending_agent in agents.items():
        for receiving_area, receiving_agent in agents.items():
            load_shapes = receiving_agent.case.bus_data.load_shape.astype(str)

            for area_name in load_shapes:
                if area_name == sending_area:
                    sending_agent.upstream_recipients.append(receiving_area)

    return agents
def _agent_results(
    agents: dict[str, AreaAgent],
) -> dict[str, PowerFlowResult]:
    return {
        area_name: agent.result
        for area_name, agent in agents.items()
        if agent.result is not None
    }


def _agent_boundaries(
    agents: dict[str, AreaAgent],
) -> dict[str, BoundaryVars]:
    return {
        area_name: agent.boundary
        for area_name, agent in agents.items()
        if agent.boundary is not None
    }

def _solve_all_pool(
    name: str,
    case: Case,
    objective: Any,
    kwargs: dict,
) -> tuple[str, Optional[PowerFlowResult]]:
    return name, safe_area_solve(name, case, objective, **kwargs)


def _solve_all_parallel(
    agents: dict[str, AreaAgent],
    objective: Any,
    uniform_solve_kwargs: bool = False,
    **kwargs,
) -> dict[str, Optional[PowerFlowResult]]:
    args = []
    for agent in agents.values():
        agent_kwargs = kwargs
        if not uniform_solve_kwargs and len(agent.upstream_recipients) == 0:
            agent_kwargs = kwargs.copy()
            agent_kwargs.pop("free_swing_voltage", None)
        args.append((agent.name, agent.case, objective, agent_kwargs))

    with mp.Pool() as pool:
        solved = pool.starmap(_solve_all_pool, args)

    return dict(solved)


def _solve_all_loop(
    agents: dict[str, AreaAgent],
    objective: Any,
    **kwargs,
) -> dict[str, Optional[PowerFlowResult]]:
    return {
        area_name: agent.solve(objective, **kwargs)
        for area_name, agent in agents.items()
    }


def _solve_iteration(
    agents: dict[str, AreaAgent],
    objective: Any,
    solve_callback: Optional[Callable],
    parallel: bool,
    solve_kwargs: dict,
    uniform_solve_kwargs: bool = False,
) -> dict[str, Optional[PowerFlowResult]]:
    cases = {area: agent.case for area, agent in agents.items()}
    if solve_callback is not None:
        return solve_callback(cases, objective, **solve_kwargs)

    if parallel:
        return _solve_all_parallel(
            agents,
            objective,
            uniform_solve_kwargs=uniform_solve_kwargs,
            **solve_kwargs,
        )

    return _solve_all_loop(agents, objective, **solve_kwargs)


def _record_iteration_results(
    agents: dict[str, AreaAgent],
    iteration_results: dict[str, Optional[PowerFlowResult]],
    iteration: int,
    failed_areas: set[str],
) -> bool:
    """Store successful local results and return whether any solve failed."""
    iteration_solve_failed = False

    for area_name, result in iteration_results.items():
        if result is None:
            iteration_solve_failed = True
            failed_areas.add(area_name)

            logger.warning(
                "area %s failed on iteration %d; "
                "retaining its previous successful result, if any.",
                area_name,
                iteration,
            )
            continue

        agents[area_name].set_result(result)

    return iteration_solve_failed
def _get_root_areas(
    area_info: dict[str, dict[str, list]],
) -> set[str]:
    return {
        area_name for area_name, info in area_info.items() if not info.get("up_areas")
    }


def _aggregate_root_objective(
    results: dict[str, PowerFlowResult],
    root_areas: set[str],
) -> Optional[float]:
    objective_values = [
        result.objective_value
        for area_name, result in results.items()
        if area_name in root_areas
        and result is not None
        and getattr(result, "objective_value", None) is not None
    ]

    return sum(objective_values) if objective_values else None


def _concat_field(
    res_list: list,
    field: str,
    valid_names: Optional[set],
    name_cols: tuple = ("name",),
    bus_id_col: Optional[str] = None,
    branch_cols: Optional[tuple] = None,
) -> Optional[pd.DataFrame]:
    """Concatenate and deduplicate a result DataFrame field.

    ``branch_cols`` marks branch-indexed data. For such data, the first
    available from/to endpoint columns are used for filtering and
    deduplication.

    For bus-indexed data, the first available column in ``name_cols`` is
    used to remove dummy boundary buses.
    """
    frames = [
        getattr(result, field)
        for result in res_list
        if getattr(result, field, None) is not None
    ]

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    is_branch_data = branch_cols is not None

    from_col = None
    to_col = None

    if is_branch_data:
        from_col = next(
            (col for col in ("from_name", "fb") if col in df.columns),
            None,
        )
        to_col = next(
            (col for col in ("to_name", "tb", "name") if col in df.columns),
            None,
        )

    if valid_names is not None:
        if is_branch_data and from_col and to_col:
            df = df.loc[
                df[from_col].astype(str).isin(valid_names)
                & df[to_col].astype(str).isin(valid_names)
            ]
        elif not is_branch_data:
            name_col = next(
                (col for col in name_cols if col in df.columns),
                None,
            )
            if name_col is not None:
                df = df.loc[df[name_col].astype(str).isin(valid_names)]

    if df.empty:
        return None

    if is_branch_data:
        key = [col for col in (from_col, to_col) if col is not None]
    elif bus_id_col and bus_id_col in df.columns:
        key = [bus_id_col]
    else:
        key = [col for col in ("name", "id") if col in df.columns]

    if key:
        if "t" in df.columns:
            key.append("t")
        df = df.drop_duplicates(subset=key)

    if is_branch_data and "tb" in df.columns:
        sort_cols = ["tb"]
        if "fb" in df.columns:
            sort_cols.append("fb")

        df = df.sort_values(
            sort_cols,
            na_position="last",
        )
    elif bus_id_col and bus_id_col in df.columns:
        df = df.sort_values(
            [bus_id_col],
            na_position="last",
        )

    return df.reset_index(drop=True)


def combine_powerflow_results(
    results: Iterable[PowerFlowResult],
    case_ref: Optional[Case] = None,
    objective_value: Optional[float] = None,
) -> Optional[PowerFlowResult]:
    """Combine per-area PowerFlowResult objects into a single aggregated result.

    Covers all fields of :class:`~distopf.results.PowerFlowResult` including
    batteries, loads, capacitors, regulators, currents, and duals.
    Dummy boundary nodes inserted by the ENAPP decomposition are stripped using
    ``case_ref.bus_data.name`` when provided.
    """
    res_list = list(results)
    if not res_list:
        return None

    valid_names: Optional[set] = (
        set(case_ref.bus_data.name.astype(str).to_list())
        if case_ref is not None and case_ref.bus_data is not None
        else None
    )

    def bus(field: str, *, id_col: str = "id") -> Optional[pd.DataFrame]:
        return _concat_field(
            res_list, field, valid_names, name_cols=("name",), bus_id_col=id_col
        )

    def branch(field: str) -> Optional[pd.DataFrame]:
        return _concat_field(
            res_list,
            field,
            valid_names,
            name_cols=(),
            branch_cols=("from_name", "to_name", "fb", "tb"),
        )

    def dual(field: str) -> Optional[pd.DataFrame]:
        # Dual variable frames are branch-indexed; treat same as branch.
        return branch(field)

    # ------------------------------------------------------------------
    # Bus-level results
    # ------------------------------------------------------------------
    voltages = bus("voltages", id_col="id")
    voltage_angles = bus("voltage_angles", id_col="id")
    active_loads = bus("active_power_loads", id_col="id")
    reactive_loads = bus("reactive_power_loads", id_col="id")
    p_gens = bus("active_power_generation", id_col="id")
    q_gens = bus("reactive_power_generation", id_col="id")
    p_bats = bus("battery_active_power", id_col="id")
    q_bats = bus("battery_reactive_power", id_col="id")
    p_discharge = bus("p_discharge", id_col="id")
    p_charge = bus("p_charge", id_col="id")
    soc = bus("soc", id_col="id")
    q_caps = bus("capacitor_reactive_power", id_col="id")
    currents_df = bus("currents", id_col="id")
    current_angles_df = bus("current_angles", id_col="id")

    # ------------------------------------------------------------------
    # Branch-level results
    # ------------------------------------------------------------------
    p_flows = branch("active_power_flows")
    q_flows = branch("reactive_power_flows")
    tap_ratios = branch("tap_ratios")
    reg_taps = branch("reg_taps")

    # ------------------------------------------------------------------
    # Scalar / sparse tables that have no bus-name column — just concat
    # ------------------------------------------------------------------
    def _scalar_concat(field: str) -> Optional[pd.DataFrame]:
        frames = [
            getattr(r, field, None)
            for r in res_list
            if getattr(r, field, None) is not None
        ]
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        return df.drop_duplicates().reset_index(drop=True) if not df.empty else None

    z_caps = _scalar_concat("z_caps")
    u_caps = _scalar_concat("u_caps")

    # ------------------------------------------------------------------
    # Dual variable tables (branch-indexed)
    # ------------------------------------------------------------------
    dual_p = dual("dual_power_balance_p")
    dual_q = dual("dual_power_balance_q")
    dual_vd = dual("dual_voltage_drop")
    dual_vlo = dual("dual_voltage_limits_lower")
    dual_vhi = dual("dual_voltage_limits_upper")

    # ------------------------------------------------------------------
    # Scalar metadata
    # ------------------------------------------------------------------
    if objective_value is None:
        obj_vals = []
        for r in res_list:
            obj_val = getattr(r, "objective_value", None)
            if obj_val is not None:
                obj_vals.append(float(obj_val))
        objective_value = sum(obj_vals) if obj_vals else None

    converged_all = all(getattr(r, "converged", True) for r in res_list)

    return PowerFlowResult(
        voltages=voltages,
        voltage_angles=voltage_angles,
        active_power_flows=p_flows,
        reactive_power_flows=q_flows,
        active_power_generation=p_gens,
        reactive_power_generation=q_gens,
        active_power_loads=active_loads,
        reactive_power_loads=reactive_loads,
        battery_active_power=p_bats,
        battery_reactive_power=q_bats,
        p_discharge=p_discharge,
        p_charge=p_charge,
        soc=soc,
        capacitor_reactive_power=q_caps,
        tap_ratios=tap_ratios,
        reg_taps=reg_taps,
        z_caps=z_caps,
        u_caps=u_caps,
        currents=currents_df,
        current_angles=current_angles_df,
        dual_power_balance_p=dual_p,
        dual_power_balance_q=dual_q,
        dual_voltage_drop=dual_vd,
        dual_voltage_limits_lower=dual_vlo,
        dual_voltage_limits_upper=dual_vhi,
        objective_value=objective_value,
        converged=converged_all,
        solver="enapp",
        result_type="opf",
        case=case_ref,
    )


def _remap_area_results_to_global_ids(
    results: dict[str, PowerFlowResult],
    case: opf.Case,
    area_info: dict[str, dict[str, list]],
) -> None:
    if case.bus_data is None:
        return

    name_to_global_id = {
        str(row["name"]): int(row["id"])
        for _, row in case.bus_data.loc[:, ["id", "name"]].iterrows()
    }

    for result in results.values():
        if result is None:
            continue

        voltages = getattr(result, "voltages", None)
        if voltages is not None and "name" in voltages.columns:
            voltages = voltages.copy()
            voltages["id"] = voltages["name"].astype(str).map(name_to_global_id)
            result.voltages = voltages

        for flow_attr in ("active_power_flows", "reactive_power_flows"):
            flows = getattr(result, flow_attr, None)

            if flows is None:
                continue

            if not {"from_name", "to_name"}.issubset(flows.columns):
                continue

            flows = flows.copy()
            flows["fb"] = flows["from_name"].astype(str).map(name_to_global_id)
            flows["tb"] = flows["to_name"].astype(str).map(name_to_global_id)
            setattr(result, flow_attr, flows)

    _reconstruct_boundary_flows(
        results,
        area_info,
        case,
        name_to_global_id,
    )


def _reconstruct_boundary_flows(
    all_results: dict,
    area_info: dict,
    case: "opf.Case",
    name_to_global_id: dict,
) -> None:
    """Fix dummy-swing from_name/fb in boundary branch rows so they survive the
    valid-names filter inside ``combine_powerflow_results``.

    In each downstream area the boundary branch is stored with
    ``from_name = <upstream_area_name>`` (the dummy SWING node).  We replace
    that with the real upstream bus name, looked up from the global case's
    ``branch_data``.  The rows are also given correct global ``fb``/``tb``
    IDs so the plotting pipeline can index branches consistently.

    Mutations are made in-place on the ``PowerFlowResult`` objects in
    ``all_results``.
    """
    if case.branch_data is None:
        return

    bd = case.branch_data
    # Ensure from_name/to_name are present (added by clean_model_data in decompose)
    has_names = "from_name" in bd.columns and "to_name" in bd.columns
    if not has_names:
        return

    for area_name, info in area_info.items():
        for up_area in info.get("up_areas", []):
            src_bus = str(info["up_buses"][0])

            # Real upstream bus: the global branch whose to_name is src_bus.
            real_branch = bd.loc[bd["to_name"].astype(str) == src_bus]
            if real_branch.empty:
                continue
            real_from_name = str(real_branch.iloc[0]["from_name"])
            real_fb = name_to_global_id.get(real_from_name)
            real_tb = name_to_global_id.get(src_bus)
            if real_fb is None or real_tb is None:
                continue

            ar = all_results.get(area_name)
            if ar is None:
                continue

            for flow_attr in ("active_power_flows", "reactive_power_flows"):
                df: Optional[pd.DataFrame] = getattr(ar, flow_attr, None)
                if df is None or "from_name" not in df.columns:
                    continue
                mask = df["from_name"].astype(str) == str(up_area)
                if not mask.any():
                    continue
                df = df.copy()
                df.loc[mask, "from_name"] = real_from_name
                df.loc[mask, "to_name"] = src_bus
                df.loc[mask, "fb"] = real_fb
                df.loc[mask, "tb"] = real_tb
                setattr(ar, flow_attr, df)


def _finalize_result(
    result: Optional[PowerFlowResult],
    *,
    case: opf.Case,
    objective_value: Optional[float],
    converged: bool,
    iterations: int,
    runtime: float,
    solver_status: str,
    termination_condition: str,
    area_results: dict[str, PowerFlowResult],
    boundary_errors: list[float],
    iteration_summaries: Optional[pd.DataFrame] = None,
    area_iteration_summaries: Optional[pd.DataFrame] = None,
    parallel_used: bool,
    area_solve_failed: bool,
    failed_areas: set[str],
    solver: str = "enapp",
    backend: Optional[str] = None,
    metadata_prefix: str = "enapp",
) -> PowerFlowResult:
    if result is None:
        result = PowerFlowResult(
            objective_value=objective_value,
            converged=converged,
            solver=solver,
            result_type="opf",
            case=case,
        )

    result.case = case
    result.objective_value = objective_value
    result.solve_time = runtime
    result.iterations = iterations
    result.converged = converged
    result.solver = solver
    result.result_type = "opf"
    result.solver_status = solver_status
    result.termination_condition = termination_condition
    result.backend = backend if backend is not None else solver

    result.iteration_summaries = iteration_summaries
    result.area_iteration_summaries = area_iteration_summaries
    result.boundary_error_per_iter = boundary_errors
    result.raw_result = {
        "area_results": area_results,
        "boundary_error_per_iter": boundary_errors,
        "iteration_summaries": iteration_summaries,
        "area_iteration_summaries": area_iteration_summaries,
        f"{metadata_prefix}_iterations": iterations,
        f"{metadata_prefix}_runtime": runtime,
        f"{metadata_prefix}_parallel_used": parallel_used,
        "area_solve_failed": area_solve_failed,
        "failed_areas": sorted(failed_areas),
    }

    return result


def dampen_boundaries(boundaries, boundaries_last, alpha=0.5):
    boundaries_damped = {}
    val_cols = ["a", "b", "c"]
    for area, boundary in boundaries.items():
        boundary_last = boundaries_last.get(area)
        if boundary_last is None:
            boundaries_damped[area] = boundary
            continue
        s_up_damped = boundary.s_up.copy()
        v_dn_damped = boundary.v_down.copy()
        s_dn_damped = boundary.s_down.copy()
        v_up_damped = boundary.v_up.copy()
        s_up_damped.loc[:, val_cols] = boundary.s_up.loc[
            :, val_cols
        ] * alpha + boundary_last.s_up.loc[:, val_cols] * (1 - alpha)
        v_dn_damped.loc[:, val_cols] = boundary.v_down.loc[
            :, val_cols
        ] * alpha + boundary_last.v_down.loc[:, val_cols] * (1 - alpha)
        s_dn_damped.loc[:, val_cols] = boundary.s_down.loc[
            :, val_cols
        ] * alpha + boundary_last.s_down.loc[:, val_cols] * (1 - alpha)
        v_up_damped.loc[:, val_cols] = boundary.v_up.loc[
            :, val_cols
        ] * alpha + boundary_last.v_up.loc[:, val_cols] * (1 - alpha)
        boundaries_damped[area] = BoundaryVars(
            s_up_damped, v_dn_damped, v_up_damped, s_dn_damped
        )
    return boundaries_damped



_finalize_enapp_result = _finalize_result
finalize_result = _finalize_result
