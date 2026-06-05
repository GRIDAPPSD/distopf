import multiprocessing as mp
from copy import deepcopy
import numpy as np
import pandas as pd
import warnings
from distopf.distributed.spatial.decompose import decompose
from distopf.api import Case
from distopf.results import PowerFlowResult
from dataclasses import dataclass
from typing import Callable, Iterable, Optional
from time import perf_counter
import distopf as opf


@dataclass
class BoundaryVars:
    s_up: pd.DataFrame
    v_down: pd.DataFrame

    def __sub__(self, other):
        dv = pd.merge(
            self.v_down,
            other.v_down,
            how="left",
            on=["name", "t"],
            suffixes=("", "_prev"),
        )
        dv.a = dv.a - dv.a_prev
        dv.b = dv.b - dv.b_prev
        dv.c = dv.c - dv.c_prev
        dv = dv.loc[:, ["name", "t", "a", "b", "c"]]
        ds = pd.merge(
            self.s_up, other.s_up, how="left", on=["name", "t"], suffixes=("", "_prev")
        )
        ds.a = ds.a - ds.a_prev
        ds.b = ds.b - ds.b_prev
        ds.c = ds.c - ds.c_prev
        ds = ds.loc[:, ["name", "t", "a", "b", "c"]]
        return BoundaryVars(ds, dv)

    def __abs__(self):
        s = deepcopy(self.s_up)
        s.loc[:, ["a", "b", "c"]] = self.s_up.loc[:, ["a", "b", "c"]].apply(abs)
        v = deepcopy(self.v_down)
        v.loc[:, ["a", "b", "c"]] = self.v_down.loc[:, ["a", "b", "c"]].apply(abs)
        return BoundaryVars(s, v)


def parse_v_up(case: Case, result: PowerFlowResult):
    assert case.bus_data is not None
    swing = case.bus_data.loc[
        case.bus_data.bus_type.isin([opf.SWING_BUS, opf.SWING_FREE]), "name"
    ].to_list()[0]
    v = result.voltages
    if v is None:
        return pd.DataFrame(columns=["name", "t", "a", "b", "c"])
    v = v.loc[v.name == swing, ["name", "t", "a", "b", "c"]]
    return v


def parse_s_dn(case: Case, result: PowerFlowResult, down_buses: list):
    p = result.active_power_flows
    q = result.reactive_power_flows
    if p is None or q is None:
        return pd.DataFrame(columns=["name", "t", "a", "b", "c"])
    p = p.loc[p["to_name"].isin(down_buses), ["to_name", "t", "a", "b", "c"]]
    q = q.loc[q["to_name"].isin(down_buses), ["to_name", "t", "a", "b", "c"]]
    s = p.copy()
    for ph in ["a", "b", "c"]:
        s[ph] = p[ph] + 1j * q[ph]
    s["name"] = s.to_name
    s = s.loc[:, ["name", "t", "a", "b", "c"]]
    return s


def parse_v_dn(case: Case, result: PowerFlowResult, down_buses: list):
    assert case.bus_data is not None
    v = result.voltages
    if v is None:
        return pd.DataFrame(columns=["name", "t", "a", "b", "c"])
    v = v.loc[v.name.isin(down_buses), ["name", "t", "a", "b", "c"]]
    return v


def parse_s_up(case: Case, result: PowerFlowResult):
    assert case.bus_data is not None
    swing = case.bus_data.loc[
        case.bus_data.bus_type.isin([opf.SWING_BUS, opf.SWING_FREE]), "name"
    ].to_list()[0]
    p = result.active_power_flows
    q = result.reactive_power_flows
    if p is None or q is None:
        return pd.DataFrame(columns=["name", "t", "a", "b", "c"])
    p = p.loc[p["from_name"] == swing, ["from_name", "t", "a", "b", "c"]]
    q = q.loc[q["from_name"] == swing, ["from_name", "t", "a", "b", "c"]]
    s = p.copy()
    for ph in ["a", "b", "c"]:
        s[ph] = p[ph] + 1j * q[ph]
    s["name"] = s.from_name
    s = s.loc[:, ["name", "t", "a", "b", "c"]]
    return s


def _concat_field(
    res_list: list,
    field: str,
    valid_names: Optional[set],
    name_cols: tuple = ("name",),
    bus_id_col: Optional[str] = None,
    branch_cols: Optional[tuple] = None,
) -> Optional[pd.DataFrame]:
    """Concatenate a DataFrame field from multiple PowerFlowResult objects.

    Parameters
    ----------
    res_list : list[PowerFlowResult]
        Area results to merge.
    field : str
        PowerFlowResult attribute name.
    valid_names : set or None
        Set of global bus names used to filter out dummy boundary nodes.
        Pass None to skip filtering.
    name_cols : tuple of str
        Columns checked when filtering by valid_names (bus-level data).
        Only rows where ALL of these columns are in valid_names are kept.
        For branch data pass as empty tuple and use branch_cols instead.
    bus_id_col : str or None
        When set, deduplicate by (bus_id_col,) or (bus_id_col, "t").
    branch_cols : tuple of str or None
        For branch/flow data: candidate column names for from/to keys.
        The first existing pair is used for filtering and deduplication.
    """
    frames = [
        getattr(r, field) for r in res_list if getattr(r, field, None) is not None
    ]
    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)

    # Filter out dummy boundary nodes using valid bus names.
    if valid_names is not None:
        if branch_cols:
            # branch / flow frames: keep rows where both endpoints are real.
            from_candidates = ("from_name", "fb")
            to_candidates = ("to_name", "tb", "name")
            fc = next((c for c in from_candidates if c in df.columns), None)
            tc = next((c for c in to_candidates if c in df.columns), None)
            if fc and tc:
                df = df.loc[
                    df[fc].astype(str).isin(valid_names)
                    & df[tc].astype(str).isin(valid_names)
                ]
        elif name_cols:
            # bus-level frames: at least one name column must exist.
            nc = next((c for c in name_cols if c in df.columns), None)
            if nc:
                df = df.loc[df[nc].astype(str).isin(valid_names)]

    if df.empty:
        return None

    # Deduplicate.
    if branch_cols:
        fc = next((c for c in ("from_name", "fb") if c in df.columns), None)
        tc = next((c for c in ("to_name", "tb", "name") if c in df.columns), None)
        key = [c for c in (fc, tc) if c]
    elif bus_id_col and bus_id_col in df.columns:
        key = [bus_id_col]
    else:
        key = [c for c in ("name", "id") if c in df.columns]

    if key:
        if "t" in df.columns:
            key = key + ["t"]
        df = df.drop_duplicates(subset=key).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Sort branch data: tb primary, fb secondary. This aligns with case.branch_data order.
    # Critical for plotting: _process_branch_data reindexes by `tb - 1`.
    if branch_cols and "tb" in df.columns:
        sort_cols = ["tb"]
        if "fb" in df.columns:
            sort_cols.append("fb")
        df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    # Sort bus data by id for consistency.
    elif bus_id_col and bus_id_col in df.columns:
        df = df.sort_values([bus_id_col], na_position="last").reset_index(drop=True)

    return df


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


def send_s_up(cases: dict[str, Case], boundaries: dict[str, BoundaryVars]):
    for up_case in cases.values():
        for sending_area in up_case.bus_data.load_shape:
            if sending_area not in cases.keys():
                continue
            s_up = deepcopy(boundaries[sending_area].s_up)
            up_case.schedules = add_s_to_schedules(
                up_case.schedules, s_up, sending_area
            )
    return cases


def send_v_up(models: dict[str, Case], boundaries: dict[str, BoundaryVars]):
    for up_model in models.values():
        for sending_area in up_model.bus_data.load_shape:
            if sending_area not in models.keys():
                continue
            v_down = deepcopy(boundaries[sending_area].v_down)
            up_model.schedules = add_v_down_to_schedules(
                up_model.schedules, v_down, sending_area
            )
    return models


def send_v_down(models: dict[str, Case], boundaries: dict[str, BoundaryVars]):
    for sending_area, boundary in boundaries.items():
        for down_name in boundary.v_down.name:
            assert down_name in models.keys()
            v = deepcopy(boundary.v_down)
            models[down_name].schedules = add_v_swing_to_schedules(
                models[down_name].schedules, v, down_name
            )
    return models


def send_s_down(models: dict[str, Case], boundaries: dict[str, BoundaryVars]):
    for sending_area, boundary in boundaries.items():
        # s_down is not defined; if needed, implement or remove this logic
        pass
    return models


# def add_v_swing_to_schedules(schedules, v, receiving_area):
#     v = v.loc[v.name == receiving_area, ["t", "a", "b", "c"]]
#     v.index = v.t
#     v = v.loc[:, ["a", "b", "c"]]
#     for t in v.index.unique():
#         v_t = v.loc[t]
#         if isinstance(v_t, pd.DataFrame):
#             v_t = v_t.iloc[0]
#         schedules.loc[schedules.time == t, ["v_a", "v_b", "v_c"]] = v_t.to_numpy()
#     return schedules

def add_v_swing_to_schedules(schedules, v, receiving_area):
    v_swing = (
        v.loc[v.name == receiving_area, ["t", "a", "b", "c"]]
        .drop_duplicates(subset=["t"], keep="first")
        .rename(columns={"t": "time", "a": "v_a", "b": "v_b", "c": "v_c"})
        .set_index("time")
    )
    schedules = schedules.set_index("time")
    schedules.loc[v_swing.index, ["v_a", "v_b", "v_c"]] = v_swing[
        ["v_a", "v_b", "v_c"]
    ]
    return schedules.reset_index()

def add_v_down_to_schedules(schedules, v, sending_area):
    v = deepcopy(v)
    v.index = v.t
    v = v.loc[:, ["a", "b", "c"]]
    for t in v.index.unique():
        # Get the row(s) for this time step
        v_t = v.loc[t]
        # If multiple rows per time (shouldn't happen for boundary), take first
        if isinstance(v_t, pd.DataFrame):
            v_t = v_t.iloc[0]
        schedules.loc[
            schedules.time == t, [f"{sending_area}.{ph}.v" for ph in "abc"]
        ] = v_t.to_numpy()
    return schedules


# def add_s_to_schedules(schedules, s, sending_area):
#     s = deepcopy(s)
#     s.index = s.t
#     s = s.loc[:, ["a", "b", "c"]]
#     p_down = s.apply(np.real)
#     q_down = s.apply(np.imag)
#     for t in s.index.unique():
#         # Get row(s) for this time step
#         s_t = s.loc[t]
#         p_t = p_down.loc[t]
#         q_t = q_down.loc[t]
#         # If DataFrames (multiple rows), take first
#         if isinstance(s_t, pd.DataFrame):
#             s_t = s_t.iloc[0]
#         if isinstance(p_t, pd.DataFrame):
#             p_t = p_t.iloc[0]
#         if isinstance(q_t, pd.DataFrame):
#             q_t = q_t.iloc[0]
#         schedules.loc[
#             schedules.time == t, [f"{sending_area}.{ph}.p" for ph in "abc"]
#         ] = p_t.to_numpy()
#         schedules.loc[
#             schedules.time == t, [f"{sending_area}.{ph}.q" for ph in "abc"]
#         ] = q_t.to_numpy()
#     return schedules

def add_s_to_schedules(schedules, s, sending_area):
    p_cols = [f"{sending_area}.{ph}.p" for ph in "abc"]
    q_cols = [f"{sending_area}.{ph}.q" for ph in "abc"]
    
    s_prep = (
        s.loc[:, ["t", "a", "b", "c"]]
        .drop_duplicates(subset=["t"], keep="first")
        .rename(columns={"t": "time"})
        .set_index("time")
    )
    
    # Extract real and imaginary parts
    p_data = s_prep[["a", "b", "c"]].apply(np.real)
    p_data.columns = p_cols
    q_data = s_prep[["a", "b", "c"]].apply(np.imag)
    q_data.columns = q_cols
    
    schedules = schedules.set_index("time")
    schedules.loc[s_prep.index, p_cols] = p_data
    schedules.loc[s_prep.index, q_cols] = q_data
    return schedules.reset_index()

def local_to_global(results: dict, x_map_to_global: dict, n_x: int):
    x = np.ones(n_x) * np.inf
    for area, result in results.items():
        local_indexes = x_map_to_global[area][:, 0]
        global_indexes = x_map_to_global[area][:, 1]
        if hasattr(result, "x"):
            x[global_indexes] = result.x[local_indexes]
        else:
            # fallback: try voltages or another property if needed
            pass
    return x


def _solve(_models, _model_name, objective, kwargs, conn):
    try:
        result = _models[_model_name].run_opf(objective=objective, **kwargs)
    except Exception as exc:
        conn.send(("error", _model_name, repr(exc)))
        conn.close()
        raise RuntimeError(f"ENAPP area solve failed for {_model_name}") from exc
    conn.send(("ok", _model_name, result))
    conn.close()


def _solve_pool(_cases, _area_name, objective, kwargs):
    try:
        result = _cases[_area_name].run_opf(objective=objective, **kwargs)
        # Multiprocessing workers must return pickle-safe objects. Some
        # backends (notably Pyomo) attach non-picklable solver/model objects
        # to PowerFlowResult for diagnostics. They are not used by ENAPP
        # boundary exchange, so strip them here before returning to parent.
        if hasattr(result, "raw_result"):
            result.raw_result = None
        if hasattr(result, "model"):
            result.model = None
        return result
    except Exception as exc:
        return None
        # raise RuntimeError(f"ENAPP area solve failed for {_area_name}") from exc


def _solve_all_parallel(cases, objective, **kwargs):
    all_results = {}
    with mp.Pool() as pool:
        args = [(cases, area_name, objective, kwargs) for area_name in cases.keys()]
        all_results_list = pool.starmap(_solve_pool, args)
    for area_name, result in zip(cases.keys(), all_results_list):
        all_results[area_name] = result
    return all_results


def _solve_all_loop(models, objective, **kwargs):
    all_results = {}
    for area_name in models.keys():
        try:
            all_results[area_name] = models[area_name].run_opf(
                objective=objective, **kwargs
            )
        except Exception as exc:
            all_results[area_name] = None
            # raise RuntimeError(f"ENAPP area solve failed for {area_name}") from exc
    return all_results


def parse_all_boundaries(models, all_results, area_info):
    boundaries = {}
    for area_name in models.keys():
        down_buses = area_info[area_name]["down_areas"]
        result = all_results[area_name]
        s_up = parse_s_up(models[area_name], result)
        v_dn = parse_v_dn(models[area_name], result, down_buses)
        boundaries[area_name] = BoundaryVars(s_up=s_up, v_down=v_dn)
    return boundaries


def send_all_boundaries(cases, boundaries):
    cases = send_s_up(cases, boundaries)
    cases = send_v_down(cases, boundaries)
    return cases


def calculate_boundary_deviation(boundaries, boundaries_prev):
    diff_maxes = []
    for area_name in boundaries.keys():
        diff = abs(boundaries[area_name] - boundaries_prev[area_name])
        p = diff.s_up.loc[:, ["a", "b", "c"]].apply(np.real)
        q = diff.s_up.loc[:, ["a", "b", "c"]].apply(np.imag)
        p_max = np.max(p)
        q_max = np.max(q)
        v_max = np.max(diff.v_down.loc[:, ["a", "b", "c"]])
        diff_maxes.append(np.nanmax([v_max, p_max, q_max]))
    diff_max = max(diff_maxes)
    return diff_max


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


def solve_enapp(
    case: opf.Case,
    area_info: dict[str, dict[str, list]],
    objective: Callable,
    tol: float,
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
    **kwargs,
) -> PowerFlowResult:
    """Solve a decomposed OPF/PF problem with ENAPP using modern Case API.

    Parameters
    ----------
    case : opf.Case
        Full network case.
    area_info : dict
        ENAPP area topology and boundary definitions.
    objective : Callable | str
        Objective accepted by ``Case.run_opf``.
    tol : float
        Boundary convergence tolerance.
    max_iterations : int, default 100
        Maximum ENAPP coordination iterations.
    parallel : bool, default True
        Whether to attempt multiprocessing for per-area solves.

    Returns
    -------
    PowerFlowResult
        Aggregated network-level result. Additional ENAPP metadata is attached:
        ``area_results``, ``boundary_error_per_iter``, ``enapp_iterations``,
        ``enapp_runtime``, and ``enapp_parallel_used``.
    """
    tic = perf_counter()
    sources = {area_name: data["up_buses"][0] for area_name, data in area_info.items()}
    cases = decompose(case, sources)
    # for _m in models.values():
    #     plot_network(_m).show()
    all_results = {}
    all_results_next = {}
    boundaries = {}  # type: dict[str, BoundaryVars]
    # cost_per_iter = []
    boundary_error_per_iter = []
    parallel_enabled = parallel
    converged = False
    it = -1
    for it in range(max_iterations):
        if solve_callback is not None:
            all_results_next = solve_callback(cases, objective, **kwargs)
        elif parallel_enabled:
            all_results_next = _solve_all_parallel(cases, objective, **kwargs)
        else:
            all_results_next = _solve_all_loop(cases, objective, **kwargs)
        for area_name, result in all_results_next.items():
            if result is None:
                print(f"{area_name} solve failed. Using last iteration result.")
                continue
            all_results[area_name] = all_results_next[area_name]
        if len(all_results.keys()) < len(cases.keys()):
            # Not all areas solved; aggregate partial results and return early
            n_solved = len(all_results)
            n_total = len(cases)
            print(
                f"Only {n_solved}/{n_total} areas solved. Returning aggregated partial result."
            )
            root_areas = [
                a for a, info in area_info.items() if not info.get("up_areas")
            ]
            fun_vals = []
            for area_name, ar in all_results.items():
                if area_name not in root_areas:
                    continue
                if (
                    ar is not None
                    and hasattr(ar, "objective_value")
                    and ar.objective_value is not None
                ):
                    fun_vals.append(ar.objective_value)
            fun = sum(fun_vals) if fun_vals else None
            partial_result = combine_powerflow_results(
                [ar for ar in all_results.values() if ar is not None],
                case_ref=case,
                objective_value=fun,
            )
            if partial_result is None:
                partial_result = PowerFlowResult(
                    objective_value=fun,
                    converged=False,
                    solver="enapp",
                    result_type="opf",
                    case=case,
                )
            partial_result.case = case
            partial_result.objective_value = fun
            partial_result.converged = False
            partial_result.solver = "enapp"
            partial_result.result_type = "opf"
            partial_result.solver_status = "failure"
            partial_result.termination_condition = "incomplete_solve"
            partial_result.backend = "enapp"
            return partial_result
        boundaries_prev = deepcopy(boundaries)
        boundaries = parse_all_boundaries(cases, all_results, area_info)
        cases = send_all_boundaries(cases, boundaries)
        if iteration_callback is not None:
            iteration_callback(it, cases, all_results, boundaries)
        if it < 1:
            continue
        diff_max = calculate_boundary_deviation(boundaries, boundaries_prev)
        # print(diff_max)
        boundary_error_per_iter.append(diff_max)
        if diff_max < tol:
            # print(f"Solved after {it} iterations.")
            converged = True
            break
    # Aggregate per-area objective values if available (each area returns a PowerFlowResult)
    # Only sum objectives for root areas (areas with no upstream areas) to avoid
    # double-counting cost contributions at dummy boundary swings.
    root_areas = [a for a, info in area_info.items() if not info.get("up_areas")]
    fun_vals = []
    for area_name, ar in all_results.items():
        if area_name not in root_areas:
            continue
        if hasattr(ar, "objective_value") and ar.objective_value is not None:
            fun_vals.append(ar.objective_value)
    fun = sum(fun_vals) if fun_vals else None
    # Remap local-area fb/tb/id columns to global bus IDs so that plotting
    # routines (which index by id-1) work correctly on the aggregated result.
    if case is not None and hasattr(case, "bus_data") and case.bus_data is not None:
        name_to_global_id: dict[str, int] = {
            str(r[1]): int(r[0])
            for r in case.bus_data.loc[:, ["id", "name"]].to_numpy()
        }
        for ar in all_results.values():
            # --- voltages: id column ---
            if getattr(ar, "voltages", None) is not None:
                v = ar.voltages.copy()
                if "name" in v.columns:
                    v["id"] = v["name"].astype(str).map(name_to_global_id)
                    ar.voltages = v
            # --- branch flows: fb/tb columns ---
            for flow_attr in ("active_power_flows", "reactive_power_flows"):
                df = getattr(ar, flow_attr, None)
                if (
                    df is not None
                    and "from_name" in df.columns
                    and "to_name" in df.columns
                ):
                    df = df.copy()
                    df["fb"] = df["from_name"].astype(str).map(name_to_global_id)
                    df["tb"] = df["to_name"].astype(str).map(name_to_global_id)
                    setattr(ar, flow_attr, df)

        # Reconstruct boundary-branch rows (dummy swing → real upstream bus)
        # so they are not dropped by the valid-names filter during aggregation.
        _reconstruct_boundary_flows(all_results, area_info, case, name_to_global_id)
    # Build an aggregated PowerFlowResult from per-area results for convenience
    aggregated_result = combine_powerflow_results(
        all_results.values(), case_ref=case, objective_value=fun
    )
    if aggregated_result is None:
        aggregated_result = PowerFlowResult(
            objective_value=fun,
            converged=converged,
            solver="enapp",
            result_type="opf",
            case=case,
        )

    runtime = perf_counter() - tic
    aggregated_result.case = case
    aggregated_result.objective_value = fun
    aggregated_result.solve_time = runtime
    aggregated_result.iterations = it + 1
    aggregated_result.converged = converged
    aggregated_result.solver = "enapp"
    aggregated_result.result_type = "opf"
    aggregated_result.solver_status = "optimal" if converged else "max_iterations"
    aggregated_result.termination_condition = (
        "converged" if converged else "max_iterations"
    )
    aggregated_result.backend = "enapp"

    # Distributed diagnostics are kept in raw_result for compatibility.
    aggregated_result.raw_result = {
        "area_results": all_results,
        "boundary_error_per_iter": boundary_error_per_iter,
        "enapp_iterations": it + 1,
        "enapp_runtime": runtime,
        "enapp_parallel_used": parallel_enabled,
    }

    return aggregated_result
