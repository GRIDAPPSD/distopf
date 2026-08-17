import multiprocessing as mp
from copy import deepcopy
import numpy as np
import pandas as pd
from distopf.distributed.spatial.decompose import decompose
from distopf.api import Case
from distopf.results import PowerFlowResult
from dataclasses import dataclass
from typing import Callable, Iterable, Optional
from time import perf_counter
import distopf as opf
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    logger.addHandler(_handler)


def _validate_boundary_frame_pair(
    current: pd.DataFrame,
    previous: pd.DataFrame,
    field_name: str,
) -> None:
    keys = ["name", "t"]

    missing_current = set(keys) - set(current.columns)
    missing_previous = set(keys) - set(previous.columns)
    if missing_current or missing_previous:
        raise ValueError(
            f"{field_name} is missing boundary key columns: "
            f"current={sorted(missing_current)}, "
            f"previous={sorted(missing_previous)}"
        )

    if current.duplicated(keys).any():
        raise ValueError(f"{field_name} contains duplicate boundary rows by {keys}")

    if previous.duplicated(keys).any():
        raise ValueError(
            f"Previous {field_name} contains duplicate boundary rows by {keys}"
        )

    current_keys = pd.MultiIndex.from_frame(current[keys])
    previous_keys = pd.MultiIndex.from_frame(previous[keys])

    missing_from_current = previous_keys.difference(current_keys)
    missing_from_previous = current_keys.difference(previous_keys)

    if len(missing_from_current) or len(missing_from_previous):
        raise ValueError(
            f"{field_name} boundary rows do not match between iterations. "
            f"Missing from current: {missing_from_current.tolist()}; "
            f"missing from previous: {missing_from_previous.tolist()}"
        )


@dataclass
class BoundaryVars:
    s_up: pd.DataFrame
    v_down: pd.DataFrame

    def __sub__(self, other):
        _validate_boundary_frame_pair(
            self.v_down,
            other.v_down,
            "v_down",
        )
        _validate_boundary_frame_pair(
            self.s_up,
            other.s_up,
            "s_up",
        )

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
            self.s_up,
            other.s_up,
            how="left",
            on=["name", "t"],
            suffixes=("", "_prev"),
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


def _get_swing_name(case: Case) -> str:
    if case.bus_data is None:
        raise ValueError("Case has no bus_data")

    swing_names = case.bus_data.loc[
        case.bus_data.bus_type.isin([opf.SWING_BUS, opf.SWING_FREE]),
        "name",
    ].tolist()

    if len(swing_names) != 1:
        raise ValueError(f"Expected exactly one swing bus, found {len(swing_names)}")

    return str(swing_names[0])


def parse_v_up(case: Case, result: PowerFlowResult):
    swing = _get_swing_name(case)
    v = result.voltages

    if v is None:
        return pd.DataFrame(columns=["name", "t", "a", "b", "c"])

    return v.loc[
        v.name.astype(str) == swing,
        ["name", "t", "a", "b", "c"],
    ]


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
    swing = _get_swing_name(case)
    p = result.active_power_flows
    q = result.reactive_power_flows

    if p is None or q is None:
        return pd.DataFrame(columns=["name", "t", "a", "b", "c"])

    p = p.loc[
        p["from_name"].astype(str) == swing,
        ["from_name", "t", "a", "b", "c"],
    ]
    q = q.loc[
        q["from_name"].astype(str) == swing,
        ["from_name", "t", "a", "b", "c"],
    ]

    s = p.copy()
    for phase in "abc":
        s[phase] = p[phase] + 1j * q[phase]

    s["name"] = s["from_name"]
    return s.loc[:, ["name", "t", "a", "b", "c"]]


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


def add_v_swing_to_schedules(schedules, v, receiving_area):
    v_rows = v.loc[
        v.name == receiving_area,
        ["t", "a", "b", "c"],
    ]

    assert not v_rows.duplicated(["t"]).any(), (
        f"Duplicate voltage boundary values for area {receiving_area}"
    )

    v_swing = v_rows.rename(
        columns={
            "t": "time",
            "a": "v_a",
            "b": "v_b",
            "c": "v_c",
        }
    ).set_index("time")

    schedules = schedules.set_index("time")

    assert v_swing.index.isin(schedules.index).all(), (
        f"Voltage boundary times are missing from the schedule "
        f"for area {receiving_area}"
    )

    schedules.loc[
        v_swing.index,
        ["v_a", "v_b", "v_c"],
    ] = v_swing[["v_a", "v_b", "v_c"]]

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


def add_s_to_schedules(schedules, s, sending_area):
    p_cols = [f"{sending_area}.{phase}.p" for phase in "abc"]
    q_cols = [f"{sending_area}.{phase}.q" for phase in "abc"]

    s_rows = s.loc[:, ["t", "a", "b", "c"]]

    assert not s_rows.duplicated(["t"]).any(), (
        f"Duplicate power boundary values for area {sending_area}"
    )

    s_prep = s_rows.rename(columns={"t": "time"}).set_index("time")
    schedules = schedules.set_index("time")

    assert s_prep.index.isin(schedules.index).all(), (
        f"Power boundary times are missing from the schedule for area {sending_area}"
    )

    p_data = s_prep[["a", "b", "c"]].apply(np.real)
    p_data.columns = p_cols
    q_data = s_prep[["a", "b", "c"]].apply(np.imag)
    q_data.columns = q_cols

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


def _solve_pool(area_name, area_case, objective, kwargs):
    try:
        result = area_case.run_opf(objective=objective, **kwargs)

        # Multiprocessing workers must return pickle-safe objects.
        if hasattr(result, "raw_result"):
            result.raw_result = None
        if hasattr(result, "model"):
            result.model = None
        return result

    except Exception:
        logger.exception("ENAPP solve failed for area %s", area_name)
        return None


def _solve_all_parallel(cases, objective, **kwargs):
    area_names = list(cases)
    args = [
        (area_name, cases[area_name], objective, kwargs) for area_name in area_names
    ]

    with mp.Pool() as pool:
        results = pool.starmap(_solve_pool, args)

    return dict(zip(area_names, results))


def _solve_all_loop(cases, objective, **kwargs):
    all_results = {}

    for area_name, area_case in cases.items():
        try:
            all_results[area_name] = area_case.run_opf(objective=objective, **kwargs)
        except Exception:
            logger.exception("ENAPP solve failed for area %s", area_name)
            all_results[area_name] = None

    return all_results


def _safe_frame_max(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    values = frame.to_numpy(
        dtype=float,
        na_value=np.nan,
    ).ravel()
    values = values[~np.isnan(values)]
    if values.size == 0:
        return 0.0
    return float(np.max(values))


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
    for area_name in boundaries:
        if area_name not in boundaries_prev:
            raise ValueError(f"Previous boundary values are missing area {area_name}")
        diff = abs(boundaries[area_name] - boundaries_prev[area_name])
        p = diff.s_up.loc[:, ["a", "b", "c"]].apply(np.real)
        q = diff.s_up.loc[:, ["a", "b", "c"]].apply(np.imag)
        v = diff.v_down.loc[:, ["a", "b", "c"]]
        p_max = _safe_frame_max(p)
        q_max = _safe_frame_max(q)
        v_max = _safe_frame_max(v)
        diff_maxes.append(float(np.nanmax([v_max, p_max, q_max])))
    return max(diff_maxes, default=0.0)


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


def _finalize_enapp_result(
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
    parallel_used: bool,
    area_solve_failed: bool,
    failed_areas: set[str],
) -> PowerFlowResult:
    if result is None:
        result = PowerFlowResult(
            objective_value=objective_value,
            converged=converged,
            solver="enapp",
            result_type="opf",
            case=case,
        )

    result.case = case
    result.objective_value = objective_value
    result.solve_time = runtime
    result.iterations = iterations
    result.converged = converged
    result.solver = "enapp"
    result.result_type = "opf"
    result.solver_status = solver_status
    result.termination_condition = termination_condition
    result.backend = "enapp"

    result.raw_result = {
        "area_results": area_results,
        "boundary_error_per_iter": boundary_errors,
        "enapp_iterations": iterations,
        "enapp_runtime": runtime,
        "enapp_parallel_used": parallel_used,
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
        s_up_damped.loc[:, val_cols] = boundary.s_up.loc[
            :, val_cols
        ] * alpha + boundary_last.s_up.loc[:, val_cols] * (1 - alpha)
        v_dn_damped.loc[:, val_cols] = boundary.v_down.loc[
            :, val_cols
        ] * alpha + boundary_last.v_down.loc[:, val_cols] * (1 - alpha)
        boundaries_damped[area] = BoundaryVars(s_up_damped, v_dn_damped)
    return boundaries_damped


def solve_enapp(
    case: opf.Case,
    area_info: dict[str, dict[str, list]],
    objective: Callable,
    swing_voltage_slack_penalty: float = 1e6,
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
    logger.setLevel(logging.INFO if verbose_enapp else logging.WARNING)

    tic = perf_counter()
    sources = {area_name: info["up_buses"][0] for area_name, info in area_info.items()}
    cases = decompose(case, sources)

    if "free_swing_voltage" in kwargs:
        raise TypeError("free_swing_voltage is controlled by solve_enapp")

    solve_kwargs = {
        **kwargs,
        "free_swing_voltage": True,
        "swing_voltage_slack_penalty": (swing_voltage_slack_penalty),
    }

    root_areas = _get_root_areas(area_info)
    all_results = {}
    boundaries = {}
    boundary_error_per_iter = []

    failed_areas = set()
    any_area_solve_failed = False
    converged = False

    parallel_used = parallel and solve_callback is None

    for iterations in range(1, max_iterations + 1):
        iteration_solve_failed = False
        if solve_callback is not None:
            iteration_results = solve_callback(cases, objective, **solve_kwargs)
        elif parallel:
            iteration_results = _solve_all_parallel(cases, objective, **solve_kwargs)
        else:
            iteration_results = _solve_all_loop(cases, objective, **solve_kwargs)

        for area_name, result in iteration_results.items():
            if result is None:
                iteration_solve_failed = True
                any_area_solve_failed = True
                failed_areas.add(area_name)

                logger.warning(
                    "ENAPP area %s failed on iteration %d; "
                    "retaining its previous successful result, if any.",
                    area_name,
                    iterations,
                )
                continue

            all_results[area_name] = result

        # At least one area has never produced a successful result.
        if len(all_results) < len(cases):
            _remap_area_results_to_global_ids(all_results, case, area_info)
            objective_value = _aggregate_root_objective(all_results, root_areas)
            partial_result = combine_powerflow_results(
                all_results.values(), case_ref=case, objective_value=objective_value
            )
            runtime = perf_counter() - tic
            logger.warning(
                "Only %d/%d ENAPP areas solved; returning a partial result.",
                len(all_results),
                len(cases),
            )
            return _finalize_enapp_result(
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
                parallel_used=parallel_used,
                area_solve_failed=True,
                failed_areas=failed_areas,
            )

        previous_boundaries = deepcopy(boundaries)
        boundaries_last = deepcopy(boundaries)
        boundaries = parse_all_boundaries(cases, all_results, area_info)
        boundaries = dampen_boundaries(boundaries, boundaries_last, alpha=1.0)
        swing_voltage_errors = {}
        for area_name, result in all_results.items():
            area_errors = {phase: 0.0 for phase in "abc"}
            v_result = parse_v_up(cases[area_name], result)
            schedules = cases[area_name].schedules

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
        cases = send_all_boundaries(cases, boundaries)

        if iteration_callback is not None:
            iteration_callback(iterations, cases, all_results, boundaries)

        if iterations == 1:
            continue

        boundary_error = calculate_boundary_deviation(
            boundaries,
            previous_boundaries,
        )
        boundary_error_per_iter.append(boundary_error)
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
        # A failed area solve prevents this run from being
        # reported as converged, even if stale boundary values meet tol.
        if boundary_error < tol and not iteration_solve_failed:
            logger.info(
                "ENAPP converged after %d iterations with boundary error %.6e",
                iterations,
                boundary_error,
            )
            converged = True
            break

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

    return _finalize_enapp_result(
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
        parallel_used=parallel_used,
        area_solve_failed=any_area_solve_failed,
        failed_areas=failed_areas,
    )
