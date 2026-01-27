import multiprocessing as mp
from copy import deepcopy
from scipy.optimize import OptimizeResult
import numpy as np
import pandas as pd
from distopf.spatial_decomposition.decompose import decompose
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
    p = result.p_flows
    q = result.q_flows
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
    p = result.p_flows
    q = result.q_flows
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


def combine_powerflow_results(
    results: Iterable[PowerFlowResult],
    case_ref: Optional[Case] = None,
    objective_value: Optional[float] = None,
) -> Optional[PowerFlowResult]:
    """Combine multiple PowerFlowResult objects into a single canonical result.

    - Concatenates voltages, p_flows, q_flows, p_gens, q_gens when present.
    - Filters out dummy area nodes using `case_ref.bus_data.name` when provided.
    - Deduplicates by (name,t) for voltages and by branch/time identifiers for flows.
    """
    res_list = list(results)
    if len(res_list) == 0:
        return None

    valid_names = (
        set(case_ref.bus_data.name.astype(str).to_list())
        if case_ref is not None
        else None
    )

    # Voltages
    v_list = [
        r.voltages
        for r in res_list
        if hasattr(r, "voltages") and r.voltages is not None
    ]
    v_all = pd.concat(v_list, ignore_index=True) if v_list else None
    if v_all is not None:
        if valid_names is not None:
            v_all = v_all.loc[v_all.name.astype(str).isin(valid_names)]
        if "t" in v_all.columns:
            v_all = v_all.drop_duplicates(subset=["name", "t"]).reset_index(drop=True)
        else:
            v_all = v_all.drop_duplicates(subset=["name"]).reset_index(drop=True)

    # Power flows
    p_list = [
        r.p_flows for r in res_list if hasattr(r, "p_flows") and r.p_flows is not None
    ]
    q_list = [
        r.q_flows for r in res_list if hasattr(r, "q_flows") and r.q_flows is not None
    ]
    p_all = pd.concat(p_list, ignore_index=True) if p_list else None
    q_all = pd.concat(q_list, ignore_index=True) if q_list else None
    if p_all is not None and q_all is not None:
        from_col = (
            "from_name"
            if "from_name" in p_all.columns
            else ("fb" if "fb" in p_all.columns else None)
        )
        to_col = (
            "to_name"
            if "to_name" in p_all.columns
            else (
                "name"
                if "name" in p_all.columns
                else ("tb" if "tb" in p_all.columns else None)
            )
        )
        if from_col is not None and to_col is not None and valid_names is not None:
            p_all = p_all.loc[
                p_all[from_col].astype(str).isin(valid_names)
                & p_all[to_col].astype(str).isin(valid_names)
            ]
            q_all = q_all.loc[
                q_all[from_col].astype(str).isin(valid_names)
                & q_all[to_col].astype(str).isin(valid_names)
            ]
        dup_cols = [
            c
            for c in ("fb", "tb", "id", "from_name", "to_name", "name")
            if c in p_all.columns
        ]
        if "t" in p_all.columns and dup_cols:
            p_all = p_all.drop_duplicates(subset=dup_cols + ["t"]).reset_index(
                drop=True
            )
            q_all = q_all.drop_duplicates(subset=dup_cols + ["t"]).reset_index(
                drop=True
            )
        elif dup_cols:
            p_all = p_all.drop_duplicates(subset=dup_cols).reset_index(drop=True)
            q_all = q_all.drop_duplicates(subset=dup_cols).reset_index(drop=True)

    # Generators
    pg_list = [
        r.p_gens for r in res_list if hasattr(r, "p_gens") and r.p_gens is not None
    ]
    qg_list = [
        r.q_gens for r in res_list if hasattr(r, "q_gens") and r.q_gens is not None
    ]
    p_gens = pd.concat(pg_list, ignore_index=True) if pg_list else None
    q_gens = pd.concat(qg_list, ignore_index=True) if qg_list else None
    if p_gens is not None and valid_names is not None:
        p_gens = p_gens.loc[p_gens.name.astype(str).isin(valid_names)]
        if "t" in p_gens.columns:
            p_gens = p_gens.drop_duplicates(subset=["id", "t"]).reset_index(drop=True)
        else:
            p_gens = p_gens.drop_duplicates(subset=["id"]).reset_index(drop=True)
    if q_gens is not None and valid_names is not None:
        q_gens = q_gens.loc[q_gens.name.astype(str).isin(valid_names)]
        if "t" in q_gens.columns:
            q_gens = q_gens.drop_duplicates(subset=["id", "t"]).reset_index(drop=True)
        else:
            q_gens = q_gens.drop_duplicates(subset=["id"]).reset_index(drop=True)

    # objective_value: sum unless provided
    if objective_value is None:
        obj_vals = [
            r.objective_value
            for r in res_list
            if hasattr(r, "objective_value") and r.objective_value is not None
        ]
        objective_value = sum(obj_vals) if obj_vals else None

    from distopf.results import PowerFlowResult

    aggregated = PowerFlowResult(
        voltages=v_all,
        p_flows=p_all,
        q_flows=q_all,
        p_gens=p_gens,
        q_gens=q_gens,
        objective_value=objective_value,
        converged=all([getattr(r, "converged", True) for r in res_list]),
        solver="enapp",
        result_type="opf",
        raw_result=None,
        model=None,
        case=case_ref,
    )
    return aggregated


def send_s_up(models: dict[str, Case], boundaries: dict[str, BoundaryVars]):
    for up_model in models.values():
        for sending_area in up_model.bus_data.load_shape:
            if sending_area not in models.keys():
                continue
            s_up = deepcopy(boundaries[sending_area].s_up)
            up_model.schedules = add_s_to_schedules(
                up_model.schedules, s_up, sending_area
            )
    return models


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
    v = v.loc[v.name == receiving_area, ["t", "a", "b", "c"]]
    v.index = v.t
    v = v.loc[:, ["a", "b", "c"]]
    for t in v.index:
        schedules.loc[schedules.time == t, ["v_a", "v_b", "v_c"]] = v.loc[t].to_numpy()
    return schedules


def add_v_down_to_schedules(schedules, v, sending_area):
    v = deepcopy(v)
    v.index = v.t
    v = v.loc[:, ["a", "b", "c"]]
    schedules.loc[:, [f"{sending_area}.{ph}.v" for ph in "abc"]] = v
    for t in v.index:
        schedules.loc[
            schedules.time == t, [f"{sending_area}.{ph}.v" for ph in "abc"]
        ] = v.loc[t].to_numpy()
    return schedules


def add_s_to_schedules(schedules, s, sending_area):
    s = deepcopy(s)
    s.index = s.t
    s = s.loc[:, ["a", "b", "c"]]
    p_down = s.apply(np.real)
    q_down = s.apply(np.imag)
    schedules.loc[:, [f"{sending_area}.{ph}.p" for ph in "abc"]] = p_down
    schedules.loc[:, [f"{sending_area}.{ph}.q" for ph in "abc"]] = q_down
    for t in s.index:
        schedules.loc[
            schedules.time == t, [f"{sending_area}.{ph}.p" for ph in "abc"]
        ] = np.array(p_down.loc[t])
        schedules.loc[
            schedules.time == t, [f"{sending_area}.{ph}.q" for ph in "abc"]
        ] = np.array(q_down.loc[t])
    return schedules


def local_to_global(results: dict, x_map_to_global: dict, n_x: int):
    x = np.ones(n_x) * np.inf
    for area, result in results.items():
        local_indexes = x_map_to_global[area][:, 0]
        global_indexes = x_map_to_global[area][:, 1]
        # result.x is now result.x or another property, update as needed
        if hasattr(result, "x"):
            x[global_indexes] = result.x[local_indexes]
        else:
            # fallback: try voltages or another property if needed
            pass
    return x


def _solve(_models, _model_name, objective, kwargs, conn):
    result = _models[_model_name].run_opf(objective=objective, **kwargs)
    conn.send(result)
    conn.close()


def _solve_pool(_models, _model_name, objective, kwargs):
    return _models[_model_name].run_opf(objective=objective, **kwargs)


def _cvxpy_solve_all_parallel(models, objective, **kwargs):
    all_results = {}
    with mp.Pool() as pool:
        args = [(models, area_name, objective, kwargs) for area_name in models.keys()]
        all_results_list = pool.starmap(_solve_pool, args)
    for area_name, result in zip(models.keys(), all_results_list):
        all_results[area_name] = result
    return all_results


def _cvxpy_solve_all_loop(models, objective, **kwargs):
    all_results = {}
    for area_name in models.keys():
        all_results[area_name] = models[area_name].run_opf(
            objective=objective, **kwargs
        )
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


def send_all_boundaries(models, boundaries):
    models = send_s_up(models, boundaries)
    models = send_v_down(models, boundaries)
    # update_models is deprecated and removed
    return models


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


def solve_enapp(
    case: opf.Case,
    area_info: dict[str, dict[str, list]],
    objective: Callable,
    tol: float,
    **kwargs,
):
    tic = perf_counter()
    sources = {area_name: data["up_buses"][0] for area_name, data in area_info.items()}
    cases = decompose(case, sources)
    # for _m in models.values():
    #     plot_network(_m).show()
    all_results = {}

    # update_models is deprecated and removed
    x_all = None  # Placeholder, will be set when convergence is reached
    boundaries = {}  # type: dict[str, BoundaryVars]
    # cost_per_iter = []
    boundary_error_per_iter = []
    for it in range(100):
        all_results = _cvxpy_solve_all_parallel(cases, objective, **kwargs)
        boundaries_prev = deepcopy(boundaries)
        boundaries = parse_all_boundaries(cases, all_results, area_info)
        cases = send_all_boundaries(cases, boundaries)
        if it < 1:
            continue
        diff_max = calculate_boundary_deviation(boundaries, boundaries_prev)
        # print(diff_max)
        boundary_error_per_iter.append(diff_max)
        if diff_max < tol:
            # print(f"Solved after {it} iterations.")
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
    # Ensure per-area voltage DataFrames include global bus ids so downstream
    # plotting/merging (which may rely on the original `id` column) works even
    # after local remapping performed during decomposition.
    if case is not None and hasattr(case, "bus_data") and case.bus_data is not None:
        name_to_global_id = {
            str(r[1]): r[0] for r in case.bus_data.loc[:, ["id", "name"]].to_numpy()
        }
        for ar in all_results.values():
            if hasattr(ar, "voltages") and ar.voltages is not None:
                v = ar.voltages.copy()
                if "name" in v.columns:
                    # Map names back to global ids where possible
                    mapped = v["name"].astype(str).map(name_to_global_id)
                    if "id" in v.columns:
                        # Prefer the global id when available
                        v.loc[:, "id"] = mapped.where(~mapped.isna(), v["id"])  # type: ignore
                    else:
                        v.loc[:, "id"] = mapped
                    ar.voltages = v
    # Build an aggregated PowerFlowResult from per-area results for convenience
    aggregated_result = combine_powerflow_results(
        all_results.values(), case_ref=case, objective_value=fun
    )

    result = OptimizeResult(
        fun=fun,
        # success=(prob.status == "optimal"),
        # message=prob.status,
        x=x_all,
        nit=it,
        area_results=all_results,
        runtime=perf_counter() - tic,
        aggregated_result=aggregated_result,
    )
    return result


# def main():
#     base_path = Path("33bus")
#     branch_data = pd.read_csv(base_path / "branch_data.csv")
#     branch_data.drop(
#         index=branch_data.loc[branch_data.status == "open"].index, inplace=True
#     )
#     bus_data = pd.read_csv(base_path / "bus_data2.csv")
#     gen_data = pd.read_csv(base_path / "gen_data.csv")
#     battery_data = pd.read_csv(base_path / "battery_data.csv")
#     schedules = pd.read_csv(base_path / "schedules.csv")
#     tou_rates = pd.read_csv(base_path / "tou_rates.csv")
#     demand_charge = 12.96  # $/kW per month
#     # start_time = 9
#     # n_steps = 15
#     start_time = 12
#     n_steps = 1
#     bus_data.v_max = 1.05
#     bus_data.v_min = 0.95

#     m = LinDistModelMP(
#         branch_data=branch_data,
#         bus_data=bus_data,
#         gen_data=gen_data,
#         bat_data=battery_data,
#         schedules=schedules,
#         start_step=start_time,
#         n_steps=n_steps,
#     )

#     # plot_network(m).show()
#     area_info_ = {
#         "area1": {
#             "up_areas": [],
#             "down_areas": ["area2", "area3"],
#             "up_buses": [1],
#             "down_buses": [5, 19],
#         },
#         "area2": {
#             "up_areas": ["area1"],
#             "down_areas": [],
#             "up_buses": [5],
#             "down_buses": [],
#         },
#         "area3": {
#             "up_areas": ["area1"],
#             "down_areas": [],
#             "up_buses": [19],
#             "down_buses": [],
#         },
#     }

#     sources = {
#         "area1": 1,
#         "area2": 5,
#         "area3": 19,
#     }

#     # area_models = decompose(m, sources)
#     result_c = cvxpy_solve(
#         m,
#         cost_min,
#         solver=cp.CLARABEL,
#         objective=cost_min,
#         demand_charge=demand_charge,
#         cost_curve=tou_rates.C.to_numpy(),
#     )
#     print(result_c.fun, " in ", result_c.runtime)
#     result_enapp = solve_enapp(
#         m,
#         area_info_,
#         tol=1e-6,
#         solver=cp.CLARABEL,
#         objective=cost_min,
#         demand_charge=demand_charge,
#         cost_curve=tou_rates.C.to_numpy(),
#     )
#     print(result_enapp.fun)

#     def at_time(df: pd.DataFrame, t):
#         names = [name for name in df.columns if name != "t"]
#         _df = deepcopy(df.loc[df.t == t, names])
#         return _df

#     # v_d = at_time(m.get_voltages(result_enapp.x), 12)
#     # v_c = at_time(m.get_voltages(result_c.x), 12)
#     # compare_voltages(v_c, v_d).show()
#     # v_d = at_time(m.get_voltages(result_enapp.x), 13)
#     # v_c = at_time(m.get_voltages(result_c.x), 13)
#     # compare_voltages(v_c, v_d).show()
#     np.savetxt("copf33x.csv", result_c.x)
#     np.savetxt("dopf33x.csv", result_enapp.x)
#     px.scatter(result_c.x - result_enapp.x).show()
#     print()


# if __name__ == "__main__":
#     main()
