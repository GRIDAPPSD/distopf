import multiprocessing as mp
from copy import deepcopy
from scipy.optimize import OptimizeResult
import numpy as np
import pandas as pd
from distopf.matrix_models.matrix_bess.solvers import cvxpy_solve
from distopf.matrix_models.matrix_bess.spatial_decomposition.decompose import decompose
from distopf.matrix_models.matrix_bess.lindist_mp import LinDistMP
from dataclasses import dataclass
from typing import Callable
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


def update_models(models: dict[str, LinDistMP]):
    for model in models.values():
        model.build()
    return models


def parse_v_up(model: LinDistMP, x: np.ndarray):
    swing = model.bus.loc[
        model.bus.bus_type.isin([opf.SWING_BUS, opf.SWING_FREE]), "name"
    ].to_list()[0]
    v = model.get_voltages(x)
    v = v.loc[v.name == swing, ["name", "t", "a", "b", "c"]]
    return v


def parse_s_dn(model: LinDistMP, x: np.ndarray, down_buses: list):
    s = model.get_apparent_power_flows(x)
    s = s.loc[s["to_name"].isin(down_buses), ["to_name", "t", "a", "b", "c"]]
    s["name"] = s.to_name
    s = s.loc[:, ["name", "t", "a", "b", "c"]]
    return s


def parse_v_dn(model: LinDistMP, x: np.ndarray, down_buses: list):
    v = model.get_voltages(x)
    v = v.loc[v.name.isin(down_buses), ["name", "t", "a", "b", "c"]]
    return v


def parse_s_up(model: LinDistMP, x: np.ndarray):
    swing = model.bus.loc[
        model.bus.bus_type.isin([opf.SWING_BUS, opf.SWING_FREE]), "name"
    ].to_list()[0]
    s = model.get_apparent_power_flows(x)
    s = s.loc[s["from_name"] == swing, ["from_name", "t", "a", "b", "c"]]
    s["name"] = s.from_name
    s = s.loc[:, ["name", "t", "a", "b", "c"]]
    return s


def send_s_up(models: dict[str, LinDistMP], boundaries: dict[str, BoundaryVars]):
    for up_model in models.values():
        for sending_area in up_model.bus.load_shape:
            if sending_area not in models.keys():
                continue
            s_up = deepcopy(boundaries[sending_area].s_up)
            up_model.schedules = add_s_to_schedules(
                up_model.schedules, s_up, sending_area
            )
    return models


def send_v_up(models: dict[str, LinDistMP], boundaries: dict[str, BoundaryVars]):
    for up_model in models.values():
        for sending_area in up_model.bus.load_shape:
            if sending_area not in models.keys():
                continue
            v_up = deepcopy(boundaries[sending_area].v_up)
            up_model.schedules = add_v_down_to_schedules(
                up_model.schedules, v_up, sending_area
            )
    return models


def send_v_down(models: dict[str, LinDistMP], boundaries: dict[str, BoundaryVars]):
    for sending_area, boundary in boundaries.items():
        for down_name in boundary.v_down.name:
            assert down_name in models.keys()
            v = deepcopy(boundary.v_down)
            models[down_name].schedules = add_v_swing_to_schedules(
                models[down_name].schedules, v, down_name
            )
    return models


def send_s_down(models: dict[str, LinDistMP], boundaries: dict[str, BoundaryVars]):
    for sending_area, boundary in boundaries.items():
        for down_name in boundary.s_down.name:
            assert down_name in models.keys()
            s = deepcopy(boundary.s_down)
            models[down_name].schedules = add_s_to_schedules(
                models[down_name].schedules, s, sending_area
            )
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


def build_local_to_global_map(primary_model: LinDistMP, models: dict[str, LinDistMP]):
    global_j_to_name_map = {
        bus_id - 1: str(bus_name)
        for bus_id, bus_name in primary_model.bus_data.loc[:, ["id", "name"]].to_numpy()
    }
    maps = [
        "x_maps",
        "v_map",
        "pl_map",
        "ql_map",
        "pg_map",
        "qg_map",
        "qc_map",
        "charge_map",
        "discharge_map",
        "pb_map",
        "qb_map",
        "soc_map",
        "vx_map",
    ]
    x_map_to_global = {area_name: np.zeros((0, 2)) for area_name in models.keys()}
    if len(primary_model.x_maps.keys()) == 0:
        primary_model.initialize_variable_index_pointers()
    for area, model in models.items():
        local_j_to_name_map = {
            bus_id - 1: str(bus_name)
            for bus_id, bus_name in model.bus_data.loc[:, ["id", "name"]].to_numpy()
        }
        if len(model.x_maps.keys()) == 0:
            model.initialize_variable_index_pointers()
        for var_map in maps:
            local_map = deepcopy(model.__dict__[var_map])
            global_map = deepcopy(primary_model.__dict__[var_map])
            for t in local_map.keys():
                for ph in local_map[t].keys():
                    if var_map == "x_maps":
                        df_g = deepcopy(global_map[t][ph])
                        df_g["bus_name"] = df_g.bj.apply(global_j_to_name_map.get)
                        df_l = deepcopy(local_map[t][ph])
                        df_l["bus_name"] = df_l.bj.apply(local_j_to_name_map.get)
                        df_m = pd.merge(df_g, df_l, how="inner", on="bus_name")
                        l2g_pij = df_m.loc[:, ["pij_y", "pij_x"]].to_numpy()
                        l2g_qij = df_m.loc[:, ["qij_y", "qij_x"]].to_numpy()
                        x_map_to_global[area] = np.r_[
                            x_map_to_global[area], l2g_pij, l2g_qij
                        ].astype(int)
                    else:
                        df_g = pd.DataFrame(
                            data={
                                "j": global_map[t][ph].index,
                                "idx": global_map[t][ph],
                            }
                        )
                        df_g["bus_name"] = df_g.j.apply(global_j_to_name_map.get)
                        df_l = pd.DataFrame(
                            data={"j": local_map[t][ph].index, "idx": local_map[t][ph]}
                        )
                        df_l["bus_name"] = df_l.j.apply(local_j_to_name_map.get)
                        df_m = pd.merge(df_g, df_l, how="inner", on="bus_name")
                        l2g = df_m.loc[:, ["idx_y", "idx_x"]].to_numpy()
                        x_map_to_global[area] = np.r_[
                            x_map_to_global[area], l2g
                        ].astype(int)
    return x_map_to_global


def local_to_global(results: dict, x_map_to_global: dict, n_x: int):
    x = np.ones(n_x) * np.inf
    for area, result in results.items():
        local_indexes = x_map_to_global[area][:, 0]
        global_indexes = x_map_to_global[area][:, 1]
        x[global_indexes] = result.x[local_indexes]
    return x


def _solve(_models, _model_name, objective, kwargs, conn):
    conn.send(cvxpy_solve(_models[_model_name], obj_func=objective, **kwargs))
    conn.close()


def _solve_pool(_models, _model_name, objective, kwargs):
    return cvxpy_solve(_models[_model_name], obj_func=objective, **kwargs)


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
        all_results[area_name] = cvxpy_solve(
            models[area_name], obj_func=objective, **kwargs
        )
    return all_results


def parse_all_boundaries(models, all_results, area_info):
    boundaries = {}
    for area_name in models.keys():
        down_buses = area_info[area_name]["down_areas"]
        s_up = parse_s_up(models[area_name], all_results[area_name].x)
        v_dn = parse_v_dn(models[area_name], all_results[area_name].x, down_buses)
        v_up = parse_v_up(models[area_name], all_results[area_name].x)
        s_dn = parse_s_dn(models[area_name], all_results[area_name].x, down_buses)
        boundaries[area_name] = BoundaryVars(s_up=s_up, v_down=v_dn)
    return boundaries


def send_all_boundaries(models, boundaries):
    models = send_s_up(models, boundaries)
    models = send_v_down(models, boundaries)
    models = update_models(models)
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
    model: LinDistMP,
    area_info: dict[str, dict[str, list]],
    objective: Callable,
    tol: float,
    **kwargs,
):
    tic = perf_counter()
    sources = {area_name: data["up_buses"][0] for area_name, data in area_info.items()}
    models = decompose(model, sources)
    # for _m in models.values():
    #     plot_network(_m).show()
    local_to_global_map = build_local_to_global_map(model, models)
    all_results = {}

    models = update_models(models)
    boundaries = {}
    x_all = np.zeros(model.n_x)
    cost_per_iter = []
    boundary_error_per_iter = []
    for it in range(100):
        all_results = _cvxpy_solve_all_parallel(models, objective, **kwargs)
        boundaries_prev = deepcopy(boundaries)
        boundaries = parse_all_boundaries(models, all_results, area_info)
        models = send_all_boundaries(models, boundaries)
        if it < 1:
            continue
        diff_max = calculate_boundary_deviation(boundaries, boundaries_prev)
        # print(diff_max)
        boundary_error_per_iter.append(diff_max)
        if diff_max < tol:
            # print(f"Solved after {it} iterations.")
            x_all = local_to_global(all_results, local_to_global_map, model.n_x)
            break
    fun = objective(model, x_all, **kwargs)
    result = OptimizeResult(
        fun=fun,
        # success=(prob.status == "optimal"),
        # message=prob.status,
        x=x_all,
        nit=it,
        area_results=all_results,
        runtime=perf_counter() - tic,
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
