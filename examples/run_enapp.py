import multiprocessing as mp
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import distopf as opf
from distopf.matrix_models.multiperiod.solvers import cvxpy_solve
from distopf.matrix_models.multiperiod.lindist_loads_mp import LinDistMPL
from distopf.matrix_models.multiperiod.lindist_mp import LinDistMP
from distopf.matrix_models.multiperiod.objectives import cp_obj_cost_min
from distopf.matrix_models.multiperiod.spatial_decomposition.enapp import solve_enapp
from distopf.importer import create_case
from distopf import CASES_DIR
import plotly.express as px
import cvxpy as cp


def main():
    base_path = CASES_DIR / "csv" / "ieee33"
    case = create_case(base_path, start_step=12, n_steps=2, delta_t=1)
    case.branch_data.drop(
        index=case.branch_data.loc[case.branch_data.status == "open"].index,
        inplace=True,
    )
    demand_charge = 12.96  # $/kW per month
    case.bus_data.v_max = 1.05
    case.bus_data.v_min = 0.95
    case.gen_data.control_variable = "PQ"
    case.bat_data.control_variable = "P"
    m = LinDistMP(case=case)
    m.build()
    # plot_network(m).show()
    area_info_ = {
        "area1": {
            "up_areas": [],
            "down_areas": ["area2", "area3"],
            "up_buses": ["1"],
            "down_buses": [5, 19],
        },
        "area2": {
            "up_areas": ["area1"],
            "down_areas": [],
            "up_buses": ["5"],
            "down_buses": [],
        },
        "area3": {
            "up_areas": ["area1"],
            "down_areas": [],
            "up_buses": ["19"],
            "down_buses": [],
        },
    }

    # area_models = decompose(m, sources)
    result_c = cvxpy_solve(
        m,
        cp_obj_cost_min,
        solver=cp.CLARABEL,
        demand_charge=demand_charge,
        cost_curve=case.schedules.price.to_numpy(),
    )
    print(result_c.fun, " in ", result_c.runtime)
    result_enapp = solve_enapp(
        m,
        area_info_,
        tol=1e-6,
        solver=cp.CLARABEL,
        objective=cp_obj_cost_min,
        demand_charge=demand_charge,
        cost_curve=case.schedules.price.to_numpy(),
    )
    print(result_enapp.fun)

    def at_time(df: pd.DataFrame, t):
        names = [name for name in df.columns if name != "t"]
        _df = deepcopy(df.loc[df.t == t, names])
        return _df

    v_d = m.get_voltages(result_enapp.x)
    v_c = m.get_voltages(result_c.x)
    print(m.get_q_gens(result_enapp.x))
    print(m.get_q_gens(result_c.x))

    opf.compare_voltages(v_c, v_d, t=12).show(renderer="browser")
    opf.compare_voltages(v_c, v_d, t=13).show(renderer="browser")
    # np.savetxt("copf33x.csv", result_c.x)
    # np.savetxt("dopf33x.csv", result_enapp.x)
    px.scatter(result_c.x - result_enapp.x).show(renderer="browser")
    print()


if __name__ == "__main__":
    main()
