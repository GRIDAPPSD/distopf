from copy import deepcopy
from pathlib import Path
import pandas as pd
import distopf as opf
from distopf.matrix_models.matrix_bess.lindist_mp import LinDistMP
from distopf.matrix_models.matrix_bess.objectives import cp_obj_cost_min

from distopf.distributed.spatial.enapp import solve_enapp
from distopf.api import create_case
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
    area_info = {
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

    result_c = case.run_opf(
        objective=cp_obj_cost_min,
        solver=cp.CLARABEL,
        demand_charge=demand_charge,
        cost_curve=case.schedules.price.to_numpy(),
        formulation="lindist_mp",
    )
    print(result_c.objective_value, " in ", result_c.solve_time)
    result_enapp = solve_enapp(
        case,
        area_info,
        tol=1e-6,
        solver=cp.CLARABEL,
        objective=cp_obj_cost_min,
        demand_charge=demand_charge,
        cost_curve=case.schedules.price.to_numpy(),
        formulation="lindist_mp",
    )
    print(result_enapp.objective_value)

    v_d = result_enapp.voltages
    v_c = result_c.voltages
    print(result_enapp.q_gens)
    print(result_c.q_gens)

    opf.compare_voltages(v_c, v_d, t=12).show(renderer="browser")
    opf.compare_voltages(v_c, v_d, t=13).show(renderer="browser")
    # np.savetxt("copf33x.csv", result_c.x)
    # np.savetxt("dopf33x.csv", result_enapp.x)
    # px.scatter(result_c.x - result_enapp.x).show(renderer="browser")
    print()


if __name__ == "__main__":
    main()
