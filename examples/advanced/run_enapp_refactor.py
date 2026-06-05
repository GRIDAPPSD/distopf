import distopf as opf
from distopf.matrix_models.matrix_bess.objectives import cp_obj_cost_min
from distopf.distributed.spatial.enapp import solve_enapp
from distopf.api import create_case
from distopf import CASES_DIR
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
    if case.bat_data is not None:
        case.bat_data.control_variable = "P"

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

    # Full-case OPF (backend auto-selected or force multiperiod)
    result_c = case.run_opf(
        objective=cp_obj_cost_min,
        wrapper="matrix_bess",
        solver=cp.CLARABEL,
        demand_charge=demand_charge,
        cost_curve=case.schedules.price.to_numpy(),
    )
    print("central objective:", result_c.objective_value)

    # Distributed ENAPP
    result_enapp = solve_enapp(
        case,
        area_info_,
        tol=1e-6,
        solver=cp.CLARABEL,
        objective=cp_obj_cost_min,
        demand_charge=demand_charge,
        cost_curve=case.schedules.price.to_numpy(),
    )

    print("enapp objective:", result_enapp.objective_value)

    # Compare voltages if available
    v_c = result_c.voltages if hasattr(result_c, "voltages") else None
    v_enapp = result_enapp.voltages

    if v_c is not None and v_enapp is not None:
        opf.compare_voltages(v_c, v_enapp, t=12).show(renderer="browser")
        opf.compare_voltages(v_c, v_enapp, t=13).show(renderer="browser")

    # Print generator reactive outputs if present
    enapp_meta = (
        result_enapp.raw_result if isinstance(result_enapp.raw_result, dict) else {}
    )
    area_results = enapp_meta.get("area_results", {})
    if isinstance(area_results, dict):
        for area, ar in area_results.items():
            print(area, "q_gens:")
            if hasattr(ar, "q_gens"):
                print(ar.q_gens)

    if hasattr(result_c, "q_gens"):
        print("central q_gens:")
        print(result_c.q_gens)


if __name__ == "__main__":
    main()
