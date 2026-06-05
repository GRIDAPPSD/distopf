from pathlib import Path
import distopf as opf
from distopf.pyomo_models.objectives import (
    gen_cost_rule,
    substation_cost_objective_rule,
    total_cost_rule,
)

from distopf.distributed.spatial.enapp import Case, solve_enapp, PowerFlowResult
from distopf.distributed.temporal import solve_tenapp_aprx, energy_cost_min

OUTPUT_DIR = Path("scratch/enapp_123_isolated")
PHASES = ("a", "b", "c")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    case = opf.create_case(
        opf.CASES_DIR / "csv" / "ieee123_30der",
        n_steps=1,
        start_step=0,
        ignore_bat=True,
        # ignore_schedule=True,
        # ignore_gen=True,
    )
    case.branch_data.loc[case.branch_data.fb == 1, ["sa_max", "sb_max", "sc_max"]] = (
        0.000001
    )
    case.schedules["PV"] = 1.0
    case.schedules["default"] = 1.0
    case.schedules["price"] = 0
    case.modify(gen_mult=5, control_variable="PQ", v_min=0.0, v_max=2.0)
    case.gen_data["cost"] = 10
    print("Total Load: \n")
    print(case.bus_data.loc[:, ["pl_a", "pl_b", "pl_c"]].sum())
    print("Total Gen: \n")
    print(case.gen_data.loc[:, ["pa", "pb", "pc"]].sum())
    # case.reg_data.loc[case.reg_data.tb == 127, ["tap_a", "tap_b", "tap_c"]] = [0.0, 0.0, 0.0]
    # case.reg_data.loc[case.reg_data.tb == 127, ["ratio_a", "ratio_b", "ratio_c"]] = [1.0, 1.0, 1.0]

    area_info = {
        "area1": {
            "up_areas": [],
            "down_areas": ["area2", "area3"],
            "up_buses": ["150"],
            "down_buses": ["152", "135"],
        },
        "area2": {
            "up_areas": ["area1"],
            "down_areas": ["area4"],
            "up_buses": ["152"],
            "down_buses": ["160"],
        },
        "area3": {
            "up_areas": ["area1"],
            "down_areas": [],
            "up_buses": ["135"],
            "down_buses": [],
        },
        "area4": {
            "up_areas": ["area2"],
            "down_areas": [],
            "up_buses": ["160"],
            "down_buses": [],
        },
    }

    result_c = case.run_opf(
        objective=total_cost_rule,
        formulation="lindist",
    )
    print(result_c.objective_value, " in ", result_c.solve_time)
    # result_c.plot_network().show(renderer="browser")

    def iteration_callback(
        it, cases: dict[str, Case], all_results: dict[str, PowerFlowResult], boundaries
    ) -> None:
        print("ENAPP iteration ", it)
        # for area_name, boundary_vars in boundaries.items():
        # print(area_name)
        # print(f"objective value: {all_results[area_name].objective_value}")
        # print(f"{area_name} s_up")
        # print(boundary_vars.s_up)
        # print(f"{area_name} v_down")
        # print(boundary_vars.v_down)
        # all_results[area_name].plot_network().show(renderer="browser")
        # print()

    result_enapp = solve_enapp(
        case,
        area_info,
        tol=1e-6,
        # objective=substation_cost_objective_rule,
        objective="min_loss",
        wrapper="pyomo",
        formulation="lindist",
        iteration_callback=iteration_callback,
    )
    print(result_enapp.objective_value)

    v_d = result_enapp.voltages
    v_c = result_c.voltages
    # print(result_enapp.q_gens)
    # print(result_c.q_gens)
    # print(v_c)
    # print(v_d)
    opf.compare_voltages(v_c, v_d, t=1).show(renderer="browser")
    print()


if __name__ == "__main__":
    main()
