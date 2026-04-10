from pathlib import Path
import distopf as opf
from distopf.pyomo_models.objectives import substation_cost_objective_rule

from distopf.distributed.spatial.enapp_copy import Case, solve_enapp, PowerFlowResult

OUTPUT_DIR = Path("scratch/enapp_123_debug")
PHASES = ("a", "b", "c")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    case = opf.create_case(
        opf.CASES_DIR / "csv" / "ieee123_30der_bat",
        n_steps=24,
        start_step=0,
        # ignore_bat=True,
        # ignore_gen=True,
    )
    demand_charge = 0  # $/p.u. power
    # case.bus_data.v_max = 2
    # case.bus_data.v_min = 0
    case.gen_data.control_variable = "PQ"
    case.bat_data.control_variable = "P"
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
        objective=substation_cost_objective_rule,
        demand_charge=demand_charge,
        cost_curve=case.schedules.price.to_numpy(),
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
        objective=substation_cost_objective_rule,
        demand_charge=demand_charge,
        cost_curve=case.schedules.price.to_numpy(),
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
    # opf.compare_voltages(v_c, v_d, t=1).show(renderer="browser")
    print()


if __name__ == "__main__":
    main()
