import distopf as opf

from distopf import (
    plot_voltages,
    plot_gens,
    plot_network,
    plot_polar,
)
case = opf.DistOPFCase(
    data_path=opf.CASES_DIR / "csv" / "ieee123_30der",
    objective_function=opf.cp_obj_loss,
    control_variable="PQ",
)
results = case.run(raw_result=True)
print(f"Objective value: {case.results.fun}")
# plot_voltages(case.v, t=0).show(renderer="browser")
# plot_gens(case.p_flow, case.).show(renderer="browser")
case.plot_voltages().show(renderer="browser")
case.plot_power_flows().show(renderer="browser")
plot_gens(case.p_gens, case.q_gens).show(renderer="browser")
plot_polar(case.p_gens, case.q_gens).show(renderer="browser")
# plot_gens(case.p_bat, case.q_bat).show(renderer="browser")
case.plot_network(show_reactive_power=True).show(renderer="browser")
