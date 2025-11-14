import distopf as opf
import pyomo.environ as pyo
from distopf.pyomo_models.lindist import create_lindist_model
from distopf.importer import create_case
from distopf.pyomo_models import constraints
from distopf.pyomo_models.results import OpfResult
from distopf import (
    plot_voltages,
    # plot_gens,
    # plot_network,
    # plot_polar,
)
from distopf.matrix_models.multiperiod.solvers import cp_obj_loss, cvxpy_solve
# from distopf.matrix_models.multiperiod.lindist_mp import LinDistMP
from distopf.matrix_models.multiperiod.lindist_loads_mp import LinDistMPL
from time import perf_counter

t0 = perf_counter()
case = create_case(data_path=opf.CASES_DIR / "csv" / "ieee123_30der", n_steps=1, start_step=12)
# case.bus_data.v_max = 2
# case.bus_data.v_min = 0
case.schedules.default = 1
case.schedules.PV = 1
case.gen_data.control_variable = "PQ"
matrix_model = LinDistMPL(
    case=case
)
t1 = perf_counter()
results_matrix = cvxpy_solve(matrix_model, obj_func=cp_obj_loss)
t2 = perf_counter()
print(f"Objective value: {results_matrix.fun}")
v = matrix_model.get_voltages(results_matrix.x)
p_gen = matrix_model.get_p_gens(results_matrix.x)
q_gen = matrix_model.get_q_gens(results_matrix.x)
# opf.plot_voltages(v).show(renderer="browser")
# case_matrix.plot_power_flows().show(renderer="browser")
# opf.plot_gens(p_gen, q_gen).show(renderer="browser")
# plot_polar(case_matrix.p_gens, case_matrix.q_gens).show(renderer="browser")
# case_matrix.plot_network(show_reactive_power=True).show(renderer="browser")
t3 = perf_counter()
model = create_lindist_model(case)
constraints.add_p_flow_constraints(model)
constraints.add_q_flow_constraints(model)
# Node Voltages
constraints.add_voltage_limits(model)
constraints.add_voltage_drop_constraints(model)
constraints.add_swing_bus_constraints(model)
# Loads, Capacitors and Regulators
constraints.add_cvr_load_constraints(model)
constraints.add_capacitor_constraints(model)
constraints.add_regulator_constraints(model)
# Generators
constraints.add_generator_limits(model)
constraints.add_generator_constant_p_constraints_q_control(model)
constraints.add_generator_constant_q_constraints_p_control(model)
# constraints.add_circular_generator_constraints_pq_control(model)
constraints.add_octagonal_inverter_constraints_pq_control(model)
# Battery models
constraints.add_battery_constant_q_constraints_p_control(model)
constraints.add_battery_energy_constraints(model)
constraints.add_battery_net_p_bat_equal_phase_constraints(model)
constraints.add_battery_power_limits(model)
constraints.add_battery_soc_limits(model)


def loss_objective_rule(model):
    """
    Calculate total system losses using the resistance parameters.
    For each branch-phase combination, calculates (P² + Q²) * R
    """
    total_loss = 0
    for _id, ph in model.branch_phase_set:
        for t in model.time_set:
            total_loss += (model.p_flow[_id, ph, t] ** 2) * model.r[_id, ph + ph]
            total_loss += (model.q_flow[_id, ph, t] ** 2) * model.r[_id, ph + ph]
    return total_loss


model.objective = pyo.Objective(
    rule=loss_objective_rule,
    sense=pyo.minimize,
)
# Solve the model
opt = pyo.SolverFactory("ipopt")
t4 = perf_counter()
results = opt.solve(model)
t5 = perf_counter()
# Extract and display results
if results.solver.status == pyo.SolverStatus.ok:
    print("Optimization successful!")
    print(f"Objective value: {pyo.value(model.objective)}")
    # # data = get_all_results(model, case)
    res = OpfResult(model)
    # s = res.p_flow.copy()
    # s = s.drop(["a", "b", "c"], axis=1)
    # s["a"] = res.p_flow.a + 1j * res.q_flow.a
    # s["b"] = res.p_flow.b + 1j * res.q_flow.b
    # s["c"] = res.p_flow.c + 1j * res.q_flow.c
    # s["tb"] = s["id"]
    # s["fb"] = s["tb"].map(model.from_bus_map)
    plot_voltages(res.voltages, t=0).show(renderer="browser")
    # plot_gens(res.p_flow, res.q_flow).show(renderer="browser")
    # plot_polar(res.p_flow, res.q_flow).show(renderer="browser")
    # plot_gens(res.p_gen, res.q_gen).show(renderer="browser")
    # plot_polar(res.p_gen, res.q_gen).show(renderer="browser")
    # # plot_gens(res.p_bat, res.q_bat).show(renderer="browser")
    # plot_network(
    #     case,
    #     res.voltages,
    #     s,
    #     p_gen=res.p_gen,
    #     q_gen=res.q_gen,
    #     show_reactive_power=True,
    # ).show(renderer="browser")

else:
    print("Optimization failed!")

print("Matrix solve:")
print(f" - setup time: {t1 - t0}")
print(f" - solve time: {t2 - t1}")
print(f" - total time: {t2 - t0}")
print("Pyomo solve:")
print(f" - setup time: {t4 - t3}")
print(f" - solve time: {t5 - t4}")
print(f" - total time: {t5 - t3}")