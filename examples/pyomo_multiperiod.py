import distopf as opf
import pyomo.environ as pyo
from distopf.pyomo_models.lindist import create_lindist_model
from distopf.importer import  create_case
from distopf.pyomo_models import constraints
from distopf.pyomo_models.results import OpfResult
from distopf import (
    plot_voltages,
    plot_gens,
    plot_network,
    plot_polar,
)

case = create_case(data_path=opf.CASES_DIR / "csv" / "ieee123_30der")
case.gen_data.control_variable = "PQ"
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
constraints.add_circular_generator_constraints_pq_control(model)
# constraints.add_octagonal_inverter_constraints_pq_control(model)
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
results = opt.solve(model)

# Extract and display results
if results.solver.status == pyo.SolverStatus.ok:
    print("Optimization successful!")
    print(f"Objective value: {pyo.value(model.objective)}")
    # data = get_all_results(model, case)
    sol = OpfResult(model)
    plot_voltages(sol.voltages, t=0).show(renderer="browser")
    plot_gens(sol.p_flow, sol.q_flow).show(renderer="browser")
    plot_polar(sol.p_flow, sol.q_flow).show(renderer="browser")
    plot_gens(sol.p_gen, sol.q_gen).show(renderer="browser")
    plot_polar(sol.p_gen, sol.q_gen).show(renderer="browser")
    # plot_gens(res.p_bat, res.q_bat).show(renderer="browser")
    plot_network(
        case,
        v = sol.voltages,
        p_flow = sol.p_flow,
        q_flow = sol.q_flow,
        p_gen=sol.p_gen,
        q_gen=sol.q_gen,
        show_reactive_power=True,
    ).show(renderer="browser")

else:
    print("Optimization failed!")
