import distopf as opf
import pyomo.environ as pyo
from distopf.pyomo_models.lindist import create_lindist_model
from distopf.importer import create_case
from distopf.pyomo_models.constraints import (
    add_capacitor_constraints,
    add_circular_generator_constraints_pq_control,
    add_cvr_load_constraints,
    # add_octagonal_inverter_constraints_pq_control,
    add_generator_constant_p_constraints_q_control,
    add_generator_constant_q_constraints_p_control,
    add_p_flow_constraints,
    add_q_flow_constraints,
    add_swing_bus_constraints,
    add_voltage_drop_constraints,
    add_regulator_constraints,
    add_generator_limits,
    add_voltage_limits,
)
from distopf.pyomo_models.results import (
    get_voltages,
    get_values,
)
from distopf import (
    plot_voltages,
    plot_gens,
    # plot_network,
    plot_polar,
)

case = create_case(data_path=opf.CASES_DIR / "csv" / "ieee123_30der")
model = create_lindist_model(case)
add_p_flow_constraints(model)
add_q_flow_constraints(model)
add_voltage_drop_constraints(model)
add_swing_bus_constraints(model)
add_cvr_load_constraints(model)
add_generator_constant_p_constraints_q_control(model)
add_generator_constant_q_constraints_p_control(model)
add_circular_generator_constraints_pq_control(model)
# add_octagonal_inverter_constraints_pq_control(model)
add_capacitor_constraints(model)
add_regulator_constraints(model)
# bounds
add_voltage_limits(model)
add_generator_limits(model)


def loss_objective_rule(model):
    """
    Calculate total system losses using the resistance parameters.
    For each branch-phase combination, calculates (P² + Q²) * R
    """
    total_loss = 0
    for _id, ph in model.branch_phase_set:
        total_loss += (model.p_flow[_id, ph] ** 2) * model.r[_id, ph + ph]
        total_loss += (model.q_flow[_id, ph] ** 2) * model.r[_id, ph + ph]
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
    v = get_voltages(model.v)
    v2 = get_values(model.v)
    p_flow = get_values(model.p_flow)
    q_flow = get_values(model.q_flow)
    p_gen = get_values(model.p_gen)
    q_gen = get_values(model.q_gen)
    plot_voltages(v).show(renderer="browser")
    plot_gens(p_flow, q_flow).show(renderer="browser")
    plot_gens(p_gen, q_gen).show(renderer="browser")
    plot_polar(p_gen, q_gen).show(renderer="browser")

else:
    print("Optimization failed!")
