import distopf as opf
import pyomo.environ as pyo
from distopf.pyomo_models.common.factory import create_lindist_model
from distopf.api import create_case
from distopf.pyomo_models.common.results import PyoResult
from distopf import (
    plot_voltages,
    plot_gens,
    plot_network,
    plot_polar,
)

case = create_case(data_path=opf.CASES_DIR / "csv" / "ieee123_30der")
case.gen_data.control_variable = "PQ"
model = create_lindist_model(case)


def loss_objective_rule(model):
    """
    Calculate total system losses using the resistance parameters.
    For each branch-phase combination, calculates (P² + Q²) * R
    """
    total_loss = 0
    for fb, tb, phase in model.branch_phase_set:
        for t in model.time_set:
            total_loss += (
                model.p_flow[fb, tb, phase, t] ** 2 * model.r[fb, tb, phase + phase]
            )
            total_loss += (
                model.q_flow[fb, tb, phase, t] ** 2 * model.r[fb, tb, phase + phase]
            )
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
    # print("Optimization successful!")
    # print(f"Objective value: {pyo.value(model.objective)}")
    # data = get_all_results(model, case)
    sol = PyoResult(model, results)
    plot_voltages(sol.voltages, t=0).show(renderer="browser")
    plot_gens(sol.p_flow, sol.q_flow).show(renderer="browser")
    plot_polar(sol.p_flow, sol.q_flow).show(renderer="browser")
    plot_gens(sol.p_gen, sol.q_gen).show(renderer="browser")
    plot_polar(sol.p_gen, sol.q_gen).show(renderer="browser")
    # plot_gens(res.p_bat, res.q_bat).show(renderer="browser")
    plot_network(
        case,
        v=sol.voltages,
        p_flow=sol.p_flow,
        q_flow=sol.q_flow,
        p_gen=sol.p_gen,
        q_gen=sol.q_gen,
        show_reactive_power=True,
    ).show(renderer="browser")

else:
    print("Optimization failed!")
