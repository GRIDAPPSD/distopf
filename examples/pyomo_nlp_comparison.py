import distopf as opf
import pyomo.environ as pyo
from distopf.pyomo_models.objectives import loss_objective_rule
from distopf.pyomo_models.lindist import create_lindist_model
from distopf.importer import create_case
from distopf.pyomo_models.nl_branchflow_prebuilt import NLBranchFlow
from distopf.pyomo_models.lindist_loads import LinDistPyoMPL
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

case = create_case(data_path=opf.CASES_DIR / "csv" / "ieee123_30der", start_step=12)

lindist = LinDistPyoMPL(case)
nlbf = NLBranchFlow(case)
m1 = lindist.model
m2 = nlbf.model


m1.objective = pyo.Objective(
    rule=loss_objective_rule,
    sense=pyo.minimize,
)
m2.objective = pyo.Objective(
    rule=loss_objective_rule,
    sense=pyo.minimize,
)
# Solve the model
opt = pyo.SolverFactory("ipopt")
# results1 = opt.solve(m1)

# Extract and display results
# if results1.solver.status == pyo.SolverStatus.ok:
#     print("Optimization successful!")
#     print(f"Objective value: {pyo.value(m1.objective)}")
#     # data = get_all_results(model, case)
#     v = get_voltages(m1.v2)
#     v2 = get_values(m1.v2)
#     p_flow = get_values(m1.p_flow)
#     q_flow = get_values(m1.q_flow)
#     p_gen = get_values(m1.p_gen)
#     q_gen = get_values(m1.q_gen)
#     plot_voltages(v, t=12).show(renderer="browser")
#     # plot_gens(p_flow, q_flow).show(renderer="browser")
#     # plot_gens(p_gen, q_gen).show(renderer="browser")
#     # plot_polar(p_gen, q_gen).show(renderer="browser")

# else:
#     print("Linear Optimization failed!")
results2 = opt.solve(m2)
if results2.solver.status == pyo.SolverStatus.ok:
    print("Optimization successful!")
    print(f"Objective value: {pyo.value(m2.objective)}")
    # data = get_all_results(model, case)
    v = get_voltages(m2.v2)
    v2 = get_values(m2.v2)
    p_flow = get_values(m2.p_flow)
    q_flow = get_values(m2.q_flow)
    p_gen = get_values(m2.p_gen)
    q_gen = get_values(m2.q_gen)
    plot_voltages(v, t=12).show(renderer="browser")
    # plot_gens(p_flow, q_flow).show(renderer="browser")
    # plot_gens(p_gen, q_gen).show(renderer="browser")
    # plot_polar(p_gen, q_gen).show(renderer="browser")
else:
    print("Non-linear Optimization failed!")

