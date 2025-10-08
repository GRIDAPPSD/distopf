import distopf as opf
import pyomo.environ as pyo
from distopf.pyomo_models.pyomo_lindist import create_lindist_model
from distopf.importer import Case, create_case
from distopf.pyomo_models.lindist_constraints import (
    add_capacitor_constraints,
    add_circular_generator_constraints,
    add_cvr_load_constraints,
    add_generator_capability_constraints,
    add_octagonal_inverter_constraints,
    add_power_flow_constraints,
    add_swing_bus_constraints,
    add_thermal_limits,
    add_voltage_drop_constraints,
)
from distopf.pyomo_models.results import get_voltages, get_apparent_power_flows, get_p_gens, get_all_results
from distopf import plot_voltages, plot_power_flows, plot_ders, plot_gens, plot_network, plot_polar

case = opf.DistOPFCase(
    data_path=opf.CASES_DIR / "csv" / "ieee123_30der",
    objective_functions=opf.cp_obj_loss,
    control_variable="PQ",
    v_min=0.8,
    v_max=1.05
)
_case = Case(
    branch_data=case.branch_data,
    bus_data=case.bus_data,
    gen_data=case.gen_data,
    cap_data=case.cap_data,
    reg_data=case.reg_data,
)
model = create_lindist_model(_case)
add_power_flow_constraints(model, case.bus_data, case.branch_data, case.gen_data)
add_voltage_drop_constraints(model, case.branch_data)
add_swing_bus_constraints(model, case.bus_data)
add_octagonal_inverter_constraints(model, case.gen_data)
model.objective = pyo.Objective(
    expr=0,
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
    v = get_voltages(model, case)
    fig = plot_voltages(v)
    fig.show(renderer="browser")

else:
    print("Optimization failed!")
