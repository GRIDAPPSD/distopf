import distopf as opf
import pyomo.environ as pyo
from distopf.pyomo_models.common.factory import create_lindist_model
from distopf.pyomo_models.common.results import PyoResult

# Load case data
case = opf.create_case(data_path=opf.CASES_DIR / "csv" / "ieee123_30der", n_steps=24)

# Example 1: Standard LP model (no MI)
model_lp = create_lindist_model(case, circular_constraints=False)
m = model_lp
m.obj = pyo.Objective(expr=0)  # Feasibility only
solver = pyo.SolverFactory("glpk")
results_lp = solver.solve(m)
result_lp = PyoResult(m, results_lp)
voltages = result_lp.voltages

# Example 2: Capacitor switching MILP
model_cap = create_lindist_model(
    case, control_capacitors=True, circular_constraints=False
)
m = model_cap
# Minimize total capacitor reactive power
m.obj = pyo.Objective(
    expr=sum(
        m.q_cap[device, ph, t]
        for device, ph in m.cap_device_phase_set
        for t in m.time_set
    ),
    sense=pyo.minimize,
)
solver = pyo.SolverFactory("cbc")
results_cap = solver.solve(m)
result_cap = PyoResult(m, results_cap)
cap_schedule = result_cap.q_cap  # or result_cap.u_cap for switching status

# Example 3: Regulator tap MILP
model_reg = create_lindist_model(
    case,
    control_regulators=True,
    reg_tap_change_limit=2,
    circular_constraints=False,
)
m = model_reg
m.obj = pyo.Objective(expr=0)
solver = pyo.SolverFactory("gurobi")
results_reg = solver.solve(m)
result_reg = PyoResult(m, results_reg)
reg_taps = result_reg.u_reg  # binary tap selection variables

# Example 4: Full MILP with both
model_full = create_lindist_model(
    case,
    control_capacitors=True,
    control_regulators=True,
    reg_tap_change_limit=3,
    circular_constraints=False,
)
print(f"Model has {len(list(model_full.component_objects(pyo.Var)))} variable types")
