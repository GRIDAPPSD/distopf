import distopf as opf
import pyomo.environ as pyo
from distopf.pyomo_models import LinDistModel
from distopf.pyomo_models.results import PyoResult

# Load case data
case = opf.create_case(data_path=opf.CASES_DIR / "csv" / "ieee123_30der", n_steps=24)

# Example 1: Standard LP model (no MI)
model_lp = LinDistModel(case, circular_constraints=False)
m = model_lp.model
m.obj = pyo.Objective(expr=0)  # Feasibility only
solver = pyo.SolverFactory("glpk")
solver.solve(m)
result_lp = PyoResult(m)
voltages = result_lp.voltages

# Example 2: Capacitor switching MILP
model_cap = LinDistModel(case, cap_mi=True, circular_constraints=False)
m = model_cap.model
# Minimize total capacitor reactive power
m.obj = pyo.Objective(
    expr=sum(m.q_cap[_id, ph, t] for (_id, ph) in m.cap_phase_set for t in m.time_set),
    sense=pyo.minimize,
)
solver = pyo.SolverFactory("cbc")
solver.solve(m)
result_cap = PyoResult(m)
cap_schedule = result_cap.q_cap  # or result_cap.u_cap for switching status

# Example 3: Regulator tap MILP
model_reg = LinDistModel(
    case, reg_mi=True, reg_tap_change_limit=2, circular_constraints=False
)
m = model_reg.model
m.obj = pyo.Objective(expr=0)
solver = pyo.SolverFactory("gurobi")
solver.solve(m)
result_reg = PyoResult(m)
reg_taps = result_reg.u_reg  # binary tap selection variables

# Example 4: Full MILP with both
model_full = LinDistModel(
    case, cap_mi=True, reg_mi=True, reg_tap_change_limit=3, circular_constraints=False
)
print(
    f"Model has {len(list(model_full.model.component_objects(pyo.Var)))} variable types"
)
