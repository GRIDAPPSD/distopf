from distopf.api import create_case
from distopf.pyomo_models import create_lindist_model, add_constraints
from distopf import CASES_DIR
from distopf.pyomo_models.solvers import solve
from distopf.pyomo_models.objectives import loss_objective


case = create_case(CASES_DIR / "dss" / "ieee123_dss" / "Run_IEEE123Bus.DSS")
model = create_lindist_model(case)
add_constraints(model)
model.objective = loss_objective
result = solve(model)
