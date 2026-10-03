from distopf.api import create_case
from distopf.pyomo_models.common.factory import create_lindist_model
from distopf import CASES_DIR
from distopf.pyomo_models.common.solvers import solve
from distopf.pyomo_models.common.objectives import add_loss_objective


case = create_case(CASES_DIR / "dss" / "ieee123_dss" / "Run_IEEE123Bus.DSS")
model = create_lindist_model(case)
add_loss_objective(model)
result = solve(model)
