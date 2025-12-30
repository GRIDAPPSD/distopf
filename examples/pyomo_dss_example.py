from distopf.importer import create_case
from distopf.pyomo_models.lindist_loads import LinDistPyoMPL
from distopf import CASES_DIR
from distopf.pyomo_models.solvers import solve
from distopf.pyomo_models.objectives import loss_objective
import pyomo.environ as pyo



case = create_case(CASES_DIR / "dss" / "ieee123_dss" / "Run_IEEE123Bus.DSS")
m = LinDistPyoMPL(case)
m.model.objective = loss_objective
result = solve(m.model)
