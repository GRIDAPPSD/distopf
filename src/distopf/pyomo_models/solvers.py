from distopf.pyomo_models.protocol import LindistModelProtocol
from distopf.pyomo_models.results import PyoResult
import pyomo.environ as pyo


def solve(model: LindistModelProtocol, solver="ipopt", duals=True, verbose=False) -> PyoResult:
    if solver is None:
        solver = "ipopt"
    if duals:
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # Solve the model
    solver_factory = pyo.SolverFactory(solver)
    if solver == "gurobi":
        solver_factory.options["NonConvex"] = 2
        solver_factory.options["FuncNonlinear"] = 1
    results = solver_factory.solve(model, tee=verbose)

    if results.solver.status == pyo.SolverStatus.ok:
        obj_value = pyo.value(model.objective)
        if verbose:
            print("Optimization successful!")
            print(f"Objective value: {obj_value}")
        res = PyoResult(model, objective_value=obj_value, extract_duals=duals)

    else:
        raise ValueError(results.solver.status)
    return res
