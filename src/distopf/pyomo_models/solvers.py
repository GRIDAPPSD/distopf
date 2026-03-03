from distopf.pyomo_models.protocol import LindistModelProtocol
from distopf.pyomo_models.results import PyoResult
import pyomo.environ as pyo


def solve(model: LindistModelProtocol, solver="ipopt", duals=True) -> PyoResult:
    # t0 = perf_counter()
    # Solve the model
    results = pyo.SolverFactory(solver).solve(model)
    if duals:
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # t1 = perf_counter()
    # Extract and display results
    if results.solver.status == pyo.SolverStatus.ok:
        print("Optimization successful!")
        obj_value = pyo.value(model.objective)
        print(f"Objective value: {obj_value}")
        res = PyoResult(model, objective_value=obj_value)

    else:
        raise ValueError(results.solver.status)
    return res
