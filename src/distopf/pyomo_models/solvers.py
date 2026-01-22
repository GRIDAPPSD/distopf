from distopf.pyomo_models.protocol import LindistModelProtocol
from distopf.pyomo_models.results import PyoResult
import pyomo.environ as pyo
from time import perf_counter


def solve(model: LindistModelProtocol) -> PyoResult:
    # t0 = perf_counter()
    # Solve the model
    results = pyo.SolverFactory("ipopt").solve(model)
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
