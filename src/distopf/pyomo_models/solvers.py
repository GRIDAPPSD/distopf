from distopf.pyomo_models.protocol import LindistModelProtocol
from distopf.pyomo_models.results import OpfResult
import pyomo.environ as pyo
from time import perf_counter


def solve(model: LindistModelProtocol) -> OpfResult:
    # t0 = perf_counter()
    # Solve the model
    results = pyo.SolverFactory("ipopt").solve(model)
    # t1 = perf_counter()
    # Extract and display results
    if results.solver.status == pyo.SolverStatus.ok:
        print("Optimization successful!")
        print(f"Objective value: {pyo.value(model.objective)}")
        res = OpfResult(model)

    else:
        raise ValueError(results.solver.status)
    return res
