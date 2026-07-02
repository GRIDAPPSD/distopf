from distopf.pyomo_models.protocol import LindistModelProtocol
from distopf.pyomo_models.results import PyoResult
from pyomo.common.tee import capture_output
import pyomo.environ as pyo
import time
from io import StringIO


def solve(
    model: LindistModelProtocol, solver="ipopt", duals=True, verbose=False
) -> PyoResult:
    if solver is None:
        solver = "ipopt"
    if duals:
        model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # Solve the model
    solver_factory = pyo.SolverFactory(solver)
    if solver == "gurobi":
        solver_factory.options["NonConvex"] = 2
        solver_factory.options["FuncNonlinear"] = 1

    buf = StringIO()
    t0 = time.perf_counter()
    with capture_output(buf):
        results = solver_factory.solve(model, tee=True)
    solve_time = time.perf_counter() - t0
    log = buf.getvalue()

    if results.solver.status != pyo.SolverStatus.ok:
        raise ValueError(results.solver.status)

    obj_value = pyo.value(model.objective)
    if verbose:
        print("Optimization successful!")
        print(f"Objective value: {obj_value}")
    res = PyoResult(
        model,
        results,
        solve_time=solve_time,
        log=log,
        objective_value=obj_value,
        extract_duals=duals,
    )
    return res
