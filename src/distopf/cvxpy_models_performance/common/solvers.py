"""Solve a vectorized CVXPY model."""

from __future__ import annotations

import time

import cvxpy as cp

from distopf.cvxpy_models_performance.common.model import CvxModel
from distopf.cvxpy_models_performance.common.results import CvxResult

_OK_STATUSES = (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


def build_problem(model: CvxModel) -> cp.Problem:
    """Assemble the ``cp.Problem`` from the model's objective and constraints."""
    expression = model.objective_expression
    if expression is None:
        raise ValueError("No objective set. Call set_objective(model, ...) first.")
    sense = getattr(model, "objective_sense", "minimize")
    objective = (
        cp.Minimize(expression) if sense == "minimize" else cp.Maximize(expression)
    )
    model.problem = cp.Problem(objective, model.all_constraints())
    return model.problem


def _has_integers(model: CvxModel) -> bool:
    return any(spec.boolean for spec in model.var_specs.values())


def _default_solver(model: CvxModel, problem: cp.Problem) -> str:
    if not _has_integers(model):
        return "SCIP"
    return "HIGHS" if (problem.is_qp()) else "CLARABEL"


def _cyipopt_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("cyipopt") is not None


def _is_dnlp(problem: cp.Problem) -> bool:
    try:
        return bool(problem.is_dnlp())
    except AttributeError:
        return False


def _solve_nlp(model: CvxModel, problem: cp.Problem, verbose: bool) -> float:
    """Solve a non-convex (DNLP) problem with IPOPT through CVXPY >= 1.9."""
    if not hasattr(cp, "IPOPT") or not _is_dnlp(problem):
        raise NotImplementedError(
            "The model is not DCP and CVXPY cannot solve it as a nonlinear "
            "program. Native NLP needs cvxpy >= 1.9 and a model that satisfies "
            "the DNLP rules. Alternatively build with socp_relaxation=True, or "
            "use the Pyomo backend with IPOPT."
        )
    if not _cyipopt_available():
        raise NotImplementedError(
            "Solving the exact BranchFlow model needs IPOPT through CVXPY. "
            "Install it with `pip install cyipopt` (requires the IPOPT library; "
            "see https://www.cvxpy.org/install/) or build with "
            "socp_relaxation=True."
        )
    if _has_integers(model):
        raise NotImplementedError(
            "Discrete controls with the exact BranchFlow model are not supported "
            "by the CVXPY backend. Use the Pyomo backend with a MINLP solver."
        )
    problem.solve(nlp=True, solver=cp.IPOPT, verbose=verbose)
    return float(problem.value)


def solve(
    model: CvxModel,
    solver: str | None = None,
    duals: bool = True,
    verbose: bool = False,
) -> CvxResult:
    """Solve the model with CVXPY (same semantics as ``cvxpy_models.solve``)."""
    problem = build_problem(model)
    nlp = not problem.is_dcp()

    t0 = time.perf_counter()
    if nlp:
        _solve_nlp(model, problem, verbose)
    else:
        if solver is None or str(solver).lower() == "ipopt":
            solver = _default_solver(model, problem)
        problem.solve(solver=str(solver).upper(), verbose=verbose)
    solve_time = time.perf_counter() - t0

    if problem.status not in _OK_STATUSES:
        raise ValueError(problem.status)

    obj_value = float(problem.value)
    if verbose:
        print("Optimization successful!")
        print(f"Objective value: {obj_value}")
    return CvxResult(
        model,
        problem,
        solve_time=solve_time,
        log="",
        objective_value=obj_value,
        extract_duals=duals and not nlp and not _has_integers(model),
    )
