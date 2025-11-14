from collections.abc import Callable
from time import perf_counter

import pyomo.environ as pe
import cvxpy as cp
import numpy as np
from scipy.optimize import OptimizeResult, linprog
from scipy.sparse import csr_array
from distopf import LinDistModelCapMI
from distopf.matrix_models.base import LinDistBase


def cvxpy_solve(
    model: LinDistBase,
    obj_func: Callable,
    **kwargs,
) -> OptimizeResult:
    """
    Solve a convex optimization problem using cvxpy.
    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    obj_func : handle to the objective function
    kwargs :

    Returns
    -------
    result: scipy.optimize.OptimizeResult

    """
    m = model
    tic = perf_counter()
    solver = kwargs.get("solver", cp.CLARABEL)
    x0 = kwargs.get("x0", None)
    if x0 is None:
        lin_res = lp_solve(m, np.zeros(m.n_x))
        if not lin_res.success:
            raise ValueError(lin_res.message)
        x0 = lin_res.x.copy()
    x = cp.Variable(shape=(m.n_x,), name="x", value=x0)
    g = [m.a_eq @ x - m.b_eq.flatten() == 0]
    # lb = [x[i] >= m.bounds[i][0] for i in range(m.n_x)]
    # ub = [x[i] <= m.bounds[i][1] for i in range(m.n_x)]
    lb = [x >= m.x_min]
    ub = [x <= m.x_max]
    g_inequality = []
    if m.a_ub is not None and m.b_ub is not None:
        if m.a_ub.shape[0] != 0 and m.a_ub.shape[1] != 0:
            g_inequality = [m.a_ub @ x - m.b_ub <= 0]
    expression = obj_func(m, x, **kwargs)
    prob = cp.Problem(cp.Minimize(expression), g + g_inequality + ub + lb)
    prob.solve(verbose=False, solver=solver)

    x_res = x.value
    result = OptimizeResult(
        fun=prob.value,
        success=(prob.status == "optimal"),
        message=prob.status,
        x=x_res,
        nit=prob.solver_stats.num_iters,
        runtime=perf_counter() - tic,
    )
    return result


def cvxpy_mi_solve(
    model: LinDistModelCapMI,
    obj_func: Callable,
    **kwargs,
) -> OptimizeResult:
    """
    Solve a convex optimization problem using cvxpy.
    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    obj_func : handle to the objective function
    kwargs :

    Returns
    -------
    result: scipy.optimize.OptimizeResult

    """
    m = model
    tic = perf_counter()
    solver = kwargs.get("solver")
    x0 = kwargs.get("x0", None)
    if x0 is None:
        lin_res = lp_solve(m, np.zeros(m.n_x))
        if not lin_res.success:
            raise ValueError(lin_res.message)
        x0 = lin_res.x.copy()
    x = cp.Variable(shape=(m.n_x,), name="x", value=x0)
    n_u = len(m.cap_buses["a"]) + len(m.cap_buses["b"]) + len(m.cap_buses["c"])
    u_c = cp.Variable(shape=(n_u,), name="u_c", value=np.ones(n_u), boolean=True)
    u_idxs = np.r_[m.uc_map["a"], m.uc_map["b"], m.uc_map["c"]]
    gu = [x[u_idxs] == u_c]
    g_ineq = [csr_array(m.a_ineq) @ x - m.b_ineq.flatten() <= 0]
    g = [csr_array(m.a_eq) @ x - m.b_eq.flatten() == 0]
    lb = [x[i] >= m.bounds[i][0] for i in range(m.n_x)]
    ub = [x[i] <= m.bounds[i][1] for i in range(m.n_x)]

    error_percent = kwargs.get("error_percent", np.zeros(3))
    target = kwargs.get("target", None)
    expression = obj_func(m, x, target=target, error_percent=error_percent)
    prob = cp.Problem(cp.Minimize(expression), g + ub + lb + gu + g_ineq)
    prob.solve(verbose=False, solver=solver)

    x_res = x.value
    result = OptimizeResult(
        fun=prob.value,
        success=(prob.status == "optimal"),
        message=prob.status,
        x=x_res,
        nit=prob.solver_stats.num_iters,
        runtime=perf_counter() - tic,
    )
    return result


def pf(model: LinDistBase) -> OptimizeResult:
    c = np.zeros(model.n_x)
    tic = perf_counter()
    res = linprog(c, A_eq=csr_array(model.a_eq), b_eq=model.b_eq.flatten())
    if not res.success:
        raise ValueError(res.message)
    runtime = perf_counter() - tic
    res["runtime"] = runtime
    return res


def lp_solve(
    model: LinDistBase,
    c: np.ndarray | Callable = None,
    **kwargs,
) -> OptimizeResult:
    """
    Solve a linear program using scipy.optimize.linprog and having the objective function:
        Min c^T x
    Parameters
    ----------
    model : LinDistModel
    c :  1-D array
        The coefficients of the linear objective function to be minimized.
    Returns
    -------
    result : OptimizeResult
        A :class:`scipy.optimize.OptimizeResult` consisting of the fields
        below. Note that the return types of the fields may depend on whether
        the optimization was successful, therefore it is recommended to check
        `OptimizeResult.status` before relying on the other fields:

        x : 1-D array
            The values of the decision variables that minimizes the
            objective function while satisfying the constraints.
        fun : float
            The optimal value of the objective function ``c @ x``.
        slack : 1-D array
            The (nominally positive) values of the slack variables,
            ``b_ub - A_ub @ x``.
        con : 1-D array
            The (nominally zero) residuals of the equality constraints,
            ``b_eq - A_eq @ x``.
        success : bool
            ``True`` when the algorithm succeeds in finding an optimal
            solution.
        status : int
            An integer representing the exit status of the algorithm.

            ``0`` : Optimization terminated successfully.

            ``1`` : Iteration limit reached.

            ``2`` : Problem appears to be infeasible.

            ``3`` : Problem appears to be unbounded.

            ``4`` : Numerical difficulties encountered.

        nit : int
            The total number of iterations performed in all phases.
        message : str
            A string descriptor of the exit status of the algorithm.
    """
    if isinstance(c, Callable):
        c = c(model)
    if c is None:
        c = np.zeros(model.n_x)
    tic = perf_counter()
    res = linprog(
        c,
        A_eq=csr_array(model.a_eq),
        b_eq=model.b_eq.flatten(),
        A_ub=model.a_ub,
        b_ub=model.b_ub,
        bounds=model.bounds,
    )
    if not res.success:
        raise ValueError(res.message)
    runtime = perf_counter() - tic
    res["runtime"] = runtime
    return res


def pyomo_solve(
    model: LinDistBase,
    obj_func: Callable,
    **kwargs,
) -> OptimizeResult:
    m = model
    tic = perf_counter()
    solver = kwargs.get("solver", "ipopt")
    x0 = kwargs.get("x0", None)
    if x0 is None:
        lin_res = lp_solve(m, np.zeros(m.n_x))
        if not lin_res.success:
            raise ValueError(lin_res.message)
        x0 = lin_res.x.copy()

    cm = pe.ConcreteModel()
    cm.n_xk = pe.RangeSet(0, model.n_x - 1)
    cm.xk = pe.Var(cm.n_xk, initialize=x0)
    cm.constraints = pe.ConstraintList()
    for i in range(model.n_x):
        cm.constraints.add(cm.xk[i] <= model.x_max[i])
        cm.constraints.add(cm.xk[i] >= model.x_min[i])

    def equality_rule(_cm, i):
        if model.a_eq[[i], :].nnz > 0:
            return model.b_eq[i] == sum(
                _cm.xk[j] * model.a_eq[i, j]
                for j in range(model.n_x)
                if model.a_eq[i, j]
            )
        return pe.Constraint.Skip

    def inequality_rule(_cm, i):
        if model.a_ub[[i], :].nnz > 0:
            return model.b_ub[i] >= sum(
                _cm.xk[j] * model.a_ub[i, j]
                for j in range(model.n_x)
                if model.a_ub[i, j]
            )
        return pe.Constraint.Skip

    cm.equality = pe.Constraint(cm.n_xk, rule=equality_rule)
    if model.a_ub.shape[0] != 0:
        cm.ineq_set = pe.RangeSet(0, model.a_ub.shape[0] - 1)
        cm.inequality = pe.Constraint(cm.ineq_set, rule=inequality_rule)
    cm.objective = pe.Objective(expr=obj_func(model, cm.xk, **kwargs))
    opt = pe.SolverFactory(solver)
    opt.solve(cm)

    x_dict = cm.xk.extract_values()
    x_res = np.zeros(len(x_dict))
    for key, value in x_dict.items():
        x_res[key] = value

    result = OptimizeResult(
        fun=float(pe.value(cm.objective)),
        # success=(prob.status == "optimal"),
        # message=prob.status,
        x=x_res,
        # nit=prob.solver_stats.num_iters,
        runtime=perf_counter() - tic,
    )
    return result
