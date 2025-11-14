from time import perf_counter
from typing import Optional, Callable
import cvxpy as cp
import numpy as np
from scipy.optimize import OptimizeResult, linprog
from scipy.sparse import csr_array
from distopf.matrix_models.multiperiod import LinDistBaseMP


def cvxpy_solve(
    model: LinDistBaseMP,
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
    g = [csr_array(m.a_eq) @ x - m.b_eq.flatten() == 0]
    g_inequality = []
    if m.a_ub is not None and m.b_ub is not None:
        if m.a_ub.shape[0] != 0 and m.a_ub.shape[1] != 0:
            g_inequality = [m.a_ub @ x - m.b_ub <= 0]

    lb = [x >= m.x_min]
    ub = [x <= m.x_max]
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


def lp_solve(model: LinDistBaseMP, c: Optional[np.ndarray] = None) -> OptimizeResult:
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
    if isinstance(c, Callable):  # type: ignore
        c = c(model)  # type: ignore
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
    )  # type: ignore
    if not res.success:
        raise ValueError(res.message)
    runtime = perf_counter() - tic
    res["runtime"] = runtime
    return res


def pf(model) -> OptimizeResult:
    c = np.zeros(model.n_x)
    tic = perf_counter()
    res = linprog(c, A_eq=csr_array(model.a_eq), b_eq=model.b_eq.flatten())  # type: ignore
    if not res.success:
        raise ValueError(res.message)
    runtime = perf_counter() - tic
    res["runtime"] = runtime
    return res
