from collections.abc import Collection
import pyomo.environ as pe
import cvxpy as cp
import numpy as np
from distopf import LinDistModel
from distopf.matrix_models.base import LinDistBase


def gradient_load_min(model: LinDistModel, *args, **kwargs) -> np.ndarray:
    """
    Gradient of the objective function to minimize the load at the substation.
    c has a 1 for each active power flow out of the substation.
    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ

    Returns
    -------
    c :  1-D array
        The coefficients of the linear objective function to be minimized.
    """
    c = np.zeros(model.n_x)
    for ph in "abc":
        if model.phase_exists(ph):
            c[model.idx("pij", model.swing_bus, ph)] = 1
    return c


def gradient_curtail(model: LinDistModel, *args, **kwargs) -> np.ndarray:
    """
    Gradient of the objective function to minimize curtailment of DERs.
    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ

    Returns
    -------
    c :  1-D array
        The coefficients of the linear objective function to be minimized.

    """

    all_pg_idx = np.array([])
    for a in "abc":
        if not model.phase_exists(a):
            continue
        all_pg_idx = np.r_[all_pg_idx, model.pg_map[a].to_numpy()]
    all_pg_idx = all_pg_idx.astype(int)
    c = np.zeros(model.n_x)
    c[all_pg_idx] = -1
    return c


# ~~~ CVXPY Objectives ~~~
def cp_obj_loss(model: LinDistModel, xk: cp.Variable, **kwargs) -> cp.Expression:
    """

    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable
    kwargs :

    Returns
    -------
    f: cp.Expression
        Expression to be minimized

    """
    index_list = []
    r_list = np.array([])
    for a in "abc":
        if not model.phase_exists(a):
            continue
        i = model.x_maps[a].bi
        j = model.x_maps[a].bj
        r_list = np.append(r_list, np.array(model.r[a + a][i, j]).flatten())
        r_list = np.append(r_list, np.array(model.r[a + a][i, j]).flatten())
        index_list = np.append(index_list, model.x_maps[a].pij.to_numpy().flatten())
        index_list = np.append(index_list, model.x_maps[a].qij.to_numpy().flatten())
    r = np.array(r_list)
    ix = np.array(index_list).astype(int)
    if isinstance(xk, cp.Variable):
        return cp.vdot(r, xk[ix] ** 2)
    else:
        return np.vdot(r, xk[ix] ** 2)


def cp_obj_loss_old(model: LinDistModel, xk: cp.Variable, **kwargs) -> cp.Expression:
    """

    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable
    kwargs :

    Returns
    -------
    f: cp.Expression
        Expression to be minimized

    """
    f_list = []
    for j in range(1, model.nb):
        for a in "abc":
            if model.phase_exists(a, j):
                i = model.idx("bi", j, a)[0]
                f_list.append(
                    model.r[a + a][i, j] * (xk[model.idx("pij", j, a)[0]] ** 2)
                )
                f_list.append(
                    model.r[a + a][i, j] * (xk[model.idx("qij", j, a)[0]] ** 2)
                )
    return cp.sum(f_list)


def cp_obj_target_p_3ph(
    model: LinDistModel, xk: cp.Variable, **kwargs
) -> cp.Expression:
    """

    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable
    kwargs :

    Returns
    -------
    f: cp.Expression
        Expression to be minimized

    """
    f = cp.Constant(0)
    target = kwargs["target"]
    if not isinstance(target, Collection):
        raise TypeError(f"target must be a size 3 array. Instead got {target}.")
    if len(target) != 3:
        raise TypeError(f"target must be a size 3 array. Instead got {target}.")

    error_percent = kwargs.get("error_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            p = 0
            for out_branch in model.branches_out_of_j("pij", 0, ph):
                p = p + xk[out_branch]
            f += (target[i] - p * (1 + error_percent[i] / 100)) ** 2
    return f


def cp_obj_target_p_total(
    model: LinDistModel | LinDistModel, xk: cp.Variable, **kwargs
) -> cp.Expression:
    """

    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable
    kwargs :

    Returns
    -------
    f: cp.Expression
        Expression to be minimized

    """
    actual = 0
    target = kwargs["target"]
    if not isinstance(target, (int, float)):
        raise TypeError(
            f"target must be a float or integer value. Instead got {target}."
        )
    error_percent = kwargs.get("error_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            p = 0
            for out_branch in model.branches_out_of_j("pij", 0, ph):
                p = p + xk[out_branch]
            actual += p
    f = (target - actual * (1 + error_percent[0] / 100)) ** 2
    return f


def cp_obj_target_q_3ph(
    model: LinDistModel, xk: cp.Variable, **kwargs
) -> cp.Expression:
    """

    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable
    kwargs :

    Returns
    -------
    f: cp.Expression
        Expression to be minimized

    """
    target = kwargs["target"]
    if not isinstance(target, Collection):
        raise TypeError(f"target must be a size 3 array. Instead got {target}.")
    if len(target) != 3:
        raise TypeError(f"target must be a size 3 array. Instead got {target}.")
    error_percent = kwargs.get("error_percent", np.zeros(3))
    f = cp.Constant(0)
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            q = 0
            for out_branch in model.branches_out_of_j("qij", 0, ph):
                q = q + xk[out_branch]
            f += (target[i] - q * (1 + error_percent[i] / 100)) ** 2
    return f


def cp_obj_target_q_total(
    model: LinDistModel, xk: cp.Variable, **kwargs
) -> cp.Expression:
    """
    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable
    kwargs :

    Returns
    -------
    f: cp.Expression
        Expression to be minimized

    """
    actual = 0
    target = kwargs["target"]
    if not isinstance(target, (int, float)):
        raise TypeError(
            f"target must be a float or integer value. Instead got {target}."
        )
    error_percent = kwargs.get("error_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            q = 0
            for out_branch in model.branches_out_of_j("qij", 0, ph):
                q = q + xk[out_branch]
            actual += q
    f = (target - actual * (1 + error_percent[0] / 100)) ** 2
    return f


# def cp_obj_curtail(model: LinDistModel, xk: cp.Variable, **kwargs) -> cp.Expression:
#     """
#     Objective function to minimize curtailment of DERs.
#     Min sum((P_der_max - P_der)^2)
#     Parameters
#     ----------
#     model : LinDistModel, or LinDistModelP, or LinDistModelQ
#     xk : cp.Variable
#
#     Returns
#     -------
#     f: cp.Expression
#         Expression to be minimized
#     """
#     f = cp.Constant(0)
#     for i in range(model.ctr_var_start_idx, model.n_x):
#         f += (model.bounds[i][1] - xk[i]) ** 2
#     return f


def cp_obj_curtail(model: LinDistModel, xk: cp.Variable, **kwargs) -> cp.Expression:
    """
    Objective function to minimize curtailment of DERs.
    Min sum((P_der_max - P_der)^2)
    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable

    Returns
    -------
    f: cp.Expression
        Expression to be minimized
    """

    all_pg_idx = np.array([])
    for a in "abc":
        if not model.phase_exists(a):
            continue
        all_pg_idx = np.r_[all_pg_idx, model.pg_map[a].to_numpy()]
    all_pg_idx = all_pg_idx.astype(int)
    return cp.sum((model.x_max[all_pg_idx] - xk[all_pg_idx]) ** 2)


def cp_obj_curtail_lp(model: LinDistModel, xk: cp.Variable, **kwargs) -> cp.Expression:
    """
    Objective function to minimize curtailment of DERs.
    Min sum((P_der_max - P_der)^2)
    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable

    Returns
    -------
    f: cp.Expression
        Expression to be minimized
    """

    all_pg_idx = np.array([])
    for a in "abc":
        if not model.phase_exists(a):
            continue
        all_pg_idx = np.r_[all_pg_idx, model.pg_map[a].to_numpy()]
    all_pg_idx = all_pg_idx.astype(int)
    return cp.sum((model.x_max[all_pg_idx] - xk[all_pg_idx]))


def cp_obj_none(*args, **kwargs) -> cp.Constant:
    """
    For use with cvxpy_solve() to run a power flow with no optimization.

    Returns
    -------
    constant 0
    """
    return cp.Constant(0)


# ~~~ PYOMO Objectives ~~~


def pyo_obj_loss(model: LinDistBase, x: pe.Var, **kwargs) -> float:
    """

    Parameters
    ----------
    model : LinDistModel
    x : pe.Var
    kwargs :

    Returns
    -------
    cost : float

    """
    index_list = []
    r_list = np.array([])
    for a in "abc":
        if not model.phase_exists(a):
            continue
        i = model.x_maps[a].bi
        j = model.x_maps[a].bj
        r_list = np.append(r_list, np.array(model.r[a + a][i, j]).flatten())
        r_list = np.append(r_list, np.array(model.r[a + a][i, j]).flatten())
        index_list = np.append(index_list, model.x_maps[a].pij.to_numpy().flatten())
        index_list = np.append(index_list, model.x_maps[a].qij.to_numpy().flatten())
    r = np.array(r_list)
    ix = np.array(index_list).astype(int)
    terms = []
    for i in range(len(ix)):
        terms.append(r * x[ix[i] ** 2])
    return sum(terms)


def pyo_obj_curtail(model: LinDistBase, x: pe.Var, **kwargs) -> float:
    """
    Objective function to minimize curtailment of DERs.
    Min sum((P_der_max - P_der)^2)
    Parameters
    ----------
    model : LinDistBase
    x : pe.Var

    Returns
    -------
    cost : float
    """

    all_pg_idx = np.array([])
    for a in "abc":
        if not model.phase_exists(a):
            continue
        all_pg_idx = np.r_[all_pg_idx, model.pg_map[a].to_numpy()]
    all_pg_idx = all_pg_idx.astype(int)
    terms = []
    for i in range(len(all_pg_idx)):
        terms.append((model.x_max[all_pg_idx[i]] - x[all_pg_idx[i]]) ** 2)
    return sum(terms)
