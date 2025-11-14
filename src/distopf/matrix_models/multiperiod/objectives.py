import cvxpy as cp
import numpy as np
from distopf.matrix_models.multiperiod import LinDistBaseMP


# cost = pd.read_csv("cost_data.csv")
def gradient_load_min(model):
    c = np.zeros(model.n_x)
    for ph in "abc":
        if model.phase_exists(ph):
            c[model.branches_out_of_j("pij", 0, ph)] = 1
    return c


def gradient_curtail(model):
    c = np.zeros(model.n_x)
    for i in range(
        model.p_der_start_phase_idx["a"],
        model.p_der_start_phase_idx["c"] + len(model.gen_buses["c"]),
    ):
        c[i] = -1
    return c


def gradient_battery_efficiency(model: LinDistBaseMP, xk: cp.Variable, **kwargs):
    """
    Parameters
    ----------
    model : LinDistModel, or LinDistModelP, or LinDistModelQ
    xk : cp.Variable
    kwargs :

    Returns
    -------
    c: numpy array, gradient of objective function c'x

    """
    if "start_step" in model.__dict__.keys():
        start_step = model.start_step
    else:
        start_step = 0
    c = np.zeros(model.n_x)
    for t in range(start_step, start_step + model.n_steps):
        for a in "abc":
            if not model.phase_exists(a):
                continue
            charging_efficiency = model.bat.loc[
                model.charge_map[t][a].index, f"nc_{a}"
            ].to_numpy()
            discharging_efficiency = model.bat.loc[
                model.discharge_map[t][a].index, f"nd_{a}"
            ].to_numpy()
            c[model.charge_map[t][a].to_numpy()] = 1 - charging_efficiency  # type: ignore
            c[model.discharge_map[t][a].to_numpy()] = (1 / discharging_efficiency) - 1  # type: ignore
    return c


# ~~~ Quadratic objective with linear constraints for use with solve_quad()~~~


def cp_obj_loss(model: LinDistBaseMP, xk: cp.Variable, **kwargs) -> cp.Expression:
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
    if "start_step" in model.__dict__.keys():
        start_step = model.start_step
    else:
        start_step = 0
    index_list: list[int] = []
    r_list = np.array([])
    for t in range(start_step, start_step + model.n_steps):
        for a in "abc":
            if not model.phase_exists(a):
                continue
            i = model.x_maps[t][a].bi
            j = model.x_maps[t][a].bj
            r_list = np.append(r_list, np.array(model.r[a + a][i, j]).flatten())
            r_list = np.append(r_list, np.array(model.r[a + a][i, j]).flatten())
            index_list = np.append(
                index_list, model.x_maps[t][a].pij.to_numpy().flatten()
            )  # type: ignore
            index_list = np.append(
                index_list, model.x_maps[t][a].qij.to_numpy().flatten()
            )  # type: ignore
    r = np.array(r_list)
    ix = np.array(index_list).astype(int)
    if isinstance(xk, cp.Variable):
        return cp.vdot(r, xk[ix] ** 2)
    else:
        return np.vdot(r, xk[ix] ** 2)


def cp_battery_efficiency(
    model: LinDistBaseMP, xk: cp.Variable, **kwargs
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
    if "start_step" in model.__dict__.keys():
        start_step = model.start_step
    else:
        start_step = 0
    vec1_list:list[float] = []
    index_list:list[int] = []
    for t in range(start_step, start_step + model.n_steps):
        for a in "abc":
            if not model.phase_exists(a):
                continue
            charging_efficiency = model.bat.loc[
                model.charge_map[t][a].index, f"nc_{a}"
            ].to_numpy()
            discharging_efficiency = model.bat.loc[
                model.discharge_map[t][a].index, f"nd_{a}"
            ].to_numpy()
            vec1_list.extend((1 - charging_efficiency))  # type: ignore
            vec1_list.extend(((1 / discharging_efficiency) - 1))  # type: ignore
            index_list.extend(model.charge_map[t][a].to_numpy())
            index_list.extend(model.discharge_map[t][a].to_numpy())
    vec1 = np.array(vec1_list)
    ix = np.array(index_list)
    if isinstance(xk, cp.Variable):
        return 1e-3 * cp.vdot(vec1, xk[ix])
    else:
        return 1e-3 * np.vdot(vec1, xk[ix])


def cp_obj_loss_batt(model: LinDistBaseMP, xk: cp.Variable, **kwargs) -> cp.Expression:
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
    return cp_obj_loss(model, xk) + cp_battery_efficiency(model, xk)


def charge_batteries(model, xk, **kwargs) -> cp.Expression:
    f_list = []
    for t in range(model.n_steps):
        for a in "abc":
            if not model.phase_exists(a):
                continue
            f_list.append(-cp.sum(xk[model.soc_map[t][a].to_numpy()]))
    return cp.sum(f_list)


def cp_obj_target_p_3ph(model, xk, **kwargs):
    f = cp.Constant(0)
    target = kwargs["target"]
    loss_percent = kwargs.get("loss_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            p = 0
            for out_branch in model.branches_out_of_j("pij", 0, ph):
                p = p + xk[out_branch]
            f += (target[i] - p * (1 + loss_percent[i] / 100)) ** 2
    return f


def cp_obj_target_p_total(model, xk, **kwargs):
    actual = 0
    target = kwargs["target"]
    loss_percent = kwargs.get("loss_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            p = 0
            for out_branch in model.branches_out_of_j("pij", 0, ph):
                p = p + xk[out_branch]
            actual += p
    f = (target - actual * (1 + loss_percent[0] / 100)) ** 2
    return f


def cp_obj_target_q_3ph(model, xk, **kwargs):
    target_q = kwargs["target"]
    loss_percent = kwargs.get("loss_percent", np.zeros(3))
    f = cp.Constant(0)
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            q = 0
            for out_branch in model.branches_out_of_j("qij", 0, ph):
                q = q + xk[out_branch]
            f += (target_q[i] - q * (1 + loss_percent[i] / 100)) ** 2
    return f


def cp_obj_target_q_total(model, xk, **kwargs):
    actual = 0
    target = kwargs["target"]
    loss_percent = kwargs.get("loss_percent", np.zeros(3))
    for i, ph in enumerate("abc"):
        if model.phase_exists(ph):
            q = 0
            for out_branch in model.branches_out_of_j("qij", 0, ph):
                q = q + xk[out_branch]
            actual += q
    f = (target - actual * (1 + loss_percent[0] / 100)) ** 2
    return f


def cp_obj_curtail(model: LinDistBaseMP, xk: cp.Variable, **kwargs) -> cp.Expression:
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

    if "start_step" in model.__dict__.keys():
        start_step = model.start_step
    else:
        start_step = 0
    all_pg_idx = np.array([])
    for t in range(start_step, start_step + model.n_steps):
        for a in "abc":
            if not model.phase_exists(a):
                continue
            all_pg_idx = np.r_[all_pg_idx, model.pg_map[t][a].to_numpy()]
    all_pg_idx = all_pg_idx.astype(int)
    return cp.sum((model.x_max[all_pg_idx] - xk[all_pg_idx]) ** 2)


def cp_obj_curtail_lp(model: LinDistBaseMP, xk: cp.Variable, **kwargs) -> cp.Expression:
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

    if "start_step" in model.__dict__.keys():
        start_step = model.start_step
    else:
        start_step = 0
    all_pg_idx = np.array([])
    for t in range(start_step, start_step + model.n_steps):
        for a in "abc":
            if not model.phase_exists(a):
                continue
            all_pg_idx = np.r_[all_pg_idx, model.pg_map[t][a].to_numpy()]
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
