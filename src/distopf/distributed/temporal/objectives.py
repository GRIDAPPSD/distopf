"""Objective functions for temporal decomposition algorithms.

These objectives augment distopf's base objectives with temporal decomposition terms
(duals from boundary conditions between time periods).
"""

import pandas as pd
import cvxpy as cp
import numpy as np


def energy_cost_min(model, xk, **kwargs):
    """Energy cost minimization objective.

    Minimizes cost of energy drawn from substation across all time periods.
    """
    cost_curve = kwargs.get("cost_curve")
    if cost_curve is None:
        raise ValueError("cost_curve required in kwargs")

    delta_t = model.delta_t if hasattr(model, "delta_t") else 1.0
    period_cost = cost_curve * delta_t

    edges = []
    costs = []
    for t in range(model.start_step, model.start_step + model.n_steps):
        for a in "abc":
            if not model.phase_exists(a):
                continue
            # Get power flow indices from swing bus (substation)
            # Swing bus is index 0 (from bus at substation)
            x_map = model.x_maps[t][a]
            pij_idx = x_map.pij.to_numpy().flatten()
            bi_idx = x_map.bi.to_numpy().flatten()
            # Find swing bus (bi == 0) connections
            swing_mask = bi_idx == 0
            swing_edges = pij_idx[swing_mask]
            if len(swing_edges) > 0:
                edges.extend(swing_edges)
                costs.extend([period_cost[t]] * len(swing_edges))

    edges = np.array(edges, dtype=int)
    costs = np.array(costs)

    if len(edges) == 0:
        return 0.0

    if isinstance(xk, cp.Variable):
        return cp.vdot(costs, xk[edges])
    else:
        return np.vdot(costs, xk[edges])


def cp_battery_efficiency(model, xk: cp.Variable, **kwargs) -> cp.Expression:
    """Battery charging/discharging efficiency penalty.

    Adds penalty for efficiency losses: (1 - charge_eff) * p_charge + (1/discharge_eff - 1) * p_discharge
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
            # Get charge/discharge efficiencies
            charge_eff = model.bat.loc[
                model.charge_map[t][a].index, "nc_" + a
            ].to_numpy()
            discharge_eff = model.bat.loc[
                model.discharge_map[t][a].index, "nd_" + a
            ].to_numpy()

            # Penalties: (1 - eff) for charging, (1/eff - 1) for discharging
            c[model.charge_map[t][a].to_numpy()] = 1 - charge_eff
            c[model.discharge_map[t][a].to_numpy()] = (1 / discharge_eff) - 1

    c = 1e-3 * c  # Scale factor
    if isinstance(xk, cp.Variable):
        return cp.vdot(c, xk)
    else:
        return np.vdot(c, xk)


# ============================================================================
# Temporal Decomposition Augmentation Terms
# ============================================================================


def tenapp_aprx_augmentation(model, xk, approx_dual=None, **kwargs):
    """Augmentation term for approximate dual-based temporal coordination (TENAPP-APRX).

    Adds penalty on future SOC values based on approximate duals from previous iteration.
    """
    if not hasattr(model, "soc_map") or len(model.bat) == 0:
        return 0.0

    # Collect future SOC indices
    idxs = []
    for t in model.soc_map.keys():
        for a in "abc":
            if model.phase_exists(a) and a in model.soc_map[t]:
                idxs.extend(model.soc_map[t][a].to_numpy().flatten())

    idxs = np.array(idxs, dtype=int)
    if len(idxs) == 0:
        return 0.0

    if approx_dual is None:
        approx_dual = np.zeros_like(idxs, dtype=float)
    elif isinstance(approx_dual, (float, int)):
        # Handle scalar dual: broadcast to vector of appropriate length
        approx_dual = np.full_like(idxs, approx_dual, dtype=float)
    else:
        # Ensure dual vector matches index length
        approx_dual = np.asarray(approx_dual, dtype=float)
        if len(approx_dual) != len(idxs):
            approx_dual = np.zeros_like(idxs, dtype=float)

    if isinstance(xk, cp.Variable):
        return -1.0 * cp.vdot(approx_dual, xk[idxs])
    else:
        return -1.0 * np.vdot(approx_dual, xk[idxs])


def tenapp_admm_augmentation(model, xk, soc0=None, soc_end=None, **kwargs):
    """Augmentation term for ADMM-based temporal coordination (TENAPP-ADMM).

    Adds quadratic penalties on initial/final SOC values to coordinate across time periods.
    """
    if not hasattr(model, "soc_map") or len(model.bat) == 0:
        return 0.0

    weight = kwargs.get("weight", 1e2)
    cost = 0.0

    # Initial SOC penalty (if using separate soc0_map)
    if hasattr(model, "soc0_map") and soc0 is not None:
        idxs0 = []
        for t in model.soc0_map.keys():
            for a in "abc":
                if model.phase_exists(a) and a in model.soc0_map[t]:
                    idxs0.extend(model.soc0_map[t][a].to_numpy().flatten())
        idxs0 = np.array(idxs0, dtype=int)

        if len(idxs0) > 0:
            soc0_vals = np.atleast_1d(np.asarray(soc0, dtype=float))
            if len(soc0_vals) == len(idxs0):
                if isinstance(xk, cp.Variable):
                    cost += weight * cp.sum_squares(xk[idxs0] - soc0_vals)
                else:
                    cost += weight * np.sum((xk[idxs0] - soc0_vals) ** 2)

    # Final SOC penalty
    if soc_end is not None:
        idxs_end = []
        # Get final time step
        final_t = (
            max(model.soc_map.keys())
            if model.soc_map
            else model.start_step + model.n_steps - 1
        )
        for a in "abc":
            if model.phase_exists(a) and a in model.soc_map.get(final_t, {}):
                idxs_end.extend(model.soc_map[final_t][a].to_numpy().flatten())

        idxs_end = np.array(idxs_end, dtype=int)
        if len(idxs_end) > 0:
            soc_end_vals = np.atleast_1d(np.asarray(soc_end, dtype=float))
            if len(soc_end_vals) == len(idxs_end):
                if isinstance(xk, cp.Variable):
                    cost += weight * cp.sum_squares(xk[idxs_end] - soc_end_vals)
                else:
                    cost += weight * np.sum((xk[idxs_end] - soc_end_vals) ** 2)

    return cost


def tenapp_1o_augmentation(
    model,
    xk: cp.Variable,
    future_duals: pd.DataFrame | None = None,
    **kwargs,
) -> cp.Expression | float:
    """Augmentation term for first-order temporal coordination (TENAPP-1O).

    Adds penalty on SOC using dual variables from future time periods.
    """
    if future_duals is None or len(future_duals) == 0:
        return 0.0

    if not hasattr(model, "soc_map") or len(model.bat) == 0:
        return 0.0

    ix = []
    duals = []

    for bat_idx in model.bat.index:
        for a in "abc":
            if not model.phase_exists(a, bat_idx):
                continue

            # Look for SOC in current time period
            t = model.start_step
            if t in model.soc_map and a in model.soc_map[t]:
                soc_series = model.soc_map[t][a]
                if bat_idx in soc_series.index:
                    soc_idx = soc_series[bat_idx]

                    # Find dual for this battery at this time
                    dual_rows = future_duals.loc[
                        (future_duals.id == bat_idx + 1) & (future_duals.t == t)
                    ]
                    if not dual_rows.empty:
                        dual_val = dual_rows["dual"].iloc[0]
                        ix.append(soc_idx)
                        duals.append(dual_val)

    if len(ix) == 0:
        return 0.0

    duals = np.array(duals, dtype=float)
    ix = np.array(ix, dtype=int)

    if isinstance(xk, cp.Variable):
        return cp.vdot(duals, xk[ix])
    else:
        return np.vdot(duals, xk[ix])


# ============================================================================
# Combined Objectives for Temporal Algorithms
# ============================================================================


def energy_cost_min_with_augmentation(model, xk, augmentation_fn, **kwargs):
    """Combine energy cost with temporal augmentation term."""
    base_cost = energy_cost_min(model, xk, **kwargs)
    augmentation = augmentation_fn(model, xk, **kwargs)
    return base_cost + augmentation


def energy_cost_min_with_efficiency(model, xk, **kwargs):
    """Energy cost + battery efficiency penalty."""
    return energy_cost_min(model, xk, **kwargs) + cp_battery_efficiency(
        model, xk, **kwargs
    )
