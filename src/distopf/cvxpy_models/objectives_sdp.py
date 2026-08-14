"""
Objective functions for the SDP BranchFlow model.

Each function takes an SdpModel and returns a cvxpy Expression
suitable for use as cp.Minimize(expr) or cp.Maximize(expr).

Mirrors distopf/pyomo_models/objectives.py in structure and naming.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from distopf.cvxpy_models.branchflow_sdp import SdpModel, _get_Z_sub


def loss_objective_sdp(m: SdpModel) -> cp.Expression:
    """
    Minimize total active power losses across all branches and time steps.

    Loss on branch (fb,tb) = trace(Zr · L_re - Zi · L_im)

    This equals the sum of per-phase I²R losses and captures mutual
    coupling between phases via the off-diagonal impedance terms.

    Equivalent to ``loss_objective_rule`` in pyomo objectives.py.
    """
    loss = 0.0
    for fb, tb in m.branch_set:
        n = len(m.branch_phase_map[fb, tb])
        if n == 0:
            continue
        Zr, Zi = _get_Z_sub(m, fb, tb)
        for t in m.time_set:
            loss = loss + cp.trace(Zr @ m.L_re[fb, tb, t] - Zi @ m.L_im[fb, tb, t])
    return loss


def substation_power_objective_sdp(m: SdpModel) -> cp.Expression:
    """
    Minimize total active power drawn from the substation (swing bus).

    Equivalent to ``substation_power_objective_rule`` in pyomo objectives.py.
    """
    total = 0.0
    for bus in m.swing_bus_set:
        ph_list = m.phase_map[bus]
        for t in m.time_set:
            # Substation injection = sum of outgoing branch power flows
            for fb_out, tb_out in m.to_bus_map.get(bus, []):
                out_ph_list = m.branch_phase_map[fb_out, tb_out]
                for i, ph in enumerate(out_ph_list):
                    total = total + m.SF_re[fb_out, tb_out, t][i, i]
    return total


def voltage_deviation_objective_sdp(m: SdpModel) -> cp.Expression:
    """
    Minimize total squared voltage deviation from 1.0 p.u.

        sum_{bus,ph,t} (W_re[bus,t][i,i] - 1)²

    Equivalent to ``voltage_deviation_objective_rule`` in pyomo objectives.py.
    """
    deviation = 0.0
    for bus in m.bus_set:
        if bus in m.swing_bus_set:
            continue
        ph_list = m.phase_map[bus]
        for t in m.time_set:
            for i in range(len(ph_list)):
                deviation = deviation + (m.W_re[bus, t][i, i] - 1.0) ** 2
    return deviation


def generation_curtailment_objective_sdp(m: SdpModel) -> cp.Expression:
    """
    Minimize total active power generation curtailment.

        sum_{bus,ph,t} (p_gen_nom - p_gen)

    Equivalent to ``generation_curtailment_objective_rule`` in pyomo objectives.py.
    """
    curtailment = 0.0
    for bus in m.gen_set:
        ph_list = m.gen_phase_map.get(bus, [])
        for t in m.time_set:
            for i, ph in enumerate(ph_list):
                p_nom = m.p_gen_nom.get((bus, ph, t), 0.0)
                curtailment = curtailment + (p_nom - m.p_gen[bus, t][i])
    return curtailment


def none_objective_sdp(m: SdpModel) -> cp.Expression:
    """Zero objective (feasibility only)."""
    return cp.Constant(0.0)
