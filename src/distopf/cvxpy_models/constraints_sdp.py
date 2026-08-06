"""
Constraint functions for the SDP BranchFlow model.

Each function follows the same pattern as constraints_nlp.py:
    - Takes an SdpModel as the first argument
    - Appends to m.constraints (master list) AND a named sub-list
    - Returns None

Main entry point: add_sdp_constraints(m, ...)
"""

from __future__ import annotations

import numpy as np
import cvxpy as cp

from distopf.cvxpy_models.sdp_branchflow import (
    SdpModel,
    _get_Z_sub,
    _get_W_parent_sub_idx,
)
from distopf.pyomo_models.nl_branchflow import ControlVariable

# ---------------------------------------------------------------------------
# Swing bus  (mirrors add_swing_bus_constraints)
# ---------------------------------------------------------------------------

# Balanced 3-phase slack bus voltage outer-product matrix at 1.0 p.u.
# W_slack = V_slack · V_slack†  for V = [1, a², a] (a = e^{j2π/3})
# Re(W_slack)
_W_SLACK_RE_3PH = np.array([
    [ 1.0,  -0.5,  -0.5],
    [-0.5,   1.0,  -0.5],
    [-0.5,  -0.5,   1.0],
])
# Im(W_slack)
_W_SLACK_IM_3PH = np.array([
    [ 0.0,             -np.sqrt(3) / 2,  np.sqrt(3) / 2],
    [ np.sqrt(3) / 2,   0.0,            -np.sqrt(3) / 2],
    [-np.sqrt(3) / 2,   np.sqrt(3) / 2,  0.0           ],
])


def add_swing_bus_constraints(m: SdpModel) -> None:
    """
    Fix swing bus voltage outer-product matrix.

    For a balanced 3-phase source at v_swing p.u., W = v_swing² · W_nominal
    where W_nominal is the unit balanced 3-phase matrix.

    For partial-phase swing buses (e.g. single-phase), the appropriate
    sub-matrix is extracted.

    Mirrors ``add_swing_bus_constraints`` in constraints_nlp.py.
    """
    new_constraints: list[cp.Constraint] = []

    for bus in m.swing_bus_set:
        ph_list = m.phase_map[bus]
        n = len(ph_list)
        if n == 0:
            continue

        for t in m.time_set:
            # Scale by swing voltage squared (per phase — use phase 'a' as reference)
            v_sq = m.v_swing.get((bus, ph_list[0], t), 1.0) ** 2

            # Extract phase-reduced sub-matrices from the 3-phase reference
            ph_idx = [{"a": 0, "b": 1, "c": 2}[ph] for ph in ph_list]
            W_re_ref = _W_SLACK_RE_3PH[np.ix_(ph_idx, ph_idx)] * v_sq
            W_im_ref = _W_SLACK_IM_3PH[np.ix_(ph_idx, ph_idx)] * v_sq

            for i in range(n):
                for j in range(n):
                    new_constraints.append(
                        m.W_re[bus, t][i, j] == W_re_ref[i, j]
                    )
                    new_constraints.append(
                        m.W_im[bus, t][i, j] == W_im_ref[i, j]
                    )

    m._add(m.swing_bus_constraints, new_constraints)


# ---------------------------------------------------------------------------
# Skew-symmetry  (structural constraint on W_im and L_im)
# ---------------------------------------------------------------------------

def add_hermitian_constraints(m: SdpModel) -> None:
    """
    Enforce W_im and L_im to be skew-symmetric (imaginary part of Hermitian matrix).

    W_im + W_im.T == 0  →  W_im[i,j] = -W_im[j,i],  W_im[i,i] = 0

    This is a structural property of outer-product matrices:
    Im(V·V†)[i,j] = Im(V_i · V_j*) = -Im(V_j · V_i*) = -Im(V·V†)[j,i]

    These constraints are added during variable creation and are grouped
    here for clarity.  They are NOT part of the physics — they are
    algebraic structure constraints.
    """
    new_constraints: list[cp.Constraint] = []

    for bus in m.bus_set:
        n = len(m.phase_map[bus])
        if n == 0:
            continue
        for t in m.time_set:
            new_constraints.append(m.W_im[bus, t] + m.W_im[bus, t].T == 0)

    for fb, tb in m.branch_set:
        n = len(m.branch_phase_map[fb, tb])
        if n == 0:
            continue
        for t in m.time_set:
            new_constraints.append(m.L_im[fb, tb, t] + m.L_im[fb, tb, t].T == 0)

    # Group under swing_bus_constraints (structural, not physics)
    # but keep them separate so they are always added first
    m._add(m.swing_bus_constraints, new_constraints)


# ---------------------------------------------------------------------------
# SDP PSD block constraint  (mirrors add_current_constraint1/2)
# ---------------------------------------------------------------------------

def add_psd_block_constraints(m: SdpModel) -> None:
    """
    Add the semidefinite (PSD) block constraint for each branch:

        M = [W_f_sub   SF    ]  ≽  0
            [SF†        L    ]

    In real-valued form this 4n×4n matrix is:

        [ W_re  -W_im   SF_re  -SF_im ]
        [ W_im   W_re   SF_im   SF_re ]  ≽  0
        [ SF_re' SF_im'  L_re   -L_im ]
        [-SF_im' SF_re'  L_im    L_re ]

    This is the SDP relaxation of the rank-1 constraint:
        [V_f; I] [V_f; I]†  ≽  0  (relaxed from == to ≽)

    When the relaxation is tight (rank = 1), this is equivalent to
    the exact nonlinear constraint P² + Q² = V² · L used in the NLP model.

    Mirrors the combination of add_current_constraint1 and
    add_current_constraint2_relaxed in constraints_nlp.py.
    """
    new_constraints: list[cp.Constraint] = []

    for fb, tb in m.branch_set:
        ph_list = m.branch_phase_map[fb, tb]
        n = len(ph_list)
        if n == 0:
            continue

        parent_idx = _get_W_parent_sub_idx(m, fb, ph_list)

        for t in m.time_set:
            Wf_re = m.W_re[fb, t][np.ix_(parent_idx, parent_idx)]
            Wf_im = m.W_im[fb, t][np.ix_(parent_idx, parent_idx)]
            Sr = m.SF_re[fb, tb, t]
            Si = m.SF_im[fb, tb, t]
            Lr = m.L_re[fb, tb, t]
            Li = m.L_im[fb, tb, t]

            # Build the 4n×4n real-valued representation of the complex PSD block
            M = cp.bmat([
                [Wf_re,  -Wf_im,  Sr,    -Si  ],
                [Wf_im,   Wf_re,  Si,     Sr  ],
                [Sr.T,    Si.T,   Lr,    -Li  ],
                [-Si.T,   Sr.T,   Li,     Lr  ],
            ])
            new_constraints.append(M >> 0)

    m._add(m.psd_block_constraints, new_constraints)


# ---------------------------------------------------------------------------
# Voltage drop  (mirrors add_voltage_drop_nlp_constraints)
# ---------------------------------------------------------------------------

def add_voltage_drop_sdp_constraints(m: SdpModel) -> None:
    """
    Add SDP voltage drop constraints for each branch (non-swing only):

        W_t = W_f_sub - SF·Z† - Z·SF† + Z·L·Z†

    In real-valued form:
        W_t_re = W_f_re - (SF_re·Zr' + SF_im·Zi') - (Zr·SF_re' + Zi·SF_im')
                        + (Zr·L_re·Zr' + Zi·L_re·Zi' - Zr·L_im·Zi' + Zi·L_im·Zr')
                        ... (see implementation)

        W_t_im = W_f_im - (SF_im·Zr' - SF_re·Zi') - (Zi·SF_re' - Zr·SF_im')
                        + (cross terms for imaginary part of Z·L·Z†)

    This is the SDP equivalent of the LinDistFlow / NLP voltage drop equations.

    Mirrors ``add_voltage_drop_nlp_constraints`` in constraints_nlp.py.
    """
    new_constraints: list[cp.Constraint] = []

    for fb, tb in m.branch_set:
        # Skip if this is a regulator branch
        # (regulator SDP constraints would be handled separately)
        ph_list = m.branch_phase_map[fb, tb]
        n = len(ph_list)
        if n == 0:
            continue

        Zr, Zi = _get_Z_sub(m, fb, tb)
        parent_idx = _get_W_parent_sub_idx(m, fb, ph_list)
        n_child = len(m.phase_map[tb])
        n_branch = len(ph_list)
        if n_branch != n_child:
            raise ValueError(
                f"Branch ({fb},{tb}): branch has {n_branch} phases {ph_list} "
                f"but to_bus {tb} has {n_child} phases {m.phase_map[tb]}. "
                f"branch_phase_map and phase_map are inconsistent."
            )
        if len(parent_idx) != n_branch:
            raise ValueError(
                f"Branch ({fb},{tb}): parent_idx={parent_idx} length "
                f"{len(parent_idx)} != branch phases {n_branch}. "
                f"Parent bus {fb} phases={m.phase_map[fb]}, "
                f"branch phases={ph_list}."
            )
        for t in m.time_set:
            # Parent bus W sub-block aligned with child phases
            Wf_re = m.W_re[fb, t][np.ix_(parent_idx, parent_idx)]
            Wf_im = m.W_im[fb, t][np.ix_(parent_idx, parent_idx)]

            Sr = m.SF_re[fb, tb, t]
            Si = m.SF_im[fb, tb, t]
            Lr = m.L_re[fb, tb, t]
            Li = m.L_im[fb, tb, t]

            # SF · Z†  (complex: (Sr + jSi)(Zr - jZi)' = Sr·Zr'+Si·Zi' + j(Si·Zr'-Sr·Zi'))
            SZH_re = Sr @ Zr.T + Si @ Zi.T
            SZH_im = Si @ Zr.T - Sr @ Zi.T

            # Z · SF†  (complex: (Zr+jZi)(Sr-jSi)' = Zr·Sr'+Zi·Si' + j(Zi·Sr'-Zr·Si'))
            ZSH_re = Zr @ Sr.T + Zi @ Si.T
            ZSH_im = Zi @ Sr.T - Zr @ Si.T

            # Z · L · Z†
            # First: A = Z·L = (Zr+jZi)(Lr+jLi) = (Zr·Lr - Zi·Li) + j(Zr·Li + Zi·Lr)
            A_re = Zr @ Lr - Zi @ Li   # Re(Z·L)
            A_im = Zr @ Li + Zi @ Lr   # Im(Z·L)
            # Then: A · Z†  = (A_re+jA_im)(Zr-jZi)' = A_re·Zr'+A_im·Zi' + j(A_im·Zr'-A_re·Zi')
            ZLZH_re = A_re @ Zr.T + A_im @ Zi.T
            ZLZH_im = A_im @ Zr.T - A_re @ Zi.T

            # Voltage drop equation (real part)
            new_constraints.append(
                m.W_re[tb, t] == Wf_re - SZH_re - ZSH_re + ZLZH_re
            )
            # Voltage drop equation (imaginary part)
            new_constraints.append(
                m.W_im[tb, t] == Wf_im - SZH_im - ZSH_im + ZLZH_im
            )

    m._add(m.voltage_drop_constraints, new_constraints)


# ---------------------------------------------------------------------------
# Power balance  (mirrors add_p_flow_nlp_constraints + add_q_flow_nlp_constraints)
# ---------------------------------------------------------------------------

def add_power_balance_sdp_constraints(m: SdpModel) -> None:
    """
    Add per-phase power balance constraints at every non-swing bus:

        p_gen[ph] - p_load[ph] - p_bat[ph]
            + P_incoming[ph] - P_loss_incoming[ph]
            = sum_outgoing(P_outgoing[ph])

        q_gen[ph] + q_cap[ph] - q_load[ph] - q_bat[ph]
            + Q_incoming[ph] - Q_loss_incoming[ph]
            = sum_outgoing(Q_outgoing[ph])

    Where the incoming power net of losses for phase ph on branch (fb,tb) is:
        P_net[ph] = SF_re[fb,tb][ph,ph] - (Zr·L·... diagonal)
        (extracted from the diagonal of SF - Z·L)

    The diagonal of (Z·L) gives the per-phase losses on the incoming branch.

    Mirrors the combined effect of add_p_flow_nlp_constraints and
    add_q_flow_nlp_constraints in constraints_nlp.py.
    """
    new_constraints: list[cp.Constraint] = []

    for bus in m.bus_set:
        ph_list = m.phase_map[bus]
        n = len(ph_list)
        if n == 0:
            continue

        # Incoming branch (parent → bus), if any
        inc_branch = None
        if bus in m.from_bus_map:
            fb_inc = m.from_bus_map[bus]
            inc_branch = (fb_inc, bus)

        # Outgoing branches (bus → children)
        out_branches = m.to_bus_map.get(bus, [])

        for t in m.time_set:
            # Build net injection expressions per phase
            for local_idx, ph in enumerate(ph_list):

                # --- Generation ---
                p_inj = (
                    m.p_gen[bus, t][local_idx]
                    if (bus, t) in m.p_gen
                    else 0.0
                )
                q_inj = (
                    m.q_gen[bus, t][local_idx]
                    if (bus, t) in m.q_gen
                    else 0.0
                )

                # --- Capacitor ---
                q_inj = q_inj + (
                    m.q_cap[bus, t][local_idx]
                    if (bus, t) in m.q_cap
                    else 0.0
                )

                # --- Battery ---
                p_inj = p_inj - (
                    m.p_bat[bus, t][local_idx]
                    if (bus, t) in m.p_bat
                    else 0.0
                )
                q_inj = q_inj - (
                    m.q_bat[bus, t][local_idx]
                    if (bus, t) in m.q_bat
                    else 0.0
                )

                # --- Load ---
                p_load = m.p_load_nom.get((bus, ph, t), 0.0)
                q_load = m.q_load_nom.get((bus, ph, t), 0.0)

                # --- Incoming power (net of line losses on incoming branch) ---
                # Net power delivered = SF_diag - diag(Z · L)
                # Real part: SF_re[k,k] - (Zr·L_re - Zi·L_im)[k,k]
                # Imag part: SF_im[k,k] - (Zr·L_im + Zi·L_re)[k,k]
                inc_P = 0.0
                inc_Q = 0.0
                if inc_branch is not None:
                    fb_inc, _ = inc_branch
                    inc_ph_list = m.branch_phase_map[fb_inc, bus]
                    if ph in inc_ph_list:
                        k = inc_ph_list.index(ph)
                        Zr_inc, Zi_inc = _get_Z_sub(m, fb_inc, bus)
                        Lr = m.L_re[fb_inc, bus, t]
                        Li = m.L_im[fb_inc, bus, t]
                        Sr = m.SF_re[fb_inc, bus, t]
                        Si = m.SF_im[fb_inc, bus, t]

                        # Loss on incoming branch for phase k (diagonal element)
                        ZL_re_kk = (
                            sum(Zr_inc[k, jj] * Lr[jj, k] for jj in range(len(inc_ph_list)))
                            - sum(Zi_inc[k, jj] * Li[jj, k] for jj in range(len(inc_ph_list)))
                        )
                        ZL_im_kk = (
                            sum(Zr_inc[k, jj] * Li[jj, k] for jj in range(len(inc_ph_list)))
                            + sum(Zi_inc[k, jj] * Lr[jj, k] for jj in range(len(inc_ph_list)))
                        )
                        inc_P = Sr[k, k] - ZL_re_kk
                        inc_Q = Si[k, k] - ZL_im_kk

                # --- Outgoing power (sum over child branches) ---
                out_P = 0.0
                out_Q = 0.0
                for fb_out, tb_out in out_branches:
                    out_ph_list = m.branch_phase_map[fb_out, tb_out]
                    if ph in out_ph_list:
                        k_out = out_ph_list.index(ph)
                        out_P = out_P + m.SF_re[fb_out, tb_out, t][k_out, k_out]
                        out_Q = out_Q + m.SF_im[fb_out, tb_out, t][k_out, k_out]

                # --- Power balance equations ---
                # Active: p_gen - p_load - p_bat + P_inc_net = P_out
                new_constraints.append(
                    p_inj - p_load + inc_P == out_P
                )
                # Reactive: q_gen + q_cap - q_load - q_bat + Q_inc_net = Q_out
                new_constraints.append(
                    q_inj - q_load + inc_Q == out_Q
                )

    m._add(m.power_balance_constraints, new_constraints)


# ---------------------------------------------------------------------------
# Voltage limits  (mirrors add_voltage_limits)
# ---------------------------------------------------------------------------

def add_voltage_limits_sdp(m: SdpModel) -> None:
    """
    Add voltage magnitude squared bounds using the W matrix diagonal:

        v_min² ≤ W_re[bus,t][i,i] ≤ v_max²

    Swing buses are excluded (their voltage is fixed by add_swing_bus_constraints).

    Mirrors ``add_voltage_limits`` in constraints_nlp.py.
    """
    new_constraints: list[cp.Constraint] = []

    for bus in m.bus_set:
        if bus in m.swing_bus_set:
            continue
        ph_list = m.phase_map[bus]
        for t in m.time_set:
            for i, ph in enumerate(ph_list):
                v_min_sq = m.v_min.get((bus, ph), 0.95) ** 2
                v_max_sq = m.v_max.get((bus, ph), 1.05) ** 2
                new_constraints.append(
                    m.W_re[bus, t][i, i] >= v_min_sq
                )
                new_constraints.append(
                    m.W_re[bus, t][i, i] <= v_max_sq
                )

    m._add(m.voltage_limit_constraints, new_constraints)


# ---------------------------------------------------------------------------
# Generator constraints  (mirrors add_generator_limits + constant p/q)
# ---------------------------------------------------------------------------

def add_generator_limits_sdp(m: SdpModel) -> None:
    """
    Add generator active and reactive power bounds.

    - ControlVariable.NONE : p_gen and q_gen fixed to nominal values
    - ControlVariable.Q    : p_gen fixed to nominal, q_gen bounded by s_rated
    - ControlVariable.P    : q_gen fixed to nominal, p_gen bounded
    - ControlVariable.PQ   : both p_gen and q_gen bounded (circular via SOC)

    Mirrors ``add_generator_limits`` in constraints_nlp.py.
    """
    new_constraints: list[cp.Constraint] = []

    for bus in m.gen_set:
        ph_list = m.gen_phase_map.get(bus, [])
        for t in m.time_set:
            for i, ph in enumerate(ph_list):
                ct = m.gen_control_type.get((bus, ph), ControlVariable.NONE)
                s_rated = m.s_rated.get((bus, ph), 1000.0)
                p_nom = m.p_gen_nom.get((bus, ph, t), 0.0)
                q_nom = m.q_gen_nom.get((bus, ph, t), 0.0)
                q_max = m.q_gen_max.get((bus, ph), s_rated)
                q_min = m.q_gen_min.get((bus, ph), -s_rated)

                if ct == ControlVariable.NONE:
                    # Fix both p and q to nominal
                    new_constraints.append(
                        m.p_gen[bus, t][i] == p_nom
                    )
                    new_constraints.append(
                        m.q_gen[bus, t][i] == q_nom
                    )
                elif ct == ControlVariable.Q:
                    # Fix p, bound q
                    new_constraints.append(
                        m.p_gen[bus, t][i] == p_nom
                    )
                    q_bound = float(
                        np.sqrt(max(0.0, s_rated**2 - p_nom**2))
                    )
                    new_constraints.append(
                        m.q_gen[bus, t][i] >= max(-q_bound, q_min)
                    )
                    new_constraints.append(
                        m.q_gen[bus, t][i] <= min(q_bound, q_max)
                    )
                elif ct == ControlVariable.P:
                    # Fix q, bound p
                    new_constraints.append(
                        m.q_gen[bus, t][i] == q_nom
                    )
                    new_constraints.append(
                        m.p_gen[bus, t][i] >= 0.0
                    )
                    new_constraints.append(
                        m.p_gen[bus, t][i] <= min(p_nom, s_rated)
                    )
                elif ct == ControlVariable.PQ:
                    # Circular SOC constraint: p² + q² ≤ s_rated²
                    new_constraints.append(
                        m.p_gen[bus, t][i] >= 0.0
                    )
                    new_constraints.append(
                        m.p_gen[bus, t][i] <= min(p_nom, s_rated)
                    )
                    new_constraints.append(
                        m.q_gen[bus, t][i] >= max(-s_rated, q_min)
                    )
                    new_constraints.append(
                        m.q_gen[bus, t][i] <= min(s_rated, q_max)
                    )
                    new_constraints.append(
                        cp.norm(
                            cp.hstack([m.p_gen[bus, t][i], m.q_gen[bus, t][i]]), 2
                        ) <= s_rated
                    )

    m._add(m.generator_limit_constraints, new_constraints)


# ---------------------------------------------------------------------------
# Capacitor constraints  (mirrors add_capacitor_constraints)
# ---------------------------------------------------------------------------

def add_capacitor_constraints_sdp(m: SdpModel) -> None:
    """
    Add voltage-dependent capacitor reactive power constraints:

        q_cap[bus, ph, t] = q_cap_nom[bus, ph] * W_re[bus, t][i, i]

    (q_cap is proportional to voltage squared — same model as NLP).

    Mirrors ``add_capacitor_constraints`` in constraints_nlp.py.
    """
    new_constraints: list[cp.Constraint] = []

    for bus in m.cap_set:
        ph_list = m.cap_phase_map.get(bus, [])
        bus_ph_list = m.phase_map[bus]
        for t in m.time_set:
            for ph in ph_list:
                if ph not in bus_ph_list:
                    continue
                i = bus_ph_list.index(ph)
                q_nom = m.q_cap_nom.get((bus, ph), 0.0)
                new_constraints.append(
                    m.q_cap[bus, t][i] == q_nom * m.W_re[bus, t][i, i]
                )

    m._add(m.capacitor_constraints, new_constraints)


# ---------------------------------------------------------------------------
# Battery constraints  (mirrors battery constraints in constraints_nlp.py)
# ---------------------------------------------------------------------------

def add_battery_constraints_sdp(m: SdpModel) -> None:
    """
    Add battery operation constraints:

    1. Net power per phase equals equal split of net discharge:
           p_bat[bus, ph, t] = (p_discharge - p_charge) / n_phases

    2. SOC dynamics:
           soc[t] = soc[t-1] + eta_c * delta_t * p_charge[t]
                              - (1/eta_d) * delta_t * p_discharge[t]

    3. SOC bounds:
           soc_min ≤ soc[t] ≤ soc_max

    4. Charge / discharge bounds:
           0 ≤ p_charge[t] ≤ s_bat_rated
           0 ≤ p_discharge[t] ≤ s_bat_rated

    5. Reactive power fixed to nominal (if control_type == P):
           q_bat[bus, ph, t] == 0

    Mirrors battery constraints in constraints_nlp.py.
    """
    new_constraints: list[cp.Constraint] = []

    for bat_id in m.bat_set:
        ph_list = m.bat_phase_map.get(bat_id, [])
        n_phases = m.battery_n_phases.get(bat_id, len(ph_list))
        bus_ph_list = m.phase_map.get(bat_id, [])
        eta_c = m.charge_efficiency.get(bat_id, 1.0)
        eta_d = m.discharge_efficiency.get(bat_id, 1.0)

        for t in m.time_set:
            s_rated_phase = m.s_bat_rated.get((bat_id, ph_list[0]), 1000.0) if ph_list else 1000.0

            # Charge / discharge bounds (already nonneg from variable declaration)
            new_constraints.append(
                m.p_charge[bat_id, t] <= s_rated_phase * n_phases
            )
            new_constraints.append(
                m.p_discharge[bat_id, t] <= s_rated_phase * n_phases
            )

            # SOC dynamics
            if t == m.start_step:
                soc_prev = m.start_soc.get(bat_id, 0.5)
            else:
                soc_prev = m.soc[bat_id, t - 1]

            new_constraints.append(
                m.soc[bat_id, t] - soc_prev
                == eta_c * m.delta_t * m.p_charge[bat_id, t]
                - (1.0 / eta_d) * m.delta_t * m.p_discharge[bat_id, t]
            )

            # SOC bounds
            new_constraints.append(
                m.soc[bat_id, t] >= m.soc_min.get(bat_id, 0.0)
            )
            new_constraints.append(
                m.soc[bat_id, t] <= m.soc_max.get(bat_id, 1.0)
            )

            # Per-phase net power (equal split)
            for ph in ph_list:
                if ph not in bus_ph_list:
                    continue
                i = bus_ph_list.index(ph)
                new_constraints.append(
                    m.p_bat[bat_id, t][i]
                    == (m.p_discharge[bat_id, t] - m.p_charge[bat_id, t]) / n_phases
                )

                # Reactive power
                ct = m.bat_control_type.get(bat_id, ControlVariable.P)
                if ct == ControlVariable.P:
                    new_constraints.append(m.q_bat[bat_id, t][i] == 0.0)
                else:
                    q_max_ph = m.q_bat_max.get((bat_id, ph), s_rated_phase)
                    q_min_ph = m.q_bat_min.get((bat_id, ph), -s_rated_phase)
                    new_constraints.append(
                        m.q_bat[bat_id, t][i] >= q_min_ph
                    )
                    new_constraints.append(
                        m.q_bat[bat_id, t][i] <= q_max_ph
                    )

    m._add(m.battery_constraints, new_constraints)


# ---------------------------------------------------------------------------
# Thermal limits  (mirrors add_circular_thermal_constraints)
# ---------------------------------------------------------------------------

def add_thermal_constraints_sdp(m: SdpModel) -> None:
    """
    Add branch thermal limit constraints using the branch power flow matrix diagonal:

        SF_re[fb,tb,t][i,i]² + SF_im[fb,tb,t][i,i]² ≤ s_branch_max[fb,tb,ph]²

    This uses CVXPY's SOC (second-order cone) constraint form:
        cp.norm([P_flow, Q_flow], 2) <= s_max

    Requires ``case.branch_data`` to have columns s_a_max, s_b_max, s_c_max.
    Branches without limits are skipped.

    Mirrors ``add_circular_thermal_constraints`` in constraints_nlp.py.
    """
    new_constraints: list[cp.Constraint] = []

    thermal_cols = {"a": "s_a_max", "b": "s_b_max", "c": "s_c_max"}
    branch_limits: dict[tuple[int, int, str], float] = {}

    for _, row in m.case.branch_data.iterrows():
        fb, tb = int(row.fb), int(row.tb)
        for ph, col in thermal_cols.items():
            if col in m.case.branch_data.columns:
                val = getattr(row, col, None)
                if val is not None and not (
                    isinstance(val, float) and np.isnan(val)
                ) and float(val) > 0:
                    branch_limits[fb, tb, ph] = float(val)

    if not branch_limits:
        return

    for fb, tb in m.branch_set:
        ph_list = m.branch_phase_map[fb, tb]
        for t in m.time_set:
            for i, ph in enumerate(ph_list):
                s_max = branch_limits.get((fb, tb, ph))
                if s_max is None:
                    continue
                new_constraints.append(
                    cp.norm(
                        cp.hstack([
                            m.SF_re[fb, tb, t][i, i],
                            m.SF_im[fb, tb, t][i, i],
                        ]),
                        2,
                    )
                    <= s_max
                )

    m._add(m.thermal_limit_constraints, new_constraints)


# ---------------------------------------------------------------------------
# Main entry point  (mirrors add_nlp_constraints)
# ---------------------------------------------------------------------------

def add_sdp_constraints(
    m: SdpModel,
    thermal_constraints: bool = False,
) -> None:
    """
    Add all constraints for the SDP BranchFlow model.

    This is the main entry point for constraint attachment, mirroring
    ``add_nlp_constraints`` in constraints_nlp.py.

    Parameters
    ----------
    m : SdpModel
        Model created by :func:`create_sdp_branchflow_model`.
    thermal_constraints : bool, default False
        If True, add branch thermal limit constraints.

    Constraint order
    ----------------
    1. Hermitian structure (skew-symmetry of imaginary parts)
    2. Swing bus voltage fix
    3. SDP PSD block (relaxed rank-1)
    4. Voltage drop (W propagation)
    5. Power balance (per bus per phase)
    6. Voltage magnitude limits
    7. Generator limits
    8. Capacitor constraints
    9. Battery constraints
    10. Thermal limits (optional)
    """
    add_hermitian_constraints(m)
    add_swing_bus_constraints(m)
    add_psd_block_constraints(m)
    add_voltage_drop_sdp_constraints(m)
    add_power_balance_sdp_constraints(m)
    add_voltage_limits_sdp(m)
    add_generator_limits_sdp(m)
    add_capacitor_constraints_sdp(m)
    add_battery_constraints_sdp(m)
    if thermal_constraints:
        add_thermal_constraints_sdp(m)