"""Branch-flow and LinDistFlow network constraints (vectorized CVXPY).

Vectorized counterpart of ``distopf.cvxpy_models.network.bfm_constraints``.
Constraint names are identical. Each constraint is built once for all branches
and time steps using sparse topology and impedance matrices, instead of one
scalar constraint per index. Row keys (without time) are recorded with every
constraint so duals can be tabulated.

Convexity notes
---------------
- LinDistFlow constraints are affine.
- ``current_constraint`` and ``current_sqr_constraint`` with ``==`` are
  non-convex (DNLP). Use ``socp_relaxation=True`` for a convex model.
- Loss terms multiply ``l_flow`` by ``cp.Parameter`` vectors holding ``cos(d)``
  and ``sin(d)``, so they stay affine in the variables.
- The PSD constraint is built per branch and time step (matrix sizes differ
  between branches), so it is not vectorized.
"""

from __future__ import annotations

from itertools import combinations
from math import sqrt

import cvxpy as cp
import numpy as np

from distopf.cvxpy_models_performance.common import common_constraints as const
from distopf.cvxpy_models_performance.common.model import (
    CvxModel,
    IndexSet,
    add_constraint,
    sparse,
)
from distopf.pyomo_models.common.data import parse_phases

sqrt2 = sqrt(2)
sqrt3 = sqrt(3)


def _flat(expression):
    return cp.vec(expression, order="C")


def _is_zero(value) -> bool:
    return isinstance(value, (int, float)) and value == 0


def _registry(m: CvxModel):
    registry = getattr(m, "_device_registry", None)
    if registry is None:
        raise RuntimeError("Device registry is not installed.")
    return registry


def _bp_keys(m: CvxModel) -> list:
    return list(m.branch_phase_set)


def _to_selector(m: CvxModel):
    """(n_branch_phase x n_bus_phase) selector of each branch's to-bus row."""
    bus_pos = m.bus_phase_set.pos
    rows, cols = [], []
    for i, (_fb, tb, ph) in enumerate(m.branch_phase_set):
        j = bus_pos.get((tb, ph))
        if j is not None:
            rows.append(i)
            cols.append(j)
    shape = (len(m.branch_phase_set), len(m.bus_phase_set))
    return sparse(rows, cols, [1.0] * len(rows), shape)


def _outgoing_matrix(m: CvxModel):
    """Sums flows leaving the to-bus of each row, including center-tap secondaries."""
    bp_pos = m.branch_phase_set.pos
    primary = getattr(m, "primary_phase_map", {})
    rows, cols = [], []
    for i, (_fb, tb, ph) in enumerate(m.branch_phase_set):
        children = m.to_bus_map[tb]
        for fb2, tb2 in children:
            j = bp_pos.get((fb2, tb2, ph))
            if j is not None:
                rows.append(i)
                cols.append(j)
        if ph not in ("s1", "s2"):
            for fb2, tb2 in children:
                if primary.get((fb2, tb2)) == ph:
                    for sec_ph in ("s1", "s2"):
                        j = bp_pos.get((fb2, tb2, sec_ph))
                        if j is not None:
                            rows.append(i)
                            cols.append(j)
    n = len(m.branch_phase_set)
    return sparse(rows, cols, [1.0] * len(rows), (n, n))


def _voltage_drop_matrices(m: CvxModel):
    """Sparse ``Zp`` and ``Zq`` such that the voltage drop is ``Zp @ P + Zq @ Q``."""
    bp_pos = m.branch_phase_set.pos
    n = len(m.branch_phase_set)
    p_rows, p_cols, p_vals = [], [], []
    q_rows, q_cols, q_vals = [], [], []

    def add(i, j, p_value, q_value):
        p_rows.append(i)
        p_cols.append(j)
        p_vals.append(p_value)
        q_rows.append(i)
        q_cols.append(j)
        q_vals.append(q_value)

    for i, (fb, tb, ph) in enumerate(m.branch_phase_set):
        if ph in ("s1", "s2"):
            other = "s2" if ph == "s1" else "s1"
            add(i, i, 2 * m.r[fb, tb, ph + ph], 2 * m.x[fb, tb, ph + ph])
            j = bp_pos.get((fb, tb, other))
            if j is not None:
                add(i, j, 2 * m.r[fb, tb, "s1s2"], 2 * m.x[fb, tb, "s1s2"])
            continue
        a = ph
        idx = "abc".index(a)
        b = "abc"[(idx + 1) % 3]
        c = "abc"[(idx - 1) % 3]
        add(i, i, 2 * m.r[fb, tb, a + a], 2 * m.x[fb, tb, a + a])
        j = bp_pos.get((fb, tb, b))
        if j is not None:
            ab = "".join(sorted(a + b))
            rab, xab = m.r[fb, tb, ab], m.x[fb, tb, ab]
            add(i, j, -rab + sqrt3 * xab, -xab - sqrt3 * rab)
        j = bp_pos.get((fb, tb, c))
        if j is not None:
            ac = "".join(sorted(a + c))
            rac, xac = m.r[fb, tb, ac], m.x[fb, tb, ac]
            add(i, j, -rac - sqrt3 * xac, -xac + sqrt3 * rac)
    return (
        sparse(p_rows, p_cols, p_vals, (n, n)),
        sparse(q_rows, q_cols, q_vals, (n, n)),
    )


def _source_voltage(m: CvxModel, for_current: bool = False):
    """Expression ``(n_branch_phase, n_steps)`` of each row's from-side v2.

    Regulated rows use ``v2_reg``. Triplex rows read the primary phase of the
    from bus. ``for_current`` also applies the primary-phase map and regulators
    to triplex rows, as the current constraints do.
    """
    reg_pos = (
        m.reg_phase_set.pos
        if hasattr(m, "reg_phase_set") and hasattr(m, "v2_reg")
        else {}
    )
    bus_pos = m.bus_phase_set.pos
    primary = getattr(m, "primary_phase_map", {})
    from_rows, from_cols, reg_rows, reg_cols = [], [], [], []
    for i, (fb, tb, ph) in enumerate(m.branch_phase_set):
        triplex = ph in ("s1", "s2")
        if (fb, tb, ph) in reg_pos and (not triplex or for_current):
            reg_rows.append(i)
            reg_cols.append(reg_pos[(fb, tb, ph)])
            continue
        from_ph = primary.get((fb, tb), ph) if (triplex or for_current) else ph
        from_rows.append(i)
        from_cols.append(bus_pos[(fb, from_ph)])
    n = len(m.branch_phase_set)
    source = (
        sparse(from_rows, from_cols, [1.0] * len(from_rows), (n, len(bus_pos))) @ m.v2
    )
    if reg_rows:
        select = sparse(
            reg_rows, reg_cols, [1.0] * len(reg_rows), (n, len(m.reg_phase_set))
        )
        source = source + select @ m.v2_reg
    return source


def _angle_terms(m: CvxModel, kind: str):
    """Terms ``l_flow[pair] * (c_cos * cos(d) + c_sin * sin(d))`` per row.

    ``kind`` is ``"p"`` (active loss), ``"q"`` (reactive loss) or ``"v"``
    (voltage drop cross term). Returns ``None`` if there are no terms.
    """
    pair_pos = m.branch_phase_pair_set.pos
    angle_pos = m.branch_angle_phase_pair_set.pos
    rows, pairs, angles, c_cos, c_sin = [], [], [], [], []
    for i, (fb, tb, ph) in enumerate(m.branch_phase_set):
        if kind == "v":
            if ph in ("s1", "s2"):
                continue
            for q1, q2 in combinations("abc", 2):
                pair = "".join(sorted(q1 + q2))
                if not (
                    q1 in m.phase_map[tb]
                    and q2 in m.phase_map[tb]
                    and (fb, tb, pair) in pair_pos
                ):
                    continue
                r1 = m.r[fb, tb, "".join(sorted(ph + q1))]
                r2 = m.r[fb, tb, "".join(sorted(ph + q2))]
                x1 = m.x[fb, tb, "".join(sorted(ph + q1))]
                x2 = m.x[fb, tb, "".join(sorted(ph + q2))]
                rows.append(i)
                pairs.append(pair_pos[(fb, tb, pair)])
                angles.append(angle_pos[(fb, tb, q1 + q2)])
                c_cos.append(2 * (r1 * r2 + x1 * x2))
                c_sin.append(2 * (r1 * x2 - x1 * r2))
            continue
        for ph2 in m.phase_map[tb]:
            pair = "".join(sorted([ph, ph2]))
            if (fb, tb, pair) not in pair_pos:
                continue
            r = m.r[fb, tb, pair]
            x = m.x[fb, tb, pair]
            rows.append(i)
            pairs.append(pair_pos[(fb, tb, pair)])
            angles.append(angle_pos[(fb, tb, ph2 + ph)])
            if kind == "p":
                c_cos.append(r)
                c_sin.append(-x)
            else:
                c_cos.append(x)
                c_sin.append(r)
    if not rows:
        return None
    k = len(rows)
    angles_arr = np.array(angles)
    coefficient = cp.reshape(
        cp.multiply(np.array(c_cos), m.cos_d[angles_arr])
        + cp.multiply(np.array(c_sin), m.sin_d[angles_arr]),
        (k, 1),
        order="C",
    )
    contribution = cp.multiply(coefficient, m.l_flow[np.array(pairs), :])
    n = len(m.branch_phase_set)
    return sparse(rows, range(k), [1.0] * k, (n, k)) @ contribution


def _voltage_drop_term2(m: CvxModel):
    """Sum over phases of ``(r^2 + x^2) * l_flow[ph2 ph2]`` per row."""
    pair_pos = m.branch_phase_pair_set.pos
    rows, cols, vals = [], [], []
    for i, (fb, tb, ph) in enumerate(m.branch_phase_set):
        if ph in ("s1", "s2"):
            continue
        for ph2 in m.phase_map[tb]:
            j = pair_pos.get((fb, tb, ph2 + ph2))
            if j is None:
                continue
            pair = "".join(sorted(ph + ph2))
            rows.append(i)
            cols.append(j)
            vals.append(m.r[fb, tb, pair] ** 2 + m.x[fb, tb, pair] ** 2)
    shape = (len(m.branch_phase_set), len(m.branch_phase_pair_set))
    return sparse(rows, cols, vals, shape) @ m.l_flow


def add_p_flow_constraints(m: CvxModel, linear: bool = True) -> None:
    """Add active-power balance from the signed injection registry."""
    registry = _registry(m)
    injection = registry.active_power_injection(m)
    outgoing = _outgoing_matrix(m) @ m.p_flow
    lhs = m.p_flow
    if not linear:
        loss = _angle_terms(m, "p")
        if loss is not None:
            lhs = lhs - loss
    rhs = outgoing
    if not _is_zero(injection):
        rhs = rhs - _to_selector(m) @ injection
    add_constraint(m, "power_balance_p", lhs == rhs, _bp_keys(m))


def add_q_flow_constraints(m: CvxModel, linear: bool = True) -> None:
    """Add reactive-power balance from the signed injection registry."""
    registry = _registry(m)
    injection = registry.reactive_power_injection(m)
    outgoing = _outgoing_matrix(m) @ m.q_flow
    lhs = m.q_flow
    if not linear:
        loss = _angle_terms(m, "q")
        if loss is not None:
            lhs = lhs - loss
    rhs = outgoing
    if not _is_zero(injection):
        rhs = rhs - _to_selector(m) @ injection
    add_constraint(m, "power_balance_q", lhs == rhs, _bp_keys(m))


def add_voltage_drop_constraints(m: CvxModel, linear: bool = True) -> None:
    """Add voltage drop constraints.

    Regulator branches use ``v2_reg`` in place of the from-bus ``v2``.
    """
    z_p, z_q = _voltage_drop_matrices(m)
    v_drop = z_p @ m.p_flow + z_q @ m.q_flow
    if not linear:
        v_drop = v_drop + _voltage_drop_term2(m)
        cross = _angle_terms(m, "v")
        if cross is not None:
            v_drop = v_drop + cross
    source = _source_voltage(m, for_current=False)
    add_constraint(
        m,
        "voltage_drop",
        _to_selector(m) @ m.v2 == source - v_drop,
        _bp_keys(m),
    )


def _diag_indices(m: CvxModel):
    pair_pos = m.branch_phase_pair_set.pos
    return np.array(
        [pair_pos[(fb, tb, ph + ph)] for fb, tb, ph in m.branch_phase_set], dtype=int
    )


def add_current_constraint1(m: CvxModel) -> None:
    """``P^2 + Q^2 == v2 * l`` (non-convex; DNLP)."""
    source = _source_voltage(m, for_current=True)
    l_diag = m.l_flow[_diag_indices(m), :]
    add_constraint(
        m,
        "current_constraint",
        cp.square(m.p_flow) + cp.square(m.q_flow) == cp.multiply(source, l_diag),
        _bp_keys(m),
    )


def _offdiagonal(m: CvxModel):
    """Positions of off-diagonal current pairs and their two diagonal entries."""
    pair_pos = m.branch_phase_pair_set.pos
    rows, first, second, keys = [], [], [], []
    for i, (fb, tb, phases) in enumerate(m.branch_phase_pair_set):
        ph1, ph2 = parse_phases(phases)[:2]
        if ph1 == ph2:
            continue
        rows.append(i)
        first.append(pair_pos[(fb, tb, ph1 + ph1)])
        second.append(pair_pos[(fb, tb, ph2 + ph2)])
        keys.append((fb, tb, phases))
    return (
        np.array(rows, dtype=int),
        np.array(first, dtype=int),
        np.array(second, dtype=int),
        keys,
    )


def add_current_constraint2(m: CvxModel) -> None:
    """``l_ab^2 == l_aa * l_bb`` (non-convex; DNLP)."""
    rows, first, second, keys = _offdiagonal(m)
    if len(rows) == 0:
        add_constraint(m, "current_sqr_constraint", [], [])
        return
    add_constraint(
        m,
        "current_sqr_constraint",
        cp.square(m.l_flow[rows, :])
        == cp.multiply(m.l_flow[first, :], m.l_flow[second, :]),
        keys,
    )


def add_current_constraint1_relaxed(m: CvxModel) -> None:
    """``P^2 + Q^2 <= v2 * l`` as a rotated second-order cone constraint."""
    source = _source_voltage(m, for_current=True)
    l_diag = m.l_flow[_diag_indices(m), :]
    add_constraint(
        m,
        "current_constraint",
        cp.SOC(
            _flat(source + l_diag),
            cp.vstack(
                [_flat(2 * m.p_flow), _flat(2 * m.q_flow), _flat(source - l_diag)]
            ),
            axis=0,
        ),
        _bp_keys(m),
    )


def add_current_constraint2_relaxed(m: CvxModel) -> None:
    """``l_ab^2 <= l_aa * l_bb`` as a rotated second-order cone constraint."""
    rows, first, second, keys = _offdiagonal(m)
    if len(rows) == 0:
        add_constraint(m, "current_sqr_constraint", [], [])
        return
    l_ab = m.l_flow[rows, :]
    l_aa = m.l_flow[first, :]
    l_bb = m.l_flow[second, :]
    add_constraint(
        m,
        "current_sqr_constraint",
        cp.SOC(
            _flat(l_aa + l_bb),
            cp.vstack([_flat(2 * l_ab), _flat(l_aa - l_bb)]),
            axis=0,
        ),
        keys,
    )


# ============ SDP relaxation ==========================================================
#
# For each branch (fb, tb) with n phases and each time step, the 4n x 4n matrix
#
#     full:   [[ Vr,  P, -Vi, -Q],        test1:  [[ Vr,  P,  0, -Q],
#              [ P.T, L,  Q.T, 0],                 [ P.T, L,  Q.T, 0],
#              [ Vi,  Q,  Vr,  P],                 [ 0,   Q,  Vr,  P],
#              [-Q.T, 0,  P.T, L]]                 [-Q.T, 0,  P.T, L]]
#
# must be positive semidefinite. Diagonals of Vr, P, Q, L reuse the (from-side)
# v2, p_flow, q_flow and l_flow variables. L's off-diagonals are l_flow entries.
# The off-diagonals of Vr, Vi, P and Q are the auxiliary variables psd_v_re,
# psd_v_im, psd_p_off and psd_q_off, unless ``diagonal_only`` fixes them to zero.
#
#   add_psd_constraints(m)                       full layout, auxiliary off-diagonals
#   add_psd_constraints(m, diagonal_only=True)   full layout, zero off-diagonals
#   add_psd_diagonal_constraints(m)              same as the line above
#   add_psd_test1_constraints(m)                 test1 layout, zero off-diagonals
#   add_psd_test1_constraints(m, False)          test1 layout, auxiliary off-diagonals
#
# All variants register the constraint ``psd_constraint`` (one piece per branch
# and time step, keyed by (fb, tb)).


def _ensure_psd_aux_vars(m: CvxModel) -> None:
    """Create the off-diagonal auxiliary variables once per model."""
    if hasattr(m, "psd_v_re"):
        return
    bp_pos = m.branch_phase_set.pos
    v_keys: list[tuple] = []
    pq_keys: list[tuple] = []
    for fb, tb in m.branch_set:
        phases = [ph for ph in "abc" if (fb, tb, ph) in bp_pos]
        v_keys.extend((fb, tb, p1 + p2) for p1, p2 in combinations(phases, 2))
        pq_keys.extend((fb, tb, p1 + p2) for p1 in phases for p2 in phases if p1 != p2)
    m.psd_v_key_set = IndexSet(v_keys)
    m.psd_pq_key_set = IndexSet(pq_keys)
    m.add_var("psd_v_re", m.psd_v_key_set)
    m.add_var("psd_v_im", m.psd_v_key_set)
    m.add_var("psd_p_off", m.psd_pq_key_set)
    m.add_var("psd_q_off", m.psd_pq_key_set)


def _to_block(entries):
    """Block matrix from a list of rows whose entries are scalars or expressions."""
    return cp.bmat(
        [
            [
                cp.reshape(
                    cp.Constant(e) if isinstance(e, (int, float)) else e,
                    (1, 1),
                    order="C",
                )
                for e in row
            ]
            for row in entries
        ]
    )


def _psd_branch_matrix(m, fb, tb, phases, t, source, layout, diagonal_only):
    """The 4n x 4n matrix for one branch at time index ``t`` (see layouts above)."""
    bp_pos = m.branch_phase_set.pos
    pair_pos = m.branch_phase_pair_set.pos
    n = len(phases)
    vr, vi, pm, qm, lm = ([[0] * n for _ in range(n)] for _ in range(5))
    for i, a in enumerate(phases):
        for j, b in enumerate(phases):
            if i == j:
                row = bp_pos[(fb, tb, a)]
                vr[i][i] = source[row, t]
                pm[i][i] = m.p_flow[row, t]
                qm[i][i] = m.q_flow[row, t]
                lm[i][i] = m.l_flow[pair_pos[(fb, tb, a + a)], t]
                continue
            pair = a + b if i < j else b + a
            lm[i][j] = m.l_flow[pair_pos[(fb, tb, pair)], t]
            if diagonal_only:
                continue
            v_row = m.psd_v_key_set.pos[(fb, tb, pair)]
            pq_row = m.psd_pq_key_set.pos[(fb, tb, a + b)]
            vr[i][j] = m.psd_v_re[v_row, t]
            vi_var = m.psd_v_im[v_row, t]
            vi[i][j] = vi_var if i < j else -vi_var
            pm[i][j] = m.psd_p_off[pq_row, t]
            qm[i][j] = m.psd_q_off[pq_row, t]

    vr_m, vi_m, p_m, q_m, l_m = (_to_block(x) for x in (vr, vi, pm, qm, lm))
    zero = cp.Constant(np.zeros((n, n)))
    return cp.bmat(
        [
            [vr_m, p_m, -vi_m, -q_m],
            [p_m.T, l_m, q_m.T, zero],
            [vi_m, q_m, vr_m, p_m],
            [-q_m.T, zero, p_m.T, l_m],
        ]
    )


def _add_psd(m: CvxModel, layout: str, diagonal_only: bool) -> None:
    if not diagonal_only:
        _ensure_psd_aux_vars(m)
    bp_pos = m.branch_phase_set.pos
    source = _source_voltage(m, for_current=True)
    record = add_constraint(m, "psd_constraint", [], [])
    for fb, tb in m.branch_set:
        phases = [ph for ph in "abc" if (fb, tb, ph) in bp_pos]
        if not phases:
            continue
        for t in range(len(m.time_set)):
            matrix = _psd_branch_matrix(
                m, fb, tb, phases, t, source, layout, diagonal_only
            )
            record.pieces.append(matrix >> 0)
            record.rows.append((fb, tb))


def add_psd_constraints(m: CvxModel, diagonal_only: bool = False) -> None:
    """PSD relaxation, full layout. ``diagonal_only`` zeroes the v, p, q off-diagonals."""
    _add_psd(m, "full", diagonal_only)


def add_psd_diagonal_constraints(m: CvxModel) -> None:
    """Testing variant of the PSD constraint with zero v, p and q off-diagonals."""
    add_psd_constraints(m, diagonal_only=True)


def add_constraints(
    model: CvxModel,
    circular_constraints: bool = True,
    thermal_constraints: bool = True,
    equality_only: bool = False,
    free_swing_voltage: bool = False,
    linear: bool = True,
    socp_relaxation: bool = False,
    voltage_slacks: bool = False,
    thermal_slacks: bool = False,
    sdp_relaxation: bool = False,
    sdp_diagonal_relaxation: bool = False,
) -> None:
    """Add the network constraints selected by the option flags.

    Always added: ``power_balance_p``, ``power_balance_q``, ``voltage_drop``.

    Current relaxation (only when ``linear=False``; first flag set wins):

    - ``sdp_test1``: ``psd_constraint`` (test1 layout, zero off-diagonals)
    - ``sdp_diagonal_relaxation``: ``psd_constraint`` (full layout, zero off-diagonals)
    - ``sdp_relaxation``: ``psd_constraint`` (full layout)
    - ``socp_relaxation``: convex ``current_constraint``, ``current_sqr_constraint``
    - none: exact (non-convex) ``current_constraint``, ``current_sqr_constraint``

    Swing bus: ``swing_voltage``, or ``swing_voltage_slack_*`` if ``free_swing_voltage``.

    Skipped entirely if ``equality_only``; otherwise:

    - thermal (if ``thermal_constraints``): ``thermal_slack_constraint`` if
      ``thermal_slacks``, else ``thermal_limit_circle`` if ``circular_constraints``,
      else ``thermal_limit_1..8``
    - ``voltage_slack_*`` if ``voltage_slacks``
    - ``voltage_limits``
    """
    add_p_flow_constraints(model, linear=linear)
    add_q_flow_constraints(model, linear=linear)
    add_voltage_drop_constraints(model, linear=linear)

    if not linear:
        if sdp_diagonal_relaxation:
            add_psd_diagonal_constraints(model)
        elif sdp_relaxation:
            add_psd_constraints(model)
        elif socp_relaxation:
            add_current_constraint1_relaxed(model)
            add_current_constraint2_relaxed(model)
        else:
            add_current_constraint1(model)
            add_current_constraint2(model)

    if free_swing_voltage:
        const.add_swing_bus_voltage_slack_constraints(model)
    else:
        const.add_swing_bus_constraints(model)
    if equality_only:
        return

    if thermal_constraints:
        if thermal_slacks:
            const.add_thermal_slack_constraints(model)
        elif circular_constraints:
            const.add_circular_thermal_constraints(model)
        else:
            const.add_octagonal_thermal_constraints(model)
    if voltage_slacks:
        const.add_voltage_slack_constraints(model)
    const.add_voltage_limits(model)
