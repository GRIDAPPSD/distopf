"""Branch-flow and LinDistFlow network constraints (CVXPY).

Mirror of ``distopf.pyomo_models.network.bfm_constraints``. Constraint names
and indices are identical to the Pyomo version so results and duals can be
compared one-to-one.

Convexity notes
---------------
- LinDistFlow constraints are affine.
- ``current_constraint`` and ``current_sqr_constraint`` with ``==`` are
  non-convex and are only built so that the model structure matches Pyomo;
  they cannot be solved with a DCP solver. Use ``socp_relaxation=True`` (the
  ``<=`` form, expressed as second-order cone constraints) for a convex model.
- Loss terms multiply ``l_flow`` by ``cp.Parameter`` objects holding
  ``cos(d)`` and ``sin(d)``, so they stay affine in the variables.
"""

from __future__ import annotations

from itertools import combinations
from math import sqrt

import cvxpy as cp

from distopf.cvxpy_models.common import common_constraints as const
from distopf.cvxpy_models.common.model import SKIP, CvxModel, add_constraint
from distopf.pyomo_models.common.data import parse_phases

sqrt2 = sqrt(2)
sqrt3 = sqrt(3)


def _voltage_drop_triplex_terms(m: CvxModel, fb, tb, ph, t):
    if ph not in ("s1", "s2"):
        return
    other = "s2" if ph == "s1" else "s1"
    r11 = m.r[fb, tb, ph + ph]
    x11 = m.x[fb, tb, ph + ph]
    dv = 2 * r11 * m.p_flow[fb, tb, ph, t] + 2 * x11 * m.q_flow[fb, tb, ph, t]
    if (fb, tb, other) not in m.branch_phase_set:
        return dv
    r12 = m.r[fb, tb, "s1s2"]
    x12 = m.x[fb, tb, "s1s2"]
    dvx = 2 * r12 * m.p_flow[fb, tb, other, t] + 2 * x12 * m.q_flow[fb, tb, other, t]
    return dv + dvx


def _voltage_drop_term1(m: CvxModel, fb, tb, ph, t):
    a = ph
    _i = "abc".index(a)
    b = "abc"[(_i + 1) % 3]
    c = "abc"[(_i - 1) % 3]
    aa = "".join(sorted(a + a))
    raa = m.r[fb, tb, aa]
    xaa = m.x[fb, tb, aa]
    voltage_drop = 2 * raa * m.p_flow[fb, tb, ph, t] + 2 * xaa * m.q_flow[fb, tb, ph, t]

    if (fb, tb, b) in m.branch_phase_set:
        ab = "".join(sorted(a + b))
        rab = m.r[fb, tb, ab]
        xab = m.x[fb, tb, ab]
        voltage_drop += (-rab + sqrt3 * xab) * m.p_flow[fb, tb, b, t]
        voltage_drop += (-xab - sqrt3 * rab) * m.q_flow[fb, tb, b, t]
    if (fb, tb, c) in m.branch_phase_set:
        ac = "".join(sorted(a + c))
        rac = m.r[fb, tb, ac]
        xac = m.x[fb, tb, ac]
        voltage_drop += (-rac - sqrt3 * xac) * m.p_flow[fb, tb, c, t]
        voltage_drop += (-xac + sqrt3 * rac) * m.q_flow[fb, tb, c, t]
    return voltage_drop


def _voltage_drop_term2(m: CvxModel, fb, tb, ph, t):
    return sum(
        [
            (
                m.r[fb, tb, "".join(sorted(ph + ph2))] ** 2
                + m.x[fb, tb, "".join(sorted(ph + ph2))] ** 2
            )
            * m.l_flow[fb, tb, ph2 + ph2, t]
            for ph2 in m.phase_map[tb]
            if (fb, tb, ph2 + ph2) in m.branch_phase_pair_set
        ]
    )


def _voltage_drop_term3(m: CvxModel, fb, tb, ph, t):
    terms = []
    for q1, q2 in combinations("abc", 2):
        pair = "".join(sorted(q1 + q2))
        if not (
            q1 in m.phase_map[tb]
            and q2 in m.phase_map[tb]
            and (fb, tb, pair) in m.branch_phase_pair_set
        ):
            continue
        cos_d = m.cos_d[fb, tb, q1 + q2]
        sin_d = m.sin_d[fb, tb, q1 + q2]
        r1 = m.r[fb, tb, "".join(sorted(ph + q1))]
        r2 = m.r[fb, tb, "".join(sorted(ph + q2))]
        x1 = m.x[fb, tb, "".join(sorted(ph + q1))]
        x2 = m.x[fb, tb, "".join(sorted(ph + q2))]
        coefficient = cos_d * (r1 * r2 + x1 * x2) + sin_d * (r1 * x2 - x1 * r2)
        terms.append(2 * m.l_flow[fb, tb, pair, t] * coefficient)
    return sum(terms)


def _active_power_loss(m: CvxModel, fb, tb, ph, t):
    terms = []
    for ph2 in m.phase_map[tb]:
        pair = "".join(sorted([ph, ph2]))
        if (fb, tb, pair) not in m.branch_phase_pair_set:
            continue
        l_flow = m.l_flow[fb, tb, pair, t]
        r = m.r[fb, tb, pair]
        x = m.x[fb, tb, pair]
        key = (fb, tb, ph2 + ph)
        terms.append(l_flow * (r * m.cos_d[key] - x * m.sin_d[key]))
    return sum(terms)


def _reactive_power_loss(m: CvxModel, fb, tb, ph, t):
    terms = []
    for ph2 in m.phase_map[tb]:
        pair = "".join(sorted([ph, ph2]))
        if (fb, tb, pair) not in m.branch_phase_pair_set:
            continue
        l_flow = m.l_flow[fb, tb, pair, t]
        r = m.r[fb, tb, pair]
        x = m.x[fb, tb, pair]
        key = (fb, tb, ph2 + ph)
        terms.append(l_flow * (x * m.cos_d[key] + r * m.sin_d[key]))
    return sum(terms)


def _secondary_outgoing(m: CvxModel, flow, tb, ph, t):
    """Flows leaving bus ``tb`` on phase ``ph``, including center-tap secondaries."""
    outgoing = sum(
        flow[fb2, tb2, ph, t]
        for fb2, tb2 in m.to_bus_map[tb]
        if (fb2, tb2, ph) in m.branch_phase_set
    )
    if ph not in ("s1", "s2"):
        for fb2, tb2 in m.to_bus_map[tb]:
            if getattr(m, "primary_phase_map", {}).get((fb2, tb2)) == ph:
                for sec_ph in ("s1", "s2"):
                    if (fb2, tb2, sec_ph) in m.branch_phase_set:
                        outgoing = outgoing + flow[fb2, tb2, sec_ph, t]
    return outgoing


def add_p_flow_constraints(m: CvxModel, linear=True) -> None:
    """Add active-power balance from the signed injection registry."""
    registry = getattr(m, "_device_registry", None)
    if registry is None:
        raise RuntimeError("Device registry is not installed.")

    def p_balance_rule(m: CvxModel, fb, tb, ph, t):
        incoming_flow = m.p_flow[fb, tb, ph, t]
        outgoing_flows = _secondary_outgoing(m, m.p_flow, tb, ph, t)
        loss = 0
        if not linear:
            loss = _active_power_loss(m, fb, tb, ph, t)
        return incoming_flow - loss == outgoing_flows - registry.active_power_injection(
            m, tb, ph, t
        )

    add_constraint(m, "power_balance_p", (m.branch_phase_set, m.time_set), p_balance_rule)


def add_q_flow_constraints(m: CvxModel, linear=True) -> None:
    """Add reactive-power balance from the signed injection registry."""
    registry = getattr(m, "_device_registry", None)
    if registry is None:
        raise RuntimeError("Device registry is not installed.")

    def q_balance_rule(m: CvxModel, fb, tb, ph, t):
        incoming_flow = m.q_flow[fb, tb, ph, t]
        outgoing_flows = _secondary_outgoing(m, m.q_flow, tb, ph, t)
        loss = 0
        if not linear:
            loss = _reactive_power_loss(m, fb, tb, ph, t)
        return (
            incoming_flow - loss
            == outgoing_flows - registry.reactive_power_injection(m, tb, ph, t)
        )

    add_constraint(m, "power_balance_q", (m.branch_phase_set, m.time_set), q_balance_rule)


def add_voltage_drop_constraints(m: CvxModel, linear=True) -> None:
    """Add voltage drop constraints.

    Regulator branches use ``v2_reg`` in place of the from-bus ``v2``.
    """
    regs_defined = hasattr(m, "reg_phase_set") and hasattr(m, "v2_reg")

    def voltage_drop_rule(m: CvxModel, fb, tb, ph, t):
        if ph in ("s1", "s2"):
            v_drop_triplex = _voltage_drop_triplex_terms(m, fb, tb, ph, t)
            from_ph = m.primary_phase_map.get((fb, tb), ph)
            return m.v2[tb, ph, t] == m.v2[fb, from_ph, t] - v_drop_triplex

        v_drop = _voltage_drop_term1(m, fb, tb, ph, t)
        if not linear:
            v_drop = v_drop + _voltage_drop_term2(m, fb, tb, ph, t)
            v_drop = v_drop + _voltage_drop_term3(m, fb, tb, ph, t)
        if regs_defined and (fb, tb, ph) in m.reg_phase_set:
            return m.v2[tb, ph, t] == m.v2_reg[fb, tb, ph, t] - v_drop
        return m.v2[tb, ph, t] == m.v2[fb, ph, t] - v_drop

    add_constraint(
        m, "voltage_drop", (m.branch_phase_set, m.time_set), voltage_drop_rule
    )


def _source_voltage(m: CvxModel, fb, tb, ph, t):
    regs_defined = hasattr(m, "reg_phase_set") and hasattr(m, "v2_reg")
    if regs_defined and (fb, tb, ph) in m.reg_phase_set:
        return m.v2_reg[fb, tb, ph, t]
    fb_ph = getattr(m, "primary_phase_map", {}).get((fb, tb), ph)
    return m.v2[fb, fb_ph, t]


def add_current_constraint1(m: CvxModel) -> None:
    """``P^2 + Q^2 == v2 * l`` (non-convex; not solvable by DCP solvers)."""

    def _rule1(m: CvxModel, fb, tb, ph, t):
        return (
            m.p_flow[fb, tb, ph, t] ** 2 + m.q_flow[fb, tb, ph, t] ** 2
            == _source_voltage(m, fb, tb, ph, t) * m.l_flow[fb, tb, ph + ph, t]
        )

    add_constraint(m, "current_constraint", (m.branch_phase_set, m.time_set), _rule1)


def add_current_constraint2(m: CvxModel) -> None:
    """``l_ab^2 == l_aa * l_bb`` (non-convex; not solvable by DCP solvers)."""

    def _rule2(m: CvxModel, fb, tb, phases, t):
        ph1, ph2 = parse_phases(phases)[:2]
        if ph1 == ph2:
            return SKIP
        return (
            m.l_flow[fb, tb, ph1 + ph2, t] ** 2
            == m.l_flow[fb, tb, ph1 + ph1, t] * m.l_flow[fb, tb, ph2 + ph2, t]
        )

    add_constraint(
        m, "current_sqr_constraint", (m.branch_phase_pair_set, m.time_set), _rule2
    )


def add_current_constraint1_relaxed(m: CvxModel) -> None:
    """``P^2 + Q^2 <= v2 * l`` as a rotated second-order cone constraint."""

    def _rule1(m: CvxModel, fb, tb, ph, t):
        v2 = _source_voltage(m, fb, tb, ph, t)
        l_flow = m.l_flow[fb, tb, ph + ph, t]
        p = m.p_flow[fb, tb, ph, t]
        q = m.q_flow[fb, tb, ph, t]
        return cp.SOC(v2 + l_flow, cp.hstack([2 * p, 2 * q, v2 - l_flow]))

    add_constraint(m, "current_constraint", (m.branch_phase_set, m.time_set), _rule1)


def add_current_constraint2_relaxed(m: CvxModel) -> None:
    """``l_ab^2 <= l_aa * l_bb`` as a rotated second-order cone constraint."""

    def _rule2(m: CvxModel, fb, tb, phases, t):
        ph1, ph2 = parse_phases(phases)[:2]
        if ph1 == ph2:
            return SKIP
        l_ab = m.l_flow[fb, tb, ph1 + ph2, t]
        l_aa = m.l_flow[fb, tb, ph1 + ph1, t]
        l_bb = m.l_flow[fb, tb, ph2 + ph2, t]
        return cp.SOC(l_aa + l_bb, cp.hstack([2 * l_ab, l_aa - l_bb]))

    add_constraint(
        m, "current_sqr_constraint", (m.branch_phase_pair_set, m.time_set), _rule2
    )


def add_constraints(
    model: CvxModel,
    circular_constraints: bool = True,
    thermal_constraints: bool = True,
    equality_only: bool = False,
    free_swing_voltage: bool = False,
    linear: bool = False,
    socp_relaxation: bool = False,
    voltage_slacks: bool = False,
    thermal_slacks: bool = False,
) -> None:
    add_p_flow_constraints(model, linear=linear)
    add_q_flow_constraints(model, linear=linear)
    add_voltage_drop_constraints(model, linear=linear)

    if not linear:
        if socp_relaxation:
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
    if thermal_constraints and not equality_only:
        if thermal_slacks:
            const.add_thermal_slack_constraints(model)
        elif circular_constraints:
            const.add_circular_thermal_constraints(model)
        else:
            const.add_octagonal_thermal_constraints(model)

    if voltage_slacks:
        const.add_voltage_slack_constraints(model)
    const.add_voltage_limits(model)
