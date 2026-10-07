"""
Constraint functions for DistOPF Pyomo models.

Each function takes a Pyomo ConcreteModel and data, and adds constraints to the model.
Functions are designed to work with models created by create_nl_branchflow_model().
"""

from itertools import combinations
import pyomo.environ as pyo  # type: ignore
from distopf.pyomo_models.common.data import parse_phases
from distopf.pyomo_models.common.protocol import LindistModelProtocol
from distopf.pyomo_models.common import common_constraints as const
from math import sqrt

sqrt2 = sqrt(2)
sqrt3 = sqrt(3)


def _voltage_drop_triplex_terms(m: LindistModelProtocol, fb, tb, ph, t):
    # Triplex (secondary) phases: 2-wire voltage drop, no sqrt(3) cross-terms
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


def _voltage_drop_term1(m: LindistModelProtocol, fb, tb, ph, t):
    # Standard 3-phase voltage drop
    # here, "a" represents the current phase,
    # b an c represent next and previous phase
    a = ph
    _i = "abc".index(a)
    b = "abc"[(_i + 1) % 3]  # next phase. If phase is "a" then "b"
    c = "abc"[(_i - 1) % 3]  # prev phase. If phase is "a" then "c"
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


def _voltage_drop_term2(m: LindistModelProtocol, fb, tb, ph, t):
    voltage_drop_term2 = sum(
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
    return voltage_drop_term2


def _voltage_drop_term3(m: LindistModelProtocol, fb, tb, ph, t):
    voltage_drop_term3 = sum(
        [
            2
            * m.l_flow[fb, tb, "".join(sorted(q1 + q2)), t]
            * (
                pyo.cos(m.d[fb, tb, q1 + q2])
                * m.r[fb, tb, "".join(sorted(ph + q1))]
                * m.r[fb, tb, "".join(sorted(ph + q2))]
                + pyo.cos(m.d[fb, tb, q1 + q2])
                * m.x[fb, tb, "".join(sorted(ph + q1))]
                * m.x[fb, tb, "".join(sorted(ph + q2))]
                + pyo.sin(m.d[fb, tb, q1 + q2])
                * m.r[fb, tb, "".join(sorted(ph + q1))]
                * m.x[fb, tb, "".join(sorted(ph + q2))]
                - pyo.sin(m.d[fb, tb, q1 + q2])
                * m.x[fb, tb, "".join(sorted(ph + q1))]
                * m.r[fb, tb, "".join(sorted(ph + q2))]
            )
            # for q1, q2 in product("abc", repeat=2)
            for q1, q2 in combinations("abc", 2)
            if q1 != q2
            and q1 in m.phase_map[tb]
            and q2 in m.phase_map[tb]
            and (fb, tb, "".join(sorted(q1 + q2))) in m.branch_phase_pair_set
        ]
    )
    return voltage_drop_term3


def _active_power_loss(m: LindistModelProtocol, fb, tb, ph, t):
    _loss_list = []
    for ph2 in m.phase_map[tb]:
        if (fb, tb, "".join(sorted([ph, ph2]))) not in m.branch_phase_pair_set:
            continue
        l_flow = m.l_flow[fb, tb, "".join(sorted([ph, ph2])), t]
        r = m.r[fb, tb, "".join(sorted([ph, ph2]))]
        x = m.x[fb, tb, "".join(sorted([ph, ph2]))]
        angle = m.d[fb, tb, ph2 + ph]
        _loss_list.append(l_flow * (r * pyo.cos(angle) - x * pyo.sin(angle)))
    return sum(_loss_list)


def _reactive_power_loss(m: LindistModelProtocol, fb, tb, ph, t):
    _loss_list = []
    for ph2 in m.phase_map[tb]:
        if (fb, tb, "".join(sorted([ph, ph2]))) not in m.branch_phase_pair_set:
            continue
        l_flow = m.l_flow[fb, tb, "".join(sorted([ph, ph2])), t]
        r = m.r[fb, tb, "".join(sorted([ph, ph2]))]
        x = m.x[fb, tb, "".join(sorted([ph, ph2]))]
        angle = m.d[fb, tb, ph2 + ph]
        _loss_list.append(l_flow * (x * pyo.cos(angle) + r * pyo.sin(angle)))
    return sum(_loss_list)


def add_p_flow_constraints(m: LindistModelProtocol, linear=True) -> None:
    """Add active-power balance from the signed injection registry."""
    registry = getattr(m, "_device_registry", None)
    if registry is None:
        raise RuntimeError("Device registry is not installed.")

    def p_balance_rule(m: LindistModelProtocol, fb, tb, ph, t):
        incoming_flow = m.p_flow[fb, tb, ph, t]
        outgoing_flows = sum(
            m.p_flow[fb2, tb2, ph, t]
            for fb2, tb2 in m.to_bus_map[tb]
            if (fb2, tb2, ph) in m.branch_phase_set
        )
        # Center-tap transformer: primary phase feeds secondary s1+s2 flows
        if ph not in ("s1", "s2"):
            for fb2, tb2 in m.to_bus_map[tb]:
                if getattr(m, "primary_phase_map", {}).get((fb2, tb2)) == ph:
                    for sec_ph in ("s1", "s2"):
                        if (fb2, tb2, sec_ph) in m.branch_phase_set:
                            outgoing_flows += m.p_flow[fb2, tb2, sec_ph, t]
        loss = 0
        if not linear:
            loss = _active_power_loss(m, fb, tb, ph, t)

        return incoming_flow - loss == outgoing_flows - registry.active_power_injection(
            m, tb, ph, t
        )

    m.power_balance_p = pyo.Constraint(
        m.branch_phase_set, m.time_set, rule=p_balance_rule
    )


def add_q_flow_constraints(m: LindistModelProtocol, linear=True) -> None:
    """Add reactive-power balance from the signed injection registry."""
    registry = getattr(m, "_device_registry", None)
    if registry is None:
        raise RuntimeError("Device registry is not installed.")

    def q_balance_rule(m: LindistModelProtocol, fb, tb, ph, t):
        incoming_flow = m.q_flow[fb, tb, ph, t]
        outgoing_flows = sum(
            m.q_flow[fb2, tb2, ph, t]
            for fb2, tb2 in m.to_bus_map[tb]
            if (fb2, tb2, ph) in m.branch_phase_set
        )

        # Center-tap transformer: primary phase feeds secondary s1+s2 flows
        if ph not in ("s1", "s2"):
            for fb2, tb2 in m.to_bus_map[tb]:
                if getattr(m, "primary_phase_map", {}).get((fb2, tb2)) == ph:
                    for sec_ph in ("s1", "s2"):
                        if (fb2, tb2, sec_ph) in m.branch_phase_set:
                            outgoing_flows += m.q_flow[fb2, tb2, sec_ph, t]
        loss = 0
        if not linear:
            loss = _reactive_power_loss(m, fb, tb, ph, t)
        return (
            incoming_flow - loss
            == outgoing_flows - registry.reactive_power_injection(m, tb, ph, t)
        )

    m.power_balance_q = pyo.Constraint(
        m.branch_phase_set, m.time_set, rule=q_balance_rule
    )


def add_voltage_drop_constraints(m: LindistModelProtocol, linear=True) -> None:
    """
    Add voltage drop constraints.
    Excludes regulator branches so they can be handled by `add_regulator_constraints`.

    v_j = v_i - sum_q 2*Re[S_ij^pq * (z_ij^pq)*]
    Simplified for LinDistFlow: v_j = v_i - 2*(r*P + x*Q)
    """
    regs_defined = hasattr(m, "reg_phase_set") and hasattr(m, "v2_reg")

    def voltage_drop_rule(m: LindistModelProtocol, fb, tb, ph, t):
        # Triplex (secondary) phases: 2-wire voltage drop, no sqrt(3) cross-terms
        if ph in ("s1", "s2"):
            v_drop_triplex = _voltage_drop_triplex_terms(m, fb, tb, ph, t)
            # For center-tap transformer branches, the from bus has the primary
            # phase (e.g. "a") not the secondary phase. If a primary phase exists use
            # that, else use `ph`.
            from_ph = m.primary_phase_map.get((fb, tb), ph)
            # Triplex line: same phase on both sides
            return m.v2[tb, ph, t] == m.v2[fb, from_ph, t] - v_drop_triplex

        v_drop = _voltage_drop_term1(m, fb, tb, ph, t)
        if not linear:
            v_drop += _voltage_drop_term2(m, fb, tb, ph, t)
            v_drop += _voltage_drop_term3(m, fb, tb, ph, t)
        if regs_defined and (fb, tb, ph) in m.reg_phase_set:
            return m.v2[tb, ph, t] == m.v2_reg[fb, tb, ph, t] - v_drop
        return m.v2[tb, ph, t] == m.v2[fb, ph, t] - v_drop

    m.voltage_drop = pyo.Constraint(
        m.branch_phase_set, m.time_set, rule=voltage_drop_rule
    )


def add_current_constraint1(m: LindistModelProtocol) -> None:
    regs_defined = hasattr(m, "reg_phase_set") and hasattr(m, "v2_reg")

    def _rule1(m: LindistModelProtocol, fb, tb, ph, t):
        # Use v2_reg for regulator branches
        if regs_defined and (fb, tb, ph) in m.reg_phase_set:
            return (
                m.p_flow[fb, tb, ph, t] ** 2 + m.q_flow[fb, tb, ph, t] ** 2
                == m.v2_reg[fb, tb, ph, t] * m.l_flow[fb, tb, ph + ph, t]
            )
        fb_ph = getattr(m, "primary_phase_map", {}).get((fb, tb), ph)
        return (
            m.p_flow[fb, tb, ph, t] ** 2 + m.q_flow[fb, tb, ph, t] ** 2
            == m.v2[fb, fb_ph, t] * m.l_flow[fb, tb, ph + ph, t]
        )

    m.current_constraint = pyo.Constraint(m.branch_phase_set, m.time_set, rule=_rule1)


def add_current_constraint2(m: LindistModelProtocol) -> None:
    def _rule2(m: LindistModelProtocol, fb, tb, phases, t):
        ph1 = parse_phases(phases)[0]
        ph2 = parse_phases(phases)[1]
        if ph1 == ph2:
            return pyo.Constraint.Skip
        return (
            m.l_flow[fb, tb, ph1 + ph2, t] ** 2
            == m.l_flow[fb, tb, ph1 + ph1, t] * m.l_flow[fb, tb, ph2 + ph2, t]
        )

    m.current_sqr_constraint = pyo.Constraint(
        m.branch_phase_pair_set, m.time_set, rule=_rule2
    )


def add_current_constraint1_relaxed(m: LindistModelProtocol) -> None:
    regs_defined = hasattr(m, "reg_phase_set") and hasattr(m, "v2_reg")

    def _rule1(m: LindistModelProtocol, fb, tb, ph, t):
        # Use v2_reg for regulator branches
        if regs_defined and (fb, tb, ph) in m.reg_phase_set:
            return (
                m.p_flow[fb, tb, ph, t] ** 2 + m.q_flow[fb, tb, ph, t] ** 2
                <= m.v2_reg[fb, tb, ph, t] * m.l_flow[fb, tb, ph + ph, t]
            )
        fb_ph = getattr(m, "primary_phase_map", {}).get((fb, tb), ph)
        return (
            m.p_flow[fb, tb, ph, t] ** 2 + m.q_flow[fb, tb, ph, t] ** 2
            <= m.v2[fb, fb_ph, t] * m.l_flow[fb, tb, ph + ph, t]
        )

    m.current_constraint = pyo.Constraint(m.branch_phase_set, m.time_set, rule=_rule1)


def add_current_constraint2_relaxed(m: LindistModelProtocol) -> None:
    def _rule2(m: LindistModelProtocol, fb, tb, phases, t):
        ph1 = parse_phases(phases)[0]
        ph2 = parse_phases(phases)[1]
        if ph1 == ph2:
            return pyo.Constraint.Skip
        return (
            m.l_flow[fb, tb, ph1 + ph2, t] ** 2
            <= m.l_flow[fb, tb, ph1 + ph1, t] * m.l_flow[fb, tb, ph2 + ph2, t]
        )

    m.current_sqr_constraint = pyo.Constraint(
        m.branch_phase_pair_set, m.time_set, rule=_rule2
    )


def add_constraints(
    model: LindistModelProtocol,
    circular_constraints: bool = True,
    thermal_constraints: bool = True,
    equality_only: bool = False,
    free_swing_voltage: bool = False,
    linear: bool = False,
    socp_relaxation: bool = False,
    voltage_slacks: bool = False,
    thermal_slacks: bool = False,
) -> None:
    # Power flow constraints
    add_p_flow_constraints(model, linear=linear)
    add_q_flow_constraints(model, linear=linear)
    add_voltage_drop_constraints(model, linear=linear)

    # Current constraints
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
    # Thermal limits
    if thermal_constraints and not equality_only:
        if thermal_slacks:
            const.add_thermal_slack_constraints(model)
        elif circular_constraints:
            const.add_circular_thermal_constraints(model)
        else:
            const.add_octagonal_thermal_constraints(model)

    # Voltage limits
    if voltage_slacks:
        const.add_voltage_slack_constraints(model)
    const.add_voltage_limits(model)
