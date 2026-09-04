"""
Constraint functions for DistOPF Pyomo models.

Each function takes a Pyomo ConcreteModel and data, and adds constraints to the model.
Functions are designed to work with models created by create_lindist_model().
"""

import pyomo.environ as pyo  # type: ignore
from distopf.pyomo_models.protocol import LindistModelProtocol
from numpy import sqrt

sqrt2 = sqrt(2)
sqrt3 = sqrt(3)


def add_p_flow_constraints(m: LindistModelProtocol) -> None:
    """
    Add LinDistFlow power balance constraints.
    Active power: P_ij = sum(P_jk) + p_L - p_D
    """

    def p_balance_rule(m: LindistModelProtocol, fb, tb, ph, t):
        load = m.p_load[tb, ph, t]
        generation = m.p_gen[tb, ph, t] if (tb, ph, t) in m.p_gen else 0
        p_bat = m.p_bat[tb, ph, t] if (tb, ph, t) in m.p_bat else 0
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
        return incoming_flow == outgoing_flows + load - generation - p_bat

    m.power_balance_p = pyo.Constraint(
        m.branch_phase_set, m.time_set, rule=p_balance_rule
    )


def add_q_flow_constraints(m: LindistModelProtocol) -> None:
    """
    Add LinDistFlow power balance constraints.
    Reactive power: Q_ij = sum(Q_jk) + q_L - q_D - q_C
    """

    def q_balanced_rule(m: LindistModelProtocol, fb, tb, ph, t):
        load = m.q_load[tb, ph, t]
        generation = m.q_gen[tb, ph, t] if (tb, ph, t) in m.q_gen else 0
        q_bat = m.q_bat[tb, ph, t] if (tb, ph, t) in m.q_bat else 0
        capacitor = m.q_cap[tb, ph, t] if (tb, ph, t) in m.q_cap else 0
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
        return incoming_flow == outgoing_flows + load - generation - capacitor - q_bat

    m.power_balance_q = pyo.Constraint(
        m.branch_phase_set, m.time_set, rule=q_balanced_rule
    )


def add_voltage_drop_constraints(m: LindistModelProtocol) -> None:
    """
    Add voltage drop constraints.
    Excludes regulator branches so they can be handled by `add_regulator_constraints`.

    v_j = v_i - sum_q 2*Re[S_ij^pq * (z_ij^pq)*]
    Simplified for LinDistFlow: v_j = v_i - 2*(r*P + x*Q)
    """

    def voltage_drop_rule(m: LindistModelProtocol, fb, tb, ph, t):
        if (fb, tb, ph) in m.reg_phase_set:
            return pyo.Constraint.Skip

        # Triplex (secondary) phases: 2-wire voltage drop, no sqrt(3) cross-terms
        if ph in ("s1", "s2"):
            other = "s2" if ph == "s1" else "s1"
            self_pair = ph + ph  # "s1s1" or "s2s2"
            cross_pair = "s1s2"

            voltage_drop = (
                2 * m.r[fb, tb, self_pair] * m.p_flow[fb, tb, ph, t]
                + 2 * m.x[fb, tb, self_pair] * m.q_flow[fb, tb, ph, t]
            )
            if (fb, tb, other) in m.branch_phase_set:
                voltage_drop += (
                    2 * m.r[fb, tb, cross_pair] * m.p_flow[fb, tb, other, t]
                    + 2 * m.x[fb, tb, cross_pair] * m.q_flow[fb, tb, other, t]
                )

            # For center-tap transformer branches, the from bus has the primary
            # phase (e.g. "a") not the secondary phase.
            primary_ph = m.primary_phase_map.get((fb, tb), None)
            if primary_ph is not None:
                # Center-tap xfmr: reference primary phase voltage at from bus
                return m.v2[tb, ph, t] == m.v2[fb, primary_ph, t] - voltage_drop
            else:
                # Triplex line: same phase on both sides
                return m.v2[tb, ph, t] == m.v2[fb, ph, t] - voltage_drop

        # Standard 3-phase voltage drop
        # here, "a" represents the current phase,
        # b an c represent next and previous phase
        a = ph
        _i = "abc".index(a)
        b = "abc"[(_i + 1) % 3]  # next phase. If phase is "a" then "b"
        c = "abc"[(_i - 1) % 3]  # prev phase. If phase is "a" then "c"
        aa = "".join(sorted(a + a))

        voltage_drop = (
            2 * m.r[fb, tb, aa] * m.p_flow[fb, tb, ph, t]
            + 2 * m.x[fb, tb, aa] * m.q_flow[fb, tb, ph, t]
        )
        if (fb, tb, b) in m.branch_phase_set:
            ab = "".join(sorted(a + b))
            voltage_drop += (-m.r[fb, tb, ab] + sqrt3 * m.x[fb, tb, ab]) * m.p_flow[
                fb, tb, b, t
            ]
            voltage_drop += (-m.x[fb, tb, ab] - sqrt3 * m.r[fb, tb, ab]) * m.q_flow[
                fb, tb, b, t
            ]
        if (fb, tb, c) in m.branch_phase_set:
            ac = "".join(sorted(a + c))
            voltage_drop += (-m.r[fb, tb, ac] - sqrt3 * m.x[fb, tb, ac]) * m.p_flow[
                fb, tb, c, t
            ]
            voltage_drop += (-m.x[fb, tb, ac] + sqrt3 * m.r[fb, tb, ac]) * m.q_flow[
                fb, tb, c, t
            ]

        return m.v2[tb, ph, t] == m.v2[fb, ph, t] - voltage_drop

    m.voltage_drop = pyo.Constraint(
        m.branch_phase_set, m.time_set, rule=voltage_drop_rule
    )


def add_swing_bus_constraints(m: LindistModelProtocol) -> None:
    """
    Add swing bus voltage constraints.

    Sets voltage at swing bus to specified values.
    """

    def swing_voltage_rule(m: LindistModelProtocol, _id, ph, t):
        """Fix swing bus voltages.

        `m.v_swing` is stored as voltage magnitude (p.u.), while `m.v2` is
        voltage magnitude squared.
        """
        if _id not in m.swing_bus_set:
            return pyo.Constraint.Skip
        return m.v2[_id, ph, t] == m.v_swing[_id, ph, t] ** 2

    m.swing_voltage = pyo.Constraint(
        m.swing_phase_set, m.time_set, rule=swing_voltage_rule
    )


def add_voltage_limits(m: LindistModelProtocol) -> None:
    """Add voltage bounds (for voltage magnitude squared)"""

    def voltage_limits(m: LindistModelProtocol, _id, ph, t):
        return (m.v_min[_id, ph] ** 2, m.v2[_id, ph, t], m.v_max[_id, ph] ** 2)

    m.voltage_limits = pyo.Constraint(m.bus_phase_set, m.time_set, rule=voltage_limits)
