"""
Constraint functions for DistOPF Pyomo models.

Each function takes a Pyomo ConcreteModel and data, and adds constraints to the model.
Functions are designed to work with models created by create_lindist_model().
"""

import pyomo.environ as pyo  # type: ignore
from distopf.pyomo_models.lindist import ControlVariable
from distopf.pyomo_models.protocol import LindistModelProtocol
from distopf.pyomo_models import common_constraints as const
from numpy import sqrt

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
        if (fb, tb, ph) in m.reg_phase_set:
            return m.v2[tb, ph, t] == m.v2_reg[fb, tb, ph, t] - v_drop
        return m.v2[tb, ph, t] == m.v2[fb, ph, t] - v_drop

    m.voltage_drop = pyo.Constraint(
        m.branch_phase_set, m.time_set, rule=voltage_drop_rule
    )



def add_constraints(
    model: pyo.ConcreteModel,
    circular_constraints: bool = False,
    thermal_constraints: bool = False,
    equality_only: bool = False,
    control_capacitors: bool = False,
    control_regulators: bool = False,
    reg_tap_change_limit: int | None = None,
    free_swing_voltage: bool = False,
    free_boundary_loads: bool = False,
) -> None:
    """Add constraints to a LinDistFlow Pyomo model.

    This is the main constraint builder that replaces multiple model classes.
    Use this for flexible constraint configuration.

    Parameters
    ----------
    model : LindistModelProtocol
        Pyomo ConcreteModel created by create_lindist_model()
    circular_constraints : bool, default True
        If True, use circular (quadratic) constraints for generators, batteries,
        and thermal limits. If False, use octagonal (linear) approximations.
        Circular requires NLP solver (IPOPT), octagonal works with LP/MILP solvers.
    thermal_constraints : bool, default False
        If True, add thermal limit constraints on branch power flows.
    equality_only : bool, default False
        If True, only add equality constraints (power flow, voltage drop, loads).
        Skips voltage limits, thermal limits, and generator capacity limits.
        Useful for penalty-based optimization approaches.
    control_capacitors : bool, default False
        Enable capacitor switching control (adds binary variables and constraints)
    control_regulators : bool, default False
        Enable regulator tap control (adds integer variables and constraints)
    reg_tap_change_limit : int or None, default None
        Max tap change per timestep (only applies if reg_mi=True)
    reg_tap_change_limit : int or None, default None
        Max tap change per timestep (only applies if reg_mi=True)

    Examples
    --------
    >>> model = create_lindist_model(case)
    >>> add_constraints(model)  # Standard constraints (circular, no thermal)

    >>> model = create_lindist_model(case, control_capacitors=True)
    >>> add_constraints(model, cap_mi=True, thermal_constraints=True)
    """

    # Power flow constraints
    add_p_flow_constraints(model)
    add_q_flow_constraints(model)

    # Thermal limits
    if thermal_constraints and not equality_only:
        if circular_constraints:
            const.add_circular_thermal_constraints(model)
        else:
            const.add_octagonal_thermal_constraints(model)

    # Voltage constraints
    if not equality_only:
        const.add_voltage_limits(model)
    add_voltage_drop_constraints(model)
    if free_swing_voltage:
        const.add_swing_bus_voltage_slack_constraints(model)
    else:
        const.add_swing_bus_constraints(model)

    # Loads
    const.add_cvr_load_constraints(model, free_boundary_loads)

    # Capacitors
    if control_capacitors:
        const.add_capacitor_mi_constraints(model)
        const.add_capacitor_mccormick_constraints(model)
        const.add_capacitor_z_bounds(model)
    else:
        const.add_capacitor_constraints(model)

    # Regulators
    if control_regulators:
        const.add_regulator_tap_sos1_constraints(model)
        if reg_tap_change_limit is not None:
            const.add_regulator_tap_change_limit_constraints(
                model, max_tap_change=reg_tap_change_limit
            )
    else:
        const.add_regulator_constraints(model)

    # Generators
    if not equality_only:
        const.add_generator_limits(model)
    const.add_generator_constant_p_constraints_q_control(model)
    const.add_generator_constant_q_constraints_p_control(model)
    if not equality_only:
        if circular_constraints:
            const.add_circular_generator_constraints_pq_control(model)
        else:
            const.add_octagonal_inverter_constraints_pq_control(model)

    # Batteries
    const.add_battery_constant_q_constraints_p_control(model)
    const.add_battery_energy_constraints(model)
    const.add_battery_net_p_bat_equal_phase_constraints(model)
    if not equality_only:
        const.add_battery_power_limits(model)
        const.add_battery_soc_limits(model)
        if circular_constraints:
            const.add_circular_battery_constraints(model)
