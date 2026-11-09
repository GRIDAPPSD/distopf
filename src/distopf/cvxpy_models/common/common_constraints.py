"""Network-wide constraints (CVXPY).

Mirror of ``distopf.pyomo_models.common.common_constraints``. Every function
takes a ``CvxModel`` and adds named constraint containers to it.
"""

from __future__ import annotations

from math import sqrt

from distopf.cvxpy_models.common.model import SKIP, CvxModel, VarDict, add_constraint, index_product

sqrt2 = sqrt(2)
sqrt3 = sqrt(3)


def add_voltage_limits(m: CvxModel) -> None:
    """Add voltage bounds (for voltage magnitude squared)."""

    def voltage_limits(m: CvxModel, _id, ph, t):
        return (m.v_min[_id, ph] ** 2, m.v2[_id, ph, t], m.v_max[_id, ph] ** 2)

    add_constraint(m, "voltage_limits", (m.bus_phase_set, m.time_set), voltage_limits)


# ============ Forwarders to device modules ============================================


def add_cvr_load_constraints(m: CvxModel, free_boundary_loads: bool = False) -> None:
    from distopf.cvxpy_models.devices.load import add_cvr_load_constraints as add

    add(m, free_boundary_loads)


def add_generator_limits(m: CvxModel) -> None:
    from distopf.cvxpy_models.devices.generator import add_gen_limits

    add_gen_limits(m)


def add_octagonal_inverter_constraints_pq_control(m: CvxModel) -> None:
    from distopf.cvxpy_models.devices.generator import add_gen_octagon_constraints

    add_gen_octagon_constraints(m)


def add_circular_generator_constraints_pq_control(m: CvxModel) -> None:
    from distopf.cvxpy_models.devices.generator import add_gen_circle_constraints

    add_gen_circle_constraints(m)


def add_capacitor_constraints(m: CvxModel) -> None:
    from distopf.cvxpy_models.devices.capacitor import add_capacitor_constraints as add

    add(m)


def add_capacitor_mi_constraints(m: CvxModel) -> None:
    from distopf.cvxpy_models.devices.capacitor import (
        add_capacitor_mi_constraints as add,
    )

    add(m)


def add_capacitor_mccormick_constraints(m: CvxModel) -> None:
    from distopf.cvxpy_models.devices.capacitor import (
        add_capacitor_mccormick_constraints as add,
    )

    add(m)


def add_capacitor_z_bounds(m: CvxModel) -> None:
    from distopf.cvxpy_models.devices.capacitor import add_capacitor_z_bounds as add

    add(m)


def add_capacitor_constraints_auto(m: CvxModel) -> None:
    """Add McCormick constraints if ``cap_mi_enabled`` else the standard model."""
    if getattr(m, "cap_mi_enabled", False):
        add_capacitor_mi_constraints(m)
        add_capacitor_mccormick_constraints(m)
        add_capacitor_z_bounds(m)
    else:
        add_capacitor_constraints(m)


def add_swing_bus_constraints(m: CvxModel) -> None:
    """Fix swing bus voltages. ``v_swing`` is a magnitude; ``v2`` is squared."""

    def swing_voltage_rule(m: CvxModel, _id, ph, t):
        if _id not in m.swing_bus_set:
            return SKIP
        return m.v2[_id, ph, t] == m.v_swing[_id, ph, t] ** 2

    add_constraint(
        m, "swing_voltage", (m.swing_phase_set, m.time_set), swing_voltage_rule
    )


# ============ Thermal Line Constraints ================================================


def _has_thermal_limit(m: CvxModel, fb, tb, ph) -> bool:
    if not hasattr(m, "s_branch_max"):
        return False
    limit = m.s_branch_max.get((fb, tb, ph), None)
    return limit is not None and limit > 0


def add_octagonal_thermal_constraints(m: CvxModel) -> None:
    """Eight linear inequalities approximating ``|S_ij| <= S_max``."""
    if not hasattr(m, "s_branch_max"):
        return

    c = sqrt2 - 1

    coefficients = (
        (c, 1.0),
        (1.0, c),
        (1.0, -c),
        (c, -1.0),
        (-c, -1.0),
        (-1.0, -c),
        (-1.0, c),
        (-c, 1.0),
    )

    def make_rule(a: float, b: float):
        def rule(m: CvxModel, fb, tb, ph, t):
            if not _has_thermal_limit(m, fb, tb, ph):
                return SKIP
            return (
                a * m.p_flow[fb, tb, ph, t] + b * m.q_flow[fb, tb, ph, t]
                <= m.s_branch_max[fb, tb, ph]
            )

        return rule

    for i, (a, b) in enumerate(coefficients, start=1):
        add_constraint(
            m,
            f"thermal_limit_{i}",
            (m.branch_phase_set, m.time_set),
            make_rule(a, b),
        )


def add_circular_thermal_constraints(m: CvxModel) -> None:
    """Exact quadratic limit ``P^2 + Q^2 <= S_max^2`` (a convex cone constraint)."""
    import cvxpy as cp

    if not hasattr(m, "s_branch_max"):
        return

    def thermal_circle(m: CvxModel, fb, tb, ph, t):
        if not _has_thermal_limit(m, fb, tb, ph):
            return SKIP
        return (
            cp.norm(cp.hstack([m.p_flow[fb, tb, ph, t], m.q_flow[fb, tb, ph, t]]), 2)
            <= m.s_branch_max[fb, tb, ph]
        )

    add_constraint(
        m, "thermal_limit_circle", (m.branch_phase_set, m.time_set), thermal_circle
    )


# ============ Linear Slack Constraints ================================================


def add_thermal_slack_constraints(m: CvxModel, derate_factor=1) -> None:
    """Soft thermal limit: ``P_flow <= S_max * derate_factor + slack``.

    The Pyomo version of this function indexes ``s_branch_max`` and ``p_flow``
    with two-element keys on a three-element set and is broken for the current
    indexing. This version uses the intended ``(fb, tb, phase, t)`` keys.
    """
    if not hasattr(m, "s_branch_max"):
        return

    m.thermal_slack = VarDict(
        index_product(m.branch_phase_set, m.time_set), nonneg=True, initialize=0.0
    )

    def thermal_slack_rule(m: CvxModel, fb, tb, ph, t):
        if not _has_thermal_limit(m, fb, tb, ph):
            return SKIP
        return (
            m.p_flow[fb, tb, ph, t]
            <= m.s_branch_max[fb, tb, ph] * derate_factor + m.thermal_slack[fb, tb, ph, t]
        )

    add_constraint(
        m,
        "thermal_slack_constraint",
        (m.branch_phase_set, m.time_set),
        thermal_slack_rule,
    )


def add_voltage_slack_constraints(m: CvxModel) -> None:
    """Soft voltage bounds using a single nonnegative slack per bus/phase/time."""
    m.v2_slack = VarDict(
        index_product(m.bus_phase_set, m.time_set), nonneg=True, initialize=0.0
    )

    def voltage_slack_under_rule(m: CvxModel, _id, ph, t):
        return m.v2[_id, ph, t] >= m.v_min[_id, ph] ** 2 - m.v2_slack[_id, ph, t]

    def voltage_slack_over_rule(m: CvxModel, _id, ph, t):
        return m.v2[_id, ph, t] <= m.v_max[_id, ph] ** 2 + m.v2_slack[_id, ph, t]

    add_constraint(
        m,
        "voltage_slack_under",
        (m.bus_phase_set, m.time_set),
        voltage_slack_under_rule,
    )
    add_constraint(
        m, "voltage_slack_over", (m.bus_phase_set, m.time_set), voltage_slack_over_rule
    )


def add_swing_bus_voltage_slack_constraints(m: CvxModel) -> None:
    """Soft swing-bus voltage: ``v2`` within ``v_swing^2`` plus or minus a slack."""
    m.swing_v2_slack = VarDict(
        index_product(m.swing_phase_set, m.time_set), nonneg=True, initialize=0.0
    )

    def swing_voltage_slack_under_rule(m: CvxModel, _id, ph, t):
        return (
            m.v2[_id, ph, t]
            >= m.v_swing[_id, ph, t] ** 2 - m.swing_v2_slack[_id, ph, t]
        )

    def swing_voltage_slack_over_rule(m: CvxModel, _id, ph, t):
        return (
            m.v2[_id, ph, t]
            <= m.v_swing[_id, ph, t] ** 2 + m.swing_v2_slack[_id, ph, t]
        )

    add_constraint(
        m,
        "swing_voltage_slack_under",
        (m.swing_phase_set, m.time_set),
        swing_voltage_slack_under_rule,
    )
    add_constraint(
        m,
        "swing_voltage_slack_over",
        (m.swing_phase_set, m.time_set),
        swing_voltage_slack_over_rule,
    )
