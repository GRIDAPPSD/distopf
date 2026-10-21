"""Network-wide constraints (vectorized CVXPY).

Vectorized counterpart of ``distopf.cvxpy_models.common.common_constraints``.
"""

from __future__ import annotations

from math import sqrt

import cvxpy as cp
import numpy as np

from distopf.cvxpy_models_performance.common.model import (
    CvxModel,
    add_constraint,
    add_ranged_constraint,
    column,
)

sqrt2 = sqrt(2)
sqrt3 = sqrt(3)


def add_voltage_limits(m: CvxModel) -> None:
    """Add voltage bounds (for voltage magnitude squared)."""
    add_ranged_constraint(
        m,
        "voltage_limits",
        column(m.v_min**2),
        m.v2,
        column(m.v_max**2),
        list(m.bus_phase_set),
    )


# ============ Forwarders to device modules ============================================


def add_cvr_load_constraints(m: CvxModel, free_boundary_loads: bool = False) -> None:
    from distopf.cvxpy_models_performance.devices.load import (
        add_cvr_load_constraints as add,
    )

    add(m, free_boundary_loads)


def add_generator_limits(m: CvxModel) -> None:
    from distopf.cvxpy_models_performance.devices.generator import add_gen_limits

    add_gen_limits(m)


def add_octagonal_inverter_constraints_pq_control(m: CvxModel) -> None:
    from distopf.cvxpy_models_performance.devices.generator import (
        add_gen_octagon_constraints,
    )

    add_gen_octagon_constraints(m)


def add_circular_generator_constraints_pq_control(m: CvxModel) -> None:
    from distopf.cvxpy_models_performance.devices.generator import (
        add_gen_circle_constraints,
    )

    add_gen_circle_constraints(m)


def add_capacitor_constraints(m: CvxModel) -> None:
    from distopf.cvxpy_models_performance.devices.capacitor import (
        add_capacitor_constraints as add,
    )

    add(m)


def add_capacitor_mi_constraints(m: CvxModel) -> None:
    from distopf.cvxpy_models_performance.devices.capacitor import (
        add_capacitor_mi_constraints as add,
    )

    add(m)


def add_capacitor_mccormick_constraints(m: CvxModel) -> None:
    from distopf.cvxpy_models_performance.devices.capacitor import (
        add_capacitor_mccormick_constraints as add,
    )

    add(m)


def add_capacitor_z_bounds(m: CvxModel) -> None:
    from distopf.cvxpy_models_performance.devices.capacitor import (
        add_capacitor_z_bounds as add,
    )

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
    rows = m.bus_phase_set.indices(m.swing_phase_set)
    add_constraint(
        m,
        "swing_voltage",
        m.v2[rows, :] == m.v_swing**2 if len(rows) else [],
        list(m.swing_phase_set),
    )


# ============ Thermal Line Constraints ================================================


def _thermal_rows(m: CvxModel) -> np.ndarray:
    """Positions of branch-phase rows with a positive thermal limit."""
    if not hasattr(m, "s_branch_max"):
        return np.array([], dtype=int)
    limit = m.s_branch_max
    return np.flatnonzero(~np.isnan(limit) & (limit > 0))


def _thermal_keys(m: CvxModel, rows: np.ndarray) -> list:
    return [m.branch_phase_set[i] for i in rows]


def add_octagonal_thermal_constraints(m: CvxModel) -> None:
    """Eight linear inequalities approximating ``|S_ij| <= S_max``."""
    rows = _thermal_rows(m)
    if len(rows) == 0:
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
    limit = column(m.s_branch_max[rows])
    p = m.p_flow[rows, :]
    q = m.q_flow[rows, :]
    keys = _thermal_keys(m, rows)
    for i, (a, b) in enumerate(coefficients, start=1):
        add_constraint(m, f"thermal_limit_{i}", a * p + b * q <= limit, keys)


def add_circular_thermal_constraints(m: CvxModel) -> None:
    """Exact quadratic limit ``P^2 + Q^2 <= S_max^2`` (a convex cone constraint)."""
    rows = _thermal_rows(m)
    if len(rows) == 0:
        return
    add_constraint(
        m,
        "thermal_limit_circle",
        cp.norm(cp.stack([m.p_flow[rows, :], m.q_flow[rows, :]]), 2, axis=0)
        <= column(m.s_branch_max[rows]),
        _thermal_keys(m, rows),
    )


# ============ Linear Slack Constraints ================================================


def add_thermal_slack_constraints(m: CvxModel, derate_factor=1) -> None:
    """Soft thermal limit: ``P_flow <= S_max * derate_factor + slack``."""
    if not hasattr(m, "s_branch_max"):
        return

    m.add_var("thermal_slack", m.branch_phase_set, nonneg=True, value=0.0)
    rows = _thermal_rows(m)
    add_constraint(
        m,
        "thermal_slack_constraint",
        m.p_flow[rows, :]
        <= column(m.s_branch_max[rows]) * derate_factor + m.thermal_slack[rows, :],
        _thermal_keys(m, rows),
    )


def add_voltage_slack_constraints(m: CvxModel) -> None:
    """Soft voltage bounds using a single nonnegative slack per bus/phase/time."""
    m.add_var("v2_slack", m.bus_phase_set, nonneg=True, value=0.0)
    keys = list(m.bus_phase_set)
    add_constraint(
        m,
        "voltage_slack_under",
        m.v2 >= column(m.v_min**2) - m.v2_slack,
        keys,
    )
    add_constraint(
        m,
        "voltage_slack_over",
        m.v2 <= column(m.v_max**2) + m.v2_slack,
        keys,
    )


def add_swing_bus_voltage_slack_constraints(m: CvxModel) -> None:
    """Soft swing-bus voltage: ``v2`` within ``v_swing^2`` plus or minus a slack."""
    m.add_var("swing_v2_slack", m.swing_phase_set, nonneg=True, value=0.0)
    rows = m.bus_phase_set.indices(m.swing_phase_set)
    keys = list(m.swing_phase_set)
    v2 = m.v2[rows, :]
    add_constraint(
        m,
        "swing_voltage_slack_under",
        v2 >= m.v_swing**2 - m.swing_v2_slack,
        keys,
    )
    add_constraint(
        m,
        "swing_voltage_slack_over",
        v2 <= m.v_swing**2 + m.swing_v2_slack,
        keys,
    )
