"""Network-wide constraints (vectorized CVXPY): voltage, swing bus and thermal limits.

Device-specific constraints live in ``devices/``. Each function here builds one
named constraint for the whole network and all time steps.
"""

from __future__ import annotations

from math import sqrt

import cvxpy as cp
import numpy as np

from distopf.cvxpy_models_performance.common.model import (
    CvxModel,
    add_constraint,
    add_constraint_on_rows,
    add_ranged_constraint,
    column,
    select_rows,
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


def add_swing_bus_constraints(m: CvxModel) -> None:
    """Fix swing bus voltages: ``v2 == v_swing^2`` (``v_swing`` is a magnitude)."""
    rows = m.bus_phase_set.indices(m.swing_phase_set)
    add_constraint_on_rows(
        m,
        "swing_voltage",
        (rows, list(m.swing_phase_set)),
        lambda r: m.v2[r, :] == m.v_swing**2,
    )


# ============ Thermal Line Constraints ================================================


def _thermal_selection(m: CvxModel) -> tuple[np.ndarray, list]:
    """Branch-phase rows (and keys) that have a positive thermal limit."""
    if not hasattr(m, "s_branch_max"):
        return np.array([], dtype=int), []
    limit = m.s_branch_max
    return select_rows(m.branch_phase_set, ~np.isnan(limit) & (limit > 0))


# Coefficients (a, b) of the eight half-planes a*P + b*Q <= S_max around the circle.
_OCTAGON_COEFFICIENTS = (
    (sqrt2 - 1, 1.0),
    (1.0, sqrt2 - 1),
    (1.0, -(sqrt2 - 1)),
    (sqrt2 - 1, -1.0),
    (-(sqrt2 - 1), -1.0),
    (-1.0, -(sqrt2 - 1)),
    (-1.0, sqrt2 - 1),
    (-(sqrt2 - 1), 1.0),
)


def add_octagonal_thermal_constraints(m: CvxModel) -> None:
    """Eight linear inequalities approximating ``|S_ij| <= S_max``."""
    rows, keys = _thermal_selection(m)
    if len(rows) == 0:
        return
    limit = column(m.s_branch_max[rows])
    p = m.p_flow[rows, :]
    q = m.q_flow[rows, :]
    for i, (a, b) in enumerate(_OCTAGON_COEFFICIENTS, start=1):
        add_constraint(m, f"thermal_limit_{i}", a * p + b * q <= limit, keys)


def add_circular_thermal_constraints(m: CvxModel) -> None:
    """Exact quadratic limit ``P^2 + Q^2 <= S_max^2`` (a convex cone constraint)."""
    selection = _thermal_selection(m)
    if len(selection[0]) == 0:
        return
    add_constraint_on_rows(
        m,
        "thermal_limit_circle",
        selection,
        lambda r: cp.norm(cp.stack([m.p_flow[r, :], m.q_flow[r, :]]), 2, axis=0)
        <= column(m.s_branch_max[r]),
    )


# ============ Linear Slack Constraints ================================================


def add_thermal_slack_constraints(m: CvxModel, derate_factor=1) -> None:
    """Soft thermal limit: ``P_flow <= S_max * derate_factor + slack``."""
    if not hasattr(m, "s_branch_max"):
        return

    m.add_var("thermal_slack", m.branch_phase_set, nonneg=True, value=0.0)
    add_constraint_on_rows(
        m,
        "thermal_slack_constraint",
        _thermal_selection(m),
        lambda r: m.p_flow[r, :]
        <= column(m.s_branch_max[r]) * derate_factor + m.thermal_slack[r, :],
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
