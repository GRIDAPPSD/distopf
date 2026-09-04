"""MPSSD device constraints extracted from the legacy constraints module."""

from __future__ import annotations

import pyomo.environ as pyo  # type: ignore
from numpy import sqrt

from distopf.pyomo_models.model_types import ControlVariable
from distopf.pyomo_models.protocol import LindistModelProtocol


sqrt2 = sqrt(2)


def add_mpssd_constant_p_constraints_q_control(m: LindistModelProtocol) -> None:
    """Fix active MPSSD power for non-P-controlled ports."""

    def rule(m, device, phase, time):
        control = m.mpssd_control_type[device, phase]
        if control in (ControlVariable.NONE, ControlVariable.Q):
            return m.p_mpssd[device, phase, time] == m.p_mpssd_nom[device, phase, time]
        return pyo.Constraint.Skip

    m.mpssd_constant_p = pyo.Constraint(m.mpssd_phase_set, m.time_set, rule=rule)


def add_mpssd_constant_q_constraints_p_control(m: LindistModelProtocol) -> None:
    """Fix reactive MPSSD power for non-Q-controlled ports."""

    def rule(m, device, phase, time):
        control = m.mpssd_control_type[device, phase]
        if control in (ControlVariable.NONE, ControlVariable.P):
            return m.q_mpssd[device, phase, time] == m.q_mpssd_nom[device, phase, time]
        return pyo.Constraint.Skip

    m.mpssd_constant_q = pyo.Constraint(m.mpssd_phase_set, m.time_set, rule=rule)


def add_mpssd_limits(m: LindistModelProtocol) -> None:
    """Add rectangular P/Q operating limits for MPSSD ports."""

    def p_bounds(m, device, phase, time):
        rating = m.mpssd_s_rated[device, phase]
        return (-rating, m.p_mpssd[device, phase, time], rating)

    def q_bounds(m, device, phase, time):
        rating = m.mpssd_s_rated[device, phase]
        return (
            max(-rating, m.mpssd_q_min[device, phase]),
            m.q_mpssd[device, phase, time],
            min(rating, m.mpssd_q_max[device, phase]),
        )

    m.mpssd_p_limits = pyo.Constraint(m.mpssd_phase_set, m.time_set, rule=p_bounds)
    m.mpssd_q_limits = pyo.Constraint(m.mpssd_phase_set, m.time_set, rule=q_bounds)


def add_circular_mpssd_constraints(m: LindistModelProtocol) -> None:
    """Add exact apparent-power limits for MPSSD ports."""

    def rule(m, device, phase, time):
        return (
            m.p_mpssd[device, phase, time] ** 2 + m.q_mpssd[device, phase, time] ** 2
            <= m.mpssd_s_rated[device, phase] ** 2
        )

    m.mpssd_circle = pyo.Constraint(m.mpssd_phase_set, m.time_set, rule=rule)


def add_octagonal_mpssd_constraints(m: LindistModelProtocol) -> None:
    """Add an eight-sided linear approximation of apparent-power limits."""
    c = sqrt2 - 1

    def r1(m, device, phase, time):
        return (
            c * m.p_mpssd[device, phase, time] + m.q_mpssd[device, phase, time]
            <= m.mpssd_s_rated[device, phase]
        )

    def r2(m, device, phase, time):
        return (
            m.p_mpssd[device, phase, time] + c * m.q_mpssd[device, phase, time]
            <= m.mpssd_s_rated[device, phase]
        )

    def r3(m, device, phase, time):
        return (
            m.p_mpssd[device, phase, time] - c * m.q_mpssd[device, phase, time]
            <= m.mpssd_s_rated[device, phase]
        )

    def r4(m, device, phase, time):
        return (
            c * m.p_mpssd[device, phase, time] - m.q_mpssd[device, phase, time]
            <= m.mpssd_s_rated[device, phase]
        )

    def r5(m, device, phase, time):
        return (
            -c * m.p_mpssd[device, phase, time] - m.q_mpssd[device, phase, time]
            <= m.mpssd_s_rated[device, phase]
        )

    def r6(m, device, phase, time):
        return (
            -m.p_mpssd[device, phase, time] - c * m.q_mpssd[device, phase, time]
            <= m.mpssd_s_rated[device, phase]
        )

    def r7(m, device, phase, time):
        return (
            -m.p_mpssd[device, phase, time] + c * m.q_mpssd[device, phase, time]
            <= m.mpssd_s_rated[device, phase]
        )

    def r8(m, device, phase, time):
        return (
            -c * m.p_mpssd[device, phase, time] + m.q_mpssd[device, phase, time]
            <= m.mpssd_s_rated[device, phase]
        )

    for index, rule in enumerate((r1, r2, r3, r4, r5, r6, r7, r8), start=1):
        setattr(
            m,
            f"mpssd_oct_{index}",
            pyo.Constraint(m.mpssd_phase_set, m.time_set, rule=rule),
        )


def add_dc_bus_balance_constraints(m: LindistModelProtocol) -> None:
    """Enforce zero net active injection for each shared DC bus."""

    def rule(m, dc_bus, time):
        return (
            sum(
                m.p_mpssd[device, phase, time]
                for device, phase in m.mpssd_phase_set
                if m.mpssd_dc_bus[device, phase] == dc_bus
            )
            == 0
        )

    m.dc_bus_balance = pyo.Constraint(m.dc_bus_set, m.time_set, rule=rule)
