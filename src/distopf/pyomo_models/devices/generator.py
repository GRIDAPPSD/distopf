"""Generator device provider for dispatch, limits, and control policies."""

from __future__ import annotations

import warnings
from typing import Any

import pyomo.environ as pyo  # type: ignore

from numpy import sqrt

from distopf.pyomo_models.common.protocol import LindistModelProtocol
from distopf.pyomo_models.common.model_types import ControlVariable
from distopf.pyomo_models.common.data import parse_phases, phase_tuples
from distopf.pyomo_models.common.model_types import CONTROL_VARIABLE_MAP
from distopf.utils.ngon import ngon_line_equations


sqrt2 = sqrt(2)


def create_generator_parameters(model: Any, case: Any) -> None:
    """Create the complete legacy-compatible generator parameter surface."""
    p_data, q_data, rating, q_min, q_max, control, cost = {}, {}, {}, {}, {}, {}, {}
    for _, row in case.gen_data.iterrows():
        for phase in parse_phases(str(row.phases)):
            key = (row.id, phase)
            if key not in model.gen_phase_set:
                continue
            s = getattr(row, f"s_{phase}_max", 1000.0)
            rating[key] = s
            q_min[key] = getattr(row, f"q_{phase}_min", -s)
            q_max[key] = getattr(row, f"q_{phase}_max", s)
            control[key] = CONTROL_VARIABLE_MAP[getattr(row, "control_variable", "")]
            cost[key] = getattr(row, "cost", 0.0)
            for time in model.time_set:
                multiplier = 1.0
                shape = getattr(row, "gen_shape", "PV")
                if shape in case.schedules.columns and time in case.schedules.index:
                    try:
                        multiplier = float(case.schedules.at[time, shape])
                    except (TypeError, ValueError):
                        warnings.warn(
                            f"Non-numeric generator schedule {shape!r}; using 1.0"
                        )
                p_data[(key[0], key[1], time)] = (
                    getattr(row, f"p_{phase}", 0.0) * multiplier
                )
                q_data[(key[0], key[1], time)] = getattr(row, f"q_{phase}", 0.0)
    model.p_gen_nom = pyo.Param(
        model.gen_phase_set, model.time_set, initialize=p_data, default=0.0
    )
    model.q_gen_nom = pyo.Param(
        model.gen_phase_set, model.time_set, initialize=q_data, default=0.0
    )
    model.s_rated = pyo.Param(model.gen_phase_set, initialize=rating, default=1000.0)
    model.q_gen_min = pyo.Param(model.gen_phase_set, initialize=q_min, default=-1000.0)
    model.q_gen_max = pyo.Param(model.gen_phase_set, initialize=q_max, default=1000.0)
    model.gen_control_type = pyo.Param(
        model.gen_phase_set, initialize=control, default=0
    )
    model.gen_cost = pyo.Param(model.gen_phase_set, initialize=cost, default=0.0)


class GeneratorProvider:
    """Own generator parameters, variables, operating constraints, and injection."""

    name = "generators"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        if not hasattr(model, "gen_phase_set"):
            model.gen_phase_set = pyo.Set(
                initialize=phase_tuples(case.gen_data), dimen=2
            )
        if not hasattr(model, "gen_set"):
            model.gen_set = pyo.Set(
                initialize=sorted({device for device, _ in model.gen_phase_set})
            )
        if not hasattr(model, "p_gen"):
            model.p_gen = pyo.Var(
                model.gen_phase_set, model.time_set, domain=pyo.NonNegativeReals
            )
        if not hasattr(model, "q_gen"):
            model.q_gen = pyo.Var(model.gen_phase_set, model.time_set, initialize=0)
        if not hasattr(model, "p_gen_nom"):
            create_generator_parameters(model, case)
        if not hasattr(model, "gen_phase_pair_set"):
            model.gen_phase_pair_set = pyo.Set(
                initialize=[
                    (device, left, right)
                    for device in model.gen_set
                    for left, right in zip(
                        [
                            phase
                            for phase in ("a", "b", "c")
                            if (device, phase) in model.gen_phase_set
                        ],
                        [
                            phase
                            for phase in ("a", "b", "c")
                            if (device, phase) in model.gen_phase_set
                        ][1:],
                    )
                ],
                dimen=3,
            )
        if not hasattr(model, "gen_phase_lock"):
            model.gen_phase_lock = pyo.Param(
                model.gen_set,
                initialize={device: False for device in model.gen_set},
                within=pyo.Boolean,
                mutable=True,
            )

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        key = (bus, phase, time)
        return model.p_gen[key] if key in model.p_gen else 0

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        key = (bus, phase, time)
        return model.q_gen[key] if key in model.q_gen else 0

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.gen_phase_set) == 0:
            return
        equality_only = getattr(config, "equality_only", False) if config else False
        if not equality_only and not hasattr(model, "p_gen_limits"):
            add_generator_limits(model)
        if not hasattr(model, "constant_p_gen"):
            add_generator_constant_p_constraints_q_control(model)
        if not hasattr(model, "constant_q_gen"):
            add_generator_constant_q_constraints_p_control(model)
        if equality_only:
            return
        if getattr(config, "circular_constraints", True) if config else True:
            if not hasattr(model, "gen_circle_constraint"):
                add_circular_generator_constraints_pq_control(model)
        elif not hasattr(model, "gen_octagon_1"):
            add_octagonal_inverter_constraints_pq_control(model)


# Constraints ------------------------------------------------------------------------


def add_generator_limits(m: LindistModelProtocol) -> None:
    """Add generator bounds following the original base.py logic"""

    def p_gen_bounds(m: LindistModelProtocol, _id, ph, t):
        if m.gen_control_type[_id, ph] == ControlVariable.NONE:
            return pyo.Constraint.Skip
        return (
            0,
            m.p_gen[_id, ph, t],
            min(m.p_gen_nom[_id, ph, t], m.s_rated[_id, ph]),
        )

    def q_gen_bounds(m: LindistModelProtocol, _id, ph, t):
        if m.gen_control_type[_id, ph] == ControlVariable.NONE:
            return pyo.Constraint.Skip
        if m.gen_control_type[_id, ph] == ControlVariable.Q:
            q_max = sqrt(max(0, m.s_rated[_id, ph] ** 2 - m.p_gen_nom[_id, ph, t] ** 2))
            return (
                max(-q_max, m.q_gen_min[_id, ph]),
                m.q_gen[_id, ph, t],
                min(q_max, m.q_gen_max[_id, ph]),
            )
        return (
            max(-m.s_rated[_id, ph], m.q_gen_min[_id, ph]),
            m.q_gen[_id, ph, t],
            min(m.s_rated[_id, ph], m.q_gen_max[_id, ph]),
        )

    m.p_gen_limits = pyo.Constraint(m.gen_phase_set, m.time_set, rule=p_gen_bounds)
    m.q_gen_limits = pyo.Constraint(m.gen_phase_set, m.time_set, rule=q_gen_bounds)


def add_generator_constant_p_constraints(m: LindistModelProtocol) -> None:
    m.constant_p_gen = pyo.Constraint(
        m.gen_phase_set,
        m.time_set,
        rule=lambda m, _id, ph, t: m.p_gen[_id, ph, t] == m.p_gen_nom[_id, ph, t],
    )


def add_generator_constant_q_constraints(m: LindistModelProtocol) -> None:
    m.constant_q_gen = pyo.Constraint(
        m.gen_phase_set,
        m.time_set,
        rule=lambda m, _id, ph, t: m.q_gen[_id, ph, t] == m.q_gen_nom[_id, ph, t],
    )


def add_generator_constant_p_constraints_q_control(m: LindistModelProtocol) -> None:
    def _rule(m: LindistModelProtocol, _id, ph, t):
        ct = m.gen_control_type[_id, ph]
        if ct in (ControlVariable.NONE, ControlVariable.Q):
            return m.p_gen[_id, ph, t] == m.p_gen_nom[_id, ph, t]
        return pyo.Constraint.Skip

    m.constant_p_gen = pyo.Constraint(m.gen_phase_set, m.time_set, rule=_rule)


def add_generator_constant_q_constraints_p_control(m: LindistModelProtocol) -> None:
    def _rule(m: LindistModelProtocol, _id, ph, t):
        ct = m.gen_control_type[_id, ph]
        if ct in (ControlVariable.NONE, ControlVariable.P):
            return m.q_gen[_id, ph, t] == m.q_gen_nom[_id, ph, t]
        return pyo.Constraint.Skip

    m.constant_q_gen = pyo.Constraint(m.gen_phase_set, m.time_set, rule=_rule)


def add_ngon_constraints(m: LindistModelProtocol, n: int = 8) -> None:
    ngon_eqs = ngon_line_equations(n)
    for i, (a, b) in enumerate(ngon_eqs, start=1):
        if a < -1e-9:
            continue
        a, b = float(a), float(b)

        def _rule(m: LindistModelProtocol, _id, ph, t):
            if m.gen_control_type[_id, ph] != ControlVariable.PQ:
                return pyo.Constraint.Skip
            return (
                a * m.p_gen[_id, ph, t] + b * m.q_gen[_id, ph, t] <= m.s_rated[_id, ph]
            )

        setattr(
            m,
            f"gen_ngon_limit_{i}",
            pyo.Constraint(m.gen_phase_set, m.time_set, rule=_rule),
        )


def add_octagonal_inverter_constraints_pq_control(m: LindistModelProtocol) -> None:
    """
    Add octagonal inverter constraints (equation 2.14).

    Linear approximation of circular curve using 8 constraints.
    Only applied to generators with control_variable=="PQ".

    c = sqrt(2) - 1
    c * p_gen + 1 * q_gen <= s_rated
    1 * p_gen + c * q_gen <= s_rated
    1 * p_gen - c * q_gen <= s_rated
    c * p_gen - 1 * q_gen <= s_rated
    """
    c = sqrt2 - 1  # ≈ 0.4142

    # If the P-Q Plane was on a clock:
    # Line from 12:00 to 1:30. Or 90 to 45 deg.
    def _1(m: LindistModelProtocol, _id, ph, t):
        if m.gen_control_type[_id, ph] != ControlVariable.PQ:
            return pyo.Constraint.Skip
        return c * m.p_gen[_id, ph, t] + 1 * m.q_gen[_id, ph, t] <= m.s_rated[_id, ph]

    # Line from 1:30 to 3:00 on a clock. Or 45 to 0 deg.
    def _2(m: LindistModelProtocol, _id, ph, t):
        if m.gen_control_type[_id, ph] != ControlVariable.PQ:
            return pyo.Constraint.Skip
        return 1 * m.p_gen[_id, ph, t] + c * m.q_gen[_id, ph, t] <= m.s_rated[_id, ph]

    # Line from 3:00 to 4:30 on a clock. Or 0 to -45 deg.
    def _3(m: LindistModelProtocol, _id, ph, t):
        if m.gen_control_type[_id, ph] != ControlVariable.PQ:
            return pyo.Constraint.Skip
        return 1 * m.p_gen[_id, ph, t] - c * m.q_gen[_id, ph, t] <= m.s_rated[_id, ph]

    # Line from 4:30 to 6:00 on a clock. Or -45 to -90 deg.
    def _4(m: LindistModelProtocol, _id, ph, t):
        if m.gen_control_type[_id, ph] != ControlVariable.PQ:
            return pyo.Constraint.Skip
        return c * m.p_gen[_id, ph, t] - 1 * m.q_gen[_id, ph, t] <= m.s_rated[_id, ph]

    # Add all octagonal constraints
    m.gen_octagon_1 = pyo.Constraint(m.gen_phase_set, m.time_set, rule=_1)
    m.gen_octagon_2 = pyo.Constraint(m.gen_phase_set, m.time_set, rule=_2)
    m.gen_octagon_3 = pyo.Constraint(m.gen_phase_set, m.time_set, rule=_3)
    m.gen_octagon_4 = pyo.Constraint(m.gen_phase_set, m.time_set, rule=_4)


def add_circular_generator_constraints_pq_control(m: LindistModelProtocol) -> None:
    """
    Add circular generator constraints.

    Uses the exact circular constraint: p_gen² + q_gen² ≤ s_rated²
    Only applied to generators with control_variable=="PQ".
    """

    def _circle(m: LindistModelProtocol, _id, ph, t):
        if m.gen_control_type[_id, ph] != ControlVariable.PQ:
            return pyo.Constraint.Skip
        return (
            m.p_gen[_id, ph, t] ** 2 + m.q_gen[_id, ph, t] ** 2
            <= m.s_rated[_id, ph] ** 2
        )

    m.gen_circle_constraint = pyo.Constraint(m.gen_phase_set, m.time_set, rule=_circle)


__all__ = ["GeneratorProvider", "create_generator_parameters"]
