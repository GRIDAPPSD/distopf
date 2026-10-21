"""Objective expressions for vectorized CVXPY models.

Each rule takes a ``CvxModel`` and returns a CVXPY expression, built from whole
arrays instead of Python loops. Rule names match ``cvxpy_models``.

Array conventions used by the rules (``T`` is ``n_steps``):

- ``p_flow``, ``q_flow``, ``l_flow``: ``(n_index, T)``; ``r_diag``: ``(n_branch_phase,)``.
- Generator: ``p_gen``, ``q_gen``, ``gen_p_available``: ``(n_port, T)``; ``gen_cost``,
  ``gen_s_max``: ``(n_port,)``.
- Battery: ``p_bat``, ``q_bat``: ``(n_port, T)``; ``s_bat_rated``: ``(n_port,)``;
  ``soc``: ``(n_device, T)``; ``soc_min``, ``soc_max``: ``(n_device,)``.
"""

from __future__ import annotations

from typing import Any, Callable

import cvxpy as cp
import numpy as np

from distopf.cvxpy_models_performance.common.model import CvxModel, column


def _swing_rows(model: CvxModel) -> np.ndarray:
    return np.array(
        [
            i
            for i, (fb, _tb, _ph) in enumerate(model.branch_phase_set)
            if fb in model.swing_bus_set
        ],
        dtype=int,
    )


def none_rule(model: CvxModel):
    """Trivial objective function that returns 0."""
    return 0


def loss_objective_rule(model: CvxModel):
    """Total losses ``sum((P^2 + Q^2) * r)`` (convex quadratic)."""
    r = column(model.r_diag)
    return cp.sum(cp.multiply(r, cp.square(model.p_flow) + cp.square(model.q_flow)))


def loss_objective_i2r_rule(model: CvxModel):
    """Total losses based on current: ``sum(l_flow * r)`` (linear)."""
    if not hasattr(model, "l_flow"):
        raise AttributeError(
            "Model does not have l_flow attribute required for I^2*R loss calculation."
        )
    pair_pos = model.branch_phase_pair_set.pos
    rows = np.array(
        [pair_pos[(fb, tb, ph + ph)] for fb, tb, ph in model.branch_phase_set],
        dtype=int,
    )
    return cp.sum(cp.multiply(column(model.r_diag), model.l_flow[rows, :]))


def substation_power_objective_rule(model: CvxModel):
    """Total active power on branches leaving swing buses (linear)."""
    rows = _swing_rows(model)
    if len(rows) == 0:
        return 0
    return cp.sum(model.p_flow[rows, :])


def voltage_deviation_objective_rule(model: CvxModel):
    """``sum((v^2 - 1)^2)`` (convex quadratic)."""
    return cp.sum_squares(model.v2 - 1.0)


def voltage_min_objective_rule(model: CvxModel):
    """Minimize the sum of squared voltage magnitudes: ``min sum(v2)`` (linear)."""
    return cp.sum(model.v2)


def generation_curtailment_objective_rule(model: CvxModel):
    """Difference between available and actual generation (linear)."""
    if len(model.gen_device_phase_set) == 0:
        return 0
    return cp.sum(model.gen_p_available - model.p_gen)


def cost_minimization_rule(m: CvxModel):
    """Minimize energy procurement cost from swing bus plus generation cost."""
    total_cost: Any = 0
    rows = _swing_rows(m)
    if len(rows):
        total_cost = total_cost + cp.sum(
            cp.multiply(m.p_flow[rows, :], m.price.reshape(1, -1) * m.delta_t)
        )
    return total_cost + gen_cost_rule(m)


def substation_cost_objective_rule(model: CvxModel):
    """Cost of substation energy using ``schedule_price``."""
    rows = _swing_rows(model)
    if len(rows) == 0:
        return 0
    return cp.sum(
        cp.multiply(model.p_flow[rows, :], model.schedule_price.reshape(1, -1))
    )


def gen_cost_rule(model: CvxModel):
    if len(model.gen_device_phase_set) == 0:
        return 0
    return cp.sum(cp.multiply(column(model.gen_cost) * model.delta_t, model.p_gen))


def total_cost_rule(model: CvxModel):
    return gen_cost_rule(model) + substation_cost_objective_rule(model)


def generation_cost_with_substation_quadratic_penalty_objective_rule(
    model: CvxModel,
    substation_penalty_weight: float = 1e6,
):
    """Generation cost plus a quadratic penalty on substation active power."""
    rows = _swing_rows(model)
    penalty: Any = 0
    if len(rows):
        penalty = cp.sum_squares(model.p_flow[rows, :])
    return gen_cost_rule(model) + substation_penalty_weight * penalty


def demand_charge_objective_rule(model: CvxModel):
    """Demand charge on the peak substation active power (convex ``max``)."""
    if not hasattr(model, "demand_charge"):
        return 0
    rows = _swing_rows(model)
    if len(rows) == 0:
        return 0
    return cp.max(model.p_flow[rows, :]) * model.demand_charge


def combined_energy_and_demand_charge_objective_rule(model: CvxModel):
    return substation_cost_objective_rule(model) + demand_charge_objective_rule(model)


# ============ Penalty Functions for Soft Constraints ==================================


def voltage_violation_penalty(model: CvxModel, weight: float = 1e3):
    """Exact one-sided penalty ``weight * sum(pos(v2_min - v2) + pos(v2 - v2_max))``."""
    v2_min = column(model.v_min**2)
    v2_max = column(model.v_max**2)
    return weight * cp.sum(cp.pos(v2_min - model.v2) + cp.pos(model.v2 - v2_max))


def voltage_violation_penalty_quadratic(model: CvxModel, weight: float = 1e3):
    """Quadratic preference for mid-range voltages."""
    v2_min = model.v_min**2
    v2_max = model.v_max**2
    v2_mid = column((v2_min + v2_max) / 2)
    v2_range = column((v2_max - v2_min) / 2)
    return weight * cp.sum_squares(cp.multiply(model.v2 - v2_mid, 1.0 / v2_range))


def thermal_violation_penalty(model: CvxModel, weight: float = 1e3):
    """``weight * sum(pos(P^2 + Q^2 - S_max^2))`` (convex)."""
    if not hasattr(model, "s_branch_max"):
        return 0
    limit = model.s_branch_max
    rows = np.flatnonzero(~np.isnan(limit) & (limit > 0))
    if len(rows) == 0:
        return 0
    s_sq = cp.square(model.p_flow[rows, :]) + cp.square(model.q_flow[rows, :])
    return weight * cp.sum(cp.pos(s_sq - column(limit[rows] ** 2)))


def generator_violation_penalty(model: CvxModel, weight: float = 1e3):
    if len(model.gen_device_phase_set) == 0:
        return 0
    s_sq = cp.square(model.p_gen) + cp.square(model.q_gen)
    return weight * cp.sum(cp.pos(s_sq - column(model.gen_s_max**2)))


def battery_violation_penalty(model: CvxModel, weight: float = 1e3):
    if len(model.bat_phase_set) == 0:
        return 0
    s_sq = cp.square(model.p_bat) + cp.square(model.q_bat)
    return weight * cp.sum(cp.pos(s_sq - column(model.s_bat_rated**2)))


def soc_violation_penalty(model: CvxModel, weight: float = 1e3):
    if len(model.bat_set) == 0:
        return 0
    return weight * cp.sum(
        cp.pos(column(model.soc_min) - model.soc)
        + cp.pos(model.soc - column(model.soc_max))
    )


# ============ Combined Objective Factories ============================================


def create_penalized_objective(
    primary_objective_rule: Callable[[CvxModel], Any],
    voltage_weight: float | None = None,
    thermal_weight: float | None = None,
    generator_weight: float | None = None,
    battery_weight: float | None = None,
    soc_weight: float | None = None,
) -> Callable[[CvxModel], Any]:
    """Return a rule combining a primary objective with soft-constraint penalties."""

    def objective_rule(m: CvxModel):
        obj = primary_objective_rule(m)
        if voltage_weight is not None:
            obj += voltage_violation_penalty(m, voltage_weight)
        if thermal_weight is not None:
            obj += thermal_violation_penalty(m, thermal_weight)
        if generator_weight is not None:
            obj += generator_violation_penalty(m, generator_weight)
        if battery_weight is not None:
            obj += battery_violation_penalty(m, battery_weight)
        if soc_weight is not None:
            obj += soc_violation_penalty(m, soc_weight)
        return obj

    return objective_rule


# ============ Penalty Functions for ADMM ==============================================


def _schedule_target(m: CvxModel, key: str):
    """Per-time schedule array ``(n_steps,)`` or ``None`` if the schedule is absent."""
    return getattr(m, f"schedule_{key}", None)


def admm_v_up_penalty(m: CvxModel, rho: float = 1.0):
    if not hasattr(m, "swing_phase_set") or len(m.swing_phase_set) == 0:
        return 0
    rows = m.bus_phase_set.indices(m.swing_phase_set)
    return 0.5 * rho * cp.sum_squares(m.v2[rows, :] - m.v_swing**2)


def admm_s_up_penalty(m: CvxModel, rho: float = 1.0):
    swing_branches: dict[tuple[int, str], list[int]] = {}
    for i, (fb, _tb, ph) in enumerate(m.branch_phase_set):
        if fb not in m.swing_bus_set:
            continue
        swing_branches.setdefault((fb, ph), []).append(i)

    penalty: Any = 0
    for (fb, ph), rows in swing_branches.items():
        area_name = m.name_map[fb]
        p_param = _schedule_target(m, f"{area_name}.{ph}.p")
        q_param = _schedule_target(m, f"{area_name}.{ph}.q")
        if p_param is None or q_param is None:
            continue
        penalty += cp.sum_squares(cp.sum(m.p_flow[rows, :], axis=0) - p_param)
        penalty += cp.sum_squares(cp.sum(m.q_flow[rows, :], axis=0) - q_param)
    return 0.5 * rho * penalty


def admm_v_down_penalty(m: CvxModel, rho: float = 1.0):
    penalty: Any = 0
    for i, (_id, ph) in enumerate(m.bus_phase_set):
        v_param = _schedule_target(m, f"{m.name_map[_id]}.{ph}.v")
        if v_param is None:
            continue
        penalty += cp.sum_squares(m.v2[i, :] - v_param**2)
    return 0.5 * rho * penalty


def admm_s_down_penalty(m: CvxModel, rho: float = 1.0):
    penalty: Any = 0
    for i, (_id, ph) in enumerate(m.bus_phase_set):
        if _id in m.swing_bus_set:
            continue
        area_name = m.name_map[_id]
        p_param = _schedule_target(m, f"{area_name}.{ph}.p")
        q_param = _schedule_target(m, f"{area_name}.{ph}.q")
        if p_param is None or q_param is None:
            continue
        penalty += cp.sum_squares(m.p_load[i, :] - p_param)
        penalty += cp.sum_squares(m.q_load[i, :] - q_param)
    return 0.5 * rho * penalty


# ============ Convenience Functions ===================================================


def set_objective(model: CvxModel, objective: Any, sense: str = "minimize") -> None:
    """Set the model objective, replacing any existing one.

    ``objective`` is a CVXPY expression, a number, or a rule ``f(model)``
    returning one. ``sense`` is ``"minimize"`` or ``"maximize"``.
    """
    if callable(objective) and not isinstance(objective, cp.Expression):
        objective = objective(model)
    if sense not in ("minimize", "maximize"):
        raise ValueError("sense must be 'minimize' or 'maximize'")
    if not isinstance(objective, cp.Expression):
        objective = cp.Constant(objective)
    model.objective_expression = objective
    model.objective_sense = sense
    model.problem = None


def add_loss_objective(model: CvxModel) -> None:
    set_objective(model, loss_objective_rule)


def add_substation_power_objective(model: CvxModel) -> None:
    set_objective(model, substation_power_objective_rule)


def add_voltage_deviation_objective(model: CvxModel) -> None:
    set_objective(model, voltage_deviation_objective_rule)


def add_generation_cost_with_substation_quadratic_penalty_objective(
    model: CvxModel, substation_penalty_weight: float = 1e6
) -> None:
    set_objective(
        model,
        generation_cost_with_substation_quadratic_penalty_objective_rule(
            model, substation_penalty_weight=substation_penalty_weight
        ),
    )


def _add_penalized(model: CvxModel, primary, **weights: float) -> None:
    set_objective(model, create_penalized_objective(primary, **weights))


def add_penalized_loss_objective(
    model: CvxModel,
    voltage_weight: float = 1e3,
    thermal_weight: float = 1e3,
    generator_weight: float = 1e3,
    battery_weight: float = 1e3,
    soc_weight: float = 1e3,
) -> None:
    _add_penalized(
        model,
        loss_objective_rule,
        voltage_weight=voltage_weight,
        thermal_weight=thermal_weight,
        generator_weight=generator_weight,
        battery_weight=battery_weight,
        soc_weight=soc_weight,
    )


def add_penalized_substation_power_objective(
    model: CvxModel,
    voltage_weight: float = 1e3,
    thermal_weight: float = 1e3,
    generator_weight: float = 1e3,
    battery_weight: float = 1e3,
    soc_weight: float = 1e3,
) -> None:
    _add_penalized(
        model,
        substation_power_objective_rule,
        voltage_weight=voltage_weight,
        thermal_weight=thermal_weight,
        generator_weight=generator_weight,
        battery_weight=battery_weight,
        soc_weight=soc_weight,
    )


def voltage_slack_penalty(m: CvxModel, weight=1e3):
    if not hasattr(m, "v2_slack"):
        return 0
    return weight * cp.sum(m.v2_slack)


def thermal_slack_penalty(m: CvxModel, weight=1e3):
    if not hasattr(m, "thermal_slack"):
        return 0
    return weight * cp.sum(m.thermal_slack)


def swing_voltage_slack_penalty(m: CvxModel, weight=1e3):
    if not hasattr(m, "swing_v2_slack"):
        return 0
    return weight * cp.sum(m.swing_v2_slack)
