"""Objective expressions for vectorized CVXPY models.

Every objective is a plain function ``f(model) -> cvxpy expression`` built from
whole arrays. Combine them with ordinary arithmetic and pass the result to
``set_objective``::

    set_objective(m, loss(m) + 1e3 * voltage_violation_penalty(m))

Array conventions (``T`` is ``n_steps``):

- ``p_flow``, ``q_flow``, ``l_flow``: ``(n_index, T)``; ``r_diag``: ``(n_branch_phase,)``.
- Generator: ``p_gen``, ``q_gen``, ``gen_p_available``: ``(n_port, T)``; ``gen_cost``,
  ``gen_s_max``: ``(n_port,)``.
- Battery: ``p_bat``, ``q_bat``: ``(n_port, T)``; ``s_bat_rated``: ``(n_port,)``;
  ``soc``: ``(n_device, T)``; ``soc_min``, ``soc_max``: ``(n_device,)``.
- ``price`` and ``schedule_<column>``: ``(T,)``.

The old ``*_rule`` names are kept as aliases at the bottom of this file.
"""

from __future__ import annotations

from typing import Any, Callable

import cvxpy as cp
import numpy as np

from distopf.cvxpy_models_performance.common.model import CvxModel, column

# ============ Building blocks =========================================================


def _swing_rows(model: CvxModel) -> np.ndarray:
    """Rows of ``branch_phase_set`` whose from-bus is a swing bus."""
    return np.array(
        [
            i
            for i, (from_bus, _to_bus, _phase) in enumerate(model.branch_phase_set)
            if from_bus in model.swing_bus_set
        ],
        dtype=int,
    )


def substation_power(model: CvxModel):
    """Total active power on branches leaving swing buses (linear)."""
    rows = _swing_rows(model)
    if len(rows) == 0:
        return 0
    return cp.sum(model.p_flow[rows, :])


def substation_cost(model: CvxModel):
    """Cost of substation energy using ``schedule_price`` (linear)."""
    rows = _swing_rows(model)
    if len(rows) == 0:
        return 0
    return cp.sum(
        cp.multiply(model.p_flow[rows, :], model.schedule_price.reshape(1, -1))
    )


def generation_cost(model: CvxModel):
    """``sum(gen_cost * delta_t * p_gen)`` (linear)."""
    if len(model.gen_device_phase_set) == 0:
        return 0
    return cp.sum(cp.multiply(column(model.gen_cost) * model.delta_t, model.p_gen))


# ============ Objectives ==============================================================


def no_objective(model: CvxModel):
    """Trivial objective that returns 0."""
    return 0


def loss(model: CvxModel):
    """Total losses ``sum((P^2 + Q^2) * r)`` (convex quadratic)."""
    r = column(model.r_diag)
    return cp.sum(cp.multiply(r, cp.square(model.p_flow) + cp.square(model.q_flow)))


def loss_i2r(model: CvxModel):
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


def voltage_deviation(model: CvxModel):
    """``sum((v^2 - 1)^2)`` (convex quadratic)."""
    return cp.sum_squares(model.v2 - 1.0)


def voltage_min(model: CvxModel):
    """Minimize the sum of squared voltage magnitudes: ``min sum(v2)`` (linear)."""
    return cp.sum(model.v2)


def generation_curtailment(model: CvxModel):
    """Difference between available and actual generation (linear)."""
    if len(model.gen_device_phase_set) == 0:
        return 0
    return cp.sum(model.gen_p_available - model.p_gen)


def energy_and_generation_cost(model: CvxModel):
    """Substation energy cost using ``price * delta_t`` plus generation cost."""
    rows = _swing_rows(model)
    energy: Any = 0
    if len(rows):
        energy = cp.sum(
            cp.multiply(
                model.p_flow[rows, :], model.price.reshape(1, -1) * model.delta_t
            )
        )
    return energy + generation_cost(model)


def total_cost(model: CvxModel):
    """Generation cost plus substation cost using ``schedule_price``."""
    return generation_cost(model) + substation_cost(model)


def generation_cost_with_substation_penalty(
    model: CvxModel, substation_penalty_weight: float = 1e6
):
    """Generation cost plus a quadratic penalty on substation active power."""
    rows = _swing_rows(model)
    penalty: Any = 0
    if len(rows):
        penalty = cp.sum_squares(model.p_flow[rows, :])
    return generation_cost(model) + substation_penalty_weight * penalty


def demand_charge_cost(model: CvxModel):
    """Demand charge on the peak substation active power (convex ``max``)."""
    rows = _swing_rows(model)
    if not hasattr(model, "demand_charge") or len(rows) == 0:
        return 0
    return cp.max(model.p_flow[rows, :]) * model.demand_charge


def energy_and_demand_charge(model: CvxModel):
    """``schedule_price`` energy cost plus the demand charge."""
    return substation_cost(model) + demand_charge_cost(model)


# ============ Penalties for soft constraints ==========================================


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


def voltage_slack_penalty(model: CvxModel, weight: float = 1e3):
    """Penalty on the slack added by ``voltage_slacks=True``."""
    if not hasattr(model, "v2_slack"):
        return 0
    return weight * cp.sum(model.v2_slack)


def thermal_slack_penalty(model: CvxModel, weight: float = 1e3):
    """Penalty on the slack added by ``thermal_slacks=True``."""
    if not hasattr(model, "thermal_slack"):
        return 0
    return weight * cp.sum(model.thermal_slack)


def swing_voltage_slack_penalty(model: CvxModel, weight: float = 1e3):
    """Penalty on the slack added by ``free_swing_voltage=True``."""
    if not hasattr(model, "swing_v2_slack"):
        return 0
    return weight * cp.sum(model.swing_v2_slack)


def create_penalized_objective(
    primary: Callable[[CvxModel], Any],
    voltage_weight: float | None = None,
    thermal_weight: float | None = None,
    generator_weight: float | None = None,
    battery_weight: float | None = None,
    soc_weight: float | None = None,
) -> Callable[[CvxModel], Any]:
    """Return an objective ``primary(m)`` plus the penalties whose weight is set."""
    penalties = (
        (voltage_weight, voltage_violation_penalty),
        (thermal_weight, thermal_violation_penalty),
        (generator_weight, generator_violation_penalty),
        (battery_weight, battery_violation_penalty),
        (soc_weight, soc_violation_penalty),
    )

    def objective(model: CvxModel):
        total = primary(model)
        for weight, penalty in penalties:
            if weight is not None:
                total = total + penalty(model, weight)
        return total

    return objective


def add_penalized_loss_objective(
    model: CvxModel,
    voltage_weight: float = 1e3,
    thermal_weight: float = 1e3,
    generator_weight: float = 1e3,
    battery_weight: float = 1e3,
    soc_weight: float = 1e3,
) -> None:
    """Set ``loss`` plus all soft-constraint penalties as the objective."""
    set_objective(
        model,
        create_penalized_objective(
            loss,
            voltage_weight,
            thermal_weight,
            generator_weight,
            battery_weight,
            soc_weight,
        ),
    )


def add_penalized_substation_power_objective(
    model: CvxModel,
    voltage_weight: float = 1e3,
    thermal_weight: float = 1e3,
    generator_weight: float = 1e3,
    battery_weight: float = 1e3,
    soc_weight: float = 1e3,
) -> None:
    """Set ``substation_power`` plus all soft-constraint penalties as the objective."""
    set_objective(
        model,
        create_penalized_objective(
            substation_power,
            voltage_weight,
            thermal_weight,
            generator_weight,
            battery_weight,
            soc_weight,
        ),
    )


# ============ ADMM coupling penalties =================================================


def admm_v_up_penalty(m: CvxModel, rho: float = 1.0):
    """Match swing-bus ``v2`` to the squared upstream voltage target ``v_swing``."""
    if len(m.swing_phase_set) == 0:
        return 0
    rows = m.bus_phase_set.indices(m.swing_phase_set)
    return 0.5 * rho * cp.sum_squares(m.v2[rows, :] - m.v_swing**2)


def admm_s_up_penalty(m: CvxModel, rho: float = 1.0):
    """Match power leaving each swing bus phase to the ``schedule_<area>.<ph>.p/q`` targets."""
    swing_branches: dict[tuple[int, str], list[int]] = {}
    for i, (from_bus, _to_bus, phase) in enumerate(m.branch_phase_set):
        if from_bus in m.swing_bus_set:
            swing_branches.setdefault((from_bus, phase), []).append(i)

    penalty: Any = 0
    for (from_bus, phase), rows in swing_branches.items():
        area = m.name_map[from_bus]
        p_target = getattr(m, f"schedule_{area}.{phase}.p", None)
        q_target = getattr(m, f"schedule_{area}.{phase}.q", None)
        if p_target is None or q_target is None:
            continue
        penalty += cp.sum_squares(cp.sum(m.p_flow[rows, :], axis=0) - p_target)
        penalty += cp.sum_squares(cp.sum(m.q_flow[rows, :], axis=0) - q_target)
    return 0.5 * rho * penalty


def admm_v_down_penalty(m: CvxModel, rho: float = 1.0):
    """Match ``v2`` to squared ``schedule_<area>.<ph>.v`` targets at boundary buses."""
    penalty: Any = 0
    for i, (bus, phase) in enumerate(m.bus_phase_set):
        v_target = getattr(m, f"schedule_{m.name_map[bus]}.{phase}.v", None)
        if v_target is not None:
            penalty += cp.sum_squares(m.v2[i, :] - v_target**2)
    return 0.5 * rho * penalty


def admm_s_down_penalty(m: CvxModel, rho: float = 1.0):
    """Match boundary loads to ``schedule_<area>.<ph>.p/q`` targets at non-swing buses."""
    penalty: Any = 0
    for i, (bus, phase) in enumerate(m.bus_phase_set):
        if bus in m.swing_bus_set:
            continue
        area = m.name_map[bus]
        p_target = getattr(m, f"schedule_{area}.{phase}.p", None)
        q_target = getattr(m, f"schedule_{area}.{phase}.q", None)
        if p_target is None or q_target is None:
            continue
        penalty += cp.sum_squares(m.p_load[i, :] - p_target)
        penalty += cp.sum_squares(m.q_load[i, :] - q_target)
    return 0.5 * rho * penalty


# ============ Setting the objective ===================================================


def set_objective(model: CvxModel, objective: Any, sense: str = "minimize") -> None:
    """Set the model objective, replacing any existing one.

    ``objective`` is a CVXPY expression, a number, or a function ``f(model)``
    returning one. ``sense`` is ``"minimize"`` or ``"maximize"``.
    """
    if sense not in ("minimize", "maximize"):
        raise ValueError("sense must be 'minimize' or 'maximize'")
    if callable(objective) and not isinstance(objective, cp.Expression):
        objective = objective(model)
    if not isinstance(objective, cp.Expression):
        objective = cp.Constant(objective)
    model.objective_expression = objective
    model.objective_sense = sense
    model.problem = None


# ============ Backward-compatible names (same behavior as the scalar package) =========

none_rule = no_objective
loss_objective_rule = loss
loss_objective_i2r_rule = loss_i2r
substation_power_objective_rule = substation_power
substation_cost_objective_rule = substation_cost
voltage_deviation_objective_rule = voltage_deviation
voltage_min_objective_rule = voltage_min
generation_curtailment_objective_rule = generation_curtailment
cost_minimization_rule = energy_and_generation_cost
gen_cost_rule = generation_cost
total_cost_rule = total_cost
generation_cost_with_substation_quadratic_penalty_objective_rule = (
    generation_cost_with_substation_penalty
)
demand_charge_objective_rule = demand_charge_cost
combined_energy_and_demand_charge_objective_rule = energy_and_demand_charge
