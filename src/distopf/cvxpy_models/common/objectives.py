"""Objective expressions for CVXPY models.

Mirror of ``pyomo_models.common.objectives``. Each rule takes a ``CvxModel``
and returns a CVXPY expression. ``set_objective`` stores an expression (or a
rule) on the model; the problem is assembled at solve time.

Differences from Pyomo
----------------------
- The violation penalties use ``cp.pos`` (exact ``max(0, x)``) instead of the
  smooth ``(sqrt(x^2 + eps) + x) / 2`` approximation, which is not DCP. The
  two agree to about 1e-4 relative to the weight. Thermal, generator and
  battery penalties apply ``cp.pos`` to ``P^2 + Q^2 - S^2``, which is convex.
- ``demand_charge_objective_rule`` uses ``cp.max`` (valid and convex).
- The module-level prebuilt objective instances are not provided.
"""

from __future__ import annotations

from typing import Any, Callable

import cvxpy as cp

from distopf.cvxpy_models.common.model import CvxModel


def none_rule(model: CvxModel):
    """Trivial objective function that returns 0."""
    return 0


def loss_objective_rule(model: CvxModel):
    """Total losses ``sum((P^2 + Q^2) * r)`` (convex quadratic)."""
    total_loss = 0
    for fb, tb, ph in model.branch_phase_set:
        for t in model.time_set:
            r = model.r[fb, tb, ph + ph]
            total_loss += (model.p_flow[fb, tb, ph, t] ** 2) * r
            total_loss += (model.q_flow[fb, tb, ph, t] ** 2) * r
    return total_loss

def loss_objective_i2r_rule(model: CvxModel):
    """Total total losses based on current: sum(l_flow*r) (linear)."""
    if not hasattr(model, "l_flow"):
        raise AttributeError("Model does not have l_flow attribute required for I^2*R loss calculation.")
    total_loss = 0
    for fb, tb, ph in model.branch_phase_set:
        for t in model.time_set:
            r = model.r[fb, tb, ph + ph]
            total_loss += model.l_flow[fb, tb, ph + ph, t] * r
    return total_loss

def substation_power_objective_rule(model: CvxModel):
    """Total active power on branches leaving swing buses (linear)."""
    total_power = 0
    for fb, tb, ph in model.branch_phase_set:
        for t in model.time_set:
            if fb in model.swing_bus_set:
                total_power += model.p_flow[fb, tb, ph, t]
    return total_power


def voltage_deviation_objective_rule(model: CvxModel):
    """``sum((v^2 - 1)^2)`` (convex quadratic)."""
    total_deviation = 0
    for _id, ph in model.bus_phase_set:
        for t in model.time_set:
            total_deviation += (model.v2[_id, ph, t] - 1.0) ** 2
    return total_deviation


def voltage_min_objective_rule(model: CvxModel):
    """
    Minimize the sum of squared voltage magnitudes: min sum(v2).

    Uses the model's v2 variable (voltage squared), so the objective is linear.

    Parameters
    ----------
    model : CvxModel
        CVXPY model

    Returns
    -------
    CVXPY expression for the sum of v2 over all buses, phases and time steps
    """
    total_voltage = 0
    for _id, ph in model.bus_phase_set:
        for t in model.time_set:
            total_voltage += model.v2[_id, ph, t]
    return total_voltage


def generation_curtailment_objective_rule(model: CvxModel):
    """Difference between available and actual generation (linear)."""
    total_curtailment = 0
    for device, ph in model.gen_device_phase_set:
        for t in model.time_set:
            total_curtailment += (
                model.gen_p_available[device, ph, t] - model.p_gen[device, ph, t]
            )
    return total_curtailment


def cost_minimization_rule(m: CvxModel):
    """Minimize energy procurement cost from swing bus plus generation cost."""
    total_cost = 0
    for fb, tb, ph in m.branch_phase_set:
        for t in m.time_set:
            if fb in m.swing_bus_set:
                total_cost += m.p_flow[fb, tb, ph, t] * m.price[t] * m.delta_t
    for device, ph in m.gen_device_phase_set:
        for t in m.time_set:
            total_cost += m.p_gen[device, ph, t] * m.gen_cost[device, ph] * m.delta_t
    return total_cost


def substation_cost_objective_rule(model: CvxModel):
    """Cost of substation energy using ``schedule_price``."""
    total_cost = 0
    for fb, tb, ph in model.branch_phase_set:
        for t in model.time_set:
            if fb in model.swing_bus_set:
                total_cost += model.p_flow[fb, tb, ph, t] * model.schedule_price[t]
    return total_cost


def gen_cost_rule(model: CvxModel):
    generation_cost = 0
    for device, ph in model.gen_device_phase_set:
        for t in model.time_set:
            generation_cost += (
                model.p_gen[device, ph, t] * model.gen_cost[device, ph] * model.delta_t
            )
    return generation_cost


def total_cost_rule(model: CvxModel):
    return gen_cost_rule(model) + substation_cost_objective_rule(model)


def generation_cost_with_substation_quadratic_penalty_objective_rule(
    model: CvxModel,
    substation_penalty_weight: float = 1e6,
):
    """Generation cost plus a quadratic penalty on substation active power.

    The Pyomo version ignores ``substation_penalty_weight`` and hard-codes
    ``1e6``; the default here is also ``1e6`` so results agree at defaults.
    """
    substation_penalty = 0
    for fb, tb, ph in model.branch_phase_set:
        for t in model.time_set:
            if fb in model.swing_bus_set:
                substation_penalty += model.p_flow[fb, tb, ph, t] ** 2
    return gen_cost_rule(model) + substation_penalty_weight * substation_penalty


def demand_charge_objective_rule(model: CvxModel):
    """Demand charge on the peak substation active power (convex ``max``)."""
    if not hasattr(model, "demand_charge"):
        return 0
    flows = [
        model.p_flow[fb, tb, ph, t]
        for fb, tb, ph in model.branch_phase_set
        if fb in model.swing_bus_set
        for t in model.time_set
    ]
    if not flows:
        return 0
    return cp.max(cp.hstack(flows)) * model.demand_charge


def combined_energy_and_demand_charge_objective_rule(model: CvxModel):
    return substation_cost_objective_rule(model) + demand_charge_objective_rule(model)


# ============ Penalty Functions for Soft Constraints ==================================


def voltage_violation_penalty(model: CvxModel, weight: float = 1e3):
    """Exact one-sided penalty ``weight * sum(pos(v2_min - v2) + pos(v2 - v2_max))``."""
    penalty = 0
    for _id, ph in model.bus_phase_set:
        v2_min = model.v_min[_id, ph] ** 2
        v2_max = model.v_max[_id, ph] ** 2
        for t in model.time_set:
            penalty += cp.pos(v2_min - model.v2[_id, ph, t])
            penalty += cp.pos(model.v2[_id, ph, t] - v2_max)
    return weight * penalty


def voltage_violation_penalty_quadratic(model: CvxModel, weight: float = 1e3):
    """Quadratic preference for mid-range voltages."""
    penalty = 0
    for _id, ph in model.bus_phase_set:
        v2_min = model.v_min[_id, ph] ** 2
        v2_max = model.v_max[_id, ph] ** 2
        v2_mid = (v2_min + v2_max) / 2
        v2_range = (v2_max - v2_min) / 2
        for t in model.time_set:
            normalized_dev = (model.v2[_id, ph, t] - v2_mid) / v2_range
            penalty += normalized_dev**2
    return weight * penalty


def thermal_violation_penalty(model: CvxModel, weight: float = 1e3):
    """``weight * sum(pos(P^2 + Q^2 - S_max^2))`` (convex)."""
    if not hasattr(model, "s_branch_max"):
        return 0
    penalty = 0
    for fb, tb, ph in model.branch_phase_set:
        s_max_val = model.s_branch_max.get((fb, tb, ph))
        if s_max_val is None or s_max_val <= 0:
            continue
        for t in model.time_set:
            s_sq = model.p_flow[fb, tb, ph, t] ** 2 + model.q_flow[fb, tb, ph, t] ** 2
            penalty += cp.pos(s_sq - s_max_val**2)
    return weight * penalty


def generator_violation_penalty(model: CvxModel, weight: float = 1e3):
    if len(model.gen_device_phase_set) == 0:
        return 0
    penalty = 0
    for device, ph in model.gen_device_phase_set:
        s_rated_sq = model.gen_s_max[device, ph] ** 2
        for t in model.time_set:
            s_sq = model.p_gen[device, ph, t] ** 2 + model.q_gen[device, ph, t] ** 2
            penalty += cp.pos(s_sq - s_rated_sq)
    return weight * penalty


def battery_violation_penalty(model: CvxModel, weight: float = 1e3):
    if len(model.bat_phase_set) == 0:
        return 0
    penalty = 0
    for _id, ph in model.bat_phase_set:
        s_rated_sq = model.s_bat_rated[_id, ph] ** 2
        for t in model.time_set:
            s_sq = model.p_bat[_id, ph, t] ** 2 + model.q_bat[_id, ph, t] ** 2
            penalty += cp.pos(s_sq - s_rated_sq)
    return weight * penalty


def soc_violation_penalty(model: CvxModel, weight: float = 1e3):
    if len(model.bat_set) == 0:
        return 0
    penalty = 0
    for _id in model.bat_set:
        soc_min = model.soc_min[_id]
        soc_max = model.soc_max[_id]
        for t in model.time_set:
            penalty += cp.pos(soc_min - model.soc[_id, t])
            penalty += cp.pos(model.soc[_id, t] - soc_max)
    return weight * penalty


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


def admm_v_up_penalty(m: CvxModel, rho: float = 1.0):
    if not hasattr(m, "swing_phase_set"):
        return 0
    penalty = 0
    for _id, ph in m.swing_phase_set:
        for t in m.time_set:
            v_target_sq = m.v_swing[_id, ph, t] ** 2
            penalty += (m.v2[_id, ph, t] - v_target_sq) ** 2
    return 0.5 * rho * penalty


def admm_s_up_penalty(m: CvxModel, rho: float = 1.0):
    swing_branches: dict[tuple[int, str], list[tuple[int, int]]] = {}
    for fb, tb, ph in m.branch_phase_set:
        if fb not in m.swing_bus_set:
            continue
        swing_branches.setdefault((fb, ph), []).append((fb, tb))

    penalty = 0
    for (fb, ph), branches in swing_branches.items():
        area_name = m.name_map[fb]
        p_param = getattr(m, f"schedule_{area_name}.{ph}.p", None)
        q_param = getattr(m, f"schedule_{area_name}.{ph}.q", None)
        if p_param is None or q_param is None:
            continue
        for t in m.time_set:
            p_total = sum(m.p_flow[f, tt, ph, t] for f, tt in branches)
            q_total = sum(m.q_flow[f, tt, ph, t] for f, tt in branches)
            penalty += (p_total - p_param[t]) ** 2
            penalty += (q_total - q_param[t]) ** 2
    return 0.5 * rho * penalty


def admm_v_down_penalty(m: CvxModel, rho: float = 1.0):
    penalty = 0
    for _id, ph in m.bus_phase_set:
        area_name = m.name_map[_id]
        v_param = getattr(m, f"schedule_{area_name}.{ph}.v", None)
        if v_param is None:
            continue
        for t in m.time_set:
            v_target_sq = v_param[t] ** 2
            penalty += (m.v2[_id, ph, t] - v_target_sq) ** 2
    return 0.5 * rho * penalty


def admm_s_down_penalty(m: CvxModel, rho: float = 1.0):
    penalty = 0
    for _id, ph in m.bus_phase_set:
        if _id in m.swing_bus_set:
            continue
        area_name = m.name_map[_id]
        p_param = getattr(m, f"schedule_{area_name}.{ph}.p", None)
        q_param = getattr(m, f"schedule_{area_name}.{ph}.q", None)
        if p_param is None or q_param is None:
            continue
        for t in m.time_set:
            penalty += (m.p_load[_id, ph, t] - p_param[t]) ** 2
            penalty += (m.q_load[_id, ph, t] - q_param[t]) ** 2
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


def add_penalized_loss_objective(
    model: CvxModel,
    voltage_weight: float = 1e3,
    thermal_weight: float = 1e3,
    generator_weight: float = 1e3,
    battery_weight: float = 1e3,
    soc_weight: float = 1e3,
) -> None:
    set_objective(
        model,
        create_penalized_objective(
            loss_objective_rule,
            voltage_weight=voltage_weight,
            thermal_weight=thermal_weight,
            generator_weight=generator_weight,
            battery_weight=battery_weight,
            soc_weight=soc_weight,
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
    set_objective(
        model,
        create_penalized_objective(
            substation_power_objective_rule,
            voltage_weight=voltage_weight,
            thermal_weight=thermal_weight,
            generator_weight=generator_weight,
            battery_weight=battery_weight,
            soc_weight=soc_weight,
        ),
    )


def voltage_slack_penalty(m: CvxModel, weight=1e3):
    if not hasattr(m, "v2_slack"):
        return 0
    return weight * sum(m.v2_slack[_id, ph, t] for _id, ph in m.bus_phase_set for t in m.time_set)


def thermal_slack_penalty(m: CvxModel, weight=1e3):
    if not hasattr(m, "thermal_slack"):
        return 0
    return weight * sum(
        m.thermal_slack[fb, tb, ph, t]
        for fb, tb, ph in m.branch_phase_set
        for t in m.time_set
    )


def swing_voltage_slack_penalty(m: CvxModel, weight=1e3):
    if not hasattr(m, "swing_v2_slack"):
        return 0
    return weight * sum(
        m.swing_v2_slack[_id, ph, t] for _id, ph in m.swing_phase_set for t in m.time_set
    )
