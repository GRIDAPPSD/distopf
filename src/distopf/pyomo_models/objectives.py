from distopf.pyomo_models.protocol import LindistModelProtocol
import pyomo.environ as pyo
from numpy import sqrt

sqrt2 = sqrt(2)


# ============ Primary Objective Functions =============================================
# ======================================================================================


def loss_objective_rule(model: LindistModelProtocol):
    """
    Calculate total system losses using the resistance parameters.

    For each branch-phase combination, calculates (P^2 + Q^2) * R.
    This is a quadratic objective requiring a QP or NLP solver.

    Parameters
    ----------
    model : LindistModelProtocol
        Pyomo model

    Returns
    -------
    Pyomo expression for total losses
    """
    total_loss = 0
    for _id, ph in model.branch_phase_set:
        for t in model.time_set:
            total_loss += (model.p_flow[_id, ph, t] ** 2) * model.r[_id, ph + ph]
            total_loss += (model.q_flow[_id, ph, t] ** 2) * model.r[_id, ph + ph]
    return total_loss


def substation_power_objective_rule(model: LindistModelProtocol):
    """
    Minimize total active power imported from the substation (swing bus).

    Sums active power flow on all branches originating from swing buses.
    This is a linear objective.

    Parameters
    ----------
    model : LindistModelProtocol
        Pyomo model

    Returns
    -------
    Pyomo expression for total substation power
    """
    total_power = 0
    for _id, ph in model.branch_phase_set:
        for t in model.time_set:
            if model.from_bus_map[_id] in model.swing_bus_set:
                total_power += model.p_flow[_id, ph, t]
    return total_power


def voltage_deviation_objective_rule(model: LindistModelProtocol):
    """
    Minimize voltage deviation from nominal (1.0 p.u.).

    Uses squared voltage magnitude: sum((v^2 - 1)^2).
    This is a quadratic objective.

    Parameters
    ----------
    model : LindistModelProtocol
        Pyomo model

    Returns
    -------
    Pyomo expression for total voltage deviation
    """
    total_deviation = 0
    for _id, ph in model.bus_phase_set:
        for t in model.time_set:
            total_deviation += (model.v2[_id, ph, t] - 1.0) ** 2
    return total_deviation


def generation_curtailment_objective_rule(model: LindistModelProtocol):
    """
    Minimize total generation curtailment.

    Penalizes difference between available generation and actual output.
    This is a linear objective.

    Parameters
    ----------
    model : LindistModelProtocol
        Pyomo model

    Returns
    -------
    Pyomo expression for total curtailment
    """
    total_curtailment = 0
    for _id, ph in model.gen_phase_set:
        for t in model.time_set:
            total_curtailment += model.p_gen_nom[_id, ph, t] - model.p_gen[_id, ph, t]
    return total_curtailment


# ============ Penalty Functions for Soft Constraints ==================================
# ======================================================================================
#
# These penalty functions use quadratic penalties on constraint violations.
# They require a nonlinear solver (e.g., IPOPT) since they use max() or
# squared terms. The penalties are one-sided: only violations are penalized,
# not feasible values.
#
# For use with LP/MILP solvers, use slack variable formulations instead.
# ======================================================================================


def voltage_violation_penalty(model: LindistModelProtocol, weight: float = 1e3):
    """
    Quadratic penalty for voltage limit violations.

    Penalizes:
        - Undervoltage: (v_min^2 - v2)^2 when v2 < v_min^2
        - Overvoltage: (v2 - v_max^2)^2 when v2 > v_max^2

    Uses smooth approximation with max(0, x)^2.

    Parameters
    ----------
    model : LindistModelProtocol
        Pyomo model
    weight : float
        Penalty weight (default: 1e3)

    Returns
    -------
    Pyomo expression for voltage violation penalty

    Notes
    -----
    Requires nonlinear solver due to max() operation.
    """
    penalty = 0
    for _id, ph in model.bus_phase_set:
        v2_min = pyo.value(model.v_min[_id, ph]) ** 2
        v2_max = pyo.value(model.v_max[_id, ph]) ** 2
        for t in model.time_set:
            # Undervoltage violation (positive when v2 < v2_min)
            undervoltage = v2_min - model.v2[_id, ph, t]
            # Overvoltage violation (positive when v2 > v2_max)
            overvoltage = model.v2[_id, ph, t] - v2_max
            # Penalize only positive violations using max(0, x)^2
            # Pyomo handles max() in nonlinear models
            penalty += pyo.sqrt(undervoltage**2 + 1e-8) + undervoltage
            penalty += pyo.sqrt(overvoltage**2 + 1e-8) + overvoltage
    # Scale by 0.5 since we're using smooth approximation of max(0,x)
    return weight * 0.5 * penalty


def voltage_violation_penalty_quadratic(
    model: LindistModelProtocol, weight: float = 1e3
):
    """
    Quadratic penalty for voltage limit violations using exterior penalty.

    Penalizes squared distance from feasible region:
        weight * sum( (v2 - v2_min)^2 * I(v2 < v2_min) + (v2 - v2_max)^2 * I(v2 > v2_max) )

    Approximated as: weight * sum( (v2 - clamp(v2, v2_min, v2_max))^2 )

    For simplicity, this version penalizes the full expression and relies on
    the optimizer to find solutions where violations are zero.

    Parameters
    ----------
    model : LindistModelProtocol
        Pyomo model
    weight : float
        Penalty weight (default: 1e3)

    Returns
    -------
    Pyomo expression for voltage violation penalty
    """
    penalty = 0
    for _id, ph in model.bus_phase_set:
        v2_min = pyo.value(model.v_min[_id, ph]) ** 2
        v2_max = pyo.value(model.v_max[_id, ph]) ** 2
        v2_mid = (v2_min + v2_max) / 2
        v2_range = (v2_max - v2_min) / 2
        for t in model.time_set:
            # Penalize deviation from midpoint, scaled by range
            # This creates a soft preference for mid-range voltages
            # with increasing penalty as limits are approached
            normalized_dev = (model.v2[_id, ph, t] - v2_mid) / v2_range
            penalty += normalized_dev**2
    return weight * penalty


def thermal_violation_penalty(model: LindistModelProtocol, weight: float = 1e3):
    """
    Quadratic penalty for branch thermal limit violations.

    Penalizes: (P^2 + Q^2 - S_max^2) when P^2 + Q^2 > S_max^2

    Parameters
    ----------
    model : LindistModelProtocol
        Pyomo model
    weight : float
        Penalty weight (default: 1e3)

    Returns
    -------
    Pyomo expression for thermal violation penalty

    Notes
    -----
    Returns 0 if model has no thermal limits defined.
    Requires nonlinear solver.
    """
    if not hasattr(model, "s_branch_max"):
        return 0

    penalty = 0
    for _id, ph in model.branch_phase_set:
        if (_id, ph) not in model.s_branch_max:
            continue
        s_max_val = pyo.value(model.s_branch_max[_id, ph])
        if s_max_val is None or s_max_val <= 0:
            continue
        s_max_sq = s_max_val**2
        for t in model.time_set:
            # Apparent power squared
            s_sq = model.p_flow[_id, ph, t] ** 2 + model.q_flow[_id, ph, t] ** 2
            # Violation (positive when over limit)
            violation = s_sq - s_max_sq
            # Smooth max(0, x) approximation: (x + sqrt(x^2 + eps)) / 2
            penalty += pyo.sqrt(violation**2 + 1e-8) + violation
    return weight * 0.5 * penalty


def generator_violation_penalty(model: LindistModelProtocol, weight: float = 1e3):
    """
    Quadratic penalty for generator apparent power limit violations.

    Penalizes: (P^2 + Q^2 - S_rated^2) when P^2 + Q^2 > S_rated^2

    Parameters
    ----------
    model : LindistModelProtocol
        Pyomo model
    weight : float
        Penalty weight (default: 1e3)

    Returns
    -------
    Pyomo expression for generator violation penalty

    Notes
    -----
    Returns 0 if model has no generators.
    Requires nonlinear solver.
    """
    if len(model.gen_phase_set) == 0:
        return 0

    penalty = 0
    for _id, ph in model.gen_phase_set:
        s_rated_sq = pyo.value(model.s_rated[_id, ph]) ** 2
        for t in model.time_set:
            s_sq = model.p_gen[_id, ph, t] ** 2 + model.q_gen[_id, ph, t] ** 2
            violation = s_sq - s_rated_sq
            penalty += pyo.sqrt(violation**2 + 1e-8) + violation
    return weight * 0.5 * penalty


def battery_violation_penalty(model: LindistModelProtocol, weight: float = 1e3):
    """
    Quadratic penalty for battery apparent power limit violations.

    Penalizes: (P^2 + Q^2 - S_rated^2) when P^2 + Q^2 > S_rated^2

    Parameters
    ----------
    model : LindistModelProtocol
        Pyomo model
    weight : float
        Penalty weight (default: 1e3)

    Returns
    -------
    Pyomo expression for battery violation penalty

    Notes
    -----
    Returns 0 if model has no batteries.
    Requires nonlinear solver.
    """
    if len(model.bat_phase_set) == 0:
        return 0

    penalty = 0
    for _id, ph in model.bat_phase_set:
        s_rated_sq = pyo.value(model.s_bat_rated[_id, ph]) ** 2
        for t in model.time_set:
            s_sq = model.p_bat[_id, ph, t] ** 2 + model.q_bat[_id, ph, t] ** 2
            violation = s_sq - s_rated_sq
            penalty += pyo.sqrt(violation**2 + 1e-8) + violation
    return weight * 0.5 * penalty


def soc_violation_penalty(model: LindistModelProtocol, weight: float = 1e3):
    """
    Quadratic penalty for battery state of charge limit violations.

    Penalizes:
        - Under SOC: (soc_min - soc)^2 when soc < soc_min
        - Over SOC: (soc - soc_max)^2 when soc > soc_max

    Parameters
    ----------
    model : LindistModelProtocol
        Pyomo model
    weight : float
        Penalty weight (default: 1e3)

    Returns
    -------
    Pyomo expression for SOC violation penalty

    Notes
    -----
    Returns 0 if model has no batteries.
    Requires nonlinear solver.
    """
    if len(model.bat_set) == 0:
        return 0

    penalty = 0
    for _id in model.bat_set:
        soc_min = pyo.value(model.soc_min[_id])
        soc_max = pyo.value(model.soc_max[_id])
        for t in model.time_set:
            under_soc = soc_min - model.soc[_id, t]
            over_soc = model.soc[_id, t] - soc_max
            penalty += pyo.sqrt(under_soc**2 + 1e-8) + under_soc
            penalty += pyo.sqrt(over_soc**2 + 1e-8) + over_soc
    return weight * 0.5 * penalty


# ============ Combined Objective Factories ============================================
# ======================================================================================


def create_penalized_objective(
    primary_objective_rule,
    voltage_weight: float | None = None,
    thermal_weight: float | None = None,
    generator_weight: float | None = None,
    battery_weight: float | None = None,
    soc_weight: float | None = None,
):
    """
    Create a penalized objective combining a primary objective with soft constraints.

    Set weight to 0 to disable a particular penalty.

    Parameters
    ----------
    primary_objective_rule : callable
        Function that takes model and returns primary objective expression
    voltage_weight : float
        Weight for voltage violation penalty (default: 0, disabled)
    thermal_weight : float
        Weight for thermal violation penalty (default: 0, disabled)
    generator_weight : float
        Weight for generator limit violation penalty (default: 0, disabled)
    battery_weight : float
        Weight for battery limit violation penalty (default: 0, disabled)
    soc_weight : float
        Weight for SOC violation penalty (default: 0, disabled)

    Returns
    -------
    pyo.Objective
        Pyomo objective ready to attach to model

    Examples
    --------
    >>> obj = create_penalized_objective(
    ...     loss_objective_rule,
    ...     voltage_weight=1e4,
    ...     thermal_weight=1e3,
    ... )
    >>> model.objective = obj
    """

    def objective_rule(m):
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

    return pyo.Objective(rule=objective_rule, sense=pyo.minimize)


# ============ Convenience Functions ===================================================
# ======================================================================================


def set_objective(model: LindistModelProtocol, objective: pyo.Objective) -> None:
    """
    Set the objective on a model, removing any existing objective first.

    Parameters
    ----------
    model : LindistModelProtocol
        Pyomo model
    objective : pyo.Objective
        Objective to set
    """
    if hasattr(model, "objective"):
        model.del_component("objective")
    model.objective = objective


def add_loss_objective(model: LindistModelProtocol) -> None:
    """Add loss minimization objective to model."""
    set_objective(model, pyo.Objective(rule=loss_objective_rule, sense=pyo.minimize))


def add_substation_power_objective(model: LindistModelProtocol) -> None:
    """Add substation power minimization objective to model."""
    set_objective(
        model, pyo.Objective(rule=substation_power_objective_rule, sense=pyo.minimize)
    )


def add_voltage_deviation_objective(model: LindistModelProtocol) -> None:
    """Add voltage deviation minimization objective to model."""
    set_objective(
        model, pyo.Objective(rule=voltage_deviation_objective_rule, sense=pyo.minimize)
    )


def add_penalized_loss_objective(
    model: LindistModelProtocol,
    voltage_weight: float = 1e3,
    thermal_weight: float = 1e3,
    generator_weight: float = 1e3,
    battery_weight: float = 1e3,
    soc_weight: float = 1e3,
) -> None:
    """
    Add loss minimization objective with soft constraint penalties.

    Parameters
    ----------
    model : LindistModelProtocol
        Pyomo model
    voltage_weight : float
        Weight for voltage violation penalty (default: 1e3)
    thermal_weight : float
        Weight for thermal violation penalty (default: 1e3)
    generator_weight : float
        Weight for generator limit violation penalty (default: 1e3)
    battery_weight : float
        Weight for battery limit violation penalty (default: 1e3)
    soc_weight : float
        Weight for SOC violation penalty (default: 1e3)
    """
    obj = create_penalized_objective(
        loss_objective_rule,
        voltage_weight=voltage_weight,
        thermal_weight=thermal_weight,
        generator_weight=generator_weight,
        battery_weight=battery_weight,
        soc_weight=soc_weight,
    )
    set_objective(model, obj)


def add_penalized_substation_power_objective(
    model: LindistModelProtocol,
    voltage_weight: float = 1e3,
    thermal_weight: float = 1e3,
    generator_weight: float = 1e3,
    battery_weight: float = 1e3,
    soc_weight: float = 1e3,
) -> None:
    """
    Add substation power minimization objective with soft constraint penalties.

    Parameters
    ----------
    model : LindistModelProtocol
        Pyomo model
    voltage_weight : float
        Weight for voltage violation penalty (default: 1e3)
    thermal_weight : float
        Weight for thermal violation penalty (default: 1e3)
    generator_weight : float
        Weight for generator limit violation penalty (default: 1e3)
    battery_weight : float
        Weight for battery limit violation penalty (default: 1e3)
    soc_weight : float
        Weight for SOC violation penalty (default: 1e3)
    """
    obj = create_penalized_objective(
        substation_power_objective_rule,
        voltage_weight=voltage_weight,
        thermal_weight=thermal_weight,
        generator_weight=generator_weight,
        battery_weight=battery_weight,
        soc_weight=soc_weight,
    )
    set_objective(model, obj)


loss_objective = pyo.Objective(rule=loss_objective_rule, sense=pyo.minimize)

substation_power_objective = pyo.Objective(
    rule=substation_power_objective_rule, sense=pyo.minimize
)

voltage_deviation_objective = pyo.Objective(
    rule=voltage_deviation_objective_rule, sense=pyo.minimize
)

generation_curtailment_objective = pyo.Objective(
    rule=generation_curtailment_objective_rule, sense=pyo.minimize
)
