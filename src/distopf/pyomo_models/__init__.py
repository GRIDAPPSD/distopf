"""Pyomo model builders and constraint APIs for DistOPF."""

from distopf.pyomo_models.lindist import LinDistModel, add_constraints, create_lindist_model
from distopf.pyomo_models.objectives import (
    add_generation_cost_with_substation_quadratic_penalty_objective,
    cost_minimization_rule,
    create_penalized_objective,
    generation_cost_with_substation_quadratic_penalty_objective,
    generation_cost_with_substation_quadratic_penalty_objective_rule,
    loss_objective,
    loss_objective_rule,
    none_rule,
    set_objective,
)
from distopf.pyomo_models.results import PyoResult, get_values, get_voltages
from distopf.pyomo_models.solvers import solve

# Canonical shared/formulation-specific constraint exports.
from distopf.pyomo_models.common_constraints import (
    add_battery_constant_q_constraints_p_control,
    add_battery_energy_constraints,
    add_battery_net_p_bat_constraints,
    add_battery_net_p_bat_equal_phase_constraints,
    add_battery_power_limits,
    add_battery_soc_limits,
    add_capacitor_constraints,
    add_circular_generator_constraints_pq_control,
    add_cvr_load_constraints,
    add_generator_constant_p_constraints,
    add_generator_constant_p_constraints_q_control,
    add_generator_constant_q_constraints,
    add_generator_constant_q_constraints_p_control,
    add_generator_limits,
    add_octagonal_inverter_constraints_pq_control,
    add_regulator_constraints,
    add_swing_bus_constraints,
    add_voltage_limits,
)
from distopf.pyomo_models.lindist_constraints import (
    add_p_flow_constraints,
    add_q_flow_constraints,
    add_voltage_drop_constraints,
)

__all__ = [
    "LinDistModel",
    "PyoResult",
    "add_constraints",
    "add_p_flow_constraints",
    "add_q_flow_constraints",
    "add_voltage_drop_constraints",
    "add_voltage_limits",
    "add_swing_bus_constraints",
    "add_cvr_load_constraints",
    "add_generator_limits",
    "add_generator_constant_p_constraints",
    "add_generator_constant_q_constraints",
    "add_generator_constant_p_constraints_q_control",
    "add_generator_constant_q_constraints_p_control",
    "add_octagonal_inverter_constraints_pq_control",
    "add_circular_generator_constraints_pq_control",
    "add_capacitor_constraints",
    "add_battery_power_limits",
    "add_battery_soc_limits",
    "add_battery_net_p_bat_constraints",
    "add_battery_net_p_bat_equal_phase_constraints",
    "add_battery_energy_constraints",
    "add_battery_constant_q_constraints_p_control",
    "add_regulator_constraints",
    "create_lindist_model",
    "create_penalized_objective",
    "get_values",
    "get_voltages",
    "loss_objective",
    "loss_objective_rule",
    "none_rule",
    "set_objective",
    "solve",
    "cost_minimization_rule",
    "generation_cost_with_substation_quadratic_penalty_objective",
    "generation_cost_with_substation_quadratic_penalty_objective_rule",
    "add_generation_cost_with_substation_quadratic_penalty_objective",
]
