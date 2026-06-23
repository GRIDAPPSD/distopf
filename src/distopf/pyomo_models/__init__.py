"""
Pyomo-based optimal power flow models for DistOPF.

This module provides Pyomo ConcreteModel builders with modular constraints
for non-linear optimization using IPOPT or other NLP solvers.

Quick Start
-----------
>>> import distopf as opf
>>> from distopf import pyomo_models as pyo_opf
>>>
>>> case = opf.create_case(opf.CASES_DIR / "csv" / "ieee123_30der")
>>> model = pyo_opf.create_lindist_model(case)
>>> pyo_opf.add_constraints(model)
>>> model.objective = pyo_opf.loss_objective
>>> result = pyo_opf.solve(model)
"""

# Model creation
from distopf.pyomo_models.lindist import (
    create_lindist_model,
    add_constraints,
    LinDistModel,
)

# Constraint functions - Power flow
from distopf.pyomo_models.constraints import (
    add_p_flow_constraints,
    add_q_flow_constraints,
    add_voltage_drop_constraints,
    add_swing_bus_constraints,
)

# Constraint functions - Voltage and limits
from distopf.pyomo_models.constraints import (
    add_voltage_limits,
    add_generator_limits,
)

# Constraint functions - Loads and devices
from distopf.pyomo_models.constraints import (
    add_cvr_load_constraints,
    add_capacitor_constraints,
    add_regulator_constraints,
)

# Constraint functions - Generators
from distopf.pyomo_models.constraints import (
    add_generator_constant_p_constraints,
    add_generator_constant_q_constraints,
    add_generator_constant_p_constraints_q_control,
    add_generator_constant_q_constraints_p_control,
    add_octagonal_inverter_constraints_pq_control,
    add_circular_generator_constraints_pq_control,
)

# Constraint functions - Batteries
from distopf.pyomo_models.constraints import (
    add_battery_power_limits,
    add_battery_soc_limits,
    add_battery_net_p_bat_constraints,
    add_battery_net_p_bat_equal_phase_constraints,
    add_battery_energy_constraints,
    add_battery_constant_q_constraints_p_control,
)

# Results extraction
from distopf.pyomo_models.results import PyoResult, get_values, get_voltages

# Objectives
from distopf.pyomo_models.objectives import (
    none_rule,
    loss_objective,
    loss_objective_rule,
    create_penalized_objective,
    set_objective,
)

# Solver
from distopf.pyomo_models.solvers import solve


__all__ = [
    # Model creation
    "create_lindist_model",
    "add_constraints",
    "LinDistModel",
    # Power flow constraints
    "add_p_flow_constraints",
    "add_q_flow_constraints",
    "add_voltage_drop_constraints",
    "add_swing_bus_constraints",
    # Voltage and limits
    "add_voltage_limits",
    "add_generator_limits",
    # Loads and devices
    "add_cvr_load_constraints",
    "add_capacitor_constraints",
    "add_regulator_constraints",
    # Generator constraints
    "add_generator_constant_p_constraints",
    "add_generator_constant_q_constraints",
    "add_generator_constant_p_constraints_q_control",
    "add_generator_constant_q_constraints_p_control",
    "add_octagonal_inverter_constraints_pq_control",
    "add_circular_generator_constraints_pq_control",
    # Battery constraints
    "add_battery_power_limits",
    "add_battery_soc_limits",
    "add_battery_net_p_bat_constraints",
    "add_battery_net_p_bat_equal_phase_constraints",
    "add_battery_energy_constraints",
    "add_battery_constant_q_constraints_p_control",
    # Results
    "PyoResult",
    "get_values",
    "get_voltages",
    # Objectives
    "none_rule",
    "loss_objective",
    "loss_objective_rule",
    "create_penalized_objective",
    "set_objective",
    # Solver
    "solve",
]
