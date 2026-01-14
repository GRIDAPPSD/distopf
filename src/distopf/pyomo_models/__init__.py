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
>>> pyo_opf.add_standard_constraints(model)
>>> model.objective = pyo_opf.loss_objective
>>> result = pyo_opf.solve(model)
"""

# Model creation
from distopf.pyomo_models.lindist import create_lindist_model

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
from distopf.pyomo_models.results import OpfResult, get_values, get_voltages

# Objectives
from distopf.pyomo_models.objectives import loss_objective, loss_objective_rule

# Solver
from distopf.pyomo_models.solvers import solve


def add_standard_constraints(model) -> None:
    """
    Add standard LinDistFlow constraints to a Pyomo model.

    This is a convenience function that adds the most commonly used
    constraints for a basic OPF problem. For more control, add
    constraints individually.

    Adds:
    - Power flow balance (P and Q)
    - Voltage drop constraints
    - Swing bus constraints
    - CVR load constraints
    - Voltage limits
    - Capacitor and regulator constraints
    - Generator limits and control constraints
    - Battery constraints (if batteries present)

    Parameters
    ----------
    model : pyo.ConcreteModel
        Pyomo model created by create_lindist_model()
    """
    # Power flow
    add_p_flow_constraints(model)
    add_q_flow_constraints(model)

    # Voltages
    add_voltage_drop_constraints(model)
    add_swing_bus_constraints(model)
    add_voltage_limits(model)

    # Loads and devices
    add_cvr_load_constraints(model)
    add_capacitor_constraints(model)
    add_regulator_constraints(model)

    # Generators
    add_generator_limits(model)
    add_generator_constant_p_constraints_q_control(model)
    add_generator_constant_q_constraints_p_control(model)
    add_circular_generator_constraints_pq_control(model)

    # Batteries (safe to call even if no batteries)
    add_battery_power_limits(model)
    add_battery_soc_limits(model)
    add_battery_net_p_bat_equal_phase_constraints(model)
    add_battery_energy_constraints(model)
    add_battery_constant_q_constraints_p_control(model)


__all__ = [
    # Model creation
    "create_lindist_model",
    # Convenience function
    "add_standard_constraints",
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
    "OpfResult",
    "get_values",
    "get_voltages",
    # Objectives
    "loss_objective",
    "loss_objective_rule",
    # Solver
    "solve",
]
