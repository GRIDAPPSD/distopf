"""Vectorized CVXPY optimal power flow models.

Same providers and constraint names as :mod:`distopf.cvxpy_models`, but variables
are arrays (``cp.Variable`` of shape ``(n_index, n_steps)``) and constraints are
built once for the whole network with sparse matrices.
"""

from .common.factory import create_lindist_model, create_nl_branchflow_model
from .common.model import CvxModel
from .common.objectives import (
    generation_curtailment,
    generation_curtailment_objective_rule,
    loss,
    loss_i2r,
    loss_objective_rule,
    no_objective,
    none_rule,
    set_objective,
    substation_cost,
    substation_cost_objective_rule,
    substation_power,
    substation_power_objective_rule,
    total_cost,
    total_cost_rule,
    voltage_deviation,
    voltage_deviation_objective_rule,
    voltage_min,
    voltage_min_objective_rule,
)
from .common.results import CvxResult, get_values, get_voltages
from .common.solvers import solve

__all__ = [
    "CvxModel",
    "CvxResult",
    "create_lindist_model",
    "create_nl_branchflow_model",
    "generation_curtailment",
    "generation_curtailment_objective_rule",
    "get_values",
    "get_voltages",
    "loss",
    "loss_i2r",
    "loss_objective_rule",
    "no_objective",
    "none_rule",
    "set_objective",
    "solve",
    "substation_cost",
    "substation_cost_objective_rule",
    "substation_power",
    "substation_power_objective_rule",
    "total_cost",
    "total_cost_rule",
    "voltage_deviation",
    "voltage_deviation_objective_rule",
    "voltage_min",
    "voltage_min_objective_rule",
]
