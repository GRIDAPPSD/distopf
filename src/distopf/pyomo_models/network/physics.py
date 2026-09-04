"""Formulation-specific physics exports organized under the network package."""

from distopf.pyomo_models.lindist_constraints import (
    add_constraints as add_lindist_constraints,
    add_p_flow_constraints,
    add_q_flow_constraints,
    add_voltage_drop_constraints,
)
from distopf.pyomo_models.nl_bfm_constraints import (
    add_nlp_constraints,
)

__all__ = [
    "add_constraints",
    "add_lindist_constraints",
    "add_nlp_constraints",
    "add_p_flow_constraints",
    "add_q_flow_constraints",
    "add_voltage_drop_constraints",
]

# Generic alias for callers selecting the LinDistFlow physics package.
add_constraints = add_lindist_constraints
