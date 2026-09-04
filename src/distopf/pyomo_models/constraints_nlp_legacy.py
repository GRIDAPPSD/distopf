"""Backward-compatible imports for the legacy NLP constraints module."""

from distopf.pyomo_models.common_constraints import *  # noqa: F401,F403
from distopf.pyomo_models.nl_bfm_constraints import (  # noqa: F401
    add_p_flow_constraints as add_p_flow_nlp_constraints,
    add_q_flow_constraints as add_q_flow_nlp_constraints,
    add_voltage_drop_constraints as add_voltage_drop_nlp_constraints,
    add_current_constraint1,
    add_current_constraint1_relaxed,
    add_current_constraint2,
    add_current_constraint2_relaxed,
    add_nlp_constraints,
)
