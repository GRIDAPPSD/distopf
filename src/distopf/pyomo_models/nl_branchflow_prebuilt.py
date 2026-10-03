"""Thin wrapper for backward compatibility with NLBranchFlow model creation.

This module provides a simple wrapper around the factory pattern for creating
and constraining nonlinear BranchFlow models. For new code, prefer using
create_nl_branchflow_model() and add_nlp_constraints() directly.
"""

from distopf.pyomo_models.nl_branchflow import create_nl_branchflow_model
from distopf.pyomo_models.constraints_nlp import add_nlp_constraints
from distopf.api import Case


class NLBranchFlow:
    """Thin wrapper for creating a fully-constrained nonlinear BranchFlow model.

    This class provides backward compatibility. For new code, use:
        model = create_nl_branchflow_model(case)
        add_nlp_constraints(model, ...)

    Parameters
    ----------
    case : Case
        The power system case data
    circular_constraints : bool, default True
        Use circular (quadratic) constraints for generators/batteries
    control_regulators : bool, default False
        Enable regulator tap control (requires MINLP solver)
    control_capacitors : bool, default False
        Enable capacitor switching (requires MINLP solver)
    """

    def __init__(
        self,
        case: Case,
        circular_constraints: bool = True,
        control_regulators: bool = False,
        control_capacitors: bool = False,
    ):
        self.model = create_nl_branchflow_model(case)
        add_nlp_constraints(
            self.model,
            circular_constraints=circular_constraints,
            control_regulators=control_regulators,
            control_capacitors=control_capacitors,
        )
        self.case = case
