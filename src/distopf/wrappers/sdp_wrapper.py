"""
SDP BranchFlow wrapper.

Mirrors PyomoWrapper in structure and public API so that
Case.run_opf(formulation='sdp_branchflow') works identically
to other formulations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

from distopf.wrappers.base import Wrapper

if TYPE_CHECKING:
    import pandas as pd
    from distopf.results import PowerFlowResult


class SdpWrapper(Wrapper):
    """CVXPY/SDP wrapper for the SDP BranchFlow OPF formulation."""

    def solve(
        self,
        objective: Optional[Any] = None,
        control_regulators: bool = False,
        control_capacitors: bool = False,
        raw_result: bool = False,
        **kwargs: Any,
    ) -> Union["PowerFlowResult", Any]:
        """
        Run SDP BranchFlow OPF.

        Parameters
        ----------
        objective : str, callable, or None
            Optimization objective. Supported strings:
            'loss', 'substation', 'voltage_deviation', 'curtail'
        control_regulators : bool
            Not supported in the convex SDP formulation.
            A warning is issued and the flag is ignored.
        control_capacitors : bool
            Voltage-dependent capacitor constraints are always included.
            Binary switching is not supported (breaks convexity).
        raw_result : bool
            If True, return the raw cp.Problem object instead of
            PowerFlowResult.
        **kwargs
            solver : str or None
                CVXPY solver name ('CLARABEL', 'SCS', 'MOSEK').
                Default: auto-selected (CLARABEL > SCS > MOSEK > CVXOPT).
            thermal_constraints : bool, default False
                Add branch thermal limit (SOC) constraints.
            verbose : bool, default False
                Print solver output.
            Any remaining kwargs are forwarded to cp.Problem.solve()
            (e.g. max_iters=50000, eps_abs=1e-8).

        Returns
        -------
        PowerFlowResult
            Unified result object. If raw_result=True, returns the
            underlying cp.Problem instead.
        """
        import warnings

        from distopf.cvxpy_models.sdp_branchflow import create_sdp_branchflow_model
        from distopf.cvxpy_models.constraints_sdp import add_sdp_constraints
        from distopf.cvxpy_models.solvers_sdp import solve_sdp

        if control_regulators:
            warnings.warn(
                "control_regulators=True is not supported in the SDP formulation "
                "(integer variables break convexity). Ignoring.",
                UserWarning,
                stacklevel=2,
            )

        if control_capacitors:
            warnings.warn(
                "control_capacitors=True: binary capacitor switching is not "
                "supported in the convex SDP formulation. "
                "Continuous voltage-dependent capacitor constraints will be used.",
                UserWarning,
                stacklevel=2,
            )

        thermal_constraints = kwargs.pop("thermal_constraints", False)
        solver_name = kwargs.pop("solver", None)
        verbose = kwargs.pop("verbose", False)        
        kwargs.pop("duals", None)
        kwargs.pop("model_type", None)
        # Remaining kwargs forwarded to cp.Problem.solve()

        # Build model and variables
        self.model = create_sdp_branchflow_model(self.case)

        # Add all constraints
        add_sdp_constraints(
            self.model,
            thermal_constraints=thermal_constraints,
        )

        # Resolve and evaluate objective expression
        obj_fn = self._resolve_objective(objective)
        obj_expr = obj_fn(self.model)

        # Solve — returns PowerFlowResult directly
        result = solve_sdp(
            self.model,
            objective=obj_expr,
            solver=solver_name,
            verbose=verbose,
            **kwargs,  # e.g. max_iters, eps_abs
        )

        self.result = result

        if raw_result:
            return result.raw_result  # cp.Problem object

        return result

    def _resolve_objective(self, objective: Any):
        """Resolve objective string or callable to an SDP objective function.

        Mirrors PyomoWrapper._resolve_objective in structure.
        """
        from distopf.cvxpy_models.objectives_sdp import (
            none_objective_sdp,
            loss_objective_sdp,
            substation_power_objective_sdp,
            voltage_deviation_objective_sdp,
            generation_curtailment_objective_sdp,
        )

        if objective is None:
            return none_objective_sdp

        if callable(objective):
            return objective

        objective_map = {
            "loss":               loss_objective_sdp,
            "loss_min":           loss_objective_sdp,
            "substation":         substation_power_objective_sdp,
            "substation_power":   substation_power_objective_sdp,
            "voltage_deviation":  voltage_deviation_objective_sdp,
            "curtail":            generation_curtailment_objective_sdp,
            "curtail_min":        generation_curtailment_objective_sdp,
            "curtailment":        generation_curtailment_objective_sdp,
        }

        obj_lower = objective.lower().strip()
        if obj_lower in objective_map:
            return objective_map[obj_lower]

        raise ValueError(
            f"Unknown SDP objective: '{objective}'. "
            f"Supported: {', '.join(objective_map.keys())}"
        )

    # ------------------------------------------------------------------
    # Result accessor helpers  (mirror PyomoWrapper)
    # ------------------------------------------------------------------

    def get_voltages(self) -> "pd.DataFrame":
        return self.result.voltages

    def get_p_flows(self) -> "pd.DataFrame":
        return self.result.active_power_flows

    def get_q_flows(self) -> "pd.DataFrame":
        return self.result.reactive_power_flows

    def get_p_gens(self) -> "pd.DataFrame":
        return self.result.active_power_generation

    def get_q_gens(self) -> "pd.DataFrame":
        return self.result.reactive_power_generation