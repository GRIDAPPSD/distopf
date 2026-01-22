"""Pyomo backend for NLP-capable OPF (IPOPT solver)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union
from distopf.backends.base import Backend

if TYPE_CHECKING:
    import pandas as pd
    from distopf.results import PowerFlowResult


class PyomoBackend(Backend):
    """Pyomo/IPOPT backend for NLP optimization."""

    def solve(
        self,
        objective: Optional[Any] = None,
        control_regulators: bool = False,
        control_capacitors: bool = False,
        raw_result: bool = False,
        **kwargs: Any,
    ) -> Union[PowerFlowResult, Any]:
        """Run OPF using Pyomo/IPOPT backend (NLP-capable).

        Parameters
        ----------
        objective : str, callable, or None
            Optimization objective function
        control_regulators : bool
            Not supported in pyomo backend (ignored with warning)
        control_capacitors : bool
            Not supported in pyomo backend (ignored with warning)
        raw_result : bool
            If True, return raw Pyomo OpfResult object instead of PowerFlowResult
        **kwargs
            Additional solver options (note: solver is always IPOPT)

        Returns
        -------
        PowerFlowResult or raw result
            If raw_result=False: PowerFlowResult with all results
            If raw_result=True: Pyomo OpfResult object with all variable results
        """
        from distopf.pyomo_models import (
            create_lindist_model,
            add_standard_constraints,
            solve,
        )
        from distopf.results import PowerFlowResult
        import pyomo.environ as pyo  # type: ignore[import-untyped]

        # Warn about unsupported parameters
        if control_regulators:
            self._warn_unsupported("pyomo", "control_regulators")
        if control_capacitors:
            self._warn_unsupported("pyomo", "control_capacitors")
        if "solver" in kwargs:
            self._warn_unsupported("pyomo", "solver kwarg (uses IPOPT)")

        # Create Pyomo model (uses gen_data.control_variable per-row)
        self.model = create_lindist_model(self.case)
        add_standard_constraints(self.model)

        # Set objective
        obj_rule = self._resolve_objective(objective)
        self.model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # Solve
        self.result = solve(self.model)

        if raw_result:
            return self.result

        # Extract results (result is OpfResult from solve())
        voltages_df = self.get_voltages()
        p_flows_df = self.get_p_flows()
        q_flows_df = self.get_q_flows()
        p_gens = self.get_p_gens()
        q_gens = self.get_q_gens()

        # Get metadata from pyomo result
        objective_value = None
        solve_time = None
        if hasattr(self.result, "objective_value"):
            objective_value = self.result.objective_value
        if hasattr(self.result, "solve_time"):
            solve_time = self.result.solve_time

        return PowerFlowResult(
            voltages=voltages_df,
            p_flows=p_flows_df,
            q_flows=q_flows_df,
            p_gens=p_gens,
            q_gens=q_gens,
            objective_value=objective_value,
            converged=True,
            solver="ipopt",
            solve_time=solve_time,
            result_type="opf",
            raw_result=self.result,
            model=self.model,
            case=self.case,
        )

    def _resolve_objective(self, objective: Any) -> Any:
        """Resolve objective string to Pyomo objective function."""
        from distopf.pyomo_models.objectives import loss_objective_rule

        if objective is None:
            return loss_objective_rule

        if callable(objective):
            return objective

        # String objective - map to pyomo objective rules
        obj_lower = objective.lower().strip()
        if obj_lower in ("loss", "loss_min"):
            return loss_objective_rule

        # List unsupported objectives that work with matrix backend
        matrix_only_objectives = [
            "curtail",
            "curtail_min",
            "gen_max",
            "target_p_3ph",
            "target_q_3ph",
            "target_p_total",
            "target_q_total",
        ]
        if obj_lower in matrix_only_objectives:
            raise ValueError(
                f"Objective '{objective}' is not supported by pyomo backend. "
                f"Use backend='matrix' or backend='multiperiod' for this objective."
            )

        raise ValueError(
            f"Unknown pyomo objective: '{objective}'. "
            f"Pyomo backend supports: loss, loss_min"
        )

    def get_voltages(self) -> pd.DataFrame:
        """Extract bus voltage results from solved model."""
        return self.result.voltages

    def get_power_flows(self) -> pd.DataFrame:
        """Extract branch power flow results (P flows).

        Note: Pyomo backend returns active power flows, not complex apparent power.
        """
        return self.result.p_flow

    def get_p_flows(self) -> pd.DataFrame:
        """Extract active power flow results from solved model."""
        return self.result.p_flow

    def get_q_flows(self) -> pd.DataFrame:
        """Extract reactive power flow results from solved model."""
        return self.result.q_flow

    def get_p_gens(self) -> pd.DataFrame:
        """Extract active power generation results from solved model."""
        return self.result.p_gen

    def get_q_gens(self) -> pd.DataFrame:
        """Extract reactive power generation results from solved model."""
        return self.result.q_gen

    def get_all_results(self) -> dict[str, pd.DataFrame]:
        """Get all variable results from the solved model.

        The OpfResult object dynamically extracts all Pyomo variables
        from the model. This method returns them as a dictionary.

        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary mapping variable names to their result DataFrames
        """
        return {
            attr: getattr(self.result, attr)
            for attr in dir(self.result)
            if isinstance(getattr(self.result, attr), pd.DataFrame)
        }
