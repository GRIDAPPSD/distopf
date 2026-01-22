"""Matrix backend for single-step convex OPF (CVXPY/CLARABEL)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union
from distopf.backends.base import Backend

if TYPE_CHECKING:
    import pandas as pd
    from distopf.results import PowerFlowResult


class MatrixBackend(Backend):
    """Single-period matrix backend using CVXPY/CLARABEL solver."""

    def solve(
        self,
        objective: Optional[Any] = None,
        control_regulators: bool = False,
        control_capacitors: bool = False,
        raw_result: bool = False,
        **kwargs: Any,
    ) -> Union[PowerFlowResult, Any]:
        """Run OPF using single-period matrix model (CVXPY/CLARABEL).

        Parameters
        ----------
        objective : str, callable, or None
            Optimization objective function
        control_regulators : bool
            Whether to include regulator tap control (default False)
        control_capacitors : bool
            Whether to include capacitor switching control (default False)
        raw_result : bool
            If True, return raw solver result instead of PowerFlowResult
        **kwargs
            Additional solver options

        Returns
        -------
        PowerFlowResult or raw result
            If raw_result=False: PowerFlowResult with all results
            If raw_result=True: Raw scipy OptimizeResult object
        """
        from distopf.distOPF import create_model, auto_solve
        from distopf.results import PowerFlowResult

        # Determine control variable from gen_data (uses per-row values)
        control_variable = self._infer_control_variable()

        # Create model
        self.model = create_model(
            control_variable=control_variable,
            control_regulators=control_regulators,
            control_capacitors=control_capacitors,
            branch_data=self.case.branch_data,
            bus_data=self.case.bus_data,
            gen_data=self.case.gen_data,
            cap_data=self.case.cap_data,
            reg_data=self.case.reg_data,
        )

        # Solve
        self.result = auto_solve(self.model, objective, **kwargs)

        if raw_result:
            return self.result

        # Extract results
        voltages_df = self.get_voltages()
        p_flows_df = self.get_p_flows()
        q_flows_df = self.get_q_flows()
        p_gens = self.get_p_gens()
        q_gens = self.get_q_gens()

        # Normalize: add time column to single-period results
        voltages_df = self._add_time_column(voltages_df, position=2)
        p_flows_df = self._add_time_column(p_flows_df, position=4)
        q_flows_df = self._add_time_column(q_flows_df, position=4)
        p_gens = self._add_time_column(p_gens, position=2)
        q_gens = self._add_time_column(q_gens, position=2)

        return PowerFlowResult(
            voltages=voltages_df,
            p_flows=p_flows_df,
            q_flows=q_flows_df,
            p_gens=p_gens,
            q_gens=q_gens,
            objective_value=self.result.fun if hasattr(self.result, "fun") else None,
            converged=self.result.success if hasattr(self.result, "success") else True,
            solver="clarabel",
            result_type="opf",
            raw_result=self.result,
            model=self.model,
            case=self.case,
        )

    def _infer_control_variable(self) -> str:
        """Infer model control variable from gen_data.

        If all generators have the same control_variable, use that.
        Otherwise use "" (no control) and let per-generator settings apply.
        """
        if self.case.gen_data is None or len(self.case.gen_data) == 0:
            return ""

        cv = self.case.gen_data.control_variable.unique()
        if len(cv) == 1:
            return cv[0]

        # Mixed control variables - use the most permissive
        if "PQ" in cv:
            return "PQ"
        if "P" in cv and "Q" in cv:
            return "PQ"
        if "P" in cv:
            return "P"
        if "Q" in cv:
            return "Q"
        return ""

    def get_voltages(self) -> pd.DataFrame:
        """Extract bus voltage results from solved model."""
        return self.model.get_voltages(self.result.x)

    def get_power_flows(self) -> pd.DataFrame:
        """Extract branch power flow results (complex apparent power)."""
        return self.model.get_apparent_power_flows(self.result.x)

    def get_p_flows(self) -> pd.DataFrame:
        """Extract active power flow results from solved model."""
        return self.model.get_p_flows(self.result.x)

    def get_q_flows(self) -> pd.DataFrame:
        """Extract reactive power flow results from solved model."""
        return self.model.get_q_flows(self.result.x)

    def get_p_gens(self) -> pd.DataFrame:
        """Extract active power generation results from solved model."""
        return self.model.get_p_gens(self.result.x)

    def get_q_gens(self) -> pd.DataFrame:
        """Extract reactive power generation results from solved model."""
        return self.model.get_q_gens(self.result.x)

    def get_q_caps(self) -> Optional[pd.DataFrame]:
        """Extract capacitor reactive power results from solved model."""
        return self.model.get_q_caps(self.result.x)
