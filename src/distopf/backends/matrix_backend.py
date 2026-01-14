"""Matrix backend for single-step convex OPF (CVXPY/CLARABEL)."""

from typing import Any, Optional, Tuple
import pandas as pd
from distopf.backends.base import Backend


class MatrixBackend(Backend):
    """Single-period matrix backend using CVXPY/CLARABEL solver."""

    def solve(
        self,
        objective: Optional[Any] = None,
        control_variable: Optional[str] = None,
        control_regulators: bool = False,
        control_capacitors: bool = False,
        raw_result: bool = False,
        **kwargs,
    ) -> Tuple[
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
        Optional[pd.DataFrame],
    ]:
        """Run OPF using single-period matrix model (CVXPY/CLARABEL)."""
        from distopf.distOPF import create_model, auto_solve

        # Determine control variable
        control_variable = self._get_control_variable(control_variable)

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

        # Extract results
        voltages_df = self.get_voltages()
        power_flows_df = self.get_power_flows()
        p_gens = self.get_p_gens()
        q_gens = self.get_q_gens()

        # Normalize: add time column to single-period results
        voltages_df = self._add_time_column(voltages_df, position=2)
        power_flows_df = self._add_time_column(power_flows_df, position=4)
        p_gens = self._add_time_column(p_gens, position=2)
        q_gens = self._add_time_column(q_gens, position=2)

        if raw_result:
            return self.result

        return voltages_df, power_flows_df, p_gens, q_gens

    def get_voltages(self) -> pd.DataFrame:
        """Extract bus voltage results from solved model."""
        return self.model.get_voltages(self.result.x)

    def get_power_flows(self) -> pd.DataFrame:
        """Extract branch power flow results from solved model."""
        return self.model.get_apparent_power_flows(self.result.x)

    def get_p_gens(self) -> pd.DataFrame:
        """Extract active power generation results from solved model."""
        return self.model.get_p_gens(self.result.x)

    def get_q_gens(self) -> pd.DataFrame:
        """Extract reactive power generation results from solved model."""
        return self.model.get_q_gens(self.result.x)
