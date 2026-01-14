"""Multiperiod backend for time-series OPF (batteries, schedules)."""

from typing import Any, Optional, Tuple
import pandas as pd
from distopf.backends.base import Backend


class MultiperiodBackend(Backend):
    """Multi-period matrix backend (supports batteries and schedules)."""

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
        """Run OPF using multi-period matrix model."""
        from distopf.matrix_models.multiperiod import LinDistMPL, cvxpy_solve

        # Warn about unsupported parameters
        if control_regulators:
            self._warn_unsupported("multiperiod", "control_regulators")
        if control_capacitors:
            self._warn_unsupported("multiperiod", "control_capacitors")

        # Determine control variable
        control_variable = self._get_control_variable(control_variable)

        # Update gen_data control variable if specified
        gen_data = self.case.gen_data
        if control_variable and gen_data is not None:
            gen_data = gen_data.copy()
            gen_data.control_variable = control_variable

        # Create multiperiod model
        self.model = LinDistMPL(
            branch_data=self.case.branch_data,
            bus_data=self.case.bus_data,
            gen_data=gen_data,
            cap_data=self.case.cap_data,
            reg_data=self.case.reg_data,
            bat_data=self.case.bat_data,
            schedules=self.case.schedules,
            start_step=self.case.start_step,
            n_steps=self.case.n_steps,
            delta_t=self.case.delta_t,
        )

        # Resolve objective function
        obj_func = self._resolve_objective(objective)

        # Solve
        self.result = cvxpy_solve(self.model, obj_func, **kwargs)

        # Extract results
        voltages_df = self.get_voltages()
        power_flows_df = self.get_power_flows()
        p_gens = self.get_p_gens()
        q_gens = self.get_q_gens()

        if raw_result:
            return self.result

        return voltages_df, power_flows_df, p_gens, q_gens

    def _resolve_objective(self, objective):
        """Resolve objective string to multiperiod objective function."""
        from distopf.matrix_models.multiperiod import objectives as mp_obj

        if objective is None:
            return mp_obj.cp_obj_loss

        if callable(objective):
            return objective

        # String objective
        obj_lower = objective.lower().strip()
        obj_map = {
            "loss": mp_obj.cp_obj_loss,
            "loss_min": mp_obj.cp_obj_loss,
            "loss_batt": mp_obj.cp_obj_loss_batt,
            "curtail": mp_obj.cp_obj_curtail,
            "curtail_min": mp_obj.cp_obj_curtail,
            "target_p_3ph": mp_obj.cp_obj_target_p_3ph,
            "target_q_3ph": mp_obj.cp_obj_target_q_3ph,
            "target_p_total": mp_obj.cp_obj_target_p_total,
            "target_q_total": mp_obj.cp_obj_target_q_total,
            "none": mp_obj.cp_obj_none,
        }
        if obj_lower in obj_map:
            return obj_map[obj_lower]
        raise ValueError(
            f"Unknown multiperiod objective: '{objective}'. "
            f"Supported objectives: {list(obj_map.keys())}"
        )

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
