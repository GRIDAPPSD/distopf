"""Matrix BESS wrapper for time-series OPF (batteries, schedules)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union
from distopf.wrappers.base import Wrapper

if TYPE_CHECKING:
    import pandas as pd
    from distopf.results import PowerFlowResult


class MatrixBessWrapper(Wrapper):
    """Matrix BESS wrapper (supports batteries and schedules)."""

    def solve(
        self,
        objective: Optional[Any] = None,
        control_regulators: bool = False,
        control_capacitors: bool = False,
        raw_result: bool = False,
        **kwargs: Any,
    ) -> Union[PowerFlowResult, Any]:
        """Run OPF using multi-period matrix model.

        Parameters
        ----------
        objective : str, callable, or None
            Optimization objective function
        control_regulators : bool
            Not supported in matrix_bess wrapper (ignored with warning)
        control_capacitors : bool
            Not supported in matrix_bess wrapper (ignored with warning)
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
        from distopf.matrix_models.matrix_bess import LinDistMPL, cvxpy_solve
        from distopf.results import PowerFlowResult

        # Warn about unsupported parameters
        if control_regulators:
            self._warn_unsupported("matrix_bess", "control_regulators")
        if control_capacitors:
            self._warn_unsupported("matrix_bess", "control_capacitors")

        # Create matrix_bess model (uses gen_data.control_variable per-row)
        self.model = LinDistMPL(
            branch_data=self.case.branch_data,
            bus_data=self.case.bus_data,
            gen_data=self.case.gen_data,
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

        if raw_result:
            return self.result

        return PowerFlowResult(
            voltages=self.get_voltages(),
            active_power_flows=self.get_p_flows(),
            reactive_power_flows=self.get_q_flows(),
            active_power_generation=self.get_p_gens(),
            reactive_power_generation=self.get_q_gens(),
            battery_active_power=self.get_p_batt(),
            battery_reactive_power=self.get_q_batt(),
            p_charge=self.get_p_charge(),
            p_discharge=self.get_p_discharge(),
            soc=self.get_soc(),
            objective_value=self.result.fun if hasattr(self.result, "fun") else None,
            converged=self.result.success if hasattr(self.result, "success") else True,
            solver="clarabel",
            result_type="opf",
            raw_result=self.result,
            model=self.model,
            case=self.case,
        )

    def _resolve_objective(self, objective: Any) -> Any:
        """Resolve objective string to matrix_bess objective function."""
        from distopf.matrix_models.matrix_bess import objectives as mp_obj

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
            f"Unknown matrix_bess objective: '{objective}'. "
            f"Supported objectives: {list(obj_map.keys())}"
        )

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

    # Battery-specific result methods

    def get_p_batt(self) -> pd.DataFrame:
        """Extract battery active power results from solved model."""
        return self.model.get_p_batt(self.result.x)

    def get_q_batt(self) -> pd.DataFrame:
        """Extract battery reactive power results from solved model."""
        return self.model.get_q_batt(self.result.x)

    def get_p_charge(self) -> pd.DataFrame:
        """Extract battery charging power results from solved model."""
        return self.model.get_p_charge(self.result.x)

    def get_p_discharge(self) -> pd.DataFrame:
        """Extract battery discharging power results from solved model."""
        return self.model.get_p_discharge(self.result.x)

    def get_soc(self) -> pd.DataFrame:
        """Extract battery state of charge results from solved model."""
        return self.model.get_soc(self.result.x)
