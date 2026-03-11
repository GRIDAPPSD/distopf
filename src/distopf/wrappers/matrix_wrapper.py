"""Matrix wrapper for single-step convex OPF (CVXPY/CLARABEL)."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any, Optional, Union, Callable
from distopf.wrappers.base import Wrapper

from distopf.matrix_models.base import LinDistBase
from distopf.cases import CASES_DIR
from distopf.matrix_models.lindist_capacitor_mi import LinDistModelCapMI
from distopf.matrix_models.lindist_capacitor_regulator_mi import (
    LinDistModelCapacitorRegulatorMI,
)
from distopf.matrix_models.lindist_p_gen import LinDistModelPGen
from distopf.matrix_models.lindist_q_gen import LinDistModelQGen
from distopf.matrix_models.lindist import LinDistModel
from distopf.matrix_models.solvers import (
    lp_solve,
    cvxpy_solve,
)
from distopf.matrix_models.objectives import (
    cp_obj_loss,
    cp_obj_curtail,
    cp_obj_target_p_3ph,
    cp_obj_target_p_total,
    cp_obj_target_q_3ph,
    cp_obj_target_q_total,
    gradient_load_min,
    gradient_curtail,
)

if TYPE_CHECKING:
    import pandas as pd
    from distopf.results import PowerFlowResult

# Objective function aliases for user convenience
OBJECTIVE_ALIASES: dict[str, str] = {
    # Loss minimization
    "loss": "loss_min",
    "minimize_loss": "loss_min",
    "min_loss": "loss_min",
    # Curtailment minimization
    "curtail": "curtail_min",
    "minimize_curtail": "curtail_min",
    "min_curtail": "curtail_min",
    "curtailment": "curtail_min",
    # Generation maximization
    "gen": "gen_max",
    "maximize_gen": "gen_max",
    "max_gen": "gen_max",
    # Load minimization
    "load": "load_min",
    "minimize_load": "load_min",
    "min_load": "load_min",
    # Target tracking (keep full names, but add alternatives)
    "target_p": "target_p_total",
    "target_q": "target_q_total",
    "p_target": "target_p_total",
    "q_target": "target_q_total",
}


def resolve_objective_alias(objective: str | None) -> str | None:
    """
    Resolve objective function alias to canonical name.

    Parameters
    ----------
    objective : str or None
        User-provided objective name (may be an alias)

    Returns
    -------
    str or None
        Canonical objective name, or None if input was None
    """
    if objective is None:
        return None
    objective_lower = objective.lower().strip()
    return OBJECTIVE_ALIASES.get(objective_lower, objective_lower)


def create_model(
    control_variable: str = "",
    control_regulators: bool = False,
    control_capacitors: bool = False,
    **kwargs,
) -> LinDistBase:
    """
    Create the correct LinDistModel object based on the control variable.
    Parameters
    ----------
    control_variable : str, optional : No Control Variables-None, Active Power Control-'p', Reactive Power Control-'q'
    control_regulators : bool, optional : Default False, if true use mixed integer control of regulators
    control_capacitors : bool, optional : Default False, if true use mixed integer control of capacitors
    kwargs :
        branch_data : pd.DataFrame
            DataFrame containing branch data (r and x values, limits)
        bus_data : pd.DataFrame
            DataFrame containing bus data (loads, voltages, limits)
        gen_data : pd.DataFrame
            DataFrame containing generator/DER data
        cap_data : pd.DataFrame
            DataFrame containing capacitor data
        reg_data : pd.DataFrame
            DataFrame containing regulator data
    Returns
    -------
    model: LinDistModel, or LinDistModelP, or LinDistModelQ object appropriate for the control variable
    """

    if control_capacitors and not control_regulators:
        return LinDistModelCapMI(**kwargs)
    if control_regulators:
        return LinDistModelCapacitorRegulatorMI(**kwargs)
    if control_variable is None or control_variable == "":
        return LinDistModel(**kwargs)
    if control_variable.upper() == "P":
        return LinDistModelPGen(**kwargs)
    if control_variable.upper() == "Q":
        return LinDistModelQGen(**kwargs)
    if control_variable.upper() == "PQ":
        return LinDistModel(**kwargs)
    raise ValueError(
        f"Unknown control variable '{control_variable}'. Valid options are 'P', 'Q' or None"
    )


def auto_solve(model: LinDistBase, objective_function=None, **kwargs):
    """
    Solve with selected objective function and model. Automatically chooses the appropriate function.

    Parameters
    ----------
    model : LinDistBase
    objective_function : str or Callable
    kwargs : kwargs to pass to objective function and solver function.
        solver: str
            Solver to use for solving with CVXPY. Default is CLARABEL. OSQP is also recommended.
        target:
            Used with target objectives. Target to track.
            Scalar for target_p_total and target_q_total and size-3 array for target_p_3ph, and target_q_3ph.
        error_percent:
            Used with target objectives. Percent error expected in total system load compared exact solution.

    Returns
    -------
    result: scipy.optimize.OptimizeResult

    """
    if objective_function is None:
        objective_function = np.zeros(model.n_x)
    if not isinstance(objective_function, (str, Callable, np.ndarray, list)):  # type: ignore
        raise TypeError(
            "objective_function must be a function handle, array, or string"
        )
    # Resolve aliases before looking up in maps
    if isinstance(objective_function, str):
        objective_function = resolve_objective_alias(objective_function)
    objective_function_map_gradient: dict[str, Callable] = {
        "gen_max": gradient_curtail,
        "load_min": gradient_load_min,
    }
    objective_function_map: dict[str, Callable] = {
        "loss_min": cp_obj_loss,
        "curtail_min": cp_obj_curtail,
        "target_p_3ph": cp_obj_target_p_3ph,
        "target_q_3ph": cp_obj_target_q_3ph,
        "target_p_total": cp_obj_target_p_total,
        "target_q_total": cp_obj_target_q_total,
    }
    if isinstance(objective_function, str):
        objective_function = objective_function.lower()
        if objective_function in objective_function_map.keys():
            objective_function = objective_function_map[objective_function]
        if objective_function in objective_function_map_gradient.keys():
            objective_function = objective_function_map[objective_function](model)
    if isinstance(objective_function, Callable):  # type: ignore
        if hasattr(model, "solve"):
            return model.solve(objective_function, **kwargs)
        return cvxpy_solve(model, objective_function, **kwargs)
    if isinstance(objective_function, (np.ndarray, list)):
        return lp_solve(model, objective_function)  # type: ignore


class MatrixWrapper(Wrapper):
    """Single-period matrix wrapper using CVXPY/CLARABEL solver."""

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
        reg_taps = None
        if hasattr(self.model, "get_regulator_taps"):
            reg_taps = self.model.get_regulator_taps()
        z_caps = None
        if hasattr(self.model, "get_zc"):
            z_caps = self.model.get_zc(self.result.x)
        u_caps = None
        if hasattr(self.model, "get_uc"):
            u_caps = self.model.get_uc(self.result.x)

        # Normalize: add time column to single-period results
        voltages_df = self._add_time_column(voltages_df, position=2)
        p_flows_df = self._add_time_column(p_flows_df, position=4)
        q_flows_df = self._add_time_column(q_flows_df, position=4)
        p_gens = self._add_time_column(p_gens, position=2)
        q_gens = self._add_time_column(q_gens, position=2)

        return PowerFlowResult(
            voltages=voltages_df,
            active_power_flows=p_flows_df,
            reactive_power_flows=q_flows_df,
            active_power_generation=p_gens,
            reactive_power_generation=q_gens,
            reg_taps=reg_taps,
            z_caps=z_caps,
            u_caps=u_caps,
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
