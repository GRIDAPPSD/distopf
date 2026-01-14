"""Pyomo backend for NLP-capable OPF (IPOPT solver)."""

from typing import Any, Optional, Tuple
import pandas as pd
from distopf.backends.base import Backend


class PyomoBackend(Backend):
    """Pyomo/IPOPT backend for NLP optimization."""

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
        """Run OPF using Pyomo/IPOPT backend (NLP-capable)."""
        from distopf.pyomo_models import (
            create_lindist_model,
            add_standard_constraints,
            solve,
        )
        from distopf.importer import Case
        import pyomo.environ as pyo

        # Warn about unsupported parameters
        if control_regulators:
            self._warn_unsupported("pyomo", "control_regulators")
        if control_capacitors:
            self._warn_unsupported("pyomo", "control_capacitors")
        if "solver" in kwargs:
            self._warn_unsupported("pyomo", "solver kwarg (uses IPOPT)")

        # Determine control variable
        control_variable = self._get_control_variable(control_variable)

        # Update gen_data control variable if specified
        gen_data = self.case.gen_data
        if control_variable is not None and gen_data is not None:
            gen_data = gen_data.copy()
            gen_data.control_variable = control_variable

        # Create case copy with updated gen_data
        case_copy = Case(
            self.case.branch_data,
            self.case.bus_data,
            gen_data,
            self.case.cap_data,
            self.case.reg_data,
            self.case.bat_data,
            self.case.schedules,
            start_step=self.case.start_step,
            n_steps=self.case.n_steps,
            delta_t=self.case.delta_t,
        )

        # Create Pyomo model
        self.model = create_lindist_model(case_copy)
        add_standard_constraints(self.model)

        # Set objective
        obj_rule = self._resolve_objective(objective)
        self.model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # Solve
        self.result = solve(self.model)

        # Extract results (result is OpfResult from solve())
        voltages_df = self.get_voltages()
        power_flows_df = self.get_power_flows()
        p_gens = self.get_p_gens()
        q_gens = self.get_q_gens()

        if raw_result:
            return self.result

        return voltages_df, power_flows_df, p_gens, q_gens

    def _resolve_objective(self, objective):
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
        """Extract branch power flow results from solved model."""
        return self.result.p_flow

    def get_p_gens(self) -> pd.DataFrame:
        """Extract active power generation results from solved model."""
        return self.result.p_gen

    def get_q_gens(self) -> pd.DataFrame:
        """Extract reactive power generation results from solved model."""
        return self.result.q_gen
