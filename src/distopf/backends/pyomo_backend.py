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
            Optimization objective function. Supported strings:
            'loss', 'substation', 'voltage_deviation', 'curtail'
        control_regulators : bool
            Enable mixed-integer regulator tap control
        control_capacitors : bool
            Enable mixed-integer capacitor switching
        raw_result : bool
            If True, return raw Pyomo PyoResult object instead of PowerFlowResult
        **kwargs
            Additional options:
            - circular_constraints : bool, default True
                Use circular (quadratic) constraints for generators, batteries,
                and thermal limits. If False, use octagonal (linear) approximations.
            - thermal_constraints : bool, default False
                Add thermal limit constraints on branch power flows.
            - equality_only : bool, default False
                Only add equality constraints (power flow, voltage drop).
                Skips voltage/thermal/generator limits.
            - reg_tap_change_limit : int or None
                Max tap change per timestep (only if control_regulators=True)

        Returns
        -------
        PowerFlowResult or raw result
            If raw_result=False: PowerFlowResult with all results
            If raw_result=True: Pyomo PyoResult object with all variable results
        """
        from distopf.pyomo_models import (
            create_lindist_model,
            add_constraints,
            solve,
            create_penalized_objective,
            set_objective,
        )
        from distopf.results import PowerFlowResult
        import pyomo.environ as pyo  # type: ignore[import-untyped]

        # Extract pyomo-specific options from kwargs
        circular_constraints = kwargs.pop("circular_constraints", False)
        thermal_constraints = kwargs.pop("thermal_constraints", False)
        equality_only = kwargs.pop("equality_only", False)
        reg_tap_change_limit = kwargs.pop("reg_tap_change_limit", None)
        duals = kwargs.pop("duals", False)

        voltage_weight = kwargs.pop("voltage_weight", None)
        thermal_weight = kwargs.pop("thermal_weight", None)
        generator_weight = kwargs.pop("generator_weight", None)
        battery_weight = kwargs.pop("battery_weight", None)
        soc_weight = kwargs.pop("soc_weight", None)
        solver = kwargs.pop("solver", "ipopt")

        # Create Pyomo model
        self.model = create_lindist_model(
            self.case,
            control_capacitors=control_capacitors,
            control_regulators=control_regulators,
        )

        # Add constraints with configuration
        add_constraints(
            self.model,
            circular_constraints=circular_constraints,
            thermal_constraints=thermal_constraints,
            equality_only=equality_only,
            control_capacitors=control_capacitors,
            control_regulators=control_regulators,
            reg_tap_change_limit=reg_tap_change_limit,
        )

        # Set objective
        obj_rule = self._resolve_objective(objective)
        if equality_only:
            obj = create_penalized_objective(
                obj_rule,
                voltage_weight=voltage_weight,
                thermal_weight=thermal_weight,
                generator_weight=generator_weight,
                battery_weight=battery_weight,
                soc_weight=soc_weight,
            )
            set_objective(self.model, obj)
        else:
            self.model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        # Solve
        self.result = solve(self.model, solver=solver, duals=duals)

        if raw_result:
            return self.result

        # Extract results (result is PyoResult from solve())
        voltages_df = self.get_voltages()
        p_flows_df = self.get_p_flows()
        q_flows_df = self.get_q_flows()
        p_gens = self.get_p_gens()
        q_gens = self.get_q_gens()

        # Extract additional results (may not exist in all models)
        q_caps = getattr(self.result, "q_cap", None)
        tap_ratios = getattr(self.result, "reg_ratio", None)
        p_loads = getattr(self.result, "p_load", None)
        q_loads = getattr(self.result, "q_load", None)

        # Batteries (present in the Pyomo model, but were not being surfaced)
        p_bats = getattr(self.result, "p_bat", None)
        q_bats = getattr(self.result, "q_bat", None)
        p_charge = getattr(self.result, "p_charge", None)
        p_discharge = getattr(self.result, "p_discharge", None)
        soc = getattr(self.result, "soc", None)

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
            p_loads=p_loads,
            q_loads=q_loads,
            q_caps=q_caps,
            tap_ratios=tap_ratios,
            p_bats=p_bats,
            q_bats=q_bats,
            p_charge=p_charge,
            p_discharge=p_discharge,
            soc=soc,
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
        from distopf.pyomo_models.objectives import (
            loss_objective_rule,
            substation_power_objective_rule,
            voltage_deviation_objective_rule,
            generation_curtailment_objective_rule,
        )

        if objective is None:
            return loss_objective_rule

        if callable(objective):
            return objective

        # String objective - map to pyomo objective rules
        obj_lower = objective.lower().strip()

        objective_map = {
            "loss": loss_objective_rule,
            "loss_min": loss_objective_rule,
            "substation": substation_power_objective_rule,
            "substation_power": substation_power_objective_rule,
            "voltage_deviation": voltage_deviation_objective_rule,
            "curtail": generation_curtailment_objective_rule,
            "curtail_min": generation_curtailment_objective_rule,
            "curtailment": generation_curtailment_objective_rule,
        }

        if obj_lower in objective_map:
            return objective_map[obj_lower]

        raise ValueError(
            f"Unknown pyomo objective: '{objective}'. "
            f"Supported: {', '.join(objective_map.keys())}"
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

        The PyoResult object dynamically extracts all Pyomo variables
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
