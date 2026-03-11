"""Nonlinear Pyomo wrapper for NLP-capable OPF (IPOPT/MINLP solvers)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union
from distopf.wrappers.base import Wrapper

if TYPE_CHECKING:
    import pandas as pd
    from distopf.results import PowerFlowResult


class NlpWrapper(Wrapper):
    """Nonlinear Pyomo/IPOPT wrapper for NLP optimization using BranchFlow model."""

    def solve(
        self,
        objective: Optional[Any] = None,
        control_regulators: bool = False,
        control_capacitors: bool = False,
        raw_result: bool = False,
        **kwargs: Any,
    ) -> Union[PowerFlowResult, Any]:
        """Run OPF using Pyomo/IPOPT wrapper (NLP-capable, nonlinear BranchFlow model).

        Parameters
        ----------
        objective : str, callable, or None
            Optimization objective function. Supported strings:
            'loss', 'substation', 'voltage_deviation', 'curtail'
        control_regulators : bool
            Enable mixed-integer regulator tap control (requires MINLP solver)
        control_capacitors : bool
            Enable mixed-integer capacitor switching (requires MINLP solver)
        raw_result : bool
            If True, return raw Pyomo PyoResult object instead of PowerFlowResult
        **kwargs
            Additional options:
            - circular_constraints : bool, default True
                Use circular (quadratic) constraints for generators, batteries.
            - thermal_constraints : bool, default False
                Add thermal limit constraints on branch power flows.
            - initialize : str or None, default None
                Warm-start strategy: 'fbs' to initialize from FBS results, None for default.
            - solver : str, default 'ipopt'
                Solver to use. 'ipopt' for continuous, 'bonmin'/'couenne' for MINLP.

        Returns
        -------
        PowerFlowResult or raw result
            If raw_result=False: PowerFlowResult with all results
            If raw_result=True: Pyomo PyoResult object with all variable results
        """
        from distopf.pyomo_models.nl_branchflow import create_nl_branchflow_model
        from distopf.pyomo_models.constraints_nlp import add_nlp_constraints
        from distopf.pyomo_models.solvers import solve
        from distopf.results import PowerFlowResult
        import pyomo.environ as pyo  # type: ignore[import-untyped]

        # Extract NLP-specific options from kwargs
        circular_constraints = kwargs.pop("circular_constraints", True)
        thermal_constraints = kwargs.pop("thermal_constraints", False)
        initialize = kwargs.pop("initialize", "fbs")
        solver_name = kwargs.pop("solver", "ipopt")

        # Validate solver choice for discrete controls
        if (control_regulators or control_capacitors) and solver_name == "ipopt":
            raise ValueError(
                "Discrete controls (control_regulators/control_capacitors) require a MINLP solver. "
                "Use solver='bonmin' or solver='couenne', or disable discrete controls."
            )

        # Create nonlinear BranchFlow model
        self.model = create_nl_branchflow_model(self.case)

        # Add constraints
        add_nlp_constraints(
            self.model,
            circular_constraints=circular_constraints,
            thermal_constraints=thermal_constraints,
            control_regulators=control_regulators,
            control_capacitors=control_capacitors,
        )

        # Warm-start initialization if requested
        if initialize == "fbs":
            self._initialize_from_fbs()

        # Set objective
        obj_rule = self._resolve_objective(objective)
        self.model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # Solve with appropriate solver
        self.result = solve(self.model, solver=solver_name)

        if raw_result:
            return self.result

        # Extract results
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
            objective_value=objective_value,
            converged=True,
            solver=solver_name,
            solve_time=solve_time,
            result_type="opf",
            raw_result=self.result,
            model=self.model,
            case=self.case,
        )

    def _initialize_from_fbs(self) -> None:
        """Initialize NL model variables from FBS power flow solution."""
        from distopf.fbs import fbs_solve
        import numpy as np
        from math import pi

        # Run FBS to get initial solution
        fbs_results = fbs_solve(self.case)

        # Initialize voltages
        v_data = {}
        v_reg_data = {}
        for _id, ph, t in self.model.bus_phase_set * self.model.time_set:
            v_mag = fbs_results.voltages.loc[
                (fbs_results.voltages.id == _id), ph
            ].to_numpy()[0]
            v_data[(_id, ph, t)] = v_mag**2
            if (_id, ph) in self.model.reg_phase_set:
                v_reg_data[(_id, ph, t)] = v_mag**2
        self.model.v2.set_values(v_data)
        self.model.v2_reg.set_values(v_reg_data)

        # Initialize power flows
        p_data = {}
        q_data = {}
        for _id, ph, t in self.model.branch_phase_set * self.model.time_set:
            p_flow = fbs_results.p_flows.loc[
                (fbs_results.p_flows.tb == _id), ph
            ].to_numpy()[0]
            q_flow = fbs_results.q_flows.loc[
                (fbs_results.q_flows.tb == _id), ph
            ].to_numpy()[0]
            p_data[(_id, ph, t)] = p_flow
            q_data[(_id, ph, t)] = q_flow
        self.model.p_flow.set_values(p_data)
        self.model.q_flow.set_values(q_data)

        # Initialize current magnitudes (squared)
        l_data = {}
        for _id, ph, t in self.model.branch_phase_set * self.model.time_set:
            i_mag = fbs_results.currents.loc[
                (fbs_results.currents.tb == _id), ph
            ].to_numpy()[0]
            l_data[(_id, ph + ph, t)] = i_mag**2

        # Initialize cross-phase current magnitudes
        for _id, phases, t in self.model.bus_phase_pair_set * self.model.time_set:
            ph1 = phases[0]
            ph2 = phases[1]
            if ph1 == ph2:
                continue
            if (_id, ph1 + ph1, t) in l_data and (_id, ph2 + ph2, t) in l_data:
                l_data[(_id, ph1 + ph2, t)] = np.sqrt(
                    l_data[_id, ph1 + ph1, t] * l_data[_id, ph2 + ph2, t]
                )
        self.model.l_flow.set_values(l_data)

        # Initialize current angle differences
        fbs_results.current_angles["ab"] = (
            fbs_results.current_angles.a - fbs_results.current_angles.b
        ) % 360
        fbs_results.current_angles["bc"] = (
            fbs_results.current_angles.b - fbs_results.current_angles.c
        ) % 360
        fbs_results.current_angles["ca"] = (
            fbs_results.current_angles.c - fbs_results.current_angles.a
        ) % 360
        fbs_results.current_angles["ba"] = -fbs_results.current_angles.ab
        fbs_results.current_angles["cb"] = -fbs_results.current_angles.bc
        fbs_results.current_angles["ac"] = -fbs_results.current_angles.ca
        fbs_results.current_angles["aa"] = (
            fbs_results.current_angles.a - fbs_results.current_angles.a
        )
        fbs_results.current_angles["bb"] = (
            fbs_results.current_angles.b - fbs_results.current_angles.b
        )
        fbs_results.current_angles["cc"] = (
            fbs_results.current_angles.c - fbs_results.current_angles.c
        )

        angle_data = {
            (_id, phases): float(
                fbs_results.current_angles.loc[
                    fbs_results.current_angles.tb == _id, phases
                ].tolist()[0]
            )
            * pi
            / 180
            for _id, phases in self.model.bus_angle_phase_pair_set
        }
        for key in self.model.d:
            self.model.d[key] = angle_data[key]

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
            f"Unknown objective: '{objective}'. "
            f"Supported: {', '.join(objective_map.keys())}"
        )

    def get_voltages(self) -> pd.DataFrame:
        """Extract bus voltage results from solved model."""
        return self.result.voltages

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
