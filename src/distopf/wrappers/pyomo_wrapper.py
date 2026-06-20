"""Pyomo wrapper for OPF (IPOPT solver).

Supports both LinDistFlow (linear) and BranchFlow (nonlinear) models
via the `model_type` parameter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union
from distopf.wrappers.base import Wrapper

if TYPE_CHECKING:
    import pandas as pd
    from distopf.results import PowerFlowResult


class PyomoWrapper(Wrapper):
    """Pyomo/IPOPT wrapper for OPF optimization.

    Supports two model types:
    - "lindist" (default): LinDistFlow linear approximation
    - "branchflow": Nonlinear BranchFlow model (NLP)
    """

    def solve(
        self,
        objective: Optional[Any] = None,
        control_regulators: bool = False,
        control_capacitors: bool = False,
        raw_result: bool = False,
        **kwargs: Any,
    ) -> Union[PowerFlowResult, Any]:
        """Run OPF using Pyomo/IPOPT.

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
            - model_type : str, default "lindist"
                "lindist" for LinDistFlow, "branchflow" for nonlinear BranchFlow.
            - circular_constraints : bool
                Use circular (quadratic) constraints. Default False for lindist, True for branchflow.
            - thermal_constraints : bool, default False
                Add thermal limit constraints on branch power flows.
            - equality_only : bool, default False (lindist only)
                Only add equality constraints (power flow, voltage drop).
            - reg_tap_change_limit : int or None (lindist only)
                Max tap change per timestep (only if control_regulators=True)
            - initialize : str or None (branchflow only)
                Warm-start strategy: 'fbs' to initialize from FBS results.
            - solver : str, default 'ipopt'
                Solver to use. 'ipopt' for continuous, 'bonmin'/'couenne' for MINLP.
            - duals : bool, default False
                Extract dual variables from constraints.

        Returns
        -------
        PowerFlowResult or raw result
            If raw_result=False: PowerFlowResult with all results
            If raw_result=True: Pyomo PyoResult object with all variable results
        """
        model_type = kwargs.pop("model_type", "lindist")

        if model_type == "branchflow":
            return self._solve_branchflow(
                objective=objective,
                control_regulators=control_regulators,
                control_capacitors=control_capacitors,
                raw_result=raw_result,
                **kwargs,
            )
        return self._solve_lindist(
            objective=objective,
            control_regulators=control_regulators,
            control_capacitors=control_capacitors,
            raw_result=raw_result,
            **kwargs,
        )

    def _solve_lindist(
        self,
        objective,
        control_regulators,
        control_capacitors,
        raw_result,
        **kwargs,
    ):
        """Solve using LinDistFlow model."""
        from distopf.pyomo_models import (
            create_lindist_model,
            add_constraints,
            solve,
            create_penalized_objective,
            set_objective,
        )
        import pyomo.environ as pyo  # type: ignore[import-untyped]

        circular_constraints = kwargs.pop("circular_constraints", False)
        thermal_constraints = kwargs.pop("thermal_constraints", False)
        equality_only = kwargs.pop("equality_only", False)
        reg_tap_change_limit = kwargs.pop("reg_tap_change_limit", None)
        duals = kwargs.pop("duals", False)
        verbose = kwargs.pop("verbose", False)

        voltage_weight = kwargs.pop("voltage_weight", None)
        thermal_weight = kwargs.pop("thermal_weight", None)
        generator_weight = kwargs.pop("generator_weight", None)
        battery_weight = kwargs.pop("battery_weight", None)
        soc_weight = kwargs.pop("soc_weight", None)
        solver = kwargs.pop("solver", "ipopt")

        self.model = create_lindist_model(
            self.case,
            control_capacitors=control_capacitors,
            control_regulators=control_regulators,
        )

        add_constraints(
            self.model,
            circular_constraints=circular_constraints,
            thermal_constraints=thermal_constraints,
            equality_only=equality_only,
            control_capacitors=control_capacitors,
            control_regulators=control_regulators,
            reg_tap_change_limit=reg_tap_change_limit,
        )

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

        self.result = solve(self.model, solver=solver, duals=duals, verbose=verbose)

        if raw_result:
            return self.result

        return self._build_result(solver_name=solver)

    def _solve_branchflow(
        self,
        objective,
        control_regulators,
        control_capacitors,
        raw_result,
        **kwargs,
    ):
        """Solve using nonlinear BranchFlow model."""
        from distopf.pyomo_models.nl_branchflow import create_nl_branchflow_model
        from distopf.pyomo_models.constraints_nlp import add_nlp_constraints
        from distopf.pyomo_models.solvers import solve
        import pyomo.environ as pyo  # type: ignore[import-untyped]

        circular_constraints = kwargs.pop("circular_constraints", True)
        thermal_constraints = kwargs.pop("thermal_constraints", False)
        initialize = kwargs.pop("initialize", "fbs")
        solver_name = kwargs.pop("solver", "ipopt")
        verbose = kwargs.pop("verbose", False)

        if (control_regulators or control_capacitors) and solver_name == "ipopt":
            raise ValueError(
                "Discrete controls (control_regulators/control_capacitors) require a MINLP solver. "
                "Use solver='bonmin' or solver='couenne', or disable discrete controls."
            )

        self.model = create_nl_branchflow_model(self.case)

        add_nlp_constraints(
            self.model,
            circular_constraints=circular_constraints,
            thermal_constraints=thermal_constraints,
            control_regulators=control_regulators,
            control_capacitors=control_capacitors,
        )

        if initialize == "fbs":
            self._initialize_from_fbs()

        obj_rule = self._resolve_objective(objective)
        self.model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        self.result = solve(self.model, solver=solver_name, verbose=verbose)

        if raw_result:
            return self.result

        return self._build_result(solver_name=solver_name)

    def _build_result(self, solver_name: str):
        """Build PowerFlowResult from solved pyomo model."""
        from distopf.results import PowerFlowResult

        q_caps = getattr(self.result, "q_cap", None)
        tap_ratios = getattr(self.result, "reg_ratio", None)
        p_loads = getattr(self.result, "p_load", None)
        q_loads = getattr(self.result, "q_load", None)
        p_bats = getattr(self.result, "p_bat", None)
        q_bats = getattr(self.result, "q_bat", None)
        p_charge = getattr(self.result, "p_charge", None)
        p_discharge = getattr(self.result, "p_discharge", None)
        soc = getattr(self.result, "soc", None)

        objective_value = getattr(self.result, "objective_value", None)
        solve_time = getattr(self.result, "solve_time", None)

        # Extract duals if available on the raw result
        dual_power_balance_p = getattr(self.result, "dual_power_balance_p", None)
        dual_power_balance_q = getattr(self.result, "dual_power_balance_q", None)
        dual_voltage_drop = getattr(self.result, "dual_voltage_drop", None)
        dual_voltage_limits_lower = getattr(
            self.result, "dual_voltage_limits_lower", None
        )
        dual_voltage_limits_upper = getattr(
            self.result, "dual_voltage_limits_upper", None
        )

        return PowerFlowResult(
            voltages=self.get_voltages(),
            active_power_flows=self.get_p_flows(),
            reactive_power_flows=self.get_q_flows(),
            active_power_generation=self.get_p_gens(),
            reactive_power_generation=self.get_q_gens(),
            active_power_loads=p_loads,
            reactive_power_loads=q_loads,
            capacitor_reactive_power=q_caps,
            tap_ratios=tap_ratios,
            battery_active_power=p_bats,
            battery_reactive_power=q_bats,
            p_charge=p_charge,
            p_discharge=p_discharge,
            soc=soc,
            dual_power_balance_p=dual_power_balance_p,
            dual_power_balance_q=dual_power_balance_q,
            dual_voltage_drop=dual_voltage_drop,
            dual_voltage_limits_lower=dual_voltage_limits_lower,
            dual_voltage_limits_upper=dual_voltage_limits_upper,
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
        """Initialize BranchFlow model variables from FBS power flow solution."""
        from distopf.fbs import fbs_solve
        import numpy as np
        from math import pi

        fbs_results = fbs_solve(self.case)
        assert fbs_results.voltages is not None
        assert fbs_results.active_power_flows is not None
        assert fbs_results.reactive_power_flows is not None
        assert fbs_results.currents is not None
        assert fbs_results.current_angles is not None
        branch_dimen = int(self.model.branch_phase_set.dimen)
        reg_dimen = int(self.model.reg_phase_set.dimen)

        voltages = fbs_results.voltages
        active_power_flows = fbs_results.active_power_flows
        reactive_power_flows = fbs_results.reactive_power_flows
        currents = fbs_results.currents
        current_angles = fbs_results.current_angles

        def _lookup_branch_phase(df, fb, tb, ph):
            mask = df.tb == tb
            if "fb" in df.columns:
                mask = mask & (df.fb == fb)
            vals = df.loc[mask, ph].to_numpy()
            if len(vals) == 0:
                return 0.0
            return vals[0]

        # Initialize voltages
        v_data: dict[tuple[Any, ...], float] = {}
        v_reg_data: dict[tuple[Any, ...], float] = {}
        for _id, ph, t in self.model.bus_phase_set * self.model.time_set:
            v_mag = voltages.loc[(voltages.id == _id), ph].to_numpy()[0]
            v_data[(_id, ph, t)] = v_mag**2
        if reg_dimen == 3:
            for fb, tb, ph, t in self.model.reg_phase_set * self.model.time_set:
                v_mag = voltages.loc[(voltages.id == tb), ph].to_numpy()[0]
                v_reg_data[(fb, tb, ph, t)] = v_mag**2
        else:
            for _id, ph, t in self.model.reg_phase_set * self.model.time_set:
                v_mag = voltages.loc[(voltages.id == _id), ph].to_numpy()[0]
                v_reg_data[(_id, ph, t)] = v_mag**2
        self.model.v2.set_values(v_data)
        self.model.v2_reg.set_values(v_reg_data)

        # Initialize power flows
        p_data: dict[tuple[Any, ...], float] = {}
        q_data: dict[tuple[Any, ...], float] = {}
        if branch_dimen == 3:
            for fb, tb, ph, t in self.model.branch_phase_set * self.model.time_set:
                p_flow = _lookup_branch_phase(active_power_flows, fb, tb, ph)
                q_flow = _lookup_branch_phase(reactive_power_flows, fb, tb, ph)
                p_data[(fb, tb, ph, t)] = p_flow
                q_data[(fb, tb, ph, t)] = q_flow
        else:
            for _id, ph, t in self.model.branch_phase_set * self.model.time_set:
                p_flow = active_power_flows.loc[
                    (active_power_flows.tb == _id), ph
                ].to_numpy()[0]
                q_flow = reactive_power_flows.loc[
                    (reactive_power_flows.tb == _id), ph
                ].to_numpy()[0]
                p_data[(_id, ph, t)] = p_flow
                q_data[(_id, ph, t)] = q_flow
        self.model.p_flow.set_values(p_data)
        self.model.q_flow.set_values(q_data)

        # Initialize current magnitudes (squared)
        l_data: dict[tuple[Any, ...], float] = {}
        if branch_dimen == 3:
            for fb, tb, ph, t in self.model.branch_phase_set * self.model.time_set:
                i_mag = _lookup_branch_phase(currents, fb, tb, ph)
                l_data[(fb, tb, ph + ph, t)] = i_mag**2
        else:
            for _id, ph, t in self.model.branch_phase_set * self.model.time_set:
                i_mag = currents.loc[(currents.tb == _id), ph].to_numpy()[0]
                l_data[(_id, ph + ph, t)] = i_mag**2

        # Initialize cross-phase current magnitudes
        for fb, tb, phases in self.model.branch_phase_pair_set:
            ph1 = phases[0]
            ph2 = phases[1]
            if ph1 == ph2:
                continue
            for t in self.model.time_set:
                if (fb, tb, ph1 + ph1, t) in l_data and (
                    fb,
                    tb,
                    ph2 + ph2,
                    t,
                ) in l_data:
                    l_data[(fb, tb, phases, t)] = np.sqrt(
                        l_data[fb, tb, ph1 + ph1, t] * l_data[fb, tb, ph2 + ph2, t]
                    )
        self.model.l_flow.set_values(l_data)

        # Initialize current angle differences
        current_angles["ab"] = (current_angles.a - current_angles.b) % 360
        current_angles["bc"] = (current_angles.b - current_angles.c) % 360
        current_angles["ca"] = (current_angles.c - current_angles.a) % 360
        current_angles["ba"] = -current_angles.ab
        current_angles["cb"] = -current_angles.bc
        current_angles["ac"] = -current_angles.ca
        current_angles["aa"] = current_angles.a - current_angles.a
        current_angles["bb"] = current_angles.b - current_angles.b
        current_angles["cc"] = current_angles.c - current_angles.c
        current_angles["s1s1"] = current_angles.s1 - current_angles.s1
        current_angles["s2s2"] = current_angles.s2 - current_angles.s2
        current_angles["s1s2"] = current_angles.s1 - current_angles.s2
        current_angles["s2s1"] = -current_angles.s1s2

        angle_data = {
            (fb, tb, phases): float(
                current_angles.loc[current_angles.tb == tb, phases].tolist()[0]
            )
            * pi
            / 180
            for fb, tb, phases in self.model.branch_angle_phase_pair_set
        }
        for key in self.model.d:
            self.model.d[key] = angle_data[key]

    def _resolve_objective(self, objective: Any) -> Any:
        """Resolve objective string to Pyomo objective function."""
        from distopf.pyomo_models.objectives import (
            none_rule,
            loss_objective_rule,
            substation_power_objective_rule,
            voltage_deviation_objective_rule,
            generation_curtailment_objective_rule,
        )

        if objective is None:
            return none_rule

        if callable(objective):
            return objective

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
        """Extract branch power flow results (P flows)."""
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
