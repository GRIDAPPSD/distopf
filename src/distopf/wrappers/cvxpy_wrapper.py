"""CVXPY wrapper for OPF built on ``distopf.cvxpy_models``.

Mirror of :class:`distopf.wrappers.pyomo_wrapper.PyomoWrapper`. Supports
LinDistFlow (default), and BranchFlow via ``model_type``:

- ``"lindist"``: LinDistFlow (convex).
- ``"socp"``: SOCP relaxation of BranchFlow (convex).
- ``"branchflow"``: exact BranchFlow. The model is built but is not DCP, so
  solving raises ``NotImplementedError``.
"""

from __future__ import annotations

from math import pi
from typing import Any, Optional, Union

import numpy as np

from distopf.cvxpy_models.common import objectives
from distopf.pyomo_models.common.data import parse_phases
from distopf.wrappers.pyomo_wrapper import PyomoWrapper


class CvxpyWrapper(PyomoWrapper):
    """CVXPY wrapper for OPF optimization (provider-based scalar model)."""

    def solve(
        self,
        objective: Optional[Any] = None,
        control_regulators: bool = False,
        control_capacitors: bool = False,
        raw_result: bool = False,
        **kwargs: Any,
    ) -> Union[Any, Any]:
        """Run OPF using the CVXPY model layer.

        Accepts the same arguments as :meth:`PyomoWrapper.solve`. ``solver`` is
        a CVXPY solver name (default: chosen automatically).
        """
        model_type = kwargs.pop("model_type", "lindist")
        if model_type in ("branchflow", "socp"):
            return self._solve_branchflow(
                objective=objective,
                control_regulators=control_regulators,
                control_capacitors=control_capacitors,
                raw_result=raw_result,
                socp_relaxation=model_type == "socp",
                **kwargs,
            )
        return self._solve_lindist(
            objective=objective,
            control_regulators=control_regulators,
            control_capacitors=control_capacitors,
            raw_result=raw_result,
            **kwargs,
        )

    def _combined_objective(
        self,
        objective,
        free_swing_voltage,
        swing_voltage_slack_penalty,
        rho_v_up,
        rho_s_up,
        rho_v_dn,
        rho_s_dn,
        penalties=None,
    ):
        obj_rule = self._resolve_objective(objective)
        penalties = penalties or {}

        def combined_objective_rule(_m):
            value = obj_rule(_m)
            if free_swing_voltage and swing_voltage_slack_penalty > 0:
                value += objectives.swing_voltage_slack_penalty(
                    _m, weight=swing_voltage_slack_penalty
                )
            if rho_v_up is not None:
                value += objectives.admm_v_up_penalty(_m, rho=rho_v_up)
            if rho_s_up is not None:
                value += objectives.admm_s_up_penalty(_m, rho=rho_s_up)
            if rho_v_dn is not None:
                value += objectives.admm_v_down_penalty(_m, rho=rho_v_dn)
            if rho_s_dn is not None:
                value += objectives.admm_s_down_penalty(_m, rho=rho_s_dn)
            for penalty_fn, weight in penalties.items():
                if weight is not None:
                    value += penalty_fn(_m, weight)
            return value

        return combined_objective_rule

    def _solve_lindist(
        self, objective, control_regulators, control_capacitors, raw_result, **kwargs
    ):
        from distopf.cvxpy_models.common.factory import create_lindist_model
        from distopf.cvxpy_models.common.objectives import set_objective
        from distopf.cvxpy_models.common.solvers import solve

        circular_constraints = kwargs.pop("circular_constraints", False)
        thermal_constraints = kwargs.pop("thermal_constraints", False)
        equality_only = kwargs.pop("equality_only", False)
        reg_tap_change_limit = kwargs.pop("reg_tap_change_limit", None)

        rho_v_up = kwargs.pop("rho_v_up", None)
        rho_s_dn = kwargs.pop("rho_s_dn", None)
        rho_v_dn = kwargs.pop("rho_v_dn", None)
        rho_s_up = kwargs.pop("rho_s_up", None)

        free_swing_voltage = kwargs.pop("free_swing_voltage", False)
        swing_voltage_slack_penalty = kwargs.pop("swing_voltage_slack_penalty", 0)
        free_boundary_loads = kwargs.pop("free_boundary_loads", False)
        for ignored in (
            "boundary_voltage_slack_penalty",
            "boundary_load_slack_penalty",
            "swing_power_slack_penalty",
        ):
            kwargs.pop(ignored, None)

        duals = kwargs.pop("duals", False)
        verbose = kwargs.pop("verbose", False)

        penalties = None
        if equality_only:
            penalties = {
                objectives.voltage_violation_penalty: kwargs.pop(
                    "voltage_weight", None
                ),
                objectives.thermal_violation_penalty: kwargs.pop(
                    "thermal_weight", None
                ),
                objectives.generator_violation_penalty: kwargs.pop(
                    "generator_weight", None
                ),
                objectives.battery_violation_penalty: kwargs.pop(
                    "battery_weight", None
                ),
                objectives.soc_violation_penalty: kwargs.pop("soc_weight", None),
            }
        for key in (
            "voltage_weight",
            "thermal_weight",
            "generator_weight",
            "battery_weight",
            "soc_weight",
        ):
            kwargs.pop(key, None)
        solver = kwargs.pop("solver", None)

        self.model = create_lindist_model(
            self.case,
            control_capacitors=control_capacitors,
            control_regulators=control_regulators,
            circular_constraints=circular_constraints,
            thermal_constraints=thermal_constraints,
            equality_only=equality_only,
            reg_tap_change_limit=reg_tap_change_limit,
            free_swing_voltage=free_swing_voltage,
            free_boundary_loads=free_boundary_loads,
        )
        set_objective(
            self.model,
            self._combined_objective(
                objective,
                free_swing_voltage,
                swing_voltage_slack_penalty,
                rho_v_up,
                rho_s_up,
                rho_v_dn,
                rho_s_dn,
                penalties,
            ),
        )
        self.result = solve(self.model, solver=solver, duals=duals, verbose=verbose)
        if raw_result:
            return self.result
        return self._build_result(solver_name=self._solver_name(solver))

    def _solve_branchflow(
        self, objective, control_regulators, control_capacitors, raw_result, **kwargs
    ):
        from distopf.cvxpy_models.common.factory import create_nl_branchflow_model
        from distopf.cvxpy_models.common.objectives import set_objective
        from distopf.cvxpy_models.common.solvers import solve

        socp_relaxation = kwargs.pop("socp_relaxation", False)
        sdp_relaxation = kwargs.pop("sdp_relaxation", False)
        circular_constraints = kwargs.pop("circular_constraints", True)
        thermal_constraints = kwargs.pop("thermal_constraints", False)
        initialize = kwargs.pop("initialize", "fbs")
        solver = kwargs.pop("solver", None)
        verbose = kwargs.pop("verbose", False)
        duals = kwargs.pop("duals", False)

        rho_v_up = kwargs.pop("rho_v_up", None)
        rho_s_dn = kwargs.pop("rho_s_dn", None)
        rho_v_dn = kwargs.pop("rho_v_dn", None)
        rho_s_up = kwargs.pop("rho_s_up", None)

        free_swing_voltage = kwargs.pop("free_swing_voltage", False)
        swing_voltage_slack_penalty = kwargs.pop("swing_voltage_slack_penalty", 0)
        free_boundary_loads = kwargs.pop(
            "free_boundary_loads", kwargs.pop("free_boundary_load", False)
        )

        self.model = create_nl_branchflow_model(
            self.case,
            control_capacitors=control_capacitors,
            control_regulators=control_regulators,
            circular_constraints=circular_constraints,
            thermal_constraints=thermal_constraints,
            socp_relaxation=socp_relaxation,
            sdp_relaxation=sdp_relaxation,
            free_swing_voltage=free_swing_voltage,
            free_boundary_loads=free_boundary_loads,
        )
        if initialize == "fbs":
            self._initialize_from_fbs()

        set_objective(
            self.model,
            self._combined_objective(
                objective,
                free_swing_voltage,
                swing_voltage_slack_penalty,
                rho_v_up,
                rho_s_up,
                rho_v_dn,
                rho_s_dn,
            ),
        )
        self.result = solve(self.model, solver=solver, duals=duals, verbose=verbose)
        if raw_result:
            return self.result
        return self._build_result(solver_name=self._solver_name(solver))

    @staticmethod
    def _solver_name(solver: Optional[str]) -> str:
        if solver is None or str(solver).lower() == "ipopt":
            return "cvxpy"
        return str(solver).upper()

    def _initialize_from_fbs(self) -> None:
        """Warm start BranchFlow variables and current angles from FBS."""
        from distopf.cvxpy_models.network.bfm import set_current_angles
        from distopf.fbs import fbs_solve

        fbs_results = fbs_solve(self.case)
        voltages = fbs_results.voltages
        active_power_flows = fbs_results.active_power_flows
        reactive_power_flows = fbs_results.reactive_power_flows
        currents = fbs_results.currents
        current_angles = fbs_results.current_angles
        assert voltages is not None
        assert active_power_flows is not None
        assert reactive_power_flows is not None
        assert currents is not None
        assert current_angles is not None
        model = self.model

        def _lookup_branch_phase(df, fb, tb, ph):
            mask = df.tb == tb
            if "fb" in df.columns:
                mask = mask & (df.fb == fb)
            vals = df.loc[mask, ph].to_numpy()
            return 0.0 if len(vals) == 0 else vals[0]

        for _id, ph in model.bus_phase_set:
            v_mag = voltages.loc[(voltages.id == _id), ph].to_numpy()[0]
            for t in model.time_set:
                model.v2[_id, ph, t].value = float(v_mag**2)
        for fb, tb, ph in model.reg_phase_set:
            v_mag = voltages.loc[(voltages.id == tb), ph].to_numpy()[0]
            for t in model.time_set:
                model.v2_reg[fb, tb, ph, t].value = float(v_mag**2)

        l_data: dict[tuple, float] = {}
        for fb, tb, ph in model.branch_phase_set:
            p_flow = _lookup_branch_phase(active_power_flows, fb, tb, ph)
            q_flow = _lookup_branch_phase(reactive_power_flows, fb, tb, ph)
            i_mag = _lookup_branch_phase(currents, fb, tb, ph)
            for t in model.time_set:
                model.p_flow[fb, tb, ph, t].value = float(p_flow)
                model.q_flow[fb, tb, ph, t].value = float(q_flow)
                l_data[(fb, tb, ph + ph, t)] = float(i_mag) ** 2
        for fb, tb, phases in model.branch_phase_pair_set:
            ph1, ph2 = parse_phases(phases)
            if ph1 == ph2:
                continue
            for t in model.time_set:
                if (fb, tb, ph1 + ph1, t) in l_data and (fb, tb, ph2 + ph2, t) in l_data:
                    l_data[(fb, tb, phases, t)] = float(
                        np.sqrt(l_data[fb, tb, ph1 + ph1, t] * l_data[fb, tb, ph2 + ph2, t])
                    )
        for key, value in l_data.items():
            if key in model.l_flow:
                model.l_flow[key].value = value

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
            for fb, tb, phases in model.branch_angle_phase_pair_set
        }
        set_current_angles(model, angle_data)

    def _resolve_objective(self, objective: Any) -> Any:
        """Resolve objective string to a CVXPY objective rule."""
        if objective is None:
            return objectives.none_rule
        if callable(objective):
            return objective

        obj_lower = objective.lower().strip()
        penalty_rule = (
            objectives.generation_cost_with_substation_quadratic_penalty_objective_rule
        )
        objective_map = {
            "loss": objectives.loss_objective_rule,
            "loss_min": objectives.loss_objective_rule,
            "substation": objectives.substation_power_objective_rule,
            "substation_power": objectives.substation_power_objective_rule,
            "voltage_deviation": objectives.voltage_deviation_objective_rule,
            "voltage_min": objectives.voltage_min_objective_rule,
            "min_voltage": objectives.voltage_min_objective_rule,
            "curtail": objectives.generation_curtailment_objective_rule,
            "curtail_min": objectives.generation_curtailment_objective_rule,
            "curtailment": objectives.generation_curtailment_objective_rule,
            "generation_cost_substation_penalty": penalty_rule,
            "gen_cost_substation_penalty": penalty_rule,
            "cost": objectives.cost_minimization_rule,
            "cost_min": objectives.cost_minimization_rule,
        }
        if obj_lower in objective_map:
            return objective_map[obj_lower]
        raise ValueError(
            f"Unknown cvxpy objective: '{objective}'. "
            f"Supported: {', '.join(objective_map.keys())}"
        )
