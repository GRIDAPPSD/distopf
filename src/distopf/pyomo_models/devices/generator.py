"""Generator device provider for dispatch, limits, and control policies."""

from __future__ import annotations

import warnings
from typing import Any

import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models import common_constraints
from distopf.pyomo_models.devices.data import parse_phases, phase_tuples
from distopf.pyomo_models.devices.injections import InjectionRegistry
from distopf.pyomo_models.model_types import CONTROL_VARIABLE_MAP


def create_generator_parameters(model: Any, case: Any) -> None:
    """Create the complete legacy-compatible generator parameter surface."""
    p_data, q_data, rating, q_min, q_max, control, cost = {}, {}, {}, {}, {}, {}, {}
    for _, row in case.gen_data.iterrows():
        for phase in parse_phases(str(row.phases)):
            key = (row.id, phase)
            if key not in model.gen_phase_set:
                continue
            s = getattr(row, f"s_{phase}_max", 1000.0)
            rating[key] = s
            q_min[key] = getattr(row, f"q_{phase}_min", -s)
            q_max[key] = getattr(row, f"q_{phase}_max", s)
            control[key] = CONTROL_VARIABLE_MAP[getattr(row, "control_variable", "")]
            cost[key] = getattr(row, "cost", 0.0)
            for time in model.time_set:
                multiplier = 1.0
                shape = getattr(row, "gen_shape", "PV")
                if shape in case.schedules.columns and time in case.schedules.index:
                    try:
                        multiplier = float(case.schedules.at[time, shape])
                    except (TypeError, ValueError):
                        warnings.warn(f"Non-numeric generator schedule {shape!r}; using 1.0")
                p_data[(key[0], key[1], time)] = getattr(row, f"p_{phase}", 0.0) * multiplier
                q_data[(key[0], key[1], time)] = getattr(row, f"q_{phase}", 0.0)
    model.p_gen_nom = pyo.Param(model.gen_phase_set, model.time_set, initialize=p_data, default=0.0)
    model.q_gen_nom = pyo.Param(model.gen_phase_set, model.time_set, initialize=q_data, default=0.0)
    model.s_rated = pyo.Param(model.gen_phase_set, initialize=rating, default=1000.0)
    model.q_gen_min = pyo.Param(model.gen_phase_set, initialize=q_min, default=-1000.0)
    model.q_gen_max = pyo.Param(model.gen_phase_set, initialize=q_max, default=1000.0)
    model.gen_control_type = pyo.Param(model.gen_phase_set, initialize=control, default=0)
    model.gen_cost = pyo.Param(model.gen_phase_set, initialize=cost, default=0.0)


class GeneratorProvider:
    """Own generator parameters, variables, operating constraints, and injection."""

    name = "generators"
    supported_formulations = frozenset({"lindist", "nl_bfm"})

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        if not hasattr(model, "gen_phase_set"):
            model.gen_phase_set = pyo.Set(initialize=phase_tuples(case.gen_data), dimen=2)
        if not hasattr(model, "gen_set"):
            model.gen_set = pyo.Set(
                initialize=sorted({device for device, _ in model.gen_phase_set})
            )
        if not hasattr(model, "p_gen"):
            model.p_gen = pyo.Var(model.gen_phase_set, model.time_set, domain=pyo.NonNegativeReals)
        if not hasattr(model, "q_gen"):
            model.q_gen = pyo.Var(model.gen_phase_set, model.time_set, initialize=0)
        if not hasattr(model, "p_gen_nom"):
            create_generator_parameters(model, case)
        if not hasattr(model, "gen_phase_pair_set"):
            model.gen_phase_pair_set = pyo.Set(
                initialize=[
                    (device, left, right)
                    for device in model.gen_set
                    for left, right in zip(
                        [phase for phase in ("a", "b", "c") if (device, phase) in model.gen_phase_set],
                        [phase for phase in ("a", "b", "c") if (device, phase) in model.gen_phase_set][1:],
                    )
                ],
                dimen=3,
            )
        if not hasattr(model, "gen_phase_lock"):
            model.gen_phase_lock = pyo.Param(
                model.gen_set, initialize={device: False for device in model.gen_set},
                within=pyo.Boolean, mutable=True,
            )

    def register_injections(
        self, model: Any, injections: InjectionRegistry, config: Any
    ) -> None:
        if any(provider.name == self.name for provider in injections.providers):
            return
        injections.add(
            self.name,
            p_term=lambda m, bus, phase, time: (
                m.p_gen[bus, phase, time]
                if (bus, phase, time) in m.p_gen
                else 0
            ),
            q_term=lambda m, bus, phase, time: (
                m.q_gen[bus, phase, time]
                if (bus, phase, time) in m.q_gen
                else 0
            ),
        )

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.gen_phase_set) == 0:
            return
        equality_only = getattr(config, "equality_only", False) if config else False
        if not equality_only and not hasattr(model, "p_gen_limits"):
            common_constraints.add_generator_limits(model)
        if not hasattr(model, "constant_p_gen"):
            common_constraints.add_generator_constant_p_constraints_q_control(model)
        if not hasattr(model, "constant_q_gen"):
            common_constraints.add_generator_constant_q_constraints_p_control(model)
        if equality_only:
            return
        if getattr(config, "circular_constraints", True) if config else True:
            if not hasattr(model, "gen_circle_constraint"):
                common_constraints.add_circular_generator_constraints_pq_control(model)
        elif not hasattr(model, "gen_octagon_1"):
            common_constraints.add_octagonal_inverter_constraints_pq_control(model)


__all__ = ["GeneratorProvider", "create_generator_parameters"]
