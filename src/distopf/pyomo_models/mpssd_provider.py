"""MPSSD model-component provider for the LinDistFlow formulation."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models.device_registry import DeviceProvider
from distopf.pyomo_models.injection_registry import InjectionRegistry
from distopf.pyomo_models.lindist import _parse_phases
from distopf.pyomo_models.model_types import CONTROL_VARIABLE_MAP
from distopf.pyomo_models import mpssd_constraints


class MpssdProvider(DeviceProvider):
    """Own MPSSD sets, parameters, variables, injections, and constraints."""

    name = "mpssd"
    supported_formulations = frozenset({"lindist"})

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        data = getattr(case, "mpssd_data", pd.DataFrame())
        model.mpssd_set = pyo.Set(
            initialize=[] if data.empty else data.id.astype(int).tolist()
        )
        phase_pairs = []
        bus_map: dict[tuple[int, str], list[int]] = {}
        if not data.empty:
            for _, row in data.iterrows():
                device = int(row.id)
                for phase in _parse_phases(str(row.phases)):
                    phase_pairs.append((device, phase))
                    bus_map.setdefault((int(row.bus), phase), []).append(device)
        model.mpssd_phase_set = pyo.Set(initialize=phase_pairs, dimen=2)
        model.mpssd_bus_map = bus_map
        dc_labels = sorted(
            {
                int(value)
                for value in data.get("dc_bus", pd.Series(dtype=int)).tolist()
                if pd.notna(value) and int(value) != 0
            }
        )
        model.dc_bus_set = pyo.Set(initialize=dc_labels)

        model.p_mpssd = pyo.Var(model.mpssd_phase_set, model.time_set, initialize=0)
        model.q_mpssd = pyo.Var(model.mpssd_phase_set, model.time_set, initialize=0)

        if data.empty:
            return

        values: dict[str, dict[tuple[Any, ...], Any]] = {
            "s_rated": {},
            "q_min": {},
            "q_max": {},
            "dc_bus": {},
            "control": {},
            "p_nom": {},
            "q_nom": {},
        }
        for _, row in data.iterrows():
            device = int(row.id)
            dc_bus = int(row.dc_bus) if pd.notna(row.get("dc_bus", 0)) else 0
            control = CONTROL_VARIABLE_MAP.get(row.get("control_variable", "PQ"), 3)
            for phase in _parse_phases(str(row.phases)):
                key = (device, phase)
                rating = row.get(f"s_{phase}_max", 1000.0)
                values["s_rated"][key] = rating
                values["q_min"][key] = row.get(f"q_{phase}_min", -rating)
                values["q_max"][key] = row.get(f"q_{phase}_max", rating)
                values["dc_bus"][key] = dc_bus
                values["control"][key] = control
                for time in model.time_set:
                    values["p_nom"][(*key, time)] = row.get(f"p_{phase}", 0.0)
                    values["q_nom"][(*key, time)] = row.get(f"q_{phase}", 0.0)

        model.mpssd_s_rated = pyo.Param(
            model.mpssd_phase_set, initialize=values["s_rated"], default=1000.0
        )
        model.mpssd_q_min = pyo.Param(
            model.mpssd_phase_set, initialize=values["q_min"], default=-1000.0
        )
        model.mpssd_q_max = pyo.Param(
            model.mpssd_phase_set, initialize=values["q_max"], default=1000.0
        )
        model.mpssd_dc_bus = pyo.Param(
            model.mpssd_phase_set, initialize=values["dc_bus"], default=0
        )
        model.mpssd_control_type = pyo.Param(
            model.mpssd_phase_set, initialize=values["control"], default=3
        )
        model.p_mpssd_nom = pyo.Param(
            model.mpssd_phase_set, model.time_set, initialize=values["p_nom"], default=0
        )
        model.q_mpssd_nom = pyo.Param(
            model.mpssd_phase_set, model.time_set, initialize=values["q_nom"], default=0
        )

    def register_injections(
        self, model: Any, injections: InjectionRegistry, config: Any
    ) -> None:
        injections.add(
            self.name,
            p_term=lambda m, bus, phase, time: sum(
                m.p_mpssd[device, phase, time]
                for device in m.mpssd_bus_map.get((bus, phase), [])
            ),
            q_term=lambda m, bus, phase, time: sum(
                m.q_mpssd[device, phase, time]
                for device in m.mpssd_bus_map.get((bus, phase), [])
            ),
        )

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.mpssd_phase_set) == 0:
            return
        circular = getattr(config, "circular_constraints", True) if config else True
        mpssd_constraints.add_mpssd_constant_p_constraints_q_control(model)
        mpssd_constraints.add_mpssd_constant_q_constraints_p_control(model)
        mpssd_constraints.add_mpssd_limits(model)
        if circular:
            mpssd_constraints.add_circular_mpssd_constraints(model)
        else:
            mpssd_constraints.add_octagonal_mpssd_constraints(model)
        if len(model.dc_bus_set) > 0:
            mpssd_constraints.add_dc_bus_balance_constraints(model)
