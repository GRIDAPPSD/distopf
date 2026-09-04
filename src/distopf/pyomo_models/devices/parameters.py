"""Canonical device parameter builders extracted from model factories."""

from __future__ import annotations

import warnings
from typing import Any

import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models.devices.data import parse_phases
from distopf.pyomo_models.model_types import CONTROL_VARIABLE_MAP


def create_load_parameters(model: Any, case: Any) -> None:
    p_data, q_data, cvr_p, cvr_q = {}, {}, {}, {}
    for _, row in case.bus_data.iterrows():
        for phase in parse_phases(str(row.phases)):
            if (row.id, phase) not in model.bus_phase_set:
                continue
            p_load = getattr(row, f"pl_{phase}", 0.0)
            q_load = getattr(row, f"ql_{phase}", 0.0)
            if phase in ("s1", "s2"):
                p_load += getattr(row, "pl_s1s2", 0.0) / 2
                q_load += getattr(row, "ql_s1s2", 0.0) / 2
            cvr_p[(row.id, phase)] = getattr(row, "cvr_p", 0.0)
            cvr_q[(row.id, phase)] = getattr(row, "cvr_q", 0.0)
            shape = getattr(row, "load_shape", "default")
            for time in model.time_set:
                multiplier_p = multiplier_q = 1.0
                if shape in case.schedules.columns:
                    multiplier_p = multiplier_q = case.schedules.at[time, shape]
                elif f"{shape}.{phase}.p" in case.schedules.columns:
                    multiplier_p = case.schedules.at[time, f"{shape}.{phase}.p"]
                    multiplier_q = case.schedules.at[time, f"{shape}.{phase}.q"]
                p_data[row.id, phase, time] = p_load * multiplier_p
                q_data[row.id, phase, time] = q_load * multiplier_q
    model.p_load_nom = pyo.Param(model.bus_phase_set, model.time_set, initialize=p_data, default=0.0)
    model.q_load_nom = pyo.Param(model.bus_phase_set, model.time_set, initialize=q_data, default=0.0)
    model.cvr_p = pyo.Param(model.bus_phase_set, initialize=cvr_p, default=0.0)
    model.cvr_q = pyo.Param(model.bus_phase_set, initialize=cvr_q, default=0.0)


def create_generator_parameters(model: Any, case: Any) -> None:
    p_data, q_data, rating, q_min, q_max, control = {}, {}, {}, {}, {}, {}
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
            for time in model.time_set:
                multiplier = 1.0
                shape = getattr(row, "gen_shape", "PV")
                if shape in case.schedules.columns:
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


def create_capacitor_parameters(model: Any, case: Any) -> None:
    q_data = {(row.id, phase): getattr(row, f"q_{phase}", 0.0) for _, row in case.cap_data.iterrows() for phase in parse_phases(str(row.phases))}
    model.q_cap_nom = pyo.Param(model.cap_phase_set, initialize=q_data, default=0.0)


def create_regulator_parameters(model: Any, case: Any) -> None:
    ratio = {(int(row.fb), int(row.tb), phase): getattr(row, f"ratio_{phase}", 1.0) for _, row in case.reg_data.iterrows() for phase in parse_phases(str(row.phases))}
    model.reg_ratio = pyo.Param(model.reg_phase_set, initialize=ratio, default=1.0)
