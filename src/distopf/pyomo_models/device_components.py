"""Shared device-component construction helpers for Pyomo models."""

from __future__ import annotations

from typing import Any

import pyomo.environ as pyo  # type: ignore

from distopf.pyomo_models.device_data import parse_phases


def _phase_tuples(data, id_column="id"):
    return [
        (row[id_column], phase)
        for _, row in data.iterrows()
        for phase in parse_phases(str(row.phases))
    ]


def create_device_sets(model: Any, case: Any) -> None:
    """Create device sets only; network sets belong to ``core_model``."""
    model.gen_phase_set = pyo.Set(
        initialize=_phase_tuples(case.gen_data), dimen=2
    )
    model.gen_set = pyo.Set(
        initialize=sorted({device for device, _ in model.gen_phase_set})
    )
    model.cap_phase_set = pyo.Set(
        initialize=_phase_tuples(case.cap_data), dimen=2
    )
    model.reg_phase_set = pyo.Set(
        initialize=[
            (int(row.fb), int(row.tb), phase)
            for _, row in case.reg_data.iterrows()
            for phase in parse_phases(str(row.phases))
        ],
        dimen=3,
    )
    model.bat_phase_set = pyo.Set(
        initialize=_phase_tuples(case.bat_data), dimen=2
    )
    model.bat_set = pyo.Set(initialize=case.bat_data.id.tolist())


def create_device_variables(model: Any, case: Any) -> None:
    """Create variables owned by built-in device providers."""
    model.p_gen = pyo.Var(model.gen_phase_set, model.time_set, domain=pyo.NonNegativeReals)
    model.q_gen = pyo.Var(model.gen_phase_set, model.time_set, initialize=0)
    model.q_cap = pyo.Var(model.cap_phase_set, model.time_set)
    model.p_load = pyo.Var(model.bus_phase_set, model.time_set)
    model.q_load = pyo.Var(model.bus_phase_set, model.time_set)

    model.p_charge = pyo.Var(model.bat_set, model.time_set, initialize=0)
    model.p_discharge = pyo.Var(model.bat_set, model.time_set, initialize=0)
    model.p_bat = pyo.Var(model.bat_phase_set, model.time_set, initialize=0)
    model.q_bat = pyo.Var(model.bat_phase_set, model.time_set, initialize=0)
    model.soc = pyo.Var(model.bat_set, model.time_set, initialize=0.5)


def create_device_parameters(model: Any, case: Any) -> None:
    """Create device parameters using dependency-free builders."""
    from distopf.pyomo_models.device_parameter_builders import (
        create_capacitor_parameters,
        create_generator_parameters,
        create_load_parameters,
        create_regulator_parameters,
    )

    create_load_parameters(model, case)
    create_generator_parameters(model, case)
    create_capacitor_parameters(model, case)
    create_regulator_parameters(model, case)


def create_generator_policy_parameters(model: Any) -> None:
    """Create optional generator phase-lock metadata."""
    model.gen_phase_lock = pyo.Param(
        model.gen_set,
        initialize={device: False for device in model.gen_set},
        within=pyo.Boolean,
        mutable=True,
    )
    pairs = []
    for device in model.gen_set:
        phases = [phase for phase in ("a", "b", "c") if (device, phase) in model.gen_phase_set]
        pairs.extend((device, left, right) for left, right in zip(phases, phases[1:]))
    model.gen_phase_pair_set = pyo.Set(initialize=pairs, dimen=3)
