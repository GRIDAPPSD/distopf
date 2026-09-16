"""Shared network-component creation for Pyomo formulations."""

from __future__ import annotations

import pandas as pd
import pyomo.environ as pyo  # type: ignore

from distopf.api import Case
from distopf.pyomo_models.devices.data import parse_phases


PHASE_PAIR_LABELS = ("aa", "ab", "ac", "bb", "bc", "cc", "s1s1", "s1s2", "s2s2")


def create_network_sets(model: pyo.ConcreteModel, case: Case) -> None:
    """Create only topology, time, and bus/branch phase sets."""
    model.time_set = pyo.RangeSet(case.start_step, case.start_step + case.n_steps - 1)
    model.bus_set = pyo.Set(initialize=case.bus_data.id.tolist())
    swing_mask = case.bus_data.bus_type.isin(["SWING", "SWING_FREE", "IN"])
    boundary_in_mask = case.bus_data.bus_type.isin(["SWING_FREE", "IN"])
    boundary_out_mask = case.bus_data.bus_type.isin(["OUT"])
    model.swing_bus_set = pyo.Set(initialize=case.bus_data.loc[swing_mask, "id"].tolist())
    model.swing_phase_set = pyo.Set(
        initialize=[(row.id, phase) for _, row in case.bus_data.loc[swing_mask].iterrows() for phase in parse_phases(str(row.phases))],
        dimen=2,
    )
    model.boundary_in_set = pyo.Set(initialize=case.bus_data.loc[boundary_in_mask, "id"].tolist())
    model.boundary_out_set = pyo.Set(initialize=case.bus_data.loc[boundary_out_mask, "id"].tolist())
    model.branch_set = pyo.Set(
        initialize=[(int(row.fb), int(row.tb)) for _, row in case.branch_data.iterrows()],
        dimen=2,
    )
    model.phase_pair_set = pyo.Set(initialize=PHASE_PAIR_LABELS)
    model.bus_phase_set = pyo.Set(
        initialize=[(row.id, phase) for _, row in case.bus_data.iterrows() for phase in parse_phases(str(row.phases))],
        dimen=2,
    )
    model.branch_phase_set = pyo.Set(
        initialize=[(int(row.fb), int(row.tb), phase) for _, row in case.branch_data.iterrows() for phase in parse_phases(str(row.phases))],
        dimen=3,
    )


def create_network_parameters(model: pyo.ConcreteModel, case: Case) -> None:
    """Create impedance parameters shared by LinDistFlow and BranchFlow."""
    resistance = {}
    reactance = {}
    for _, row in case.branch_data.iterrows():
        branch = (int(row.fb), int(row.tb))
        for pair in PHASE_PAIR_LABELS:
            if f"r_{pair}" in case.branch_data and f"x_{pair}" in case.branch_data:
                resistance[(*branch, pair)] = row[f"r_{pair}"]
                reactance[(*branch, pair)] = row[f"x_{pair}"]
    model.r = pyo.Param(model.branch_set, model.phase_pair_set, initialize=resistance, default=0.0)
    model.x = pyo.Param(model.branch_set, model.phase_pair_set, initialize=reactance, default=0.0)


def create_network_components(model: pyo.ConcreteModel, case: Case) -> None:
    """Create common network sets, parameters, and topology maps."""
    create_network_sets(model, case)
    create_network_parameters(model, case)

    model.primary_phase_map = {}
    if "primary_phase" in case.branch_data.columns:
        bus_phases = dict(zip(case.bus_data.id, case.bus_data.phases))
        for _, row in case.branch_data.iterrows():
            primary = getattr(row, "primary_phase", "")
            from_phases = str(bus_phases.get(int(row.fb), ""))
            if primary and not pd.isna(primary) and "s1" not in from_phases and "s2" not in from_phases:
                model.primary_phase_map[(int(row.fb), int(row.tb))] = str(primary)
    model.to_bus_map = {
        int(bus): [
            (int(row.fb), int(row.tb))
            for _, row in case.branch_data.loc[case.branch_data.fb == int(bus), ["fb", "tb"]].iterrows()
        ]
        for bus in case.bus_data.id
    }
    model.name_map = {
        int(row["id"]): str(row["name"])
        for _, row in case.bus_data[["id", "name"]].iterrows()
    }


def create_network_operating_parameters(model: pyo.ConcreteModel, case: Case) -> None:
    """Create voltage, thermal, schedule, and market parameters shared by models."""
    swing = {}
    swing_mask = case.bus_data.bus_type.isin(["SWING", "SWING_FREE", "IN"])
    for _, row in case.bus_data.loc[swing_mask].iterrows():
        for phase in parse_phases(str(row.phases)):
            for time in model.time_set:
                value = getattr(row, f"v_{phase}", 1.0)
                if f"v_{phase}" in case.schedules.columns and time in case.schedules.index:
                    scheduled = case.schedules.at[time, f"v_{phase}"]
                    if pd.notna(scheduled):
                        value = float(scheduled)
                swing[(row.id, phase, time)] = value
    model.v_swing = pyo.Param(model.swing_phase_set, model.time_set, initialize=swing, default=1.0)
    model.v_min = pyo.Param(model.bus_phase_set, initialize={(row.id, phase): getattr(row, "v_min", 0.95) for _, row in case.bus_data.iterrows() for phase in parse_phases(str(row.phases))}, default=0.95)
    model.v_max = pyo.Param(model.bus_phase_set, initialize={(row.id, phase): getattr(row, "v_max", 1.05) for _, row in case.bus_data.iterrows() for phase in parse_phases(str(row.phases))}, default=1.05)
    thermal = {}
    columns = {"a": ("s_a_max",), "b": ("s_b_max",), "c": ("s_c_max",), "s1": ("s_s1_max", "s1_max"), "s2": ("s_s2_max", "s2_max")}
    for _, row in case.branch_data.iterrows():
        for phase in parse_phases(str(row.phases)):
            for column in columns.get(phase, ()):
                if column in case.branch_data and pd.notna(getattr(row, column, None)):
                    thermal[(int(row.fb), int(row.tb), phase)] = getattr(row, column)
                    break
    if thermal:
        model.s_branch_max = pyo.Param(model.branch_phase_set, initialize=thermal, default=None)
    model.price = pyo.Param(model.time_set, initialize={time: case.schedules.at[time, "price"] for time in model.time_set if "price" in case.schedules and time in case.schedules.index}, default=0.0)
    for column in case.schedules.columns:
        if column == "time":
            continue
        setattr(model, f"schedule_{column}", pyo.Param(model.time_set, initialize={time: float(case.schedules.at[time, column]) for time in model.time_set if time in case.schedules.index}, default=0.0))
