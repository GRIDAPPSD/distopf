"""Network provider (sets, impedance, operating parameters, flow variables).

CVXPY mirror of ``distopf.pyomo_models.network.bfm``.
"""

from __future__ import annotations

from itertools import combinations_with_replacement, product
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from distopf.api import Case
from distopf.cvxpy_models.common.model import (
    CvxModel,
    IndexSet,
    ParamDict,
    VarDict,
    index_product,
)
from distopf.cvxpy_models.network import bfm_constraints
from distopf.pyomo_models.common.data import parse_phases


PHASE_PAIR_LABELS = ("aa", "ab", "ac", "bb", "bc", "cc", "s1s1", "s1s2", "s2s2")


def create_network_sets(model: CvxModel, case: Case) -> None:
    """Create only topology, time, and bus/branch phase sets."""

    model.delta_t = case.delta_t
    model.start_step = case.start_step
    model.n_steps = case.n_steps
    model.time_set = IndexSet(
        range(case.start_step, case.start_step + case.n_steps), dimen=1
    )
    model.bus_set = IndexSet(case.bus_data.id.tolist(), dimen=1)
    swing_mask = case.bus_data.bus_type.isin(["SWING", "SWING_FREE", "IN"])
    boundary_in_mask = case.bus_data.bus_type.isin(["SWING_FREE", "IN"])
    boundary_out_mask = case.bus_data.bus_type.isin(["OUT"])
    model.swing_bus_set = IndexSet(
        case.bus_data.loc[swing_mask, "id"].tolist(), dimen=1
    )
    model.swing_phase_set = IndexSet(
        [
            (row.id, phase)
            for _, row in case.bus_data.loc[swing_mask].iterrows()
            for phase in parse_phases(str(row.phases))
        ],
        dimen=2,
    )
    model.boundary_in_set = IndexSet(
        case.bus_data.loc[boundary_in_mask, "id"].tolist(), dimen=1
    )
    model.boundary_out_set = IndexSet(
        case.bus_data.loc[boundary_out_mask, "id"].tolist(), dimen=1
    )
    model.branch_set = IndexSet(
        [(int(row.fb), int(row.tb)) for _, row in case.branch_data.iterrows()],
        dimen=2,
    )
    model.phase_pair_set = IndexSet(PHASE_PAIR_LABELS, dimen=1)
    model.bus_phase_set = IndexSet(
        [
            (row.id, phase)
            for _, row in case.bus_data.iterrows()
            for phase in parse_phases(str(row.phases))
        ],
        dimen=2,
    )
    model.branch_phase_set = IndexSet(
        [
            (int(row.fb), int(row.tb), phase)
            for _, row in case.branch_data.iterrows()
            for phase in parse_phases(str(row.phases))
        ],
        dimen=3,
    )


def create_network_rx_parameters(model: CvxModel, case: Case) -> None:
    """Create impedance parameters shared by LinDistFlow and BranchFlow."""
    resistance = {}
    reactance = {}
    for _, row in case.branch_data.iterrows():
        branch = (int(row.fb), int(row.tb))
        for pair in PHASE_PAIR_LABELS:
            if f"r_{pair}" in case.branch_data and f"x_{pair}" in case.branch_data:
                resistance[(*branch, pair)] = row[f"r_{pair}"]
                reactance[(*branch, pair)] = row[f"x_{pair}"]
    model.r = ParamDict(resistance, default=0.0)
    model.x = ParamDict(reactance, default=0.0)


def create_network_operating_parameters(model: CvxModel, case: Case) -> None:
    """Create voltage, thermal, schedule, and market parameters shared by models."""
    swing = {}
    swing_mask = case.bus_data.bus_type.isin(["SWING", "SWING_FREE", "IN"])
    for _, row in case.bus_data.loc[swing_mask].iterrows():
        for phase in parse_phases(str(row.phases)):
            for time in model.time_set:
                value = getattr(row, f"v_{phase}", 1.0)
                if (
                    f"v_{phase}" in case.schedules.columns
                    and time in case.schedules.index
                ):
                    scheduled = case.schedules.at[time, f"v_{phase}"]
                    if pd.notna(scheduled):
                        value = float(scheduled)  # type: ignore[arg-type]
                swing[(row.id, phase, time)] = value
    model.v_swing = ParamDict(swing, default=1.0)
    model.v_min = ParamDict(
        {
            (row.id, phase): getattr(row, "v_min", 0.95)
            for _, row in case.bus_data.iterrows()
            for phase in parse_phases(str(row.phases))
        },
        default=0.95,
    )
    model.v_max = ParamDict(
        {
            (row.id, phase): getattr(row, "v_max", 1.05)
            for _, row in case.bus_data.iterrows()
            for phase in parse_phases(str(row.phases))
        },
        default=1.05,
    )
    thermal = {}
    columns = {
        "a": ("s_a_max",),
        "b": ("s_b_max",),
        "c": ("s_c_max",),
        "s1": ("s_s1_max", "s1_max"),
        "s2": ("s_s2_max", "s2_max"),
    }
    for _, row in case.branch_data.iterrows():
        for phase in parse_phases(str(row.phases)):
            for column in columns.get(phase, ()):
                if column in case.branch_data and pd.notna(getattr(row, column, None)):
                    thermal[(int(row.fb), int(row.tb), phase)] = getattr(row, column)
                    break
    if thermal:
        model.s_branch_max = ParamDict(thermal, default=None)
    model.price = ParamDict(
        {
            time: case.schedules.at[time, "price"]
            for time in model.time_set
            if "price" in case.schedules and time in case.schedules.index
        },
        default=0.0,
    )
    for column in case.schedules.columns:
        if column == "time":
            continue
        setattr(
            model,
            f"schedule_{column}",
            ParamDict(
                {
                    time: float(case.schedules.at[time, column])  # type: ignore[arg-type]
                    for time in model.time_set
                    if time in case.schedules.index
                },
                default=0.0,
            ),
        )


def create_network_components(model: CvxModel, case: Case) -> None:
    """Create common network sets, parameters, and topology maps."""
    create_network_sets(model, case)
    create_network_rx_parameters(model, case)
    create_network_operating_parameters(model, case)

    model.primary_phase_map = {}
    if "primary_phase" in case.branch_data.columns:
        bus_phases = dict(zip(case.bus_data.id, case.bus_data.phases))
        for _, row in case.branch_data.iterrows():
            primary = getattr(row, "primary_phase", "")
            from_phases = str(bus_phases.get(int(row.fb), ""))
            if (
                primary
                and not pd.isna(primary)
                and "s1" not in from_phases
                and "s2" not in from_phases
            ):
                model.primary_phase_map[(int(row.fb), int(row.tb))] = str(primary)
    model.to_bus_map = {
        int(bus): [
            (int(row.fb), int(row.tb))
            for _, row in case.branch_data.loc[
                case.branch_data.fb == int(bus), ["fb", "tb"]
            ].iterrows()
        ]
        for bus in case.bus_data.id
    }
    bus_names = case.bus_data["name"].map(lambda value: str(value).strip())
    dupes = sorted(set(bus_names[bus_names.duplicated()]))
    if dupes:
        raise ValueError(f"bus_data has duplicate bus names: {dupes}")
    bus_ids = case.bus_data["id"].astype(int)
    model.bus_name_to_id_map = dict(zip(bus_names, bus_ids))
    model.bus_id_to_name_map = dict(zip(bus_ids, bus_names))
    model.name_map = model.bus_id_to_name_map
    model.phase_map = {
        int(row.id): parse_phases(str(row.phases))
        for _, row in case.bus_data[["id", "phases"]].iterrows()
    }


def create_branchflow_components(model: Any, case: Case) -> None:
    """Create current and angle state used only by nonlinear BranchFlow.

    ``d`` holds current angle differences (radians) as plain floats. The
    cosine and sine of each angle are ``cp.Parameter`` objects (``cos_d`` and
    ``sin_d``) so they can be updated after the model is built, for example by
    FBS initialization, without rebuilding constraints. Use
    ``set_current_angles`` to change them.
    """
    branch_phase_pairs: list[tuple[int, int, str]] = []
    branch_angle_phase_pairs: list[tuple[int, int, str]] = []
    for _, row in case.branch_data.iterrows():
        fb, tb = int(row.fb), int(row.tb)
        phases = parse_phases(str(row.phases))
        branch_phase_pairs.extend(
            (fb, tb, phase_1 + phase_2)
            for phase_1, phase_2 in combinations_with_replacement(sorted(phases), 2)
        )
        if tb not in model.swing_bus_set:
            branch_angle_phase_pairs.extend(
                (fb, tb, phase_1 + phase_2)
                for phase_1, phase_2 in product(phases, repeat=2)
            )

    model.branch_phase_pair_set = IndexSet(branch_phase_pairs, dimen=3)
    model.branch_angle_phase_pair_set = IndexSet(branch_angle_phase_pairs, dimen=3)
    model.l_flow = VarDict(index_product(model.branch_phase_pair_set, model.time_set))
    model.d = ParamDict({key: 0.0 for key in model.branch_angle_phase_pair_set})
    model.cos_d = {key: cp.Parameter(value=1.0) for key in model.d}
    model.sin_d = {key: cp.Parameter(value=0.0) for key in model.d}


def set_current_angles(model: Any, angles: dict[tuple[int, int, str], float]) -> None:
    """Update current angle differences (radians) and their cos/sin parameters."""
    for key, value in angles.items():
        model.d[key] = float(value)
        model.cos_d[key].value = float(np.cos(value))
        model.sin_d[key].value = float(np.sin(value))


class BFMProvider:
    """Own network sets, parameters, flow variables, and flow constraints."""

    name = "bfm"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        model.cap_mi_enabled = getattr(config, "control_capacitors", False)
        model.reg_mi_enabled = getattr(config, "control_regulators", False)
        create_network_components(model, case)
        model.v2 = VarDict(
            index_product(model.bus_phase_set, model.time_set),
            nonneg=True,
            initialize=1.0,
        )
        model.p_flow = VarDict(index_product(model.branch_phase_set, model.time_set))
        model.q_flow = VarDict(
            index_product(model.branch_phase_set, model.time_set), initialize=0.0
        )
        if not getattr(config, "linear", True):
            create_branchflow_components(model, case)

    def add_constraints(self, model: Any, config: Any) -> None:
        bfm_constraints.add_constraints(
            model,
            circular_constraints=getattr(config, "circular_constraints", True),
            thermal_constraints=getattr(config, "thermal_constraints", True),
            equality_only=getattr(config, "equality_only", False),
            free_swing_voltage=getattr(config, "free_swing_voltage", False),
            linear=getattr(config, "linear", True),
            socp_relaxation=getattr(config, "socp_relaxation", False),
            voltage_slacks=getattr(config, "voltage_slacks", False),
            thermal_slacks=getattr(config, "thermal_slacks", False),
        )


__all__ = ["BFMProvider", "create_network_components", "set_current_angles"]
