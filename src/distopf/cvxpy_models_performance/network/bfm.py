"""Network provider (sets, impedance, operating parameters, flow variables).

Vectorized CVXPY version of ``distopf.cvxpy_models.network.bfm``. Parameters
are NumPy arrays aligned with the index sets (``IndexSet.pos``), and the flow
variables are arrays of shape ``(n_index, n_steps)``.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations_with_replacement, product
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from distopf.api import Case
from distopf.cvxpy_models_performance.common.model import (
    CvxModel,
    IndexSet,
)
from distopf.cvxpy_models_performance.network import bfm_constraints
from distopf.pyomo_models.common.data import parse_phases

PHASE_PAIR_LABELS = ("aa", "ab", "ac", "bb", "bc", "cc", "s1s1", "s1s2", "s2s2")


def create_network_sets(model: CvxModel, case: Case) -> None:
    """Create topology, time, and bus/branch phase sets."""

    model.delta_t = case.delta_t
    model.start_step = case.start_step
    model.n_steps = case.n_steps
    model.time_set = IndexSet(range(case.start_step, case.start_step + case.n_steps))
    bus_data = case.bus_data
    branch_data = case.branch_data
    model.bus_set = IndexSet(bus_data.id.tolist())
    swing_mask = bus_data.bus_type.isin(["SWING", "SWING_FREE", "IN"])
    boundary_in_mask = bus_data.bus_type.isin(["SWING_FREE", "IN"])
    boundary_out_mask = bus_data.bus_type.isin(["OUT"])
    model.swing_bus_set = IndexSet(bus_data.loc[swing_mask, "id"].tolist())
    model.swing_phase_set = IndexSet(
        [
            (row.id, phase)
            for row in bus_data.loc[swing_mask].itertuples()
            for phase in parse_phases(str(row.phases))
        ],
    )
    model.boundary_in_set = IndexSet(bus_data.loc[boundary_in_mask, "id"].tolist())
    model.boundary_out_set = IndexSet(bus_data.loc[boundary_out_mask, "id"].tolist())
    model.branch_set = IndexSet(
        [(int(fb), int(tb)) for fb, tb in zip(branch_data.fb, branch_data.tb)],
    )
    model.phase_pair_set = IndexSet(PHASE_PAIR_LABELS)
    model.bus_phase_set = IndexSet(
        [
            (row.id, phase)
            for row in bus_data.itertuples()
            for phase in parse_phases(str(row.phases))
        ],
    )
    model.branch_phase_set = IndexSet(
        [
            (int(row.fb), int(row.tb), phase)  # type: ignore[arg-type]
            for row in branch_data.itertuples()
            for phase in parse_phases(str(row.phases))
        ],
    )


def create_network_rx_parameters(model: CvxModel, case: Case) -> None:
    """Create impedance parameters shared by LinDistFlow and BranchFlow.

    ``model.r`` and ``model.x`` are ``defaultdict(float)`` keyed by ``(fb, tb, pair)``
    (missing pairs read as 0). ``model.r_diag`` and ``model.x_diag`` are arrays aligned with
    ``branch_phase_set`` holding the self impedance ``r_{aa}``, ``x_{aa}``.
    """
    resistance: dict[tuple, float] = defaultdict(float)
    reactance: dict[tuple, float] = defaultdict(float)
    branch_data = case.branch_data
    pairs = [
        pair
        for pair in PHASE_PAIR_LABELS
        if f"r_{pair}" in branch_data and f"x_{pair}" in branch_data
    ]
    for row in branch_data.to_dict("records"):
        branch = (int(row["fb"]), int(row["tb"]))
        for pair in pairs:
            resistance[(*branch, pair)] = row[f"r_{pair}"]
            reactance[(*branch, pair)] = row[f"x_{pair}"]
    model.r = resistance
    model.x = reactance
    model.r_diag = np.array(
        [model.r[fb, tb, ph + ph] for fb, tb, ph in model.branch_phase_set],
        dtype=float,
    )
    model.x_diag = np.array(
        [model.x[fb, tb, ph + ph] for fb, tb, ph in model.branch_phase_set],
        dtype=float,
    )


def _schedule_array(model: CvxModel, case: Case, column: str, default: float):
    schedules = case.schedules
    return np.array(
        [
            float(schedules.at[time, column])  # type: ignore[arg-type]
            if time in schedules.index
            else default
            for time in model.time_set
        ],
        dtype=float,
    )


def create_network_operating_parameters(model: CvxModel, case: Case) -> None:
    """Create voltage, thermal, schedule, and market parameters.

    Array shapes: ``v_min``, ``v_max`` are ``(n_bus_phase,)``; ``v_swing`` is
    ``(n_swing_phase, n_steps)``; ``s_branch_max`` is ``(n_branch_phase,)`` with
    NaN where no limit exists (defined only if some branch has one);
    ``price`` and ``schedule_<column>`` are ``(n_steps,)``.
    """
    bus_rows = {int(row["id"]): row for row in case.bus_data.to_dict("records")}
    n_steps = len(model.time_set)

    v_swing = np.ones((len(model.swing_phase_set), n_steps))
    for i, (bus, phase) in enumerate(model.swing_phase_set):
        nominal = bus_rows[int(bus)].get(f"v_{phase}", 1.0)
        for j, time in enumerate(model.time_set):
            value = nominal
            if f"v_{phase}" in case.schedules.columns and time in case.schedules.index:
                scheduled = case.schedules.at[time, f"v_{phase}"]
                if pd.notna(scheduled):
                    value = float(scheduled)  # type: ignore[arg-type]
            v_swing[i, j] = value
    model.v_swing = v_swing
    model.v_min = np.array(
        [bus_rows[int(bus)].get("v_min", 0.95) for bus, _ in model.bus_phase_set],
        dtype=float,
    )
    model.v_max = np.array(
        [bus_rows[int(bus)].get("v_max", 1.05) for bus, _ in model.bus_phase_set],
        dtype=float,
    )

    branch_data = case.branch_data
    columns = {
        "a": ("s_a_max",),
        "b": ("s_b_max",),
        "c": ("s_c_max",),
        "s1": ("s_s1_max", "s1_max"),
        "s2": ("s_s2_max", "s2_max"),
    }
    thermal = np.full(len(model.branch_phase_set), np.nan)
    any_thermal = False
    for row in branch_data.to_dict("records"):
        fb, tb = int(row["fb"]), int(row["tb"])
        for phase in parse_phases(str(row["phases"])):
            for column in columns.get(phase, ()):
                if column in branch_data and pd.notna(row.get(column)):
                    thermal[model.branch_phase_set.pos[(fb, tb, phase)]] = row[column]
                    any_thermal = True
                    break
    if any_thermal:
        model.s_branch_max = thermal

    if "price" in case.schedules:
        model.price = _schedule_array(model, case, "price", 0.0)
    else:
        model.price = np.zeros(n_steps)
    for column in case.schedules.columns:
        if column == "time":
            continue
        setattr(model, f"schedule_{column}", _schedule_array(model, case, column, 0.0))


def create_network_components(model: CvxModel, case: Case) -> None:
    """Create common network sets, parameters, and topology maps."""
    create_network_sets(model, case)
    create_network_rx_parameters(model, case)
    create_network_operating_parameters(model, case)

    model.primary_phase_map = {}
    if "primary_phase" in case.branch_data.columns:
        bus_phases = dict(zip(case.bus_data.id, case.bus_data.phases))
        for row in case.branch_data.itertuples():
            primary = getattr(row, "primary_phase", "")
            from_phases = str(bus_phases.get(int(row.fb), ""))  # type: ignore[arg-type]
            if (
                primary
                and not pd.isna(primary)
                and "s1" not in from_phases
                and "s2" not in from_phases
            ):
                model.primary_phase_map[(int(row.fb), int(row.tb))] = str(primary)  # type: ignore[arg-type]
    children: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for fb, tb in zip(case.branch_data.fb, case.branch_data.tb):
        children[int(fb)].append((int(fb), int(tb)))
    model.to_bus_map = children
    bus_names = case.bus_data["name"].map(lambda value: str(value).strip())
    dupes = sorted(set(bus_names[bus_names.duplicated()]))
    if dupes:
        raise ValueError(f"bus_data has duplicate bus names: {dupes}")
    bus_ids = case.bus_data["id"].astype(int)
    model.bus_name_to_id_map = dict(zip(bus_names, bus_ids))
    model.bus_id_to_name_map = dict(zip(bus_ids, bus_names))
    model.name_map = model.bus_id_to_name_map
    model.phase_map = {
        int(bus): parse_phases(str(phases))
        for bus, phases in zip(case.bus_data["id"], case.bus_data["phases"])
    }


def create_branchflow_components(model: Any, case: Case) -> None:
    """Create current and angle state used only by nonlinear BranchFlow.

    ``d`` is an array of current angle differences (radians) aligned with
    ``branch_angle_phase_pair_set``. ``cos_d`` and ``sin_d`` are ``cp.Parameter``
    vectors on the same index, so they can be updated after the model is built
    (for example by FBS initialization) without rebuilding. Use
    ``set_current_angles`` to change them.
    """
    branch_phase_pairs: list[tuple[int, int, str]] = []
    branch_angle_phase_pairs: list[tuple[int, int, str]] = []
    for row in case.branch_data.itertuples():
        fb, tb = int(row.fb), int(row.tb)  # type: ignore[arg-type]
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

    model.branch_phase_pair_set = IndexSet(branch_phase_pairs)
    model.branch_angle_phase_pair_set = IndexSet(branch_angle_phase_pairs)
    model.add_var("l_flow", model.branch_phase_pair_set)
    n_angles = len(model.branch_angle_phase_pair_set)
    model.d = np.zeros(n_angles)
    model.cos_d = cp.Parameter(n_angles, value=np.ones(n_angles))
    model.sin_d = cp.Parameter(n_angles, value=np.zeros(n_angles))


def set_current_angles(model: Any, angles: dict[tuple[int, int, str], float]) -> None:
    """Update current angle differences (radians) and their cos/sin parameters."""
    pos = model.branch_angle_phase_pair_set.pos
    for key, value in angles.items():
        model.d[pos[key]] = float(value)
    model.cos_d.value = np.cos(model.d)
    model.sin_d.value = np.sin(model.d)


class BFMProvider:
    """Own network sets, parameters, flow variables, and flow constraints."""

    name = "bfm"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        create_network_components(model, case)
        model.add_var("v2", model.bus_phase_set, nonneg=True, value=1.0)
        model.add_var("p_flow", model.branch_phase_set)
        model.add_var("q_flow", model.branch_phase_set, value=0.0)
        if not config.linear:
            create_branchflow_components(model, case)

    def add_constraints(self, model: Any, config: Any) -> None:
        bfm_constraints.add_constraints(
            model,
            circular_constraints=config.circular_constraints,
            thermal_constraints=config.thermal_constraints,
            equality_only=config.equality_only,
            free_swing_voltage=config.free_swing_voltage,
            linear=config.linear,
            socp_relaxation=config.socp_relaxation,
            voltage_slacks=config.voltage_slacks,
            thermal_slacks=config.thermal_slacks,
            sdp_relaxation=config.sdp_relaxation,
            sdp_diagonal_relaxation=config.sdp_diagonal_relaxation,
        )


__all__ = ["BFMProvider", "create_network_components", "set_current_angles"]
