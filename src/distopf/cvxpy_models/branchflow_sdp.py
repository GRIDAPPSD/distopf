"""
Semi-Definite Programming (SDP) relaxation of the three-phase Branch Flow Model.

Variables
---------
W_re[bus, t] : cp.Variable (n x n, symmetric)
    Real part of the Hermitian voltage outer-product matrix W = V·V†.
    Diagonal W_re[i,i] = |V_i|² (voltage magnitude squared).
    Off-diagonal W_re[i,j] = Re(V_i · V_j*).

W_im[bus, t] : cp.Variable (n x n, skew-symmetric enforced by constraint)
    Imaginary part of W. W_im[i,j] = Im(V_i · V_j*).
    Diagonal is zero, W_im = -W_im.T.

L_re[fb, tb, t] : cp.Variable (n x n, symmetric)
    Real part of the Hermitian current outer-product matrix L = I·I†.
    Diagonal L_re[i,i] = |I_i|² (current magnitude squared).

L_im[fb, tb, t] : cp.Variable (n x n, skew-symmetric enforced by constraint)
    Imaginary part of L.

SF_re[fb, tb, t] : cp.Variable (n x n)
    Real part of the branch complex power flow matrix S = V_f · I†.
    Diagonal SF_re[i,i] = P_flow on phase i (active power flow).

SF_im[fb, tb, t] : cp.Variable (n x n)
    Imaginary part of S.
    Diagonal SF_im[i,i] = Q_flow on phase i (reactive power flow).

p_gen[bus, t] : cp.Variable (n,)
    Active power generation per phase at bus (p.u.).

q_gen[bus, t] : cp.Variable (n,)
    Reactive power generation per phase at bus (p.u.).

q_cap[bus, t] : cp.Variable (n,)
    Capacitor reactive power injection per phase at bus (p.u.).

p_bat[bus, t] : cp.Variable (n,)
    Battery net active power injection per phase (positive = discharge).

q_bat[bus, t] : cp.Variable (n,)
    Battery net reactive power injection per phase.

p_load[bus, t] : np.ndarray (n,)  [parameter]
    Active power load per phase (p.u.).

q_load[bus, t] : np.ndarray (n,)  [parameter]
    Reactive power load per phase (p.u.).

Constraint groups (named sub-lists on SdpModel)
------------------------------------------------
swing_bus_constraints
voltage_drop_constraints
psd_block_constraints
power_balance_constraints
voltage_limit_constraints
generator_limit_constraints
generator_constant_p_constraints
generator_constant_q_constraints
capacitor_constraints
battery_constraints
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import cvxpy as cp
import numpy as np
import pandas as pd

from distopf.api import Case

# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------

PHASE_TO_IDX = {"a": 0, "b": 1, "c": 2}
IDX_TO_PHASE = {0: "a", 1: "b", 2: "c"}
SUPPORTED_PHASES = ("a", "b", "c")


def _parse_phases_abc(phases_str: str) -> list[str]:
    """Parse a phases string into a list of primary phase names (abc only).

    Triplex phases (s1, s2) are silently ignored.

    Examples
    --------
    >>> _parse_phases_abc("abc")
    ['a', 'b', 'c']
    >>> _parse_phases_abc("ac")
    ['a', 'c']
    """
    return [ch for ch in str(phases_str) if ch in PHASE_TO_IDX]


def _phase_list_to_idx(phase_list: list[str]) -> list[int]:
    """Convert list of phase strings to list of 0-based integer indices."""
    return [PHASE_TO_IDX[ph] for ph in phase_list]


# ---------------------------------------------------------------------------
# SDP Model dataclass
# ---------------------------------------------------------------------------


@dataclass
class SdpModel:
    """
    Container for the SDP BranchFlow model.

    Holds all CVXPY variables, parameters (as numpy arrays / dicts),
    topology maps, and constraint lists.  Mirrors the role of
    ``pyo.ConcreteModel`` in the Pyomo-based models.

    Attributes
    ----------
    See module docstring for variable descriptions.
    """

    # ------------------------------------------------------------------ #
    # CVXPY decision variables                                             #
    # ------------------------------------------------------------------ #
    # Voltage outer-product matrices  W = V·V†
    W_re: dict = field(default_factory=dict)  # (bus, t) -> cp.Variable (n×n sym)
    W_im: dict = field(default_factory=dict)  # (bus, t) -> cp.Variable (n×n)
    # Regulator intermediate voltage matrices  W_reg = α·W_f·α
    # (voltage after tap transform, before impedance drop)
    W_reg_re: dict = field(default_factory=dict)  # (fb, tb, t) -> cp.Variable (n×n sym)
    W_reg_im: dict = field(default_factory=dict)  # (fb, tb, t) -> cp.Variable (n×n)
    # Current outer-product matrices  L = I·I†
    L_re: dict = field(default_factory=dict)  # (fb, tb, t) -> cp.Variable (n×n sym)
    L_im: dict = field(default_factory=dict)  # (fb, tb, t) -> cp.Variable (n×n)

    # Branch power-flow matrices  SF = V_f · I†
    SF_re: dict = field(default_factory=dict)  # (fb, tb, t) -> cp.Variable (n×n)
    SF_im: dict = field(default_factory=dict)  # (fb, tb, t) -> cp.Variable (n×n)

    # Per-bus injection vectors (length = number of active phases at bus)
    p_load: dict = field(default_factory=dict)  # (bus, t) -> cp.Variable (n,)
    q_load: dict = field(default_factory=dict)  # (bus, t) -> cp.Variable (n,)
    p_gen: dict = field(default_factory=dict)  # (bus, t) -> cp.Variable (n,)
    q_gen: dict = field(default_factory=dict)  # (bus, t) -> cp.Variable (n,)
    q_cap: dict = field(default_factory=dict)  # (bus, t) -> cp.Variable (n,)
    p_bat: dict = field(default_factory=dict)  # (bus, t) -> cp.Variable (n,)
    q_bat: dict = field(default_factory=dict)  # (bus, t) -> cp.Variable (n,)

    # Battery state variables (per battery id)
    p_charge: dict = field(default_factory=dict)  # (bat_id, t) -> cp.Variable scalar
    p_discharge: dict = field(default_factory=dict)  # (bat_id, t) -> cp.Variable scalar
    soc: dict = field(default_factory=dict)  # (bat_id, t) -> cp.Variable scalar

    # ------------------------------------------------------------------ #
    # Parameters (numpy arrays / plain Python dicts — read-only)          #
    # ------------------------------------------------------------------ #
    # Impedance sub-matrices per branch (phase-reduced)
    Z_re: dict = field(default_factory=dict)  # (fb, tb) -> np.ndarray (n×n)
    Z_im: dict = field(default_factory=dict)  # (fb, tb) -> np.ndarray (n×n)

    # Scalar impedance look-up (mirrors DistOPF m.r / m.x)
    r: dict = field(default_factory=dict)  # (fb, tb, phase_pair) -> float
    x: dict = field(default_factory=dict)  # (fb, tb, phase_pair) -> float

    # Load parameters
    p_load_nom: dict = field(default_factory=dict)  # (bus, ph, t) -> float
    q_load_nom: dict = field(default_factory=dict)  # (bus, ph, t) -> float

    # CVR (Conservation Voltage Reduction) factors
    cvr_p: dict = field(default_factory=dict)  # (bus, ph) -> float
    cvr_q: dict = field(default_factory=dict)  # (bus, ph) -> float

    # Generator parameters
    p_gen_nom: dict = field(default_factory=dict)  # (bus, ph, t) -> float
    q_gen_nom: dict = field(default_factory=dict)  # (bus, ph, t) -> float
    s_rated: dict = field(default_factory=dict)  # (bus, ph) -> float
    q_gen_max: dict = field(default_factory=dict)  # (bus, ph) -> float
    q_gen_min: dict = field(default_factory=dict)  # (bus, ph) -> float
    gen_control_type: dict = field(default_factory=dict)  # (bus, ph) -> int

    # Regulator parameters
    reg_ratio: dict = field(default_factory=dict)  # (fb, tb, ph) -> float

    # Capacitor parameters
    q_cap_nom: dict = field(default_factory=dict)  # (bus, ph) -> float

    # Voltage limits
    v_min: dict = field(default_factory=dict)  # (bus, ph) -> float
    v_max: dict = field(default_factory=dict)  # (bus, ph) -> float
    v_swing: dict = field(default_factory=dict)  # (bus, ph, t) -> float

    # Battery parameters
    energy_capacity: dict = field(default_factory=dict)  # bat_id -> float
    soc_min: dict = field(default_factory=dict)  # bat_id -> float
    soc_max: dict = field(default_factory=dict)  # bat_id -> float
    start_soc: dict = field(default_factory=dict)  # bat_id -> float
    charge_efficiency: dict = field(default_factory=dict)  # bat_id -> float
    discharge_efficiency: dict = field(default_factory=dict)  # bat_id -> float
    s_bat_rated: dict = field(default_factory=dict)  # (bat_id, ph) -> float
    q_bat_max: dict = field(default_factory=dict)  # (bat_id, ph) -> float
    q_bat_min: dict = field(default_factory=dict)  # (bat_id, ph) -> float
    bat_control_type: dict = field(default_factory=dict)  # bat_id -> int
    battery_n_phases: dict = field(default_factory=dict)  # bat_id -> int

    # Electricity price
    price: dict = field(default_factory=dict)  # t -> float

    # ------------------------------------------------------------------ #
    # Topology / index sets (mirrors DistOPF m.bus_set, m.phase_map …)   #
    # ------------------------------------------------------------------ #
    bus_set: list = field(default_factory=list)  # [bus_id, ...]
    branch_set: list = field(default_factory=list)  # [(fb, tb), ...]
    gen_set: list = field(default_factory=list)  # [bus_id, ...]  buses with gens
    cap_set: list = field(default_factory=list)  # [bus_id, ...]
    bat_set: list = field(default_factory=list)  # [bat_id, ...]
    reg_set: list = field(default_factory=list)  # [(fb, tb), ...]  regulator branches
    reg_phase_map: dict = field(default_factory=dict)  # (fb, tb) -> [0, 1, 2]
    swing_bus_set: list = field(default_factory=list)  # [bus_id, ...]  (usually 1 bus)
    time_set: range = field(default_factory=lambda: range(1))
    # Phase maps
    phase_map: dict = field(default_factory=dict)
    # bus -> ['a','b','c'] (only abc phases, no triplex)

    phase_idx_map: dict = field(default_factory=dict)
    # bus -> {'a':0, 'b':1, 'c':2}  local index within that bus's phase list

    branch_phase_map: dict = field(default_factory=dict)
    # (fb, tb) -> ['a','b','c']  phases active on this branch

    gen_phase_map: dict = field(default_factory=dict)
    # bus -> ['a','b','c']  phases where generator exists

    cap_phase_map: dict = field(default_factory=dict)
    # bus -> ['a','b','c']

    bat_phase_map: dict = field(default_factory=dict)
    # bat_id -> ['a','b','c']

    # Topology helpers (mirrors DistOPF m.to_bus_map / m.from_bus_map)
    to_bus_map: dict = field(default_factory=dict)
    # bus -> [(fb, tb), ...]  outgoing branches from this bus

    from_bus_map: dict = field(default_factory=dict)
    # tb -> fb   (radial: each bus has at most one parent)

    # Scalar config
    delta_t: float = 1.0
    start_step: int = 0
    n_steps: int = 1

    # Reference to original Case
    case: Any = None  # distopf.api.Case

    # ------------------------------------------------------------------ #
    # Constraint lists                                                     #
    # ------------------------------------------------------------------ #
    constraints: list = field(default_factory=list)

    # Named sub-groups for inspection / debugging
    swing_bus_constraints: list = field(default_factory=list)
    voltage_drop_constraints: list = field(default_factory=list)
    psd_block_constraints: list = field(default_factory=list)
    power_balance_constraints: list = field(default_factory=list)
    voltage_limit_constraints: list = field(default_factory=list)
    generator_limit_constraints: list = field(default_factory=list)
    generator_constant_p_constraints: list = field(default_factory=list)
    generator_constant_q_constraints: list = field(default_factory=list)
    capacitor_constraints: list = field(default_factory=list)
    battery_constraints: list = field(default_factory=list)
    thermal_limit_constraints: list = field(default_factory=list)
    regulator_constraints: list = field(default_factory=list)
    load_constraints: list = field(default_factory=list)

    def _add(self, group: list, c: cp.Constraint | list) -> None:
        """Append constraint(s) to a named group AND the master list."""
        if isinstance(c, list):
            group.extend(c)
            self.constraints.extend(c)
        else:
            group.append(c)
            self.constraints.append(c)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_Z_sub(m: SdpModel, fb: int, tb: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the (n×n) real and imaginary impedance sub-matrices for a branch.

    The sub-matrix is indexed by the branch's active phases only.
    """
    return m.Z_re[fb, tb], m.Z_im[fb, tb]


def _get_W_parent_sub_idx(
    m: SdpModel, parent_bus: int, child_phases: list[str]
) -> list[int]:
    """Return the local indices in parent_bus's phase list that correspond to child_phases.

    Used to extract the matching sub-block of W[parent_bus] that aligns with
    the child branch's phase set.

    Example
    -------
    parent_bus has phases ['a','b','c'] (indices 0,1,2)
    child_phases = ['a','c']
    → returns [0, 2]
    """
    parent_phases = m.phase_map[parent_bus]
    return [parent_phases.index(ph) for ph in child_phases]


# ---------------------------------------------------------------------------
# Parameter creation helpers  (mirror _create_*_parameters in nl_branchflow)
# ---------------------------------------------------------------------------


def _create_impedance_parameters(m: SdpModel, case: Case) -> None:
    """Build full 3×3 R/X dicts AND phase-reduced Z sub-matrices per branch."""
    phase_pair_set = ["aa", "ab", "ac", "bb", "bc", "cc"]

    for _, row in case.branch_data.iterrows():
        fb, tb = int(row.fb), int(row.tb)

        # Scalar r/x (mirrors DistOPF m.r / m.x)
        for pp in phase_pair_set:
            r_col, x_col = f"r_{pp}", f"x_{pp}"
            if r_col in case.branch_data.columns:
                m.r[fb, tb, pp] = float(row[r_col]) if not pd.isna(row[r_col]) else 0.0
                m.x[fb, tb, pp] = float(row[x_col]) if not pd.isna(row[x_col]) else 0.0
            else:
                m.r[fb, tb, pp] = 0.0
                m.x[fb, tb, pp] = 0.0

        # Phase-reduced Z sub-matrix for the SDP voltage-drop constraint
        br_phases = m.branch_phase_map[fb, tb]
        n = len(br_phases)
        Zr = np.zeros((n, n))
        Zi = np.zeros((n, n))
        for ii, phi in enumerate(br_phases):
            for jj, phj in enumerate(br_phases):
                pp = "".join(sorted([phi, phj]))
                Zr[ii, jj] = m.r.get((fb, tb, pp), 0.0)
                Zi[ii, jj] = m.x.get((fb, tb, pp), 0.0)
        m.Z_re[fb, tb] = Zr
        m.Z_im[fb, tb] = Zi


def _create_load_parameters(m: SdpModel, case: Case) -> None:
    """Populate p_load_nom, q_load_nom, and CVR factors (mirrors _create_load_parameters)."""
    for _, row in case.bus_data.iterrows():
        phases = _parse_phases_abc(str(row.phases))
        for ph in phases:
            # CVR factors for voltage-dependent loads
            cvr_p = getattr(row, "cvr_p", 0.0) or 0.0
            cvr_q = getattr(row, "cvr_q", 0.0) or 0.0
            m.cvr_p[row.id, ph] = float(cvr_p)
            m.cvr_q[row.id, ph] = float(cvr_q)

            p_load = getattr(row, f"pl_{ph}", 0.0) or 0.0
            q_load = getattr(row, f"ql_{ph}", 0.0) or 0.0
            load_shape = getattr(row, "load_shape", "default")
            for t in m.time_set:
                mult = 1.0
                if not case.schedules.empty and load_shape in case.schedules.columns:
                    mult = float(case.schedules.at[t, load_shape])
                m.p_load_nom[row.id, ph, t] = float(p_load) * mult
                m.q_load_nom[row.id, ph, t] = float(q_load) * mult


def _create_price_parameters(m: SdpModel, case: Case) -> None:
    """Create electricity price parameters from schedules (mirrors _create_price_parameters)."""
    if not case.schedules.empty and "price" in case.schedules.columns:
        for t in m.time_set:
            m.price[t] = float(case.schedules.at[t, "price"])
    else:
        for t in m.time_set:
            m.price[t] = 0.0


def _get_gen_schedule_mult(gen_shape: str, t: int, schedules: pd.DataFrame) -> float:
    """Schedule multiplier for a generator at time t (mirrors nl_branchflow)."""
    if not gen_shape or pd.isna(gen_shape):
        return 1.0
    if gen_shape in schedules.columns:
        try:
            return float(schedules.at[t, gen_shape])
        except (TypeError, ValueError):
            return 1.0
    return 1.0


def _create_generator_parameters(m: SdpModel, case: Case) -> None:
    """Populate generator parameters (mirrors _create_generator_parameters)."""
    from distopf.pyomo_models.nl_branchflow import CONTROL_VARIABLE_MAP

    for _, row in case.gen_data.iterrows():
        phases = _parse_phases_abc(str(row.phases))
        for ph in phases:
            s_rated = getattr(row, f"s_{ph}_max", 1000.0) or 1000.0
            q_max = getattr(row, f"q_{ph}_max", s_rated) or s_rated
            q_min = getattr(row, f"q_{ph}_min", -s_rated)
            if q_min is None or pd.isna(q_min):
                q_min = -s_rated
            p_gen_val = getattr(row, f"p_{ph}", 0.0) or 0.0
            q_gen_val = getattr(row, f"q_{ph}", 0.0) or 0.0

            m.s_rated[row.id, ph] = float(s_rated)
            m.q_gen_max[row.id, ph] = float(q_max)
            m.q_gen_min[row.id, ph] = float(q_min)
            m.gen_control_type[row.id, ph] = CONTROL_VARIABLE_MAP[
                getattr(row, "control_variable", "") or ""
            ]

            gen_shape = getattr(row, "gen_shape", "PV") or "PV"
            for t in m.time_set:
                mult = _get_gen_schedule_mult(gen_shape, t, case.schedules)
                m.p_gen_nom[row.id, ph, t] = float(p_gen_val) * mult
                m.q_gen_nom[row.id, ph, t] = float(q_gen_val)


def _create_capacitor_parameters(m: SdpModel, case: Case) -> None:
    """Populate capacitor parameters (mirrors _create_capacitor_parameters)."""
    for _, row in case.cap_data.iterrows():
        for ph in _parse_phases_abc(str(row.phases)):
            m.q_cap_nom[row.id, ph] = float(getattr(row, f"q_{ph}", 0.0) or 0.0)


def _create_voltage_parameters(m: SdpModel, case: Case) -> None:
    """Populate voltage limit and swing bus parameters."""
    swing_buses = case.bus_data[case.bus_data.bus_type == "SWING"]
    for row in case.bus_data.itertuples(index=False):
        for ph in _parse_phases_abc(str(row.phases)):
            m.v_min[row.id, ph] = float(getattr(row, "v_min", 0.95) or 0.95)
            m.v_max[row.id, ph] = float(getattr(row, "v_max", 1.05) or 1.05)

    for _, row in swing_buses.iterrows():
        for ph in _parse_phases_abc(str(row.phases)):
            v_swing = float(getattr(row, f"v_{ph}", 1.0) or 1.0)
            for t in m.time_set:
                m.v_swing[row.id, ph, t] = v_swing


def _create_battery_parameters(m: SdpModel, case: Case) -> None:
    """Populate battery parameters (mirrors _create_battery_parameters)."""
    from distopf.pyomo_models.nl_branchflow import CONTROL_VARIABLE_MAP

    for _, row in case.bat_data.iterrows():
        m.energy_capacity[row.id] = float(getattr(row, "energy_capacity", 0.0))
        m.soc_min[row.id] = float(getattr(row, "min_soc", 0.0))
        m.soc_max[row.id] = float(getattr(row, "max_soc", 1.0))
        m.start_soc[row.id] = float(getattr(row, "start_soc", 0.5))
        m.charge_efficiency[row.id] = float(getattr(row, "charge_efficiency", 1.0))
        m.discharge_efficiency[row.id] = float(
            getattr(row, "discharge_efficiency", 1.0)
        )
        m.bat_control_type[row.id] = CONTROL_VARIABLE_MAP[
            getattr(row, "control_variable", "P") or "P"
        ]
        phases = _parse_phases_abc(str(row.phases))
        n_phases = len(phases)
        m.battery_n_phases[row.id] = n_phases
        s_max = float(getattr(row, "s_max", 1000.0) or 1000.0)
        q_bat_max_val = float(getattr(row, "q_max", s_max) or s_max)
        q_bat_min_val = float(getattr(row, "q_min", -s_max) or -s_max)
        for ph in phases:
            m.s_bat_rated[row.id, ph] = s_max / n_phases
            m.q_bat_max[row.id, ph] = q_bat_max_val / n_phases
            m.q_bat_min[row.id, ph] = q_bat_min_val / n_phases


def _create_regulator_parameters(m: SdpModel, case: Case) -> None:
    """
    Populate reg_ratio dict keyed by (fb, tb, ph) from case.reg_data.

    Mirrors _create_regulator_parameters in lindist.py which reads the
    ratio_{ph} columns from case.reg_data.
    """
    if case.reg_data.empty:
        return

    for _, row in case.reg_data.iterrows():
        fb, tb = int(row.fb), int(row.tb)
        for ph in _parse_phases_abc(str(row.phases)):
            m.reg_ratio[fb, tb, ph] = float(getattr(row, f"ratio_{ph}", 1.0) or 1.0)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def create_sdp_branchflow_model(case: Case) -> SdpModel:
    """
    Create and return an :class:`SdpModel` populated with all CVXPY variables
    and parameters derived from *case*.

    This is the SDP analogue of ``create_nl_branchflow_model``.

    Parameters
    ----------
    case : Case
        DistOPF case object containing network data.

    Returns
    -------
    SdpModel
        Model with all variables and parameters initialised.
        Constraints are **not** added here — call :func:`add_sdp_constraints`
        afterwards.
    """
    m = SdpModel()
    m.case = case
    m.delta_t = case.delta_t
    m.start_step = case.start_step
    m.n_steps = case.n_steps
    m.time_set = range(case.start_step, case.start_step + case.n_steps)

    # ------------------------------------------------------------------ #
    # Build index sets / topology maps                                     #
    # ------------------------------------------------------------------ #
    m.bus_set = case.bus_data.id.tolist()
    m.swing_bus_set = case.bus_data[case.bus_data.bus_type == "SWING"].id.tolist()
    m.branch_set = [
        (int(fb), int(tb))
        for fb, tb in case.branch_data.loc[:, ["fb", "tb"]].to_numpy()
    ]

    # Phase maps — abc only, no triplex
    m.phase_map = {
        int(_id): _parse_phases_abc(str(phases))
        for _id, phases in case.bus_data.loc[:, ["id", "phases"]].to_numpy()
    }
    m.phase_idx_map = {
        bus: {ph: i for i, ph in enumerate(ph_list)}
        for bus, ph_list in m.phase_map.items()
    }

    m.branch_phase_map = {}
    for _, row in case.branch_data.iterrows():
        fb, tb = int(row.fb), int(row.tb)
        m.branch_phase_map[fb, tb] = _parse_phases_abc(str(row.phases))

    # Generator / cap / bat sets and phase maps
    m.gen_set = case.gen_data.id.tolist() if not case.gen_data.empty else []
    m.gen_phase_map = {
        int(row.id): _parse_phases_abc(str(row.phases))
        for _, row in case.gen_data.iterrows()
    }
    m.cap_set = case.cap_data.id.tolist() if not case.cap_data.empty else []
    m.cap_phase_map = {
        int(row.id): _parse_phases_abc(str(row.phases))
        for _, row in case.cap_data.iterrows()
    }
    m.bat_set = case.bat_data.id.tolist() if not case.bat_data.empty else []
    m.bat_phase_map = {
        int(row.id): _parse_phases_abc(str(row.phases))
        for _, row in case.bat_data.iterrows()
    }

    # Topology helpers
    m.from_bus_map = {
        int(tb): int(fb) for fb, tb in case.branch_data.loc[:, ["fb", "tb"]].to_numpy()
    }
    m.to_bus_map = {
        int(bus_id): [
            (int(fb), int(tb))
            for fb, tb in case.branch_data.loc[
                case.branch_data.fb == int(bus_id), ["fb", "tb"]
            ].to_numpy()
        ]
        for bus_id in case.bus_data.id.to_numpy()
    }
    # Regulator sets and phase maps
    m.reg_set = []
    m.reg_phase_map = {}
    if not case.reg_data.empty:
        for _, row in case.reg_data.iterrows():
            fb, tb = int(row.fb), int(row.tb)
            ph_list = _parse_phases_abc(str(row.phases))
            m.reg_set.append((fb, tb))
            m.reg_phase_map[fb, tb] = ph_list
    _create_regulator_parameters(m, case)
    # ------------------------------------------------------------------ #
    # Parameters                                                           #
    # ------------------------------------------------------------------ #
    _create_impedance_parameters(m, case)
    _create_load_parameters(m, case)
    _create_generator_parameters(m, case)
    _create_capacitor_parameters(m, case)
    _create_voltage_parameters(m, case)
    _create_battery_parameters(m, case)
    _create_price_parameters(m, case)
    # ------------------------------------------------------------------ #
    # CVXPY decision variables                                             #
    # ------------------------------------------------------------------ #
    for bus in m.bus_set:
        n = len(m.phase_map[bus])
        if n == 0:
            continue
        for t in m.time_set:
            m.W_re[bus, t] = cp.Variable((n, n), symmetric=True, name=f"W_re_{bus}_{t}")
            m.W_im[bus, t] = cp.Variable((n, n), name=f"W_im_{bus}_{t}")
            m.p_load[bus, t] = cp.Variable(n, name=f"p_load_{bus}_{t}")
            m.q_load[bus, t] = cp.Variable(n, name=f"q_load_{bus}_{t}")

    for fb, tb in m.branch_set:
        n = len(m.branch_phase_map[fb, tb])
        if n == 0:
            continue
        for t in m.time_set:
            m.L_re[fb, tb, t] = cp.Variable(
                (n, n), symmetric=True, name=f"L_re_{fb}_{tb}_{t}"
            )
            m.L_im[fb, tb, t] = cp.Variable((n, n), name=f"L_im_{fb}_{tb}_{t}")
            m.SF_re[fb, tb, t] = cp.Variable((n, n), name=f"SF_re_{fb}_{tb}_{t}")
            m.SF_im[fb, tb, t] = cp.Variable((n, n), name=f"SF_im_{fb}_{tb}_{t}")
            # Regulator intermediate voltage matrices (only for regulator branches)
            if (fb, tb) in m.reg_phase_map:
                m.W_reg_re[fb, tb, t] = cp.Variable(
                    (n, n), symmetric=True, name=f"W_reg_re_{fb}_{tb}_{t}"
                )
                m.W_reg_im[fb, tb, t] = cp.Variable(
                    (n, n), name=f"W_reg_im_{fb}_{tb}_{t}"
                )

    for bus in m.bus_set:
        n = len(m.phase_map[bus])
        if n == 0:
            continue
        for t in m.time_set:
            # Generators
            if bus in m.gen_set:
                m.p_gen[bus, t] = cp.Variable(n, nonneg=True, name=f"p_gen_{bus}_{t}")
                m.q_gen[bus, t] = cp.Variable(n, name=f"q_gen_{bus}_{t}")
            # Capacitors
            if bus in m.cap_set:
                m.q_cap[bus, t] = cp.Variable(n, name=f"q_cap_{bus}_{t}")
            # Batteries (indexed by bat_id which equals bus id in DistOPF)
            if bus in m.bat_set:
                m.p_bat[bus, t] = cp.Variable(n, name=f"p_bat_{bus}_{t}")
                m.q_bat[bus, t] = cp.Variable(n, name=f"q_bat_{bus}_{t}")

    for bat_id in m.bat_set:
        for t in m.time_set:
            m.p_charge[bat_id, t] = cp.Variable(
                nonneg=True, name=f"p_charge_{bat_id}_{t}"
            )
            m.p_discharge[bat_id, t] = cp.Variable(
                nonneg=True, name=f"p_discharge_{bat_id}_{t}"
            )
            m.soc[bat_id, t] = cp.Variable(name=f"soc_{bat_id}_{t}")

    return m
