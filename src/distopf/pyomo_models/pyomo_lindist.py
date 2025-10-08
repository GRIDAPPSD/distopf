import pyomo.environ as pyo
from typing import Optional, Tuple, Dict, List
import pandas as pd
from dataclasses import dataclass
import distopf as opf
from numpy import sqrt, ones_like
from distopf.utils.utils import (
    handle_branch_input,
    handle_bus_input,
    handle_gen_input,
    handle_cap_input,
    handle_reg_input,
)


@dataclass
class Case:
    branch_data: Optional[pd.DataFrame] = None
    bus_data: Optional[pd.DataFrame] = None
    gen_data: Optional[pd.DataFrame] = None
    cap_data: Optional[pd.DataFrame] = None
    reg_data: Optional[pd.DataFrame] = None


def _create_phase_tuples(df: pd.DataFrame, id_col: str = "id") -> List[Tuple[str, str]]:
    """Create (id, phase) tuples from dataframe with phases column"""
    result = []
    for _, row in df.iterrows():
        result.extend([(row[id_col], phase) for phase in row.phases])
    return result


def _create_sets(
    m: pyo.ConcreteModel,
    bus: pd.DataFrame,
    branch: pd.DataFrame,
    gen: pd.DataFrame,
    cap: pd.DataFrame,
    reg: pd.DataFrame,
) -> None:
    """Create all Pyomo sets"""
    m.bus_set = pyo.Set(initialize=bus.id.tolist())
    m.phase_set = pyo.Set(initialize=["a", "b", "c"])
    m.swing_bus_set = pyo.Set(initialize=bus[bus.bus_type == "SWING"].id.tolist())
    m.branch_set = pyo.Set(initialize=bus[bus.bus_type != "SWING"].id.tolist())
    m.phase_pair_set = pyo.Set(initialize=["aa", "ab", "ac", "bb", "bc", "cc"])

    m.bus_phase_set = pyo.Set(initialize=_create_phase_tuples(bus), dimen=2)
    m.branch_phase_set = pyo.Set(initialize=_create_phase_tuples(branch, "tb"), dimen=2)
    m.gen_phase_set = pyo.Set(initialize=_create_phase_tuples(gen), dimen=2)
    m.cap_phase_set = pyo.Set(initialize=_create_phase_tuples(cap), dimen=2)
    m.reg_phase_set = pyo.Set(initialize=_create_phase_tuples(reg, "tb"), dimen=2)


def _create_parameters(m: pyo.ConcreteModel, branch: pd.DataFrame) -> None:
    """Create resistance and reactance parameters"""
    r_data, x_data = {}, {}

    for _, row in branch.iterrows():
        for phase_pair in m.phase_pair_set:
            r_col, x_col = f"r{phase_pair}", f"x{phase_pair}"

            if r_col in branch.columns and x_col in branch.columns:
                r_data[(phase_pair, row.tb)] = row[r_col]
                x_data[(phase_pair, row.tb)] = row[x_col]

    m.r = pyo.Param(m.phase_pair_set, m.branch_set, initialize=r_data, default=0.0)
    m.x = pyo.Param(m.phase_pair_set, m.branch_set, initialize=x_data, default=0.0)


def _create_variables(m: pyo.ConcreteModel) -> None:
    """Create all variables without bounds"""
    m.v = pyo.Var(m.bus_phase_set)  # Voltage magnitude squared
    m.p_flow = pyo.Var(m.branch_phase_set)
    m.q_flow = pyo.Var(m.branch_phase_set)
    m.p_gen = pyo.Var(m.gen_phase_set)
    m.q_gen = pyo.Var(m.gen_phase_set)
    m.q_cap = pyo.Var(m.cap_phase_set)
    m.v_reg = pyo.Var(m.reg_pahse_set)


def add_voltage_bounds(m: pyo.ConcreteModel, bus: pd.DataFrame) -> None:
    """Add voltage bounds (for voltage magnitude squared)"""
    for bus_id, phase in m.bus_phase_set:
        bus_row = bus[bus.id == bus_id].iloc[0]
        v_min = getattr(bus_row, "v_min", 0.95)
        v_max = getattr(bus_row, "v_max", 1.05)

        # Apply bounds to voltage squared
        m.v[bus_id, phase].setlb(v_min**2)
        m.v[bus_id, phase].setub(v_max**2)


def add_generator_bounds(m: pyo.ConcreteModel, gen: pd.DataFrame) -> None:
    """Add generator bounds following the original base.py logic"""
    if len(gen) == 0:
        return

    for phase in "abc":
        # Get generators that have this phase
        gen_phase_buses = [bus_id for bus_id, p in m.gen_phase_set if p == phase]
        if not gen_phase_buses:
            continue

        # Create lookup dictionaries for generator data by bus_id
        gen_lookup = {}
        for _, row in gen.iterrows():
            gen_lookup[row.id] = row

        # Get phase-specific arrays for generators with this phase
        s_rated_dict = {}
        p_out_dict = {}
        q_max_manual_dict = {}
        q_min_manual_dict = {}

        for bus_id in gen_phase_buses:
            if bus_id in gen_lookup:
                row = gen_lookup[bus_id]
                s_rated_dict[bus_id] = getattr(row, f"s{phase}_max", 0)
                p_out_dict[bus_id] = getattr(row, f"p{phase}", 0)
                q_max_manual_dict[bus_id] = getattr(row, f"q{phase}_max", 100e3)
                q_min_manual_dict[bus_id] = getattr(row, f"q{phase}_min", -100e3)

        for bus_id in gen_phase_buses:
            if bus_id not in gen_lookup:
                continue

            gen_row = gen_lookup[bus_id]
            mode = getattr(gen_row, "control_variable", "")

            s_rated = s_rated_dict[bus_id]
            p_out = p_out_dict[bus_id]
            q_max_manual = q_max_manual_dict[bus_id]
            q_min_manual = q_min_manual_dict[bus_id]

            # Active power bounds
            m.p_gen[bus_id, phase].setlb(0)
            m.p_gen[bus_id, phase].setub(p_out)

            # Reactive power bounds
            if mode == opf.CONSTANT_P:  # mode == "Q"
                # Use capability curve limits
                q_max_cap = sqrt(max(0, s_rated**2 - p_out**2))
                q_min_cap = -q_max_cap
                q_lower = max(q_min_cap, q_min_manual)
                q_upper = min(q_max_cap, q_max_manual)
            else:  # mode is "P", "PQ", or ""
                # Use apparent power limits
                q_lower = max(-s_rated, q_min_manual)
                q_upper = min(s_rated, q_max_manual)

            m.q_gen[bus_id, phase].setlb(q_lower)
            m.q_gen[bus_id, phase].setub(q_upper)


def create_lindist_model(case: Case) -> pyo.ConcreteModel:
    """
    Factory function to create a Pyomo ConcreteModel for linear distribution system optimization.

    Parameters
    ----------
    case : Case
        Dataclass containing network data frames

    Returns
    -------
    pyo.ConcreteModel
        Configured Pyomo model with sets, variables, and parameters
    """
    # Load and validate data frames
    branch = handle_branch_input(case.branch_data)
    bus = handle_bus_input(case.bus_data)
    gen = handle_gen_input(case.gen_data)
    cap = handle_cap_input(case.cap_data)
    reg = handle_reg_input(case.reg_data)

    m = pyo.ConcreteModel()
    _create_sets(m, bus, branch, gen, cap, reg)
    _create_parameters(m, branch)
    _create_variables(m)

    add_voltage_bounds(m, bus)
    add_generator_bounds(m, gen)

    return m
