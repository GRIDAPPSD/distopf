import pyomo.environ as pyo
from pathlib import Path
import pandas as pd
from scipy.stats.tests.test_continuous_basic import case1
from distopf.utils.utils import (
    handle_branch_input,
    handle_bus_input,
    handle_gen_input,
    handle_cap_input,
    handle_reg_input,
)
from distopf.dss_importer import DSSToCSVConverter
from .pyomo_lindist import Case, create_lindist_model
from .lindist_constraints import (
    add_cvr_load_model,
    add_generator_capability_constraints,
    add_generator_control_constraints,
    add_power_balance_constraints,
    add_swing_bus_constraints,
    add_thermal_limits,
    add_voltage_drop_constraints,
    add_voltage_limits,
)


def create_case_from_csv(data_path: Path) -> Case:
    branch_data = None
    bus_data = None
    gen_data = None
    cap_data = None
    reg_data = None
    if not data_path.exists():
        raise FileNotFoundError()
    if data_path.is_file():
        raise ValueError(
            "The variable, data_path, must point to a directory containing model CSVs."
        )
    if data_path.is_dir():
        branch_path = data_path / "branch_data.csv"
        if branch_path.exists():
            branch_data = pd.read_csv(branch_path, header=0)
        bus_path = data_path / "bus_data.csv"
        if bus_path.exists():
            bus_data = pd.read_csv(bus_path, header=0)
        gen_path = data_path / "gen_data.csv"
        if gen_path.exists():
            gen_data = pd.read_csv(gen_path, header=0)
        cap_path = data_path / "cap_data.csv"
        if cap_path.exists():
            cap_data = pd.read_csv(cap_path, header=0)
        reg_path = data_path / "reg_data.csv"
        if reg_path.exists():
            reg_data = pd.read_csv(data_path / "reg_data.csv", header=0)

    branch_data = handle_branch_input(branch_data)
    bus_data = handle_bus_input(bus_data)
    gen_data = handle_gen_input(gen_data)
    cap_data = handle_cap_input(cap_data)
    reg_data = handle_reg_input(reg_data)
    return Case(
        branch_data,
        bus_data,
        gen_data,
        cap_data,
        reg_data,
    )

def create_case_from_dss(data_path: Path) -> Case:
    branch_data = None
    bus_data = None
    gen_data = None
    cap_data = None
    reg_data = None
    if not data_path.exists():
        raise FileNotFoundError()
    if (data_path.is_file() and data_path.suffix.lower() != ".dss") or data_path.is_dir():
        raise ValueError(
            "The variable, data_path, must point to a an OpenDSS model file."
        )
    if data_path.suffix.lower() == ".dss":
        dss_parser = DSSToCSVConverter(data_path)
        branch_data = dss_parser.branch_data
        bus_data = dss_parser.bus_data
        gen_data = dss_parser.gen_data
        cap_data = dss_parser.cap_data
        reg_data = dss_parser.reg_data

    branch_data = handle_branch_input(branch_data)
    bus_data = handle_bus_input(bus_data)
    gen_data = handle_gen_input(gen_data)
    cap_data = handle_cap_input(cap_data)
    reg_data = handle_reg_input(reg_data)
    return Case(
        branch_data,
        bus_data,
        gen_data,
        cap_data,
        reg_data,
    )

# Updated factory function to allow selective constraint addition
def create_lindist_model_modular(
    case: Case,
    include_power_balance: bool = True,
    include_voltage_drop: bool = True,
    include_voltage_limits: bool = True,
    include_generator_capability: bool = True,
    include_generator_control: bool = True,
    include_swing_constraints: bool = True,
    include_thermal_limits: bool = False,
    linear_gen_approx: bool = True,
) -> pyo.ConcreteModel:
    """
    Create a Pyomo model with selectable constraints.

    Parameters
    ----------
    case : Case
        Input data
    include_power_balance : bool
        Include power balance constraints
    include_voltage_drop : bool
        Include voltage drop constraints
    include_voltage_limits : bool
        Include voltage magnitude limits
    include_generator_capability : bool
        Include generator capability constraints
    include_generator_control : bool
        Include generator control mode constraints
    include_swing_constraints : bool
        Include swing bus voltage constraints
    include_thermal_limits : bool
        Include thermal limits
    linear_gen_approx : bool
        Use linear approximation for generator capability curves
    """
    # Create basic model structure
    model = create_lindist_model(case)

    # Load data
    branch = handle_branch_input(case.branch_data)
    bus = handle_bus_input(case.bus_data)
    gen = handle_gen_input(case.gen_data)
    cap = handle_cap_input(case.cap_data)

    # Add constraints based on user selection
    if include_power_balance:
        add_power_balance_constraints(model, bus, gen, cap)

    if include_voltage_drop:
        add_voltage_drop_constraints(model, branch)

    if include_voltage_limits:
        add_voltage_limits(model, bus)

    if include_generator_capability and len(gen) > 0:
        add_generator_capability_constraints(model, gen, linear_gen_approx)

    if include_generator_control and len(gen) > 0:
        add_generator_control_constraints(model, gen)

    if include_swing_constraints:
        add_swing_bus_constraints(model, bus)

    if include_thermal_limits:
        add_thermal_limits(model, branch)

    return model
