import pandas as pd
import pyomo.environ as pyo
from typing import Optional
from distopf.importer import Case


def get_voltages(model: pyo.ConcreteModel, case: Case) -> pd.DataFrame:
    """
    Extract voltage magnitudes from solved Pyomo model.

    Parameters
    ----------
    model : pyo.ConcreteModel
        Solved Pyomo model with voltage variables
    bus_data : pd.DataFrame
        Bus data for getting bus names

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["id", "name", "a", "b", "c"] containing
        voltage magnitudes (not squared)
    """
    # Create bus lookup for names
    bus_lookup = {row.id: row.name for _, row in case.bus_data.iterrows()}

    # Get all unique bus IDs
    bus_ids = sorted(set(bus_id for (bus_id, phase) in model.bus_phase_set))

    # Initialize DataFrame
    df = pd.DataFrame(columns=["id", "name", "a", "b", "c"], index=range(len(bus_ids)))
    df["id"] = bus_ids
    df["name"] = [bus_lookup.get(bus_id, f"Bus_{bus_id}") for bus_id in bus_ids]

    # Fill voltage values (convert from squared magnitude to magnitude)
    for i, bus_id in enumerate(bus_ids):
        for phase in ["a", "b", "c"]:
            if (bus_id, phase) in model.bus_phase_set:
                v_squared = pyo.value(model.v[bus_id, phase])
                df.at[i, phase] = v_squared**0.5 if v_squared >= 0 else 0.0
            else:
                df.at[i, phase] = None

    return df


def get_p_gens(model: pyo.ConcreteModel, case: Case) -> pd.DataFrame:
    """
    Extract active power generation from solved Pyomo model.

    Parameters
    ----------
    model : pyo.ConcreteModel
        Solved Pyomo model with generator variables
    bus_data : pd.DataFrame
        Bus data for getting bus names
    gen_data : pd.DataFrame, optional
        Generator data (not currently used but kept for consistency)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["id", "name", "a", "b", "c"] containing
        active power generation values
    """
    if not hasattr(model, "p_gen") or len(model.gen_phase_set) == 0:
        return pd.DataFrame(columns=["id", "name", "a", "b", "c"])

    # Create bus lookup for names
    bus_lookup = {row.id: row.name for _, row in case.bus_data.iterrows()}

    # Get all unique bus IDs with generators
    gen_bus_ids = sorted(set(bus_id for (bus_id, phase) in model.gen_phase_set))

    # Initialize DataFrame
    df = pd.DataFrame(
        columns=["id", "name", "a", "b", "c"], index=range(len(gen_bus_ids))
    )
    df["id"] = gen_bus_ids
    df["name"] = [bus_lookup.get(bus_id, f"Bus_{bus_id}") for bus_id in gen_bus_ids]

    # Fill generation values
    for i, bus_id in enumerate(gen_bus_ids):
        for phase in ["a", "b", "c"]:
            if (bus_id, phase) in model.gen_phase_set:
                df.at[i, phase] = pyo.value(model.p_gen[bus_id, phase])
            else:
                df.at[i, phase] = None

    return df


def get_q_gens(model: pyo.ConcreteModel, case: Case) -> pd.DataFrame:
    """
    Extract reactive power generation from solved Pyomo model.

    Parameters
    ----------
    model : pyo.ConcreteModel
        Solved Pyomo model with generator variables
    bus_data : pd.DataFrame
        Bus data for getting bus names
    gen_data : pd.DataFrame, optional
        Generator data (not currently used but kept for consistency)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["id", "name", "a", "b", "c"] containing
        reactive power generation values
    """
    if not hasattr(model, "q_gen") or len(model.gen_phase_set) == 0:
        return pd.DataFrame(columns=["id", "name", "a", "b", "c"])

    # Create bus lookup for names
    bus_lookup = {row.id: row.name for _, row in case.bus_data.iterrows()}

    # Get all unique bus IDs with generators
    gen_bus_ids = sorted(set(bus_id for (bus_id, phase) in model.gen_phase_set))

    # Initialize DataFrame
    df = pd.DataFrame(
        columns=["id", "name", "a", "b", "c"], index=range(len(gen_bus_ids))
    )
    df["id"] = gen_bus_ids
    df["name"] = [bus_lookup.get(bus_id, f"Bus_{bus_id}") for bus_id in gen_bus_ids]

    # Fill generation values
    for i, bus_id in enumerate(gen_bus_ids):
        for phase in ["a", "b", "c"]:
            if (bus_id, phase) in model.gen_phase_set:
                df.at[i, phase] = pyo.value(model.q_gen[bus_id, phase])
            else:
                df.at[i, phase] = None

    return df


def get_q_caps(model: pyo.ConcreteModel, case: Case) -> pd.DataFrame:
    """
    Extract capacitor reactive power from solved Pyomo model.

    Parameters
    ----------
    model : pyo.ConcreteModel
        Solved Pyomo model with capacitor variables
    bus_data : pd.DataFrame
        Bus data for getting bus names
    cap_data : pd.DataFrame, optional
        Capacitor data (not currently used but kept for consistency)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["id", "name", "a", "b", "c"] containing
        capacitor reactive power values
    """
    if not hasattr(model, "q_cap") or len(model.cap_phase_set) == 0:
        return pd.DataFrame(columns=["id", "name", "a", "b", "c"])

    # Create bus lookup for names
    bus_lookup = {row.id: row.name for _, row in case.bus_data.iterrows()}

    # Get all unique bus IDs with capacitors
    cap_bus_ids = sorted(set(bus_id for (bus_id, phase) in model.cap_phase_set))

    # Initialize DataFrame
    df = pd.DataFrame(
        columns=["id", "name", "a", "b", "c"], index=range(len(cap_bus_ids))
    )
    df["id"] = cap_bus_ids
    df["name"] = [bus_lookup.get(bus_id, f"Bus_{bus_id}") for bus_id in cap_bus_ids]

    # Fill capacitor values
    for i, bus_id in enumerate(cap_bus_ids):
        for phase in ["a", "b", "c"]:
            if (bus_id, phase) in model.cap_phase_set:
                df.at[i, phase] = pyo.value(model.q_cap[bus_id, phase])
            else:
                df.at[i, phase] = None

    return df


def get_apparent_power_flows(model: pyo.ConcreteModel, case: Case) -> pd.DataFrame:
    """
    Extract apparent power flows from solved Pyomo model.

    Parameters
    ----------
    model : pyo.ConcreteModel
        Solved Pyomo model with power flow variables
    bus_data : pd.DataFrame
        Bus data for getting bus names
    branch_data : pd.DataFrame
        Branch data for getting from/to bus relationships and names

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["fb", "tb", "from_name", "to_name", "a", "b", "c"]
        containing complex apparent power flows (P + jQ)
    """
    if not hasattr(model, "p_flow") or not hasattr(model, "q_flow"):
        return pd.DataFrame(columns=["fb", "tb", "from_name", "to_name", "a", "b", "c"])

    # Create bus lookup for names
    bus_lookup = {row.id: row.name for _, row in case.bus_data.iterrows()}

    # Create branch lookup for from-bus relationships
    branch_lookup = {}
    for _, row in case.branch_data.iterrows():
        branch_lookup[row.tb] = {
            "fb": row.fb,
            "from_name": getattr(
                row, "from_name", bus_lookup.get(row.fb, f"Bus_{row.fb}")
            ),
            "to_name": getattr(row, "to_name", bus_lookup.get(row.tb, f"Bus_{row.tb}")),
        }

    # Get all unique to-bus IDs (branch_phase_set uses to-bus)
    to_bus_ids = sorted(set(bus_id for (bus_id, phase) in model.branch_phase_set))

    # Initialize DataFrame
    df = pd.DataFrame(
        columns=["fb", "tb", "from_name", "to_name", "a", "b", "c"],
        index=range(len(to_bus_ids)),
    )

    # Fill branch information
    for i, to_bus_id in enumerate(to_bus_ids):
        if to_bus_id in branch_lookup:
            df.at[i, "fb"] = branch_lookup[to_bus_id]["fb"]
            df.at[i, "tb"] = to_bus_id
            df.at[i, "from_name"] = branch_lookup[to_bus_id]["from_name"]
            df.at[i, "to_name"] = branch_lookup[to_bus_id]["to_name"]
        else:
            # Fallback if branch lookup fails
            df.at[i, "fb"] = None
            df.at[i, "tb"] = to_bus_id
            df.at[i, "from_name"] = "Unknown"
            df.at[i, "to_name"] = bus_lookup.get(to_bus_id, f"Bus_{to_bus_id}")

    # Fill power flow values (complex: P + jQ)
    for i, to_bus_id in enumerate(to_bus_ids):
        for phase in ["a", "b", "c"]:
            if (to_bus_id, phase) in model.branch_phase_set:
                p_val = pyo.value(model.p_flow[to_bus_id, phase])
                q_val = pyo.value(model.q_flow[to_bus_id, phase])
                df.at[i, phase] = complex(p_val, q_val)
            else:
                df.at[i, phase] = None

    # Set proper data types
    df["fb"] = df["fb"].astype("Int64")  # Nullable integer
    df["tb"] = df["tb"].astype("Int64")

    return df


# Convenience function to extract all results at once
def get_all_results(
    model: pyo.ConcreteModel,
    case: Case
) -> dict:
    """
    Extract all results from solved Pyomo model.

    Returns
    -------
    dict
        Dictionary containing all result DataFrames:
        - 'voltages': Voltage magnitudes
        - 'p_gens': Active power generation
        - 'q_gens': Reactive power generation
        - 'q_caps': Capacitor reactive power
        - 'power_flows': Apparent power flows
    """
    return {
        "voltages": get_voltages(model, case),
        "p_gens": get_p_gens(model, case),
        "q_gens": get_q_gens(model, case),
        "q_caps": get_q_caps(model, case),
        "power_flows": get_apparent_power_flows(model, case),
    }
