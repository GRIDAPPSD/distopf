import pandas as pd
import pyomo.environ as pyo
from typing import Optional
from distopf.importer import Case
from math import sqrt


def get_values(var: pyo.Var, v_warning=True):
    if var.name == "v" and v_warning:
        print(
            "The variable 'v' represents squared voltages, v=V^2. To get true voltages, use get_voltages instead. "
            "To suppress this warning set v_warning=False"
        )
    df = pd.DataFrame(columns=["id", "name", "a", "b", "c"])
    df["id"] = sorted(set([bus_id for bus_id, _ in var.index_set()]))
    df["name"] = df.id.map(var.model().name_map)
    df = df.set_index("id").sort_index()
    for bus_id, phase in var.index_set():
        df.at[bus_id, phase] = var[bus_id, phase].value
    df = df.reset_index()
    return df


def get_mp_values(var: pyo.Var, v_warning=True):
    if var.name == "v" and v_warning:
        print(
            "The variable 'v' represents squared voltages, v=V^2. To get true voltages, use get_voltages instead. "
            "To suppress this warning set v_warning=False"
        )
    df = get_mp_values_tidy(var, v_warning=False)
    df = df.pivot(index=["id", "name", "t"], columns="phase", values="value").reset_index()
    df.columns.name = None
    return df


def get_values_tidy(var: pyo.Var, v_warning=True):
    if var.name == "v" and v_warning:
        print(
            "The variable 'v' represents squared voltages, v=V^2. To get true voltages, use get_voltages instead. "
            "To suppress this warning set v_warning=False"
        )
    return pd.DataFrame(
        data=[
            [_id, var.model().name_map[_id], _ph, _val]
            for (_id, _ph), _val in var.extract_values().items()
        ],
        columns=["id", "name", "phase", "value"],
    )


def get_mp_values_tidy(var: pyo.Var, v_warning=True):
    if var.name == "v" and v_warning:
        print(
            "The variable 'v' represents squared voltages, v=V^2. To get true voltages, use get_voltages instead. "
            "To suppress this warning set v_warning=False"
        )
    return pd.DataFrame(
        data=[
            [_id, var.model().name_map[_id], t, _ph, _val]
            for (_id, _ph, t), _val in var.extract_values().items()
        ],
        columns=["id", "name", "t", "phase", "value"],
    )


def get_voltages(var: pyo.Var) -> pd.DataFrame:
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
    v = get_values_tidy(var, v_warning=False)
    v["value"] = v.value.map(sqrt)
    v = v.pivot(index=["id", "name"], columns="phase", values="value").reset_index()
    v.columns.name = None
    return v


def get_mp_voltages(var: pyo.Var) -> pd.DataFrame:
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
    v = get_mp_values_tidy(var, v_warning=False)
    v["value"] = v.value.map(sqrt)
    v = v.pivot(index=["id", "name", "t"], columns="phase", values="value").reset_index()
    v.columns.name = None
    return v


# Convenience function to extract all results at once
def get_all_results(model: pyo.ConcreteModel, case: Case) -> dict:
    """
    Extract all results from solved Pyomo model.

    Returns
    -------
    dict
        Dictionary containing all result DataFrames:
        - 'voltages': Voltage magnitudes
        - 'p_flow': Active power flows
        - 'q_flow': Reactive power flows
        - 'p_gens': Active power generation
        - 'q_gens': Reactive power generation
        - 'q_caps': Capacitor reactive power
    """
    return {
        "voltages": get_voltages(model.v),
        "p_flow": get_values(model.p_flow),
        "q_flow": get_values(model.q_flow),
        "p_gens": get_values(model.p_gen),
        "q_gens": get_values(model.q_gen),
        "q_caps": get_values(model.q_cap),
    }
