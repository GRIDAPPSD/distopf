import pandas as pd
import pyomo.environ as pyo  # type: ignore
from math import sqrt
from distopf.pyomo_models.protocol import LindistModelProtocol


class OpfResult:
    def __init__(self, model: pyo.ConcreteModel | LindistModelProtocol):
        self.voltages = get_voltages(model.v2)
        vars = [
            att
            for att in model.__dict__.keys()
            if isinstance(getattr(model, att), pyo.Var)
        ]
        for var in vars:
            setattr(self, var, get_values(getattr(model, var)))


def get_values(var: pyo.Var):
    df = get_values_tidy(var)
    df = df.pivot(
        index=["id", "name", "t"], columns="phase", values="value"
    ).reset_index()
    df.columns.name = None
    return df


def get_values_tidy_3ph(var: pyo.Var):
    return pd.DataFrame(
        data=[
            [_id, var.model().name_map[_id], t, _ph, _val]
            for (_id, _ph, t), _val in var.extract_values().items()
        ],
        columns=["id", "name", "t", "phase", "value"],
    )


def get_values_1ph(var: pyo.Var):
    return pd.DataFrame(
        data=[
            [_id, var.model().name_map[_id], t, _val]
            for (_id, t), _val in var.extract_values().items()
        ],
        columns=["id", "name", "t", "value"],
    )


def get_values_tidy(var: pyo.Var):
    if var.name == "v2":
        return pd.DataFrame(
            data=[
                [_id, var.model().name_map[_id], t, _ph, sqrt(_val)]
                for (_id, _ph, t), _val in var.extract_values().items()
            ],
            columns=["id", "name", "t", "phase", "value"],
        )
    if var.dim() == 2:
        return pd.DataFrame(
            data=[
                [_id, var.model().name_map[_id], t, "value", _val]
                for (_id, t), _val in var.extract_values().items()
            ],
            columns=["id", "name", "t", "phase", "value"],
        )
    if var.dim() == 3:
        return pd.DataFrame(
            data=[
                [_id, var.model().name_map[_id], t, _ph, _val]
                for (_id, _ph, t), _val in var.extract_values().items()
            ],
            columns=["id", "name", "t", "phase", "value"],
        )
    return pd.DataFrame(columns=["id", "name", "t", "phase", "value"])


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
    v = get_values_tidy_3ph(var)
    v["value"] = v.value.map(sqrt)
    v = v.pivot(
        index=["id", "name", "t"], columns="phase", values="value"
    ).reset_index()
    v.columns.name = None
    return v
