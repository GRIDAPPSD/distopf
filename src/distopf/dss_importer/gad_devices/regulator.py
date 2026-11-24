"""Extract OpenDSS regulator transformers into the GAD `reg_data` table (schema v1).

A Transformer is a regulator when a RegControl names it or its winding-2 tap is
not 1. Every regulator Transformer is one device with one tap: a 3-phase
regulator is one ganged device, three single-phase regulators are three devices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from distopf.gad_models.common.schema import REG_COLUMNS, empty_device_table

if TYPE_CHECKING:
    from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter

# Contract columns of `reg_data` (defined once in gad_models.common.schema).
COLUMNS = REG_COLUMNS


def _regcontrol_by_transformer(conv: GadDSSToCSVConverter) -> dict[str, str]:
    """Transformer name -> name of the one RegControl that controls it."""
    controls: dict[str, list[str]] = {}
    flag = conv.dss.RegControls.First()
    while flag:
        transformer = conv.dss.RegControls.Transformer().lower()
        controls.setdefault(transformer, []).append(conv.dss.RegControls.Name().lower())
        flag = conv.dss.RegControls.Next()
    for transformer, names in controls.items():
        if len(names) > 1:
            listed = ", ".join(f"regcontrol.{n}" for n in names)
            raise ValueError(
                f"transformer.{transformer} is controlled by more than one RegControl "
                f"({listed}); only one RegControl per regulator is supported"
            )
    return {transformer: names[0] for transformer, names in controls.items()}


def _regulator_names(conv: GadDSSToCSVConverter, controlled: set[str]) -> list[str]:
    """Names of the Transformers that are regulators, in OpenDSS order."""
    names = []
    flag = conv.dss.Transformers.First()
    while flag:
        name = conv.dss.Transformers.Name().lower()
        conv.dss.Transformers.Wdg(2)
        if name in controlled or abs(conv.dss.Transformers.Tap() - 1) > 1e-9:
            names.append(name)
        flag = conv.dss.Transformers.Next()
    return names


def _transformer_columns(conv: GadDSSToCSVConverter, name: str) -> dict:
    """Branch, phases and winding-2 tap of a regulator Transformer."""
    dss = conv.dss
    dss.Transformers.Name(name)
    specs = dss.CktElement.BusNames()
    raw_from, raw_to = conv._bus_of(specs[0]), conv._bus_of(specs[-1])
    from_name, to_name = conv._orient_edge(raw_from, raw_to)
    # A transformer written downstream -> upstream steps the voltage the other way.
    direction = 1 if from_name == raw_from else -1
    dss.Transformers.Wdg(2)
    ratio_min = dss.Transformers.MinTap()
    ratio_max = dss.Transformers.MaxTap()
    num_taps = dss.Transformers.NumTaps()
    step = (ratio_max - ratio_min) / num_taps
    return dict(
        device_name=f"transformer.{name}",
        type="regulator",
        from_name=from_name,
        to_name=to_name,
        phases=conv._phases_from_nodes(conv._split_bus_spec(specs[0])[1]),
        fb=conv.bus_names_to_index_map[from_name],
        tb=conv.bus_names_to_index_map[to_name],
        tap=int(round((dss.Transformers.Tap() - 1) / step * direction)),
        ratio_min=ratio_min,
        ratio_max=ratio_max,
        num_taps=num_taps,
    )


def _controller_columns(conv: GadDSSToCSVConverter, name: str, control: str) -> dict:
    """RegControl settings; v_reg and band are in pu of the bus line-to-neutral base.

    v_reg = vreg * ptratio assumes a line-to-neutral PT (wye regulators). It is
    informational only: the OPF chooses the tap.
    """
    dss = conv.dss
    dss.Transformers.Name(name)
    winding_buses = [conv._bus_of(spec) for spec in dss.CktElement.BusNames()]
    dss.RegControls.Name(control)
    bus = winding_buses[dss.RegControls.Winding() - 1]
    volts_per_pu = conv._nominal_voltage_bases[bus] / dss.RegControls.PTRatio()
    return dict(
        max_tap_change=dss.RegControls.MaxTapChange(),
        controller=f"regcontrol.{control}",
        v_reg=dss.RegControls.ForwardVreg() / volts_per_pu,
        band=dss.RegControls.ForwardBand() / volts_per_pu,
    )


def extract_regulator(conv: GadDSSToCSVConverter) -> pd.DataFrame:
    """One row per regulator Transformer, with exactly the `reg_data` contract columns."""
    control_of = _regcontrol_by_transformer(conv)
    rows = []
    for name in _regulator_names(conv, set(control_of)):
        row = _transformer_columns(conv, name)
        if name in control_of:
            row.update(_controller_columns(conv, name, control_of[name]))
        else:
            row.update(max_tap_change=np.nan, controller="", v_reg=np.nan, band=np.nan)
        rows.append(row)
    if not rows:
        return empty_device_table("reg_data")
    return pd.DataFrame(rows, columns=REG_COLUMNS)
