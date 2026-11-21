"""Extract OpenDSS Load elements into the GAD `load_data` table (schema v1).

One row per OpenDSS Load (never merged per bus), so several loads with
different shapes can share a bus. Loads are the one device type that keeps
per-phase values (`p_a`.. `q_s2`): a 1-element delta load becomes a single
device with its exact, unequal wye-equivalent split.

Schedules convention (shared by every device extractor)
-------------------------------------------------------
Schedule columns are named by the lowercase OpenDSS LoadShape name and hold
the P multiplier for each time step (`conv.schedule_steps` values). A Q
multiplier, if the LoadShape defines one that differs from the P multiplier,
goes in the column `<name>.q`. Extractors register every shape they reference
in `conv._schedule_columns` with `register_shape()`, and the converter turns
that dict into the `schedules` table. `conv.shape` ("yearly", "daily" or
"duty") selects which OpenDSS property supplies the shape name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from distopf.gad_models.common.schema import LOAD_COLUMNS, empty_device_table

if TYPE_CHECKING:
    from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter

# Contract columns of `load_data` (defined once in gad_models.common.schema).
COLUMNS = LOAD_COLUMNS

# Secondary (triplex) bus node -> leg.
_SECONDARY_LEG = {1: "s1", 2: "s2"}


def register_shape(conv: GadDSSToCSVConverter, name: str) -> None:
    """Add the P (and, if different, Q) multipliers of a LoadShape to the schedules."""
    p_values = conv._shape_values(name, "p")
    conv._schedule_columns[name] = p_values
    q_values = conv._shape_values(name, "q")
    if not np.array_equal(p_values, q_values):
        conv._schedule_columns[f"{name}.q"] = q_values


def _rated_to_base_ratio(
    conv: GadDSSToCSVConverter, bus: str, rated_kv: float, is_delta: bool
) -> float:
    """Bus voltage base / the load's rated voltage (line-to-line if multiphase or delta).

    The active element must be the load. Secondary (triplex) buses are 1.0.
    """
    if bus in conv.secondary_buses:
        return 1.0
    line_to_line = is_delta or conv.dss.CktElement.NumPhases() > 1
    rated_ln = rated_kv * 1000 / (3**0.5 if line_to_line else 1)
    return float(conv._nominal_voltage_bases[bus]) / rated_ln


def _delta_to_wye(nodes: list[int], n_elements: int, s_total: complex) -> dict[str, complex]:
    """Line-to-neutral power per phase of line-to-line load elements (balanced voltage).

    OpenDSS uses `n_elements` elements on consecutive node pairs, wrapping around.
    Each element's power is moved to its two phases with a +-30 degree rotation.
    """
    s_element = s_total / n_elements
    by_node: dict[int, complex] = {}
    for k in range(n_elements):
        i, j = nodes[k], nodes[(k + 1) % len(nodes)]
        sign = 1 if (j - i) % 3 == 1 else -1
        shift = np.exp(1j * sign * np.pi / 6) / np.sqrt(3)
        by_node[i] = by_node.get(i, 0) + s_element * shift.conjugate()
        by_node[j] = by_node.get(j, 0) + s_element * shift
    return {"abc"[node - 1]: by_node[node] for node in sorted(by_node)}


def _split_by_phase(
    conv: GadDSSToCSVConverter,
    bus: str,
    nodes: list[int],
    is_delta: bool,
    n_elements: int,
    s_total: complex,
) -> dict[str, complex]:
    """Per-phase complex power (pu) of a load, keyed by phase label in phase order."""
    hot = [n for n in nodes if n != 0]
    if bus in conv.secondary_buses:
        if sorted(hot) == [1, 2]:  # 240 V across both legs
            return {"s1": s_total / 2, "s2": s_total / 2}
        if len(hot) == 1 and hot[0] in _SECONDARY_LEG:  # 120 V on one leg
            return {_SECONDARY_LEG[hot[0]]: s_total}
        raise ValueError(f"Load on secondary bus {bus!r} has unsupported nodes {nodes}")
    if is_delta and len(nodes) >= 2 and 0 not in nodes:
        return _delta_to_wye(nodes, n_elements, s_total)
    phases = conv._phases_from_nodes(hot)  # no nodes means abc
    return {ph: s_total / len(phases) for ph in phases}


def extract_load(conv: GadDSSToCSVConverter) -> pd.DataFrame:
    """One row per OpenDSS Load, with exactly the `load_data` contract columns.

    Powers are per-phase pu on `conv.s_base`, nominal (rated-voltage) values
    rebased to the bus voltage base with the load's CVR exponents. The OPF
    applies the CVR itself.
    """
    dss = conv.dss
    scale = 1000 / conv.s_base
    rows = []
    flag = dss.Loads.First()
    while flag:
        specs = dss.CktElement.BusNames()
        if len(specs) > 1:
            raise ValueError("Multiple connected buses")
        bus, nodes = conv._split_bus_spec(specs[0])
        is_delta = dss.Loads.IsDelta()

        cvr_p, cvr_q = conv._load_cvr()
        ratio = _rated_to_base_ratio(conv, bus, dss.Loads.kV(), is_delta)
        s_total = complex(
            dss.Loads.kW() * ratio**cvr_p, dss.Loads.kvar() * ratio**cvr_q
        )
        by_phase = _split_by_phase(
            conv, bus, nodes, is_delta, dss.CktElement.NumPhases(), s_total * scale
        )

        shape = getattr(dss.Loads, conv.shape.capitalize())().lower()
        if shape:
            register_shape(conv, shape)

        row = dict.fromkeys(LOAD_COLUMNS, 0.0)
        for phase, power in by_phase.items():
            row[f"p_{phase}"] = power.real
            row[f"q_{phase}"] = power.imag
        row.update(
            device_name=f"load.{dss.Loads.Name()}".lower(),
            type="load",
            bus_name=bus,
            phases="".join(by_phase),
            cvr_p=cvr_p,
            cvr_q=cvr_q,
            load_shape=shape,
            status=dss.Loads.Status().name.lower(),
            load_class=str(dss.Loads.Class()),
            num_cust=dss.Loads.NumCust(),
        )
        rows.append(row)
        flag = dss.Loads.Next()

    if not rows:
        return empty_device_table("load_data")
    return pd.DataFrame(rows, columns=LOAD_COLUMNS)
