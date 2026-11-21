"""Reading shared by the generator and PV extractors (both are balanced injectors).

An injector is one OpenDSS element whose total power is split equally over its
phases, so a row needs the element's bus, phases and shape, and its powers in
per unit of the converter's ``s_base``.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter

_SECONDARY_LEG = {1: "s1", 2: "s2"}


def read_property(conv: GadDSSToCSVConverter, element: str, prop: str) -> str:
    """Value of an OpenDSS property as text, e.g. ``("Generator.g1", "Maxkvar")``."""
    conv.dss.Text.Command(f"? {element}.{prop}")
    return conv.dss.Text.Result()


def to_per_unit(conv: GadDSSToCSVConverter, kilo_value: float) -> float:
    """kW / kvar / kVA -> per unit on ``s_base``."""
    return kilo_value * 1000 / conv.s_base


def bus_and_phases(conv: GadDSSToCSVConverter, device_name: str) -> tuple[str, str]:
    """Bus name and phases of the active circuit element.

    Triplex (secondary) buses give ``s1``, ``s2`` or ``s1s2`` (240 V). A
    1-element delta device (one element across two phases) is returned on its
    two phases with a warning, because the exact wye equivalent is unequal per
    phase while a device is split equally.
    """
    bus, nodes = conv._split_bus_spec(conv.dss.CktElement.BusNames()[0])
    if bus in conv.secondary_buses:
        legs = sorted({_SECONDARY_LEG[n] for n in nodes if n in _SECONDARY_LEG})
        if not legs:
            raise ValueError(f"{device_name}: no s1/s2 node in {nodes} on bus {bus!r}")
        return bus, "".join(legs)
    phases = conv._phases_from_nodes(nodes)
    if conv.dss.CktElement.NumPhases() == 1 and len(phases) == 2:
        warnings.warn(
            f"{device_name} is a 1-element delta device on phases {phases!r}; "
            "its power is split equally over the two phases, which is approximate.",
            stacklevel=2,
        )
    return bus, phases


def register_shape(conv: GadDSSToCSVConverter, element: str) -> str:
    """Name of the element's yearly/daily/duty shape ("" if none), added to the schedules.

    Which property is read is chosen by ``conv.shape``.
    """
    name = read_property(conv, element, conv.shape).strip().lower()
    if name in ("", "none"):
        return ""
    conv._schedule_columns[name] = conv._shape_values(name)
    return name


def injector_table(rows: list[dict], columns: list[str], empty: pd.DataFrame) -> pd.DataFrame:
    """DataFrame of the rows with exactly the contract columns (``empty`` if no rows)."""
    return pd.DataFrame(rows, columns=columns) if rows else empty
