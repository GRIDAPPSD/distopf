"""Extract OpenDSS Capacitor elements into the GAD `cap_data` table (schema v1)."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd

from distopf.gad_models.common.schema import CAP_COLUMNS, empty_device_table

if TYPE_CHECKING:
    from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter

# Contract columns of `cap_data` (defined once in gad_models.common.schema).
COLUMNS = CAP_COLUMNS
_SECONDARY_NODE_LEG = {1: "s1", 2: "s2"}


def _controllers_by_capacitor(dss) -> dict[str, str]:
    """Capacitor name -> full name of the CapControl that switches it (lowercase)."""
    controllers: dict[str, str] = {}
    flag = dss.CapControls.First()
    while flag:
        capacitor = dss.CapControls.Capacitor().lower()
        controller = f"capcontrol.{dss.CapControls.Name().lower()}"
        if capacitor in controllers:
            raise ValueError(
                f"capacitor.{capacitor} has two CapControls "
                f"({controllers[capacitor]}, {controller}); only one is supported"
            )
        controllers[capacitor] = controller
        flag = dss.CapControls.Next()
    return controllers


def _phases(conv: GadDSSToCSVConverter, bus: str, nodes: list[int]) -> str:
    """Phases of a capacitor: s1/s2 legs on a secondary bus, a/b/c elsewhere."""
    if bus in conv.secondary_buses:
        legs = sorted(
            {_SECONDARY_NODE_LEG[n] for n in nodes if n in _SECONDARY_NODE_LEG}
        )
        return "".join(legs) or "s1s2"
    return conv._phases_from_nodes(nodes)


def extract_capacitor(conv: GadDSSToCSVConverter) -> pd.DataFrame:
    """One row per OpenDSS Capacitor, with exactly the `cap_data` contract columns.

    ``q`` is the total kvar in per unit on ``s_base``, rebased from the rated kV
    to the bus voltage base. ``status`` is the initial OpenDSS state, so switched
    off capacitors are kept. ``controller`` is the CapControl switching it, if any.
    """
    dss = conv.dss
    # Read the controllers first: iterating them changes the active element.
    controllers = _controllers_by_capacitor(dss)
    rows = []
    flag = dss.Capacitors.First()
    while flag:
        name = dss.Capacitors.Name().lower()
        device_name = f"capacitor.{name}"
        if dss.Capacitors.NumSteps() > 1:
            raise NotImplementedError(
                f"{device_name}: NumSteps={dss.Capacitors.NumSteps()}; multi-step "
                "capacitor banks are not supported. Model each step as its own "
                "Capacitor."
            )
        bus, nodes = conv._split_bus_spec(dss.CktElement.BusNames()[0])
        phases = _phases(conv, bus, nodes)
        is_delta = dss.Capacitors.IsDelta()
        if is_delta and dss.CktElement.NumPhases() == 1:
            if len(phases) != 2:
                raise ValueError(
                    f"{device_name}: 1-element delta capacitor needs two phase "
                    f"nodes, got {dss.CktElement.BusNames()[0]!r}"
                )
            warnings.warn(
                f"{device_name}: 1-element delta capacitor on phases {phases}; its "
                "kvar is split equally over both phases (approximate wye equivalent)",
                stacklevel=2,
            )
        ratio = conv._rated_to_base_ratio(bus, dss.Capacitors.kV(), is_delta)
        rows.append(
            dict(
                device_name=device_name,
                type="capacitor",
                bus_name=bus,
                phases=phases,
                q=dss.Capacitors.kvar() * 1000 / conv.s_base * ratio**2,
                status=int(dss.Capacitors.States()[0]),
                controller=controllers.get(name, ""),
            )
        )
        flag = dss.Capacitors.Next()
    if not rows:
        return empty_device_table("cap_data")
    return pd.DataFrame(rows, columns=CAP_COLUMNS)
