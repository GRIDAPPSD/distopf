"""Topology diagnostics for voltage-variable connectivity."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import pandas as pd


_SWING_BUS_TYPES = frozenset(("SWING", "SWING_FREE", "IN"))


def _parse_phases(value: object) -> list[str]:
    """Parse standard and triplex phase strings into phase labels."""
    phases = str(value)
    if "s1" in phases or "s2" in phases:
        parsed = []
        if "s1" in phases:
            parsed.append("s1")
        if "s2" in phases:
            parsed.append("s2")
        return parsed
    return list(phases)


def find_unconstrained_bus_phases(
    bus_data: pd.DataFrame,
    branch_data: pd.DataFrame,
    swing_bus_types: Iterable[str] = _SWING_BUS_TYPES,
) -> pd.DataFrame:
    """Find bus-phase voltage variables with no incoming voltage equation.

    LinDistFlow creates one voltage variable for every ``(bus id, phase)`` in
    ``bus_data``. A non-swing variable receives a voltage-drop equation only
    when its phase appears on a branch entering that bus. Regulator branches
    are included automatically because they are still represented in
    ``branch_data``. Center-tap transformer secondary phases are likewise
    handled by their ``s1`` and ``s2`` branch phases.

    Parameters
    ----------
    bus_data : pandas.DataFrame
        Must contain ``id``, ``phases``, and optionally ``name`` and
        ``bus_type``.
    branch_data : pandas.DataFrame
        Must contain ``tb`` and ``phases``.
    swing_bus_types : iterable of str, default=("SWING", "SWING_FREE", "IN")
        Bus types whose voltage is fixed by the swing-voltage equations.

    Returns
    -------
    pandas.DataFrame
        Rows with ``id``, ``name``, ``phase``, and ``incoming_phases`` for
        bus-phase variables that are not reached by a voltage equality.
    """
    required_columns = {"id", "phases"}
    missing = required_columns - set(bus_data.columns)
    if missing:
        raise ValueError(f"bus_data is missing required columns: {sorted(missing)}")
    missing = {"tb", "phases"} - set(branch_data.columns)
    if missing:
        raise ValueError(f"branch_data is missing required columns: {sorted(missing)}")

    incoming_phases: defaultdict[int, set[str]] = defaultdict(set)
    for row in branch_data.itertuples(index=False):
        incoming_phases[int(row.tb)].update(_parse_phases(row.phases))

    swing_types = set(swing_bus_types)
    records = []
    for row in bus_data.itertuples(index=False):
        if getattr(row, "bus_type", None) in swing_types:
            continue
        bus_id = int(row.id)
        incoming = incoming_phases[bus_id]
        for phase in _parse_phases(row.phases):
            if phase not in incoming:
                records.append(
                    {
                        "id": bus_id,
                        "name": getattr(row, "name", bus_id),
                        "phase": phase,
                        "incoming_phases": "".join(sorted(incoming)),
                    }
                )

    return pd.DataFrame(records, columns=["id", "name", "phase", "incoming_phases"])


__all__ = ["find_unconstrained_bus_phases"]
