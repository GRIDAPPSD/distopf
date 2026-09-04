"""Normalization and shared indexing helpers for Pyomo device models."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DeviceTable:
    """Normalized table with independent entity and bus identifiers."""

    kind: str
    data: pd.DataFrame
    id_column: str
    bus_column: str

    @property
    def ids(self) -> tuple[Any, ...]:
        return tuple(self.data[self.id_column].tolist())

    def rows_at_bus(self, bus_id: Any) -> pd.DataFrame:
        return self.data.loc[self.data[self.bus_column] == bus_id]


def parse_phases(phases_str: str) -> list[str]:
    """Parse standard or triplex phase labels without importing a model module."""
    if "s1" in phases_str or "s2" in phases_str:
        return [phase for phase in ("s1", "s2") if phase in phases_str]
    return list(phases_str)


def phase_tuples(data: pd.DataFrame, id_column: str = "id") -> list[tuple[Any, str]]:
    """Build entity/phase tuples from a device table."""
    return [
        (row[id_column], phase)
        for _, row in data.iterrows()
        for phase in parse_phases(str(row.get("phases", "")))
    ]


def normalize_device_table(
    data: pd.DataFrame | None,
    *,
    kind: str,
    id_column: str = "device_id",
    bus_column: str = "bus_id",
) -> DeviceTable:
    """Normalize legacy rows while permitting multiple devices per bus."""
    frame = data.copy() if data is not None else pd.DataFrame()
    if frame.empty:
        frame[id_column] = pd.Series(dtype=str)
        frame[bus_column] = pd.Series(dtype=int)
        return DeviceTable(kind, frame, id_column, bus_column)
    if bus_column not in frame:
        if "id" not in frame:
            raise ValueError(f"{kind} data requires {bus_column!r} or legacy 'id'")
        frame[bus_column] = frame["id"]
    if id_column not in frame:
        frame[id_column] = [f"{kind}_{index}" for index in range(len(frame))]
    if frame[id_column].duplicated().any():
        duplicates = frame.loc[frame[id_column].duplicated(), id_column].tolist()
        raise ValueError(f"Duplicate {kind} device IDs: {duplicates}")
    return DeviceTable(kind, frame.reset_index(drop=True), id_column, bus_column)


def create_bus_device_map(
    table: DeviceTable, phases: Iterable[str] = ("a", "b", "c", "s1", "s2")
) -> dict[tuple[Any, str], list[Any]]:
    """Create a bus/phase to device-ID map from normalized device data."""
    result: dict[tuple[Any, str], list[Any]] = {}
    for _, row in table.data.iterrows():
        row_phases = parse_phases(str(row.get("phases", "")))
        for phase in phases:
            if phase in row_phases:
                result.setdefault((row[table.bus_column], phase), []).append(
                    row[table.id_column]
                )
    return result
