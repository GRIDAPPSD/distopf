"""GadCase: the input of the gad_models factories (GAD schema v1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from distopf.gad_models.common.schema import (
    DEVICE_TABLES,
    empty_device_table,
    text_columns,
)

NETWORK_TABLES = ("branch_data", "bus_data")
OPTIONAL_TABLES = ("schedules", *DEVICE_TABLES)


def _empty_device_table(table: str):
    return field(default_factory=lambda: empty_device_table(table))


@dataclass
class GadCase:
    """Network, schedules and device tables of one GAD case.

    Device tables follow the column contract in ``gad_models.common.schema``.
    ``bus_data`` carries bus properties only (no load columns).
    """

    branch_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    bus_data: pd.DataFrame = field(default_factory=pd.DataFrame)
    schedules: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame({"time": pd.Series(dtype=int)})
    )
    pv_data: pd.DataFrame = _empty_device_table("pv_data")
    gen_data: pd.DataFrame = _empty_device_table("gen_data")
    bat_data: pd.DataFrame = _empty_device_table("bat_data")
    cap_data: pd.DataFrame = _empty_device_table("cap_data")
    reg_data: pd.DataFrame = _empty_device_table("reg_data")
    load_data: pd.DataFrame = _empty_device_table("load_data")
    start_step: int = 0
    n_steps: int = 1
    delta_t: float = 1

    @classmethod
    def from_dir(
        cls,
        path: str | Path,
        start_step: int = 0,
        n_steps: int = 1,
        delta_t: float = 1,
    ) -> GadCase:
        """Read ``<table>.csv`` files from a directory.

        ``branch_data.csv`` and ``bus_data.csv`` are required. A missing
        schedules or device table means there are none of that kind.
        """
        directory = Path(path)
        tables = {}
        for table in NETWORK_TABLES:
            csv_path = directory / f"{table}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"{table}.csv not found in {directory}")
            tables[table] = _read_table(csv_path, table)
        for table in OPTIONAL_TABLES:
            csv_path = directory / f"{table}.csv"
            if csv_path.exists():
                tables[table] = _read_table(csv_path, table)
        return cls(start_step=start_step, n_steps=n_steps, delta_t=delta_t, **tables)

    def to_dir(self, path: str | Path) -> None:
        """Write every table to ``<table>.csv`` (empty device tables keep their header)."""
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        for table in (*NETWORK_TABLES, *OPTIONAL_TABLES):
            getattr(self, table).to_csv(directory / f"{table}.csv", index=False)


def _read_table(csv_path: Path, table: str) -> pd.DataFrame:
    """Read a CSV with text columns as str (so bus "650" stays "650") and "" for empty cells."""
    text = text_columns(table)
    header = pd.read_csv(csv_path, nrows=0).columns
    text_in_file = [col for col in header if col in text]
    frame = pd.read_csv(
        csv_path,
        dtype=dict.fromkeys(text_in_file, str),
        keep_default_na=False,
        na_values=[""],
    )
    frame[text_in_file] = frame[text_in_file].fillna("")
    return frame
