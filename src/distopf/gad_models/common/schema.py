"""GAD schema v1: the column contract of every device table.

This is the single place where the column lists live. The OpenDSS extractors
(``dss_importer/gad_devices``), ``GadCase`` and the providers all import them.

Conventions: per-unit on ``s_base``; device powers are totals, split equally
over ``phases`` (``load_data`` is the exception and stores per-phase values).
``device_name`` is unique within a table and equals the lowercase OpenDSS full
name, ``"<class>.<name>"``.
"""

from __future__ import annotations

import pandas as pd

LOAD_PHASES = ("a", "b", "c", "s1", "s2")

# Columns that identify a device. Regulators use from_name/to_name instead of bus_name.
_DEVICE_COLUMNS = ["device_name", "type", "bus_name", "phases"]

PV_COLUMNS = [
    *_DEVICE_COLUMNS,
    "pmpp",
    "s_max",
    "irradiance",
    "pmpp_limit",
    "q",
    "q_min",
    "q_max",
    "control_variable",
    "pv_shape",
    "cost",
]

GEN_COLUMNS = [
    *_DEVICE_COLUMNS,
    "p",
    "s_max",
    "p_min",
    "p_max",
    "q",
    "q_min",
    "q_max",
    "control_variable",
    "gen_shape",
    "cost",
]

BAT_COLUMNS = [
    *_DEVICE_COLUMNS,
    "s_max",
    "energy_capacity",
    "p_max",
    "p_charge_max",
    "p_discharge_max",
    "min_soc",
    "max_soc",
    "start_soc",
    "charge_efficiency",
    "discharge_efficiency",
    "q_min",
    "q_max",
    "p",
    "q",
    "control_variable",
    "annual_cycle_limit",
]

CAP_COLUMNS = [*_DEVICE_COLUMNS, "q", "status", "controller"]

REG_COLUMNS = [
    "device_name",
    "type",
    "from_name",
    "to_name",
    "phases",
    "fb",
    "tb",
    "tap",
    "ratio_min",
    "ratio_max",
    "num_taps",
    "max_tap_change",
    "controller",
    "v_reg",
    "band",
]

LOAD_COLUMNS = [
    *_DEVICE_COLUMNS,
    *[f"p_{ph}" for ph in LOAD_PHASES],
    *[f"q_{ph}" for ph in LOAD_PHASES],
    "cvr_p",
    "cvr_q",
    "load_shape",
    "status",
    "load_class",
    "num_cust",
]

# Device table name -> (columns, value of the `type` column).
DEVICE_TABLES: dict[str, tuple[list[str], str]] = {
    "pv_data": (PV_COLUMNS, "pv"),
    "gen_data": (GEN_COLUMNS, "generator"),
    "bat_data": (BAT_COLUMNS, "battery"),
    "cap_data": (CAP_COLUMNS, "capacitor"),
    "reg_data": (REG_COLUMNS, "regulator"),
    "load_data": (LOAD_COLUMNS, "load"),
}

# Columns that hold text. CSVs are read with these as str so that a bus named
# "650" stays "650" and an empty cell stays "" (not NaN).
_TEXT_COLUMNS = frozenset(
    {
        "device_name",
        "type",
        "bus_name",
        "from_name",
        "to_name",
        "name",
        "phases",
        "primary_phase",
        "bus_type",
        "control_variable",
        "pv_shape",
        "gen_shape",
        "load_shape",
        "controller",
        "load_class",
    }
)


def text_columns(table: str) -> frozenset[str]:
    """Text columns of a table. ``status`` is text in branch_data and load_data only."""
    if table in ("branch_data", "load_data"):
        return _TEXT_COLUMNS | {"status"}
    return _TEXT_COLUMNS


def empty_device_table(table: str) -> pd.DataFrame:
    """Empty frame with exactly the contract columns of a device table."""
    columns, _ = DEVICE_TABLES[table]
    text = text_columns(table)
    return pd.DataFrame(
        {col: pd.Series(dtype=object if col in text else float) for col in columns}
    )
