"""Migrate a legacy distopf ``Case`` to a ``GadCase`` (GAD schema v1).

Legacy generator, capacitor and regulator tables hold per-phase values. A GAD
device is balanced (one set of totals split equally over its phases), so:

* equal per-phase values  -> one multi-phase device with the totals;
* unequal per-phase values -> one single-phase device per phase (exact).

Names are ``<type>.<legacy name or bus name>``, plus ``.<phase>`` when a device
is split. Loads leave ``bus_data`` as one ``load_data`` row per bus with
per-phase values (never split). Batteries already hold totals.

Lossy: the legacy ``balanced`` generator flag is dropped, so an unequal
generator that was forced to dispatch equally becomes independent per-phase
devices.
"""

from __future__ import annotations

import math

import pandas as pd

from distopf.gad_models.common.case import GadCase
from distopf.gad_models.common.data import parse_phases
from distopf.gad_models.common.schema import (
    DEVICE_TABLES,
    LOAD_PHASES,
    empty_device_table,
    text_columns,
)

EQUAL_TOLERANCE = 1e-12
DEFAULT_GEN_SHAPE = "PV"
DEFAULT_LOAD_SHAPE = "default"
_BUS_LOAD_COLUMNS = ("cvr_p", "cvr_q", "load_shape")


def to_gad_case(case) -> GadCase:
    """Convert a legacy ``distopf.Case`` to a ``GadCase``. ``case`` is not modified."""
    bus_names = dict(zip(case.bus_data["id"], case.bus_data["name"].astype(str)))
    schedule_columns = list(case.schedules.columns)
    load_data = _load_data(case.bus_data, schedule_columns)
    return GadCase(
        branch_data=_branch_data(case.branch_data, bus_names),
        bus_data=_bus_data(case.bus_data),
        schedules=_schedules(case.schedules, load_data),
        gen_data=_gen_data(case.gen_data, bus_names, schedule_columns),
        bat_data=_bat_data(case.bat_data, bus_names),
        cap_data=_cap_data(case.cap_data, bus_names),
        reg_data=_reg_data(case.reg_data, bus_names),
        load_data=load_data,
        start_step=case.start_step,
        n_steps=case.n_steps,
        delta_t=case.delta_t,
    )


# ---------------------------------------------------------------------------
# Network tables
# ---------------------------------------------------------------------------


def _fill_text(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    """Empty cells of text columns are "" (as ``GadCase.from_dir`` reads them)."""
    for column in text_columns(table) & set(frame.columns):
        frame[column] = frame[column].fillna("")
    return frame


def _branch_data(branch_data: pd.DataFrame, bus_names: dict) -> pd.DataFrame:
    branch = branch_data.copy()
    if "from_name" not in branch.columns:
        branch.insert(2, "from_name", branch["fb"].map(bus_names))
    if "to_name" not in branch.columns:
        branch.insert(3, "to_name", branch["tb"].map(bus_names))
    return _fill_text(branch, "branch_data")


def _bus_data(bus_data: pd.DataFrame) -> pd.DataFrame:
    """Bus properties only: the load columns move to ``load_data``."""
    load_columns = [
        column
        for column in bus_data.columns
        if column[:3] in ("pl_", "ql_") or column in _BUS_LOAD_COLUMNS
    ]
    return _fill_text(bus_data.drop(columns=load_columns), "bus_data")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _text(value) -> str:
    """Plain text of a name; a float-parsed 151.0 becomes "151"."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _blank(value) -> bool:
    return pd.isna(value) or _text(value) in ("", "nan")


def _number(
    row: pd.Series, column: str, table: str, default: float | None = None
) -> float:
    """Numeric cell. A missing column or blank cell gives ``default``, or an error if None."""
    value = row.get(column)
    if value is None or pd.isna(value):
        if default is None:
            raise ValueError(f"{table} needs column {column!r}")
        return default
    return float(value)


def _bus_name(row: pd.Series, bus_names: dict, table: str) -> str:
    """Bus of a legacy device row: ``bus_name`` if present, else the bus ``id``."""
    if "bus_name" in row.index:
        return _text(row["bus_name"])
    if row["id"] not in bus_names:
        raise ValueError(f"{table}: bus id {row['id']} is not in bus_data")
    return bus_names[row["id"]]


def _base_name(row: pd.Series, bus_name: str) -> str:
    """Legacy device name: ``device_name`` or ``name``; the bus name if there is none."""
    for column in ("device_name", "name"):
        if column in row.index and not _blank(row[column]):
            return _text(row[column])
    return bus_name


def _device_frame(table: str, rows: list[dict]) -> pd.DataFrame:
    """Frame with exactly the contract columns; cells not given are "" (text) or NaN."""
    if not rows:
        return empty_device_table(table)
    columns, device_type = DEVICE_TABLES[table]
    frame = pd.DataFrame(
        [{"type": device_type, **row} for row in rows], columns=columns
    )
    frame = _fill_text(frame, table)
    names = frame["device_name"]
    if not names.is_unique:
        raise ValueError(
            f"{table}: duplicate device names {sorted(set(names[names.duplicated()]))}"
        )
    return frame


def _is_balanced(values_by_phase: dict[str, dict[str, float]]) -> bool:
    """True when every phase has the same value for every field."""
    first, *others = values_by_phase.values()
    return all(
        math.isclose(first[field], other[field], rel_tol=0, abs_tol=EQUAL_TOLERANCE)
        for other in others
        for field in first
    )


def _device_parts(
    phases: list[str], values_by_phase: dict[str, dict[str, float]], combine=sum
) -> list[tuple[str, str, dict[str, float]]]:
    """(phases, name suffix, values) of the GAD devices for one legacy device.

    A balanced device is one part whose values are ``combine`` of the per-phase
    values (``sum`` for powers, ``min`` for a tap that is equal anyway).
    """
    if _is_balanced(values_by_phase):
        fields = values_by_phase[phases[0]]
        totals = {f: combine(v[f] for v in values_by_phase.values()) for f in fields}
        return [("".join(phases), "", totals)]
    return [(ph, f".{ph}", values_by_phase[ph]) for ph in phases]


# ---------------------------------------------------------------------------
# Balanced device tables
# ---------------------------------------------------------------------------


def _gen_shape(shape, schedule_columns: list[str]) -> str:
    """A blank legacy shape means the "PV" schedule if there is one."""
    if not _blank(shape):
        return _text(shape)
    return DEFAULT_GEN_SHAPE if DEFAULT_GEN_SHAPE in schedule_columns else ""


def _gen_data(
    gen_data: pd.DataFrame, bus_names: dict, schedule_columns: list[str]
) -> pd.DataFrame:
    rows = []
    for _, legacy in gen_data.iterrows():
        bus_name = _bus_name(legacy, bus_names, "gen_data")
        name = _base_name(legacy, bus_name)
        phases = parse_phases(legacy["phases"])
        per_phase = {}
        for ph in phases:
            s_max = _number(legacy, f"s_{ph}_max", "gen_data")
            per_phase[ph] = {
                "p": _number(legacy, f"p_{ph}", "gen_data"),
                "q": _number(legacy, f"q_{ph}", "gen_data", 0.0),
                "s_max": s_max,
                "q_min": _number(legacy, f"q_{ph}_min", "gen_data", -s_max),
                "q_max": _number(legacy, f"q_{ph}_max", "gen_data", s_max),
            }
        control = (
            ""
            if _blank(legacy.get("control_variable"))
            else _text(legacy["control_variable"])
        )
        for part_phases, suffix, values in _device_parts(phases, per_phase):
            rows.append(
                {
                    "device_name": f"generator.{name}{suffix}",
                    "bus_name": bus_name,
                    "phases": part_phases,
                    **values,
                    "p_min": 0.0,
                    "p_max": values["p"],
                    "control_variable": control,
                    "gen_shape": _gen_shape(legacy.get("gen_shape"), schedule_columns),
                    "cost": _number(legacy, "cost", "gen_data", 0.0),
                }
            )
    return _device_frame("gen_data", rows)


def _cap_data(cap_data: pd.DataFrame, bus_names: dict) -> pd.DataFrame:
    rows = []
    for _, legacy in cap_data.iterrows():
        bus_name = _bus_name(legacy, bus_names, "cap_data")
        name = _base_name(legacy, bus_name)
        phases = parse_phases(legacy["phases"])
        per_phase = {ph: {"q": _number(legacy, f"q_{ph}", "cap_data")} for ph in phases}
        for part_phases, suffix, values in _device_parts(phases, per_phase):
            rows.append(
                {
                    "device_name": f"capacitor.{name}{suffix}",
                    "bus_name": bus_name,
                    "phases": part_phases,
                    "q": values["q"],
                    "status": 1,
                }
            )
    return _device_frame("cap_data", rows)


def _reg_data(reg_data: pd.DataFrame, bus_names: dict) -> pd.DataFrame:
    rows = []
    for _, legacy in reg_data.iterrows():
        from_name = (
            _text(legacy["from_name"])
            if "from_name" in legacy.index
            else bus_names[legacy["fb"]]
        )
        to_name = (
            _text(legacy["to_name"])
            if "to_name" in legacy.index
            else bus_names[legacy["tb"]]
        )
        phases = parse_phases(legacy["phases"])
        per_phase = {
            ph: {"tap": _number(legacy, f"tap_{ph}", "reg_data", 0.0)} for ph in phases
        }
        for part_phases, suffix, values in _device_parts(
            phases, per_phase, combine=min
        ):
            rows.append(
                {
                    "device_name": f"regulator.{to_name}{suffix}",
                    "from_name": from_name,
                    "to_name": to_name,
                    "phases": part_phases,
                    "fb": legacy["fb"],
                    "tb": legacy["tb"],
                    "tap": int(values["tap"]),
                    "ratio_min": _number(legacy, "ratio_min", "reg_data", 0.9),
                    "ratio_max": _number(legacy, "ratio_max", "reg_data", 1.1),
                    "num_taps": _number(legacy, "num_taps", "reg_data", 32),
                }
            )
    return _device_frame("reg_data", rows)


# ---------------------------------------------------------------------------
# Batteries and loads
# ---------------------------------------------------------------------------


def _bat_data(bat_data: pd.DataFrame, bus_names: dict) -> pd.DataFrame:
    rows = []
    for _, legacy in bat_data.iterrows():
        bus_name = _bus_name(legacy, bus_names, "bat_data")
        s_max = _number(legacy, "s_max", "bat_data")
        control = (
            "P"
            if "control_variable" not in legacy.index
            else (
                ""
                if _blank(legacy["control_variable"])
                else _text(legacy["control_variable"])
            )
        )
        rows.append(
            {
                "device_name": f"battery.{_base_name(legacy, bus_name)}",
                "bus_name": bus_name,
                "phases": _text(legacy["phases"]),
                "s_max": s_max,
                "energy_capacity": _number(legacy, "energy_capacity", "bat_data"),
                "p_max": s_max,
                "p_charge_max": s_max,
                "p_discharge_max": s_max,
                "min_soc": _number(legacy, "min_soc", "bat_data", 0.0),
                "max_soc": _number(legacy, "max_soc", "bat_data", 1.0),
                "start_soc": _number(legacy, "start_soc", "bat_data", 0.5),
                "charge_efficiency": _number(
                    legacy, "charge_efficiency", "bat_data", 1.0
                ),
                "discharge_efficiency": _number(
                    legacy, "discharge_efficiency", "bat_data", 1.0
                ),
                "q_min": _number(legacy, "q_min", "bat_data", -s_max),
                "q_max": _number(legacy, "q_max", "bat_data", s_max),
                "p": _number(legacy, "p", "bat_data", 0.0),
                "q": _number(legacy, "q", "bat_data", 0.0),
                "control_variable": control,
                "annual_cycle_limit": _number(
                    legacy, "annual_cycle_limit", "bat_data", 365
                ),
            }
        )
    return _device_frame("bat_data", rows)


def _load_shape(shape, schedule_columns: list[str]) -> str:
    """The bus shape if it scales anything in the schedules, else "" (multiplier 1).

    A blank shape means "default". Legacy loads also read ``<shape>.p`` and
    ``<shape>.<phase>.p`` style columns, so a prefix match counts
    (``_schedules`` then renames ``<shape>.p`` to ``<shape>``).
    """
    shape = DEFAULT_LOAD_SHAPE if _blank(shape) else _text(shape)
    if any(c == shape or c.startswith(f"{shape}.") for c in schedule_columns):
        return shape
    return ""


def _schedules(schedules: pd.DataFrame, load_data: pd.DataFrame) -> pd.DataFrame:
    """Copy of the schedules where a load shape's ``<shape>.p`` column is named ``<shape>``.

    GAD convention: ``<shape>`` is the P multiplier and ``<shape>.q`` the Q multiplier.
    Only shapes used by ``load_data`` are renamed (other names may contain dots).
    """
    renames = {}
    for shape in sorted(set(load_data["load_shape"]) - {""}):
        if f"{shape}.p" not in schedules.columns:
            continue
        if shape in schedules.columns:
            raise ValueError(
                f"schedules has both {shape!r} and {shape + '.p'!r}; keep only one P column"
            )
        renames[f"{shape}.p"] = shape
    return schedules.rename(columns=renames)


def _load_data(bus_data: pd.DataFrame, schedule_columns: list[str]) -> pd.DataFrame:
    rows = []
    for _, bus in bus_data.iterrows():
        values = {}
        for ph in parse_phases(bus["phases"]):
            p = _number(bus, f"pl_{ph}", "bus_data", 0.0)
            q = _number(bus, f"ql_{ph}", "bus_data", 0.0)
            if ph in ("s1", "s2"):
                p += _number(bus, "pl_s1s2", "bus_data", 0.0) / 2
                q += _number(bus, "ql_s1s2", "bus_data", 0.0) / 2
            if p or q:
                values[ph] = (p, q)
        if not values:
            continue
        name = _text(bus["name"])
        row = {
            "device_name": f"load.{name}",
            "bus_name": name,
            "phases": "".join(ph for ph in LOAD_PHASES if ph in values),
            "cvr_p": _number(bus, "cvr_p", "bus_data", 0.0),
            "cvr_q": _number(bus, "cvr_q", "bus_data", 0.0),
            "load_shape": _load_shape(bus.get("load_shape"), schedule_columns),
            "status": "variable",
        }
        for ph in LOAD_PHASES:
            row[f"p_{ph}"], row[f"q_{ph}"] = values.get(ph, (0.0, 0.0))
        rows.append(row)
    return _device_frame("load_data", rows)
