"""GadCase -> legacy ``distopf.Case`` for one time step, so FBS can solve it.

``to_fbs_case(gad_case, t, setpoints)`` folds everything that varies in time into a
single-step case without schedules:

* loads: ``p_{ph}``/``q_{ph}`` times the schedule value at ``t`` (the ``<shape>.q``
  column scales Q when it exists). ``status == "fixed"`` and blank shapes are not
  scaled. Devices are summed per bus-phase into ``pl_*``/``ql_*``. FBS uses the linear
  CVR model ``p = p0 (1 + cvr/2 (v^2 - 1))``, so the power-weighted CVR of the devices
  on a bus-phase is exact. It is written to the ``cvr_p_{ph}``/``cvr_q_{ph}`` columns.
* generators and PV: total power split equally over the device phases, summed per bus.
* batteries: same, taking the power from the case or the setpoints.
* capacitors: ``q`` of the devices that are on, split equally over the phases.
* regulators: one legacy row per ``(fb, tb)`` with a tap per phase.

``setpoints`` (optional) overrides device values. It maps a table name to a frame
indexed by ``device_name``: ``p``/``q`` for ``pv_data``, ``gen_data``, ``bat_data``,
``u`` (0/1) for ``cap_data`` and ``tap`` for ``reg_data``. Devices or columns that
are missing keep the case value.
"""

from __future__ import annotations

import re
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from distopf.api import Case
from distopf.gad_models.common.case import GadCase

_PHASES = ("a", "b", "c", "s1", "s2")
_SWING_TYPES = ("SWING", "SWING_FREE", "IN")
_PHASE_TOKEN = re.compile(r"s1|s2|[abc]")

Setpoints = Mapping[str, pd.DataFrame]


def to_fbs_case(
    gad_case: GadCase, t: int, setpoints: Optional[Setpoints] = None
) -> Case:
    """Legacy single-step case with the device values of time step ``t``."""
    setpoints = setpoints or {}
    bus_data = gad_case.bus_data.copy()
    bus_phases = dict(zip(bus_data["name"].astype(str), bus_data["phases"].astype(str)))
    bus_ids = dict(zip(bus_data["name"].astype(str), bus_data["id"].astype(int)))

    bus_data = _with_scheduled_swing_voltage(bus_data, gad_case.schedules, t)
    bus_data = _with_loads(bus_data, gad_case, t, bus_phases)

    injections = _pv_injections(gad_case, t, setpoints)
    injections += _gen_injections(gad_case, t, setpoints)
    batteries = _bat_injections(gad_case, setpoints)
    capacitors = _cap_injections(gad_case, setpoints)

    return Case(
        branch_data=gad_case.branch_data.copy(),
        bus_data=bus_data,
        gen_data=_injection_table(injections, bus_ids, bus_phases, "gen"),
        bat_data=_injection_table(batteries, bus_ids, bus_phases, "bat"),
        cap_data=_cap_table(capacitors, bus_ids, bus_phases),
        reg_data=_reg_table(gad_case.reg_data, setpoints),
        ignore_schedule=True,
        start_step=0,
        n_steps=1,
        delta_t=gad_case.delta_t,
    )


# ---------------------------------------------------------------- helpers


def _phases_of(device_name: str, phases: str) -> list[str]:
    found = _PHASE_TOKEN.findall(str(phases).lower())
    if not found:
        raise ValueError(f"{device_name}: no phases in {phases!r}")
    return found


def _cell(row: Mapping, column: str, default: float) -> float:
    """Numeric cell, or ``default`` when the column is missing or blank."""
    value = row.get(column, np.nan)
    return default if pd.isna(value) or value == "" else float(value)


def _setpoint(
    setpoints: Setpoints, table: str, device: str, column: str, default: float
) -> float:
    """Setpoint of one device, or ``default`` when it has none."""
    frame = setpoints.get(table)
    if frame is None or device not in frame.index or column not in frame.columns:
        return default
    value = frame.at[device, column]
    return default if pd.isna(value) else float(value)


def _schedule_value(schedules: pd.DataFrame, t: int, column: str) -> float:
    rows = schedules[schedules["time"] == t]
    if rows.empty:
        raise ValueError(f"time step {t} is not in schedules")
    value = pd.to_numeric(rows[column].iloc[0], errors="coerce")
    if not np.isfinite(value):
        raise ValueError(f"schedule {column!r} is not a finite number at time step {t}")
    return float(value)


def _multiplier(schedules: pd.DataFrame, t: int, shape: str, kind: str = "p") -> float:
    """Value of ``shape`` at ``t``. Q uses ``<shape>.q`` when it exists. Blank shape: 1."""
    shape = str(shape).strip().lower()
    if not shape:
        return 1.0
    column = (
        f"{shape}.q" if kind == "q" and f"{shape}.q" in schedules.columns else shape
    )
    if column not in schedules.columns:
        raise ValueError(f"schedule {column!r} is not a column of schedules")
    return _schedule_value(schedules, t, column)


def _check_bus_phases(
    device: str, bus: str, phases: list[str], bus_phases: Mapping[str, str]
) -> None:
    if bus not in bus_phases:
        raise ValueError(f"{device}: bus {bus!r} is not in bus_data")
    missing = [ph for ph in phases if ph not in _phases_of(bus, bus_phases[bus])]
    if missing:
        raise ValueError(
            f"{device}: phases {missing} are not phases of bus {bus!r} ({bus_phases[bus]})"
        )


def _with_scheduled_swing_voltage(
    bus_data: pd.DataFrame, schedules: pd.DataFrame, t: int
) -> pd.DataFrame:
    """Swing bus voltages come from the ``v_a``/``v_b``/``v_c`` schedule columns, if any."""
    swing = bus_data["bus_type"].isin(_SWING_TYPES)
    for phase in "abc":
        column = f"v_{phase}"
        rows = schedules[schedules["time"] == t]
        if (
            column in schedules.columns
            and not rows.empty
            and pd.notna(rows[column].iloc[0])
        ):
            bus_data.loc[swing, column] = float(rows[column].iloc[0])
    return bus_data


# ------------------------------------------------------------------ loads


def _with_loads(
    bus_data: pd.DataFrame, gad_case: GadCase, t: int, bus_phases: Mapping[str, str]
) -> pd.DataFrame:
    """Add ``pl_*``, ``ql_*`` and per-phase ``cvr_*`` columns summed over the load devices."""
    schedules = gad_case.schedules
    keys = [(kind, ph) for kind in ("p", "q") for ph in _PHASES]
    power = {key: {} for key in keys}  # (kind, phase) -> bus -> sum of scaled power
    cvr_weighted = {key: {} for key in keys}  # same, sum of power * cvr

    for row in gad_case.load_data.to_dict("records"):
        device, bus = row["device_name"], str(row["bus_name"])
        phases = _phases_of(device, row["phases"])
        _check_bus_phases(device, bus, phases, bus_phases)
        fixed = str(row.get("status", "")).strip().lower() == "fixed"
        shape = "" if fixed else row.get("load_shape", "")
        for kind in ("p", "q"):
            scale = _multiplier(schedules, t, shape, kind)
            cvr = _cell(row, f"cvr_{kind}", 0.0)
            for phase in phases:
                value = _cell(row, f"{kind}_{phase}", 0.0) * scale
                power[kind, phase][bus] = power[kind, phase].get(bus, 0.0) + value
                cvr_weighted[kind, phase][bus] = (
                    cvr_weighted[kind, phase].get(bus, 0.0) + value * cvr
                )

    names = bus_data["name"].astype(str)
    for kind, load_prefix in (("p", "pl"), ("q", "ql")):
        for phase in _PHASES:
            total = names.map(power[kind, phase]).fillna(0.0)
            weighted = names.map(cvr_weighted[kind, phase]).fillna(0.0)
            bus_data[f"{load_prefix}_{phase}"] = total
            bus_data[f"cvr_{kind}_{phase}"] = np.where(
                total != 0, weighted / total.where(total != 0, 1.0), 0.0
            )
        bus_data[f"{load_prefix}_s1s2"] = 0.0
        bus_data[f"cvr_{kind}"] = 0.0
    return bus_data


# ------------------------------------- generators, PV, batteries, capacitors

# One injection: (device name, bus name, phases, total p, total q).
_Injection = tuple[str, str, list[str], float, float]


def _pv_injections(gad_case: GadCase, t: int, setpoints: Setpoints) -> list[_Injection]:
    out = []
    for row in gad_case.pv_data.to_dict("records"):
        device = row["device_name"]
        pmpp = float(row["pmpp"])
        available = (
            pmpp
            * _cell(row, "irradiance", 1.0)
            * _multiplier(gad_case.schedules, t, row.get("pv_shape", ""))
        )
        p = min(available, _cell(row, "pmpp_limit", 1.0) * pmpp)
        out.append(
            (
                device,
                str(row["bus_name"]),
                _phases_of(device, row["phases"]),
                _setpoint(setpoints, "pv_data", device, "p", p),
                _setpoint(setpoints, "pv_data", device, "q", _cell(row, "q", 0.0)),
            )
        )
    return out


def _gen_injections(
    gad_case: GadCase, t: int, setpoints: Setpoints
) -> list[_Injection]:
    out = []
    for row in gad_case.gen_data.to_dict("records"):
        device = row["device_name"]
        p = float(row["p"]) * _multiplier(
            gad_case.schedules, t, row.get("gen_shape", "")
        )
        out.append(
            (
                device,
                str(row["bus_name"]),
                _phases_of(device, row["phases"]),
                _setpoint(setpoints, "gen_data", device, "p", p),
                _setpoint(setpoints, "gen_data", device, "q", _cell(row, "q", 0.0)),
            )
        )
    return out


def _bat_injections(gad_case: GadCase, setpoints: Setpoints) -> list[_Injection]:
    out = []
    for row in gad_case.bat_data.to_dict("records"):
        device = row["device_name"]
        out.append(
            (
                device,
                str(row["bus_name"]),
                _phases_of(device, row["phases"]),
                _setpoint(setpoints, "bat_data", device, "p", _cell(row, "p", 0.0)),
                _setpoint(setpoints, "bat_data", device, "q", _cell(row, "q", 0.0)),
            )
        )
    return out


def _cap_injections(gad_case: GadCase, setpoints: Setpoints) -> list[_Injection]:
    """Capacitors as injections with p = 0 and q = q of the devices that are on."""
    out = []
    for row in gad_case.cap_data.to_dict("records"):
        device = row["device_name"]
        phases = _phases_of(device, row["phases"])
        if any(phase not in "abc" for phase in phases):
            raise ValueError(
                f"{device}: FBS supports capacitors on phases a, b, c only"
            )
        on = _setpoint(setpoints, "cap_data", device, "u", _cell(row, "status", 1.0))
        out.append((device, str(row["bus_name"]), phases, 0.0, float(row["q"]) * on))
    return out


def _sum_per_bus_phase(
    injections: list[_Injection],
    bus_phases: Mapping[str, str],
) -> dict[str, dict[str, list[float]]]:
    """bus -> phase -> [p, q], each device split equally over its phases."""
    sums: dict[str, dict[str, list[float]]] = {}
    for device, bus, phases, p, q in injections:
        _check_bus_phases(device, bus, phases, bus_phases)
        for phase in phases:
            pq = sums.setdefault(bus, {}).setdefault(phase, [0.0, 0.0])
            pq[0] += p / len(phases)
            pq[1] += q / len(phases)
    return sums


def _injection_table(
    injections: list[_Injection],
    bus_ids: Mapping[str, int],
    bus_phases: Mapping[str, str],
    kind: str,
) -> Optional[pd.DataFrame]:
    """Legacy per-phase ``gen_data`` or ``bat_data``: one row per bus. ``None`` if empty."""
    if not injections:
        return None
    rows = []
    for bus, by_phase in _sum_per_bus_phase(injections, bus_phases).items():
        row = {"id": bus_ids[bus], "name": bus}
        for phase in _PHASES:
            p, q = by_phase.get(phase, (0.0, 0.0))
            row[f"p_{phase}"], row[f"q_{phase}"] = p, q
        row["phases"] = "".join(ph for ph in _PHASES if ph in by_phase)
        row["control_variable"] = ""
        if kind == "gen":
            row["gen_shape"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def _cap_table(
    injections: list[_Injection],
    bus_ids: Mapping[str, int],
    bus_phases: Mapping[str, str],
) -> Optional[pd.DataFrame]:
    if not injections:
        return None
    rows = []
    for bus, by_phase in _sum_per_bus_phase(injections, bus_phases).items():
        row = {"id": bus_ids[bus], "name": bus}
        for phase in "abc":
            row[f"q_{phase}"] = by_phase.get(phase, (0.0, 0.0))[1]
        row["phases"] = "".join(ph for ph in "abc" if ph in by_phase)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------- regulators


def _reg_table(reg_data: pd.DataFrame, setpoints: Setpoints) -> Optional[pd.DataFrame]:
    """One legacy row per ``(fb, tb)``; each device sets the taps of its own phases."""
    if reg_data.empty:
        return None
    rows: dict[tuple[int, int], dict] = {}
    owner: dict[tuple[int, int, str], str] = {}
    for device_row in reg_data.to_dict("records"):
        device = device_row["device_name"]
        branch = (int(device_row["fb"]), int(device_row["tb"]))
        phases = _phases_of(device, device_row["phases"])
        if any(phase not in "abc" for phase in phases):
            raise ValueError(
                f"{device}: FBS supports regulators on phases a, b, c only"
            )
        ratio_range = {
            "ratio_min": _cell(device_row, "ratio_min", 0.9),
            "ratio_max": _cell(device_row, "ratio_max", 1.1),
            "num_taps": _cell(device_row, "num_taps", 32),
        }
        row = rows.setdefault(
            branch,
            {
                "fb": branch[0],
                "tb": branch[1],
                **ratio_range,
                "tap_a": 0.0,
                "tap_b": 0.0,
                "tap_c": 0.0,
            },
        )
        if any(row[key] != value for key, value in ratio_range.items()):
            raise ValueError(
                f"{device}: regulators on branch {branch} must share ratio_min, ratio_max and num_taps"
            )
        tap = _setpoint(
            setpoints, "reg_data", device, "tap", _cell(device_row, "tap", 0.0)
        )
        for phase in phases:
            if (*branch, phase) in owner:
                raise ValueError(
                    f"{device} and {owner[*branch, phase]} both regulate phase {phase} of branch {branch}"
                )
            owner[*branch, phase] = device
            row[f"tap_{phase}"] = tap
    for (fb, tb), row in rows.items():
        row["phases"] = "".join(ph for ph in "abc" if (fb, tb, ph) in owner)
    return pd.DataFrame(list(rows.values()))
