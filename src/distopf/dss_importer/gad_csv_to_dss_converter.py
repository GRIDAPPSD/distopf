"""
GAD case -> OpenDSS writer.

Writes a single .dss file from a ``GadCase`` at one time step, with one OpenDSS
element per GAD device, so power flow solutions can be compared to FBS and
OpenDSS read-back values can be compared to the GAD device values.

Network (same as ``csv_to_dss_converter``)
------------------------------------------
- swing bus        -> circuit Vsource (3-phase, voltage from ``v_a`` or the ``v_a`` schedule)
- lines/switches   -> Line with R/X matrices (ohms, length 1 km) and line charging cmatrix
- transformers     -> one single-phase Transformer per phase; regulated branches see below
- center_tap_xfmr  -> single-phase 3-winding Transformer, secondary ``sec.1.0`` / ``sec.0.2``
- triplex_line     -> 2-conductor Line on nodes 1 and 2 (s1, s2)

Devices (one OpenDSS element per device)
----------------------------------------
The element name is the part of ``device_name`` after the class prefix
(``storage.b1`` -> ``Storage.b1``). Balanced devices are multi-phase elements on
their phases, where OpenDSS also splits the total equally; ``kV`` is line-to-line
for more than one phase. Each ``device_results`` frame has an ``element`` column with
the written element name(s).

========== ============================================ =======================
table      OpenDSS element                              name
========== ============================================ =======================
gen_data   constant-PQ Generator                        ``<name>``
pv_data    constant-PQ Generator                        ``<name>``; prefixed
                                                        ``pvsystem_<name>`` if a
                                                        generator has that name
bat_data   Storage with fixed kW/kvar                   ``<name>``
cap_data   Capacitor, ``states=[1]`` or ``[0]``         ``<name>``
reg_data   Transformer with one winding-2 tap on all    ``<name>``
           its phases (replaces the branch's
           per-phase transformers for those phases)
load_data  Load; equal per-phase P and Q -> one         ``<name>``, or
           multi-phase Load, else one single-phase      ``<name>_<phase>``
           Load per phase
========== ============================================ =======================

Two elements of one OpenDSS class may not have the same name (``ValueError``).

Time step ``t`` (default ``gad_case.start_step``)
-------------------------------------------------
- A shape is a ``schedules`` column named like the lowercase shape name (the
  P multiplier); ``<shape>.q`` is the Q multiplier if that column exists, otherwise
  the P multiplier also scales Q. A blank shape, or a load with
  ``status == "fixed"``, is not scaled. A named shape without a column raises
  ``ValueError``. This holds for loads, generators and PV.
- Generator P is ``p * gen_shape(t)``; PV P is
  ``min(pmpp * irradiance * pv_shape(t), pmpp_limit * pmpp)``.
- A load with ``cvr_p`` or ``cvr_q`` nonzero is written with ``model=4``.
- The swing voltage uses the ``v_a`` schedule column if present.

Setpoints
---------
``setpoints`` maps table names ``gen_data``, ``pv_data``, ``bat_data`` (columns
``p``, ``q``), ``cap_data`` (``u``, on when above 0.5) and ``reg_data`` (``tap``) to
frames indexed by ``device_name``. They are the outputs at ``t`` and replace the
case values with no schedule on top; devices and cells that are missing keep the
case values.
"""

import re
import tempfile
from dataclasses import dataclass
from math import pi, sqrt
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from distopf.gad_models.common.case import GadCase
from distopf.gad_models.common.schema import DEVICE_TABLES
from distopf.results import PowerFlowResult

PHASE_NODE = {"a": 1, "b": 2, "c": 3, "s1": 1, "s2": 2}
LOAD_PHASES = ("a", "b", "c", "s1", "s2")
FREQUENCY_HZ = 60.0
SETPOINT_COLUMNS = {
    "gen_data": ("p", "q"),
    "pv_data": ("p", "q"),
    "bat_data": ("p", "q"),
    "cap_data": ("u",),
    "reg_data": ("tap",),
}


def _value(row: pd.Series, column: str) -> float:
    """Numeric column value; missing or NaN counts as 0."""
    value = row.get(column, 0.0)
    return 0.0 if pd.isna(value) else float(value)


def _text(row: pd.Series, column: str) -> str:
    value = row.get(column, "")
    return "" if pd.isna(value) else str(value).strip()


def _capacitance_nf(row: pd.Series):
    """Entry callable for the cmatrix (nF/km): diagonal from the line charging ``b_<phase>``."""

    def get(pi_: str, pj: str) -> float:
        if pi_ != pj:
            return 0.0
        return _value(row, f"b_{pi_}") / row["z_base"] / (2 * pi * FREQUENCY_HZ) * 1e9

    return get


def _core_shunt(row: pd.Series, phase: str) -> str:
    """No-load loss and magnetizing current as percent of kVA (kVA equals s_base)."""
    g, b = _value(row, f"g_sh_{phase}"), _value(row, f"b_sh_{phase}")
    return f"%noloadloss={g * 100:.12g} %imag={b * 100:.12g}"


def _split_phases(phases: str) -> list[str]:
    """'abc' -> ['a','b','c'], 's1s2' -> ['s1','s2']."""
    return re.findall(r"s1|s2|[abc]", str(phases).lower())


def _nodes(phases: list[str]) -> str:
    """Node suffix of a bus spec: ['a', 'c'] -> '1.3'."""
    return ".".join(str(PHASE_NODE[p]) for p in phases)


def _node(bus: str, phase: str) -> str:
    return f"{bus}.{PHASE_NODE[phase]}"


def _kv(kv_ln: float, n_phases: int) -> float:
    """Element ``kV``: line-to-neutral for one phase, line-to-line for more."""
    return kv_ln * (sqrt(3) if n_phases > 1 else 1)


def _matrix_str(entry, phases: list[str]) -> str:
    """Lower-triangular OpenDSS matrix string, e.g. '[1 | 0 2]'."""
    rows = []
    for i, phase_i in enumerate(phases):
        values = [f"{entry(phase_i, phases[j]):.12g}" for j in range(i + 1)]
        rows.append(" ".join(values))
    return "[" + " | ".join(rows) + "]"


def _line(row: pd.Series, bus_name: dict) -> str:
    """Line for abc branches and triplex (s1, s2) branches; impedances are per-unit."""
    phases = _split_phases(row["phases"])

    def entry(prefix: str):
        def get(pi: str, pj: str) -> float:
            for key in (f"{prefix}_{pi}{pj}", f"{prefix}_{pj}{pi}"):
                if key in row.index and pd.notna(row[key]):
                    return row[key] * row["z_base"]
            return 0.0

        return get

    return (
        f"New Line.{row['name']}_{int(row['fb'])}_{int(row['tb'])} phases={len(phases)} "
        f"bus1={bus_name[row['fb']]}.{_nodes(phases)} bus2={bus_name[row['tb']]}.{_nodes(phases)} "
        f"length=1 units=km rmatrix={_matrix_str(entry('r'), phases)} "
        f"xmatrix={_matrix_str(entry('x'), phases)} "
        f"cmatrix={_matrix_str(_capacitance_nf(row), phases)}"
    )


def _center_tap_transformer(row: pd.Series, bus_name: dict, bus_kv: dict) -> str:
    """Split-phase transformer: primary phase to secondary s1 (node 1) and s2 (node 2).

    The branch matrix is ``[[zp+z1, -zp], [-zp, zp+z2]]`` referred to the secondary, so
    ``zp = -z_s1s2``. Windings 1/2/3 are primary/s1/s2 with a star-equivalent impedance.
    """
    kva = row["s_base"] / 1e3
    kv_sec = bus_kv[row["tb"]]
    z_xfmr_base = (kv_sec * 1e3) ** 2 / row["s_base"]

    def z_star(prefix: str) -> tuple[float, float, float]:
        """Primary, s1, s2 star-leg values in percent for the r or x columns."""
        scale = row["z_base"] / z_xfmr_base * 100
        zp = -_value(row, f"{prefix}_s1s2")
        return (
            zp * scale,
            (_value(row, f"{prefix}_s1s1") - zp) * scale,
            (_value(row, f"{prefix}_s2s2") - zp) * scale,
        )

    primary_phase = row["primary_phase"]
    primary = _node(bus_name[row["fb"]], primary_phase)
    secondary = bus_name[row["tb"]]
    r_p, r_1, r_2 = z_star("r")
    x_p, x_1, x_2 = z_star("x")
    return (
        f"New Transformer.{row['name']}_{int(row['fb'])}_{int(row['tb'])} phases=1 windings=3 "
        f"buses=[{primary}.0, {secondary}.1.0, {secondary}.0.2] "
        f"conns=[wye, wye, wye] kVs=[{bus_kv[row['fb']]:.12g}, {kv_sec:.12g}, {kv_sec:.12g}] "
        f"kVAs=[{kva:.12g}, {kva:.12g}, {kva:.12g}] "
        f"%rs=[{r_p:.12g}, {r_1:.12g}, {r_2:.12g}] "
        f"xhl={x_p + x_1:.12g} xht={x_p + x_2:.12g} xlt={x_1 + x_2:.12g} "
        f"{_core_shunt(row, primary_phase)}"
    )


# ------------------------------------------------------------ transformers
@dataclass
class _Transformer:
    """One OpenDSS Transformer on a transformer branch: its phases and winding-2 tap ratio."""

    element: str
    phases: list[str]
    ratio: float


def _same_for_phases(row: pd.Series, columns: list[str], element: str) -> float:
    """The common value of per-phase ``columns``; unequal phases cannot be one element."""
    values = [_value(row, c) for c in columns]
    if max(values) - min(values) > 1e-9 * max(1.0, max(abs(v) for v in values)):
        raise ValueError(
            f"{element}: phase impedances differ ({columns}); "
            "split the regulator into one single-phase device per phase"
        )
    return values[0]


def _transformer_line(
    element: str,
    row: pd.Series,
    transformer: _Transformer,
    bus_name: dict,
    bus_kv: dict,
) -> str:
    """Wye-wye Transformer over ``transformer.phases`` (one or more), same on every phase."""
    phases = transformer.phases
    n = len(phases)
    r = _same_for_phases(row, [f"r_{p}{p}" for p in phases], element)
    x = _same_for_phases(row, [f"x_{p}{p}" for p in phases], element)
    shunt = {
        k: _same_for_phases(row, [f"{k}_sh_{p}" for p in phases], element)
        for k in ("g", "b")
    }
    nodes = _nodes(phases) + ".0"
    kva = n * row["s_base"] / 1e3
    kv_from, kv_to = _kv(bus_kv[row["fb"]], n), _kv(bus_kv[row["tb"]], n)
    return (
        f"New Transformer.{element} phases={n} windings=2 "
        f"buses=[{bus_name[row['fb']]}.{nodes}, {bus_name[row['tb']]}.{nodes}] "
        f"conns=[wye, wye] kVs=[{kv_from:.12g}, {kv_to:.12g}] "
        f"kVAs=[{kva:.12g}, {kva:.12g}] "
        f"%rs=[{r * 50:.12g}, {r * 50:.12g}] xhl={x * 100:.12g} "
        f"%noloadloss={shunt['g'] * 100:.12g} %imag={shunt['b'] * 100:.12g} "
        f"taps=[1, {transformer.ratio:.12g}]"
    )


def _branch_transformers(
    row: pd.Series, regulators: list[_Transformer]
) -> list[_Transformer]:
    """Transformers of a transformer branch: its regulators plus per-phase ones for the rest."""
    covered = [p for reg in regulators for p in reg.phases]
    if len(covered) != len(set(covered)):
        raise ValueError(
            f"regulators {[r.element for r in regulators]} overlap on branch {row['name']!r}"
        )
    base = f"{row['name']}_{int(row['fb'])}_{int(row['tb'])}"
    rest = [
        _Transformer(f"{base}_{p}", [p], 1.0)
        for p in _split_phases(row["phases"])
        if p not in covered
    ]
    return [*regulators, *rest]


# ------------------------------------------------------------- schedules
class _Schedule:
    """Schedule columns at one time step, looked up by lowercase shape name."""

    def __init__(self, schedules: pd.DataFrame, t: int):
        self.t = t
        if "time" in schedules.columns:
            rows = schedules[schedules["time"] == t]
        else:
            rows = schedules.loc[schedules.index.intersection([t])]
        if len(schedules) and not len(rows):
            raise ValueError(f"schedules has no row for time step {t}")
        values = rows.iloc[0] if len(rows) else pd.Series(dtype=float)
        self.values = {
            str(column).lower(): pd.to_numeric(value, errors="coerce")
            for column, value in values.items()
            if column != "time"
        }

    def column(self, name: str) -> Optional[float]:
        """Value of a column (case-insensitive), None when there is no such column."""
        value = self.values.get(name.lower())
        return None if value is None else (1.0 if pd.isna(value) else float(value))

    def multipliers(self, shape: str) -> tuple[float, float]:
        """(P, Q) multipliers of a shape: Q is ``<shape>.q`` if present, else the P one."""
        if not shape:
            return 1.0, 1.0
        p_mult = self.column(shape)
        if p_mult is None:
            raise ValueError(f"shape {shape!r} is not a column of schedules")
        q_mult = self.column(f"{shape}.q")
        return p_mult, p_mult if q_mult is None else q_mult


# ------------------------------------------------------- elements and setpoints
@dataclass
class _Device:
    """A GAD device and the OpenDSS elements written for it, ``(element, phases)`` each."""

    table: str
    device_name: str
    elements: list[tuple[str, list[str]]]
    bus_name: str
    kva_base: float
    step: float = 0.0  # regulators only: tap step in ratio per tap


def _element_name(device_name: str) -> str:
    """Part of ``device_name`` after the class prefix: 'pvsystem.pv1' -> 'pv1'."""
    return device_name.split(".", 1)[-1]


def _check_setpoints(gad_case: GadCase, setpoints: dict) -> None:
    for table, frame in setpoints.items():
        if table not in SETPOINT_COLUMNS:
            raise ValueError(
                f"setpoints table {table!r} is not supported; use {sorted(SETPOINT_COLUMNS)}"
            )
        if not set(SETPOINT_COLUMNS[table]) & set(frame.columns):
            raise ValueError(
                f"setpoints[{table!r}] needs a column in {list(SETPOINT_COLUMNS[table])}"
            )
        unknown = set(frame.index) - set(getattr(gad_case, table)["device_name"])
        if unknown:
            raise ValueError(
                f"setpoints[{table!r}] has unknown devices {sorted(unknown)}"
            )


def _setpoint(
    setpoints: dict, table: str, device_name: str, column: str, default: float
) -> float:
    """Setpoint of one device and column, ``default`` when not given or NaN."""
    frame = setpoints.get(table)
    if frame is None or column not in frame.columns or device_name not in frame.index:
        return default
    value = pd.to_numeric(frame.at[device_name, column], errors="coerce")
    return default if pd.isna(value) else float(value)


class _Writer:
    """Builds the lines of the .dss file and records the elements written per device."""

    def __init__(self, gad_case: GadCase, t: int, setpoints: dict):
        self.case = gad_case
        self.t = t
        self.setpoints = setpoints
        self.schedule = _Schedule(gad_case.schedules, t)
        bus = gad_case.bus_data
        self.bus = bus
        self.bus_name = dict(zip(bus["id"], bus["name"].astype(str)))
        self.bus_kv = dict(zip(bus["id"], bus["v_ln_base"] / 1e3))
        by_name = bus.assign(name=bus["name"].astype(str)).set_index("name")
        self.bus_row = by_name
        self.lines: list[str] = []
        self.devices: list[_Device] = []
        self.regulators: dict[tuple, list[_Transformer]] = {}
        self.used: set[tuple[str, str]] = set()

    # -- bookkeeping
    def claim(self, element_class: str, name: str) -> str:
        key = (element_class, name.lower())
        if key in self.used:
            raise ValueError(f"two {element_class} elements are named {name!r}")
        self.used.add(key)
        return name

    def add(self, table: str, row: pd.Series, elements: list[tuple[str, list[str]]]):
        bus_name = _text(row, "bus_name")
        kva_base = self.bus_row.loc[bus_name, "s_base"] / 1e3
        self.devices.append(
            _Device(table, row["device_name"], elements, bus_name, kva_base)
        )

    def bus_spec(self, row: pd.Series) -> tuple[str, list[str], float, float]:
        """(bus1 with nodes, phases, kV, kVA base) of a device row."""
        bus_name = _text(row, "bus_name")
        phases = _split_phases(row["phases"])
        bus = self.bus_row.loc[bus_name]
        kv = _kv(bus["v_ln_base"] / 1e3, len(phases))
        return f"{bus_name}.{_nodes(phases)}", phases, kv, bus["s_base"] / 1e3

    # -- network
    def write_network(self) -> None:
        swing = self.bus[self.bus["bus_type"] == "SWING"].iloc[0]
        v_swing = self.schedule.column("v_a")
        self.lines += [
            "Clear",
            f"New Circuit.distopf basekv={swing['v_ln_base'] * sqrt(3) / 1e3:.12g} "
            f"pu={swing['v_a'] if v_swing is None else v_swing:.12g} phases=3 "
            f"bus1={self.bus_name[swing['id']]} mvasc3=1e6 mvasc1=1e6",
        ]
        self.regulators = self.regulator_transformers()
        for _, row in self.case.branch_data.iterrows():
            if row["type"] == "transformer":
                key = (row["fb"], row["tb"])
                for transformer in _branch_transformers(
                    row, self.regulators.get(key, [])
                ):
                    self.claim("transformer", transformer.element)
                    self.lines.append(
                        _transformer_line(
                            transformer.element,
                            row,
                            transformer,
                            self.bus_name,
                            self.bus_kv,
                        )
                    )
            elif row["type"] == "center_tap_xfmr":
                self.lines.append(
                    _center_tap_transformer(row, self.bus_name, self.bus_kv)
                )
            else:
                self.lines.append(_line(row, self.bus_name))

    def regulator_transformers(self) -> dict[tuple, list[_Transformer]]:
        """Regulator devices as Transformers, by the (fb, tb) ids of their branch."""
        branch = self.case.branch_data
        branch_key = {
            (self.bus_name[r["fb"]], self.bus_name[r["tb"]]): (r["fb"], r["tb"])
            for _, r in branch[branch["type"] == "transformer"].iterrows()
        }
        by_branch: dict[tuple, list[_Transformer]] = {}
        for _, reg in self.case.reg_data.iterrows():
            name = reg["device_name"]
            key = branch_key.get((str(reg["from_name"]), str(reg["to_name"])))
            if key is None:
                raise ValueError(
                    f"regulator {name}: no transformer branch from "
                    f"{reg['from_name']!r} to {reg['to_name']!r}"
                )
            step = (reg["ratio_max"] - reg["ratio_min"]) / reg["num_taps"]
            tap = _setpoint(self.setpoints, "reg_data", name, "tap", reg["tap"])
            phases = _split_phases(reg["phases"])
            element = _element_name(name)
            by_branch.setdefault(key, []).append(
                _Transformer(element, phases, 1 + step * tap)
            )
            self.devices.append(
                _Device(
                    "reg_data",
                    name,
                    [(f"Transformer.{element}", phases)],
                    "",
                    0.0,
                    step,
                )
            )
        return by_branch

    # -- devices
    def write_devices(self) -> None:
        self.write_generators()
        self.write_pv()
        self.write_batteries()
        self.write_capacitors()
        self.write_loads()
        # A tight tolerance makes constant-PQ elements deliver their setpoints to ~1e-9 pu.
        self.lines += ["Set mode=snap tolerance=1e-9 maxiterations=100", "Solve"]
        self.lines += [
            f"SetkVBase bus={self.bus_name[i]} kvll={kv * sqrt(3):.12g}"
            for i, kv in self.bus_kv.items()
        ]

    def write_injector(
        self, table: str, row: pd.Series, name: str, p_pu: float, q_pu: float
    ) -> None:
        """Constant-PQ Generator for a gen or pv device with totals ``p_pu``, ``q_pu``."""
        bus1, phases, kv, scale = self.bus_spec(row)
        p_pu = _setpoint(self.setpoints, table, row["device_name"], "p", p_pu)
        q_pu = _setpoint(self.setpoints, table, row["device_name"], "q", q_pu)
        self.lines.append(
            f"New Generator.{name} phases={len(phases)} bus1={bus1} kV={kv:.12g} "
            f"kW={p_pu * scale:.12g} kvar={q_pu * scale:.12g} model=1"
        )
        self.add(table, row, [(f"Generator.{name}", phases)])

    def shaped(self, row: pd.Series, shape_column: str) -> tuple[float, float]:
        """(P multiplier, scaled Q) of a gen or pv row."""
        p_mult, q_mult = self.schedule.multipliers(_text(row, shape_column))
        return p_mult, _value(row, "q") * q_mult

    def write_generators(self) -> None:
        for _, row in self.case.gen_data.iterrows():
            p_mult, q = self.shaped(row, "gen_shape")
            name = self.claim("generator", _element_name(row["device_name"]))
            self.write_injector("gen_data", row, name, _value(row, "p") * p_mult, q)

    def write_pv(self) -> None:
        for _, row in self.case.pv_data.iterrows():
            pmpp = _value(row, "pmpp")
            p_mult, q = self.shaped(row, "pv_shape")
            available = min(
                pmpp * _value(row, "irradiance") * p_mult,
                _value(row, "pmpp_limit") * pmpp,
            )
            name = _element_name(row["device_name"])
            if ("generator", name.lower()) in self.used:
                name = f"pvsystem_{name}"
            self.write_injector(
                "pv_data", row, self.claim("generator", name), available, q
            )

    def write_batteries(self) -> None:
        for _, row in self.case.bat_data.iterrows():
            bus1, phases, kv, scale = self.bus_spec(row)
            name = self.claim("storage", _element_name(row["device_name"]))
            p = _setpoint(
                self.setpoints, "bat_data", row["device_name"], "p", _value(row, "p")
            )
            q = _setpoint(
                self.setpoints, "bat_data", row["device_name"], "q", _value(row, "q")
            )
            rating = max(_value(row, "s_max"), np.hypot(p, q)) * scale
            self.lines.append(
                f"New Storage.{name} phases={len(phases)} bus1={bus1} kV={kv:.12g} "
                f"kVA={rating:.12g} kWrated={rating:.12g} "
                f"kWhrated={_value(row, 'energy_capacity') * scale:.12g} "
                f"%stored={_value(row, 'start_soc') * 100:.12g} "
                f"%reserve={_value(row, 'min_soc') * 100:.12g} "
                f"%EffCharge={_value(row, 'charge_efficiency') * 100:.12g} "
                f"%EffDischarge={_value(row, 'discharge_efficiency') * 100:.12g} "
                f"%idlingkW=0 kW={p * scale:.12g} kvar={q * scale:.12g}"
            )
            self.add("bat_data", row, [(f"Storage.{name}", phases)])

    def write_capacitors(self) -> None:
        for _, row in self.case.cap_data.iterrows():
            bus1, phases, kv, scale = self.bus_spec(row)
            name = self.claim("capacitor", _element_name(row["device_name"]))
            status = _setpoint(
                self.setpoints,
                "cap_data",
                row["device_name"],
                "u",
                _value(row, "status"),
            )
            self.lines.append(
                f"New Capacitor.{name} phases={len(phases)} bus1={bus1} kV={kv:.12g} "
                f"kvar={_value(row, 'q') * scale:.12g} numsteps=1 "
                f"states=[{int(status > 0.5)}]"
            )
            self.add("cap_data", row, [(f"Capacitor.{name}", phases)])

    def write_loads(self) -> None:
        for _, row in self.case.load_data.iterrows():
            phases = _split_phases(row["phases"])
            bus_name = _text(row, "bus_name")
            kv_ln = self.bus_row.loc[bus_name, "v_ln_base"] / 1e3
            scale = self.bus_row.loc[bus_name, "s_base"] / 1e3
            if _text(row, "status").lower() == "fixed":
                p_mult, q_mult = 1.0, 1.0
            else:
                p_mult, q_mult = self.schedule.multipliers(_text(row, "load_shape"))
            p = {ph: _value(row, f"p_{ph}") * p_mult for ph in phases}
            q = {ph: _value(row, f"q_{ph}") * q_mult for ph in phases}
            cvr_p, cvr_q = _value(row, "cvr_p"), _value(row, "cvr_q")
            model = (
                f"model=4 cvrwatts={cvr_p:.12g} cvrvars={cvr_q:.12g}"
                if cvr_p or cvr_q
                else "model=1"
            )
            equal = (
                np.ptp(list(p.values())) < 1e-12 and np.ptp(list(q.values())) < 1e-12
            )
            # Equal phases are one multi-phase Load; unequal ones one single-phase Load each.
            groups = [phases] if equal else [[ph] for ph in phases]
            name = _element_name(row["device_name"])
            elements = []
            for group in groups:
                element = name if len(groups) == 1 else f"{name}_{group[0]}"
                self.claim("load", element)
                self.lines.append(
                    f"New Load.{element} phases={len(group)} "
                    f"bus1={bus_name}.{_nodes(group)} kV={_kv(kv_ln, len(group)):.12g} "
                    f"kW={sum(p[ph] for ph in group) * scale:.12g} "
                    f"kvar={sum(q[ph] for ph in group) * scale:.12g} "
                    f"{model} vminpu=0.5 vmaxpu=1.5"
                )
                elements.append((f"Load.{element}", group))
            self.add("load_data", row, elements)


def _build(gad_case: GadCase, t: Optional[int], setpoints: Optional[dict]):
    """The script lines and written devices of ``gad_case`` at time step ``t``."""
    setpoints = {} if setpoints is None else setpoints
    _check_setpoints(gad_case, setpoints)
    t = gad_case.start_step if t is None else t
    writer = _Writer(gad_case, t, setpoints)
    writer.write_network()
    writer.write_devices()
    return writer, t


def gad_case_to_dss(
    gad_case: GadCase,
    output_path: Union[str, Path],
    t: Optional[int] = None,
    setpoints: Optional[dict[str, pd.DataFrame]] = None,
) -> Path:
    """Write ``gad_case`` at time step ``t`` (default ``start_step``) to an OpenDSS file.

    ``setpoints`` replaces the device outputs, see the module docstring.
    """
    writer, _ = _build(gad_case, t, setpoints)
    output_path = Path(output_path)
    output_path.write_text("\n".join([*writer.lines, ""]))
    return output_path


# ---------------------------------------------------------------- read-back
def _complex_pairs(flat) -> np.ndarray:
    """OpenDSS returns complex arrays as interleaved real, imaginary values."""
    values = np.asarray(flat, dtype=float)
    return values[0::2] + 1j * values[1::2]


def _element_terminal(element: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Voltage (V), current (A) and power (kVA) of every conductor of ``element``."""
    from opendssdirect import dss

    dss.Circuit.SetActiveElement(element)
    return (
        _complex_pairs(dss.CktElement.Voltages()),
        _complex_pairs(dss.CktElement.Currents()),
        _complex_pairs(dss.CktElement.Powers()),
    )


def _branch_values(
    row: pd.Series, bus_v_base: dict, transformers: list[_Transformer]
) -> dict[str, tuple[complex, complex]]:
    """Per-unit (power, current) entering each phase of one branch, by phase name.

    Power is taken at terminal 1 (the from side) and current at terminal 2 (the branch
    current into the to bus), as FBS does. A center-tap transformer is reported like
    FBS: primary voltage times the s1/s2 secondary load currents, with the s2 power on
    the minus voltage reference.
    """
    s_base = row["s_base"]
    name = f"{row['name']}_{int(row['fb'])}_{int(row['tb'])}"
    values = {}
    if row["type"] == "center_tap_xfmr":
        v, i, _ = _element_terminal(f"Transformer.{name}")
        i_base = s_base / bus_v_base[row["tb"]]
        v_primary = v[0] / bus_v_base[row["fb"]]
        i_s1, i_s2 = -i[2] / i_base, -i[5] / i_base
        values["s1"] = (v_primary * np.conj(i_s1), i_s1)
        values["s2"] = (-v_primary * np.conj(i_s2), i_s2)
        return values

    if row["type"] == "transformer":
        elements = [(f"Transformer.{t.element}", t.phases) for t in transformers]
    else:
        elements = [(f"Line.{name}", _split_phases(row["phases"]))]
    i_base = s_base / bus_v_base[row["tb"]]
    for element, phases in elements:
        _, i, s = _element_terminal(element)
        terminal_2 = len(i) // 2
        for k, phase in enumerate(phases):
            values[phase] = (s[k] * 1e3 / s_base, -i[terminal_2 + k] / i_base)
    return values


def _solved_bus_rows(bus: pd.DataFrame, t: int) -> tuple[list[dict], list[dict]]:
    """Per-unit voltage magnitude and angle (degrees) rows of the solved circuit."""
    from opendssdirect import dss

    magnitudes, angles = [], []
    secondary = bus["phases"].astype(str).str.contains("s").to_numpy()
    for (i, name), is_sec in zip(bus[["id", "name"]].values, secondary):
        mag_row = {"id": i, "name": name, "t": t, **dict.fromkeys(PHASE_NODE, np.nan)}
        ang_row = dict(mag_row)
        dss.Circuit.SetActiveBus(str(name))
        phases = ("s1", "s2") if is_sec else ("a", "b", "c")
        pu_mag_ang = dss.Bus.puVmagAngle()
        for node, mag, ang in zip(dss.Bus.Nodes(), pu_mag_ang[0::2], pu_mag_ang[1::2]):
            if 1 <= node <= len(phases):
                phase = phases[node - 1]
                mag_row[phase] = mag
                # s2 sits on the minus reference (secondary.0.2); FBS reports it flipped.
                ang_row[phase] = (ang + 360) % 360 - 180 if phase == "s2" else ang
        magnitudes.append(mag_row)
        angles.append(ang_row)
    return magnitudes, angles


def _solved_branch_rows(writer: _Writer, t: int) -> dict[str, list[dict]]:
    """Per-unit branch rows (P, Q, current magnitude, current angle) in the FBS layout."""
    case = writer.case
    bus_v_base = dict(zip(writer.bus["id"], writer.bus["v_ln_base"]))
    tables = {"p": [], "q": [], "i_mag": [], "i_ang": []}
    for _, row in case.branch_data.sort_values("tb", kind="stable").iterrows():
        lead = {
            "fb": row["fb"],
            "tb": row["tb"],
            "from_name": writer.bus_name[row["fb"]],
            "to_name": writer.bus_name[row["tb"]],
            "t": t,
        }
        out = {key: {**lead, **dict.fromkeys(PHASE_NODE, np.nan)} for key in tables}
        transformers = (
            _branch_transformers(row, writer.regulators.get((row["fb"], row["tb"]), []))
            if row["type"] == "transformer"
            else []
        )
        for phase, (power, current) in _branch_values(
            row, bus_v_base, transformers
        ).items():
            out["p"][phase] = power.real
            out["q"][phase] = power.imag
            out["i_mag"][phase] = abs(current)
            out["i_ang"][phase] = np.angle(current, deg=True) % 360
        for key, table in tables.items():
            table.append(out[key])
    return tables


def _kva_by_node(element: str) -> dict[int, complex]:
    """Terminal-1 power (kW + j kvar into the element) at each non-ground node."""
    from opendssdirect import dss

    _, _, power = _element_terminal(element)
    n_conductors = dss.CktElement.NumConductors()
    nodes = dss.CktElement.NodeOrder()[:n_conductors]
    return {n: p for n, p in zip(nodes, power[:n_conductors]) if n != 0}


def _device_kva(device: _Device) -> complex:
    """Power into all elements of a device (kW + j kvar)."""
    return sum(sum(_kva_by_node(e).values()) for e, _ in device.elements)


def _load_rows(devices: list[_Device]) -> pd.DataFrame:
    """Per-phase P and Q (pu, consumption positive) of every load device."""
    rows = []
    for device in devices:
        row = {
            "device_name": device.device_name,
            "element": ",".join(e for e, _ in device.elements),
        }
        row.update({f"{k}_{ph}": 0.0 for k in "pq" for ph in LOAD_PHASES})
        for element, phases in device.elements:
            for node, kva in _kva_by_node(element).items():
                phase = next(ph for ph in phases if PHASE_NODE[ph] == node)
                row[f"p_{phase}"] += kva.real / device.kva_base
                row[f"q_{phase}"] += kva.imag / device.kva_base
        rows.append(row)
    columns = ["device_name", "element"] + [
        f"{k}_{ph}" for k in "pq" for ph in LOAD_PHASES
    ]
    return pd.DataFrame(rows, columns=columns).set_index("device_name")


def _device_frame(devices: list[_Device], values, columns: list[str]) -> pd.DataFrame:
    """Frame indexed by device_name with the ``element`` column and ``values(device)`` columns."""
    rows = [
        {
            "device_name": d.device_name,
            "element": ",".join(e for e, _ in d.elements),
            **dict(zip(columns, values(d))),
        }
        for d in devices
    ]
    return pd.DataFrame(rows, columns=["device_name", "element", *columns]).set_index(
        "device_name"
    )


def _injection(device: _Device) -> tuple[float, float]:
    """Injected (P, Q) in pu: OpenDSS reports power into the element."""
    kva = -_device_kva(device)
    return kva.real / device.kva_base, kva.imag / device.kva_base


def _capacitor_state(device: _Device) -> tuple[float, float]:
    """Injected Q (pu) and the on/off state of a capacitor."""
    from opendssdirect import dss

    dss.Capacitors.Name(device.elements[0][0].split(".", 1)[1])
    return _injection(device)[1], float(dss.Capacitors.States()[0])


def _regulator_tap(device: _Device) -> tuple[int]:
    """Tap position of a regulator from its winding-2 ratio."""
    from opendssdirect import dss

    dss.Transformers.Name(device.elements[0][0].split(".", 1)[1])
    dss.Transformers.Wdg(2)
    return (round((dss.Transformers.Tap() - 1) / device.step),)


def _device_results(devices: list[_Device]) -> dict[str, pd.DataFrame]:
    """Read-back frames of the solved circuit, one per device table."""
    by_table = {
        table: [d for d in devices if d.table == table] for table in DEVICE_TABLES
    }
    results = {}
    for table in ("pv_data", "gen_data", "bat_data"):
        results[table] = _device_frame(by_table[table], _injection, ["p", "q"])
    results["cap_data"] = _device_frame(
        by_table["cap_data"], _capacitor_state, ["q", "u"]
    )
    results["reg_data"] = _device_frame(by_table["reg_data"], _regulator_tap, ["tap"])
    results["load_data"] = _load_rows(by_table["load_data"])
    return results


def run_opendss_gad(
    gad_case: GadCase,
    setpoints: Optional[dict[str, pd.DataFrame]] = None,
    t: Optional[int] = None,
) -> tuple[PowerFlowResult, dict[str, pd.DataFrame]]:
    """Solve ``gad_case`` at time step ``t`` (default ``start_step``) in OpenDSS.

    Returns the network result (bus voltage magnitudes and angles, branch P/Q and
    currents, in the FBS layout, per-unit) and ``device_results``, a dict of frames
    keyed by device table name and indexed by ``device_name``:

    - ``gen_data``, ``pv_data``, ``bat_data``: ``p``, ``q`` total injection (pu)
    - ``cap_data``: ``q`` injection (pu, at the solved voltage) and ``u`` on/off
    - ``reg_data``: ``tap``
    - ``load_data``: ``p_<phase>``, ``q_<phase>`` consumption (pu, as in ``load_data``)

    Every frame also has an ``element`` column with the OpenDSS element name(s).
    ``gen_data``/``pv_data``/``bat_data``/``reg_data`` frames can be passed back as
    ``setpoints``.
    """
    from opendssdirect import dss

    with tempfile.TemporaryDirectory() as tmp:
        writer, t = _build(gad_case, t, setpoints)
        path = Path(tmp) / "gad.dss"
        path.write_text("\n".join([*writer.lines, ""]))
        dss.Text.Command("Clear")
        dss.Text.Command(f"Redirect {path}")
    magnitudes, angles = _solved_bus_rows(gad_case.bus_data, t)
    branch = _solved_branch_rows(writer, t)
    devices = _device_results(writer.devices)
    return (
        PowerFlowResult(
            voltages=pd.DataFrame(magnitudes),
            voltage_angles=pd.DataFrame(angles),
            active_power_flows=pd.DataFrame(branch["p"]),
            reactive_power_flows=pd.DataFrame(branch["q"]),
            currents=pd.DataFrame(branch["i_mag"]),
            current_angles=pd.DataFrame(branch["i_ang"]),
            solver="opendss",
            backend="opendss",
            result_type="opendss",
            converged=bool(dss.Solution.Converged()),
            case=gad_case,
        ),
        devices,
    )
