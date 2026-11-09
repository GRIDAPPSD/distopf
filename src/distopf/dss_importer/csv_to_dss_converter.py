"""
DistOPF case -> OpenDSS converter.

Writes a single .dss file from a ``Case`` so results can be compared to FBS.
Not validated against an original OpenDSS model.

Mapping
-------
- swing bus        -> circuit Vsource (3-phase, voltage from ``v_a``)
- lines/switches   -> Line with R/X matrices (ohms, length 1 km)
- transformers     -> one single-phase Transformer per phase; winding-2 tap = reg ratio
- center_tap_xfmr  -> single-phase 3-winding Transformer, secondary ``sec.1.0`` / ``sec.0.2``
- triplex_line     -> 2-conductor Line on nodes 1 and 2 (s1, s2)
- loads            -> one single-phase Load per bus phase (a, b, c, s1, s2, s1s2);
                      constant PQ, or ``model=4`` (CVR) when ``cvr_p``/``cvr_q`` are nonzero
- capacitors       -> one single-phase Capacitor per bus phase
- generators       -> one single-phase constant-PQ Generator per bus phase (p/q setpoints)
- mpssd ports      -> one single-phase constant-PQ Generator ``mpssd_<device>_<phase>`` per
                      port of ``case.mpssd_data`` (p/q setpoints, positive is injection)
- batteries        -> one multi-phase Storage per battery; fixed kW/kvar from the summed
                      ``p_*``/``q_*`` or ``p``/``q`` (OpenDSS splits them equally over the
                      phases); positive is discharging
- line charging    -> Line cmatrix from ``b_*``; transformer shunt -> %noloadloss/%imag
                      from ``g_sh_*``/``b_sh_*``; center-tap primary leg from ``-z_s1s2``
- schedules        -> a single time step ``t`` is written (default ``case.start_step``):
                      load and generator setpoints are scaled by the schedule multipliers
                      exactly as FBS does, and the swing voltage uses the ``v_a`` column.
                      Convert each time step separately.
- opf_result       -> optional OPF result replayed like ``run_fbs_with_opf_setpoints``:
                      generator, battery and OUT-bus load setpoints replace the case values
                      (CVR off at OUT buses), regulator taps come from ``reg_taps``,
                      capacitors are switched from ``u_caps`` (else inferred from
                      ``capacitor_reactive_power``), and MPSSD ports use ``p_mpssd``/
                      ``q_mpssd`` (on the result or its ``raw_result``).
"""

import re
import tempfile
from math import pi, sqrt
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from distopf.api import Case, create_case
from distopf.fbs import _opf_frames, _at_period, _replay_times
from distopf.results import PowerFlowResult

PHASE_NODE = {"a": 1, "b": 2, "c": 3, "s1": 1, "s2": 2}
LOAD_PHASES = ("a", "b", "c", "s1", "s2", "s1s2")
FREQUENCY_HZ = 60.0


def _value(row: pd.Series, column: str) -> float:
    """Numeric column value; missing or NaN counts as 0."""
    value = row.get(column, 0.0)
    return 0.0 if pd.isna(value) else float(value)


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


def _node(bus: str, phase: str) -> str:
    return f"{bus}.{PHASE_NODE[phase]}"


def _matrix_str(entry, phases: list[str]) -> str:
    """Lower-triangular OpenDSS matrix string, e.g. '[1 | 0 2]'."""
    rows = []
    for i, pi in enumerate(phases):
        values = [f"{entry(pi, phases[j]):.12g}" for j in range(i + 1)]
        rows.append(" ".join(values))
    return "[" + " | ".join(rows) + "]"


def _line(row: pd.Series, bus_name: dict) -> str:
    """Line for abc branches and triplex (s1, s2) branches; impedances are per-unit."""
    phases = _split_phases(row["phases"])
    nodes = ".".join(str(PHASE_NODE[p]) for p in phases)

    def entry(prefix: str):
        def get(pi: str, pj: str) -> float:
            for key in (f"{prefix}_{pi}{pj}", f"{prefix}_{pj}{pi}"):
                if key in row.index and pd.notna(row[key]):
                    return row[key] * row["z_base"]
            return 0.0

        return get

    return (
        f"New Line.{row['name']}_{int(row['fb'])}_{int(row['tb'])} phases={len(phases)} "
        f"bus1={bus_name[row['fb']]}.{nodes} bus2={bus_name[row['tb']]}.{nodes} "
        f"length=1 units=km rmatrix={_matrix_str(entry('r'), phases)} "
        f"xmatrix={_matrix_str(entry('x'), phases)} "
        f"cmatrix={_matrix_str(_capacitance_nf(row), phases)}"
    )


def _transformers(
    row: pd.Series, bus_name: dict, bus_kv: dict, reg_ratio: dict
) -> list[str]:
    key = (row["fb"], row["tb"])
    lines = []
    for phase in "abc":
        if phase not in row["phases"]:
            continue
        ratio = reg_ratio.get(key, {}).get(phase, 1.0)
        lines.append(
            f"New Transformer.{row['name']}_{int(row['fb'])}_{int(row['tb'])}_{phase} phases=1 windings=2 "
            f"buses=[{_node(bus_name[row['fb']], phase)}.0, {_node(bus_name[row['tb']], phase)}.0] "
            f"conns=[wye, wye] kVs=[{bus_kv[row['fb']]:.12g}, {bus_kv[row['tb']]:.12g}] "
            f"kVAs=[{row['s_base'] / 1e3:.12g}, {row['s_base'] / 1e3:.12g}] "
            f"%rs=[{row[f'r_{phase}{phase}'] * 50:.12g}, {row[f'r_{phase}{phase}'] * 50:.12g}] "
            f"xhl={row[f'x_{phase}{phase}'] * 100:.12g} {_core_shunt(row, phase)} "
            f"taps=[1, {ratio:.12g}]"
        )
    return lines


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


class _Multipliers:
    """Schedule multipliers at one time step, following the FBS rules."""

    def __init__(self, case: Case, t: int):
        sched = getattr(case, "schedules", None)
        self.t = t
        self.sched = sched if sched is not None else pd.DataFrame()
        if len(self.sched) and t not in self.sched.index:
            raise ValueError(f"schedules has no row for time step {t}")

    def _value(self, column: str) -> Optional[float]:
        if column not in self.sched.columns:
            return None
        value = pd.to_numeric(self.sched.loc[self.t, column], errors="coerce")
        return 1.0 if pd.isna(value) else float(value)

    def load(self, shape: str, phase: str, kind: str) -> float:
        """Multiplier for load kind 'p' or 'q' on ``phase`` (a, b, c, s1, s2, s1s2)."""
        if phase == "s1s2":
            return (self.load(shape, "s1", kind) + self.load(shape, "s2", kind)) / 2
        for column in (shape, f"{shape}.{phase}.{kind}", f"{shape}.{kind}"):
            value = self._value(column)
            if value is not None:
                return value
        return 1.0

    def generator(self, shape: str) -> float:
        for column in (shape, "PV"):
            if column:
                value = self._value(column)
                if value is not None:
                    return value
        return 1.0

    def swing_voltage(self, default: float) -> float:
        value = self._value("v_a")
        return default if value is None else value


def _text(row: pd.Series, column: str) -> str:
    value = row.get(column, "")
    return "" if pd.isna(value) else str(value).strip()


def _load_line(
    name: str,
    bus: str,
    phase: str,
    kv: float,
    p_kw: float,
    q_kvar: float,
    cvr_p: float,
    cvr_q: float,
) -> str:
    if phase == "s1s2":
        nodes, kv = f"{bus}.1.2", 2 * kv
    else:
        nodes = _node(bus, phase)
    model = (
        f"model=4 cvrwatts={cvr_p:.12g} cvrvars={cvr_q:.12g}"
        if cvr_p or cvr_q
        else "model=1"
    )
    return (
        f"New Load.{name}_{phase} phases=1 bus1={nodes} kV={kv:.12g} "
        f"kW={p_kw:.12g} kvar={q_kvar:.12g} {model} vminpu=0.5 vmaxpu=1.5"
    )


def _generator_line(
    name: str, bus: str, phase: str, kv: float, p_kw: float, q_kvar: float
) -> str:
    return (
        f"New Generator.{name}_{phase} phases=1 bus1={_node(bus, phase)} "
        f"kV={kv:.12g} kW={p_kw:.12g} kvar={q_kvar:.12g} model=1"
    )


def _battery_setpoints(
    row: pd.Series, saved_p: pd.DataFrame, saved_q: pd.DataFrame
) -> dict[str, tuple[float, float]]:
    """Per-phase (p, q) of a battery row: ``p_*``/``q_*`` columns, else ``p``/``q`` split equally.

    Saved OPF values (phase or ``p_<phase>``/``q_<phase>`` columns, by ``id``) replace them.
    """
    legs = _split_phases(row.get("phases", "abc"))
    if any(f"{k}_{p}" in row.index for k in "pq" for p in PHASE_NODE):
        case_values = {
            p: (row.get(f"p_{p}", 0.0), row.get(f"q_{p}", 0.0)) for p in legs
        }
    else:
        share = (row.get("p", 0.0) / len(legs), row.get("q", 0.0) / len(legs))
        case_values = {p: share for p in legs}
    return {
        p: (
            _saved(saved_p, row["id"], p, f"p_{p}", default=pq[0]),
            _saved(saved_q, row["id"], p, f"q_{p}", default=pq[1]),
        )
        for p, pq in case_values.items()
    }


def _result_frames(opf_result: Any, t: int) -> dict[str, Optional[pd.DataFrame]]:
    """OPF result frames for period ``t``."""
    return {name: _at_period(f, t) for name, f in _opf_frames(opf_result).items()}


def _indexed(frame: Optional[pd.DataFrame], key: str) -> pd.DataFrame:
    """Rows of a result frame indexed by ``key`` (first row wins); empty without a frame."""
    if frame is None or not len(frame):
        return pd.DataFrame()
    if key not in frame.columns:
        raise ValueError(f"OPF result frame must contain a {key!r} column")
    return frame.drop_duplicates(key, keep="first").set_index(key)


def _saved(saved: pd.DataFrame, key: Any, *columns: str, default=None):
    """First non-NaN value of ``columns`` in row ``key`` of ``saved``, else ``default``."""
    if key not in saved.index:
        return default
    for column in columns:
        if column in saved.columns:
            value = pd.to_numeric(saved.at[key, column], errors="coerce")
            if pd.notna(value):
                return float(value)
    return default


def _bus_load(
    row: pd.Series, phase: str, kind: str, mult: _Multipliers, saved: pd.DataFrame
) -> float:
    """Load ``kind`` ('p'/'q') in pu; saved OPF loads replace the load at OUT buses."""
    shape = _text(row, "load_shape") or "default"
    value = row.get(f"{kind}l_{phase}", 0.0) * mult.load(shape, phase, kind)
    if row["bus_type"] != "OUT":
        return value
    if phase == "s1s2":
        # Result s1/s2 loads already include half of the s1s2 load.
        split = [_saved(saved, row["id"], s) for s in ("s1", "s2")]
        return value if split == [None, None] else 0.0
    return _saved(saved, row["id"], phase, default=value)


def _capacitor_on(
    cap_id: Any,
    phase: str,
    nominal: float,
    v_min: float,
    u_caps: pd.DataFrame,
    q_caps: pd.DataFrame,
    voltages: pd.DataFrame,
) -> bool:
    """Switch state from ``u_caps``, else on when q >= half of nominal * v^2 (FBS rule)."""
    if len(u_caps):
        state = _saved(u_caps, cap_id, phase)
        return state is None or state > 0.5
    q = _saved(q_caps, cap_id, phase)
    if q is None:
        return True
    v = _saved(voltages, cap_id, phase, default=v_min)
    return q / nominal > 0.5 * v**2


def case_to_dss(
    case: Case,
    output_path: Union[str, Path],
    t: Optional[int] = None,
    opf_result: Any = None,
) -> Path:
    """Write ``case`` at time step ``t`` (default ``case.start_step``) to an OpenDSS file.

    With ``opf_result`` the OPF setpoints and device states are written instead of the
    case values, as in :func:`distopf.fbs.run_fbs_with_opf_setpoints`.
    """
    bus = case.bus_data
    bus_name = dict(zip(bus["id"], bus["name"].astype(str)))
    bus_kv = dict(zip(bus["id"], bus["v_ln_base"] / 1e3))
    bus_s_base = dict(zip(bus["id"], bus["s_base"]))
    swing = bus[bus["bus_type"] == "SWING"].iloc[0]
    t = case.start_step if t is None else t
    mult = _Multipliers(case, t)
    frames = _result_frames(opf_result, t)
    p_loads = _indexed(frames.get("active_power_loads"), "id")
    q_loads = _indexed(frames.get("reactive_power_loads"), "id")
    loads_given = bool(len(p_loads) or len(q_loads))

    reg_taps = _indexed(frames.get("reg_taps"), "tb")
    reg_ratio = {}
    for _, r in case.reg_data.iterrows():
        step = (r["ratio_max"] - r["ratio_min"]) / r["num_taps"] if len(reg_taps) else 0
        ratios = {}
        for p in "abc":
            if f"ratio_{p}" not in r:
                continue
            tap = _saved(reg_taps, r["tb"], f"tap_{p}", p)
            ratios[p] = r[f"ratio_{p}"] if tap is None else 1 + step * tap
        reg_ratio[(r["fb"], r["tb"])] = ratios

    out = [
        "Clear",
        f"New Circuit.distopf basekv={swing['v_ln_base'] * sqrt(3) / 1e3:.12g} "
        f"pu={mult.swing_voltage(swing['v_a']):.12g} phases=3 bus1={bus_name[swing['id']]} "
        "mvasc3=1e6 mvasc1=1e6",
    ]

    for _, row in case.branch_data.iterrows():
        if row["type"] == "transformer":
            out += _transformers(row, bus_name, bus_kv, reg_ratio)
        elif row["type"] == "center_tap_xfmr":
            out.append(_center_tap_transformer(row, bus_name, bus_kv))
        else:
            out.append(_line(row, bus_name))

    for _, row in bus.iterrows():
        if row["bus_type"] == "SWING":
            continue  # FBS ignores load at the source bus
        cvr_off = loads_given and row["bus_type"] == "OUT"
        for p in LOAD_PHASES:
            pl = _bus_load(row, p, "p", mult, p_loads)
            ql = _bus_load(row, p, "q", mult, q_loads)
            if pl != 0 or ql != 0:
                scale = row["s_base"] / 1e3
                out.append(
                    _load_line(
                        row["name"],
                        bus_name[row["id"]],
                        p,
                        row["v_ln_base"] / 1e3,
                        pl * scale,
                        ql * scale,
                        0.0 if cvr_off else row.get("cvr_p", 0.0),
                        0.0 if cvr_off else row.get("cvr_q", 0.0),
                    )
                )

    u_caps = _indexed(frames.get("u_caps"), "id")
    q_caps = _indexed(frames.get("capacitor_reactive_power"), "id")
    voltages = _indexed(frames.get("voltages"), "id")
    v_min = dict(zip(bus["id"], bus["v_min"])) if "v_min" in bus else {}
    for _, row in case.cap_data.iterrows():
        for p in "abc":
            nominal = row[f"q_{p}"]
            if nominal != 0 and _capacitor_on(
                row["id"],
                p,
                nominal,
                v_min.get(row["id"], 1.0),
                u_caps,
                q_caps,
                voltages,
            ):
                out.append(
                    f"New Capacitor.{row['name']}_{p} phases=1 bus1={_node(bus_name[row['id']], p)} "
                    f"kV={bus_kv[row['id']]:.12g} kvar={nominal * bus_s_base[row['id']] / 1e3:.12g}"
                )

    p_gens = _indexed(frames.get("active_power_generation"), "id")
    q_gens = _indexed(frames.get("reactive_power_generation"), "id")
    for _, row in case.gen_data.iterrows():
        p_scale = mult.generator(_text(row, "gen_shape"))
        scale = bus_s_base[row["id"]] / 1e3
        for p in PHASE_NODE:
            # Saved setpoints replace the scheduled values; no schedule on top.
            p_pu = _saved(
                p_gens, row["id"], p, default=row.get(f"p_{p}", 0.0) * p_scale
            )
            q_pu = _saved(q_gens, row["id"], p, default=row.get(f"q_{p}", 0.0))
            if p_pu != 0 or q_pu != 0:
                out.append(
                    _generator_line(
                        row["name"],
                        bus_name[row["id"]],
                        p,
                        bus_kv[row["id"]],
                        p_pu * scale,
                        q_pu * scale,
                    )
                )

    mpssd = getattr(case, "mpssd_data", None)
    if mpssd is not None and len(mpssd):
        bus_id = dict(zip(bus["name"].astype(str), bus["id"]))
        p_mpssd = _indexed(frames.get("p_mpssd"), "device_name")
        q_mpssd = _indexed(frames.get("q_mpssd"), "device_name")
        for _, row in mpssd.iterrows():
            device = _text(row, "device_name")
            i = bus_id[_text(row, "bus_name")]
            scale = bus_s_base[i] / 1e3
            for p in _split_phases(row["phases"]):
                p_pu = _saved(p_mpssd, device, p, default=_value(row, f"p_{p}"))
                q_pu = _saved(q_mpssd, device, p, default=_value(row, f"q_{p}"))
                out.append(
                    _generator_line(
                        f"mpssd_{device}",
                        bus_name[i],
                        p,
                        bus_kv[i],
                        p_pu * scale,
                        q_pu * scale,
                    )
                )

    p_bats = _indexed(frames.get("battery_active_power"), "id")
    q_bats = _indexed(frames.get("battery_reactive_power"), "id")
    if case.bat_data is not None:
        for _, row in case.bat_data.iterrows():
            scale = bus_s_base[row["id"]] / 1e3
            setpoints = _battery_setpoints(row, p_bats, q_bats)
            nodes = ".".join(str(PHASE_NODE[p]) for p in setpoints)
            kv = bus_kv[row["id"]] * (sqrt(3) if len(setpoints) > 1 else 1)
            p_total = sum(p for p, _ in setpoints.values())
            q_total = sum(q for _, q in setpoints.values())
            out.append(
                f"New Storage.{row['name']} phases={len(setpoints)} "
                f"bus1={bus_name[row['id']]}.{nodes} kV={kv:.12g} "
                f"kVA={_value(row, 's_max') * scale:.12g} "
                f"kWrated={_value(row, 's_max') * scale:.12g} "
                f"kWhrated={_value(row, 'energy_capacity') * scale:.12g} "
                f"%stored={_value(row, 'start_soc') * 100:.12g} "
                f"%reserve={_value(row, 'min_soc') * 100:.12g} "
                f"%EffCharge={_value(row, 'charge_efficiency') * 100:.12g} "
                f"%EffDischarge={_value(row, 'discharge_efficiency') * 100:.12g} "
                f"%idlingkW=0 kW={p_total * scale:.12g} kvar={q_total * scale:.12g}"
            )

    out += ["Set mode=snap", "Solve"]
    out += [
        f"SetkVBase bus={bus_name[i]} kvll={kv * sqrt(3):.12g}"
        for i, kv in bus_kv.items()
    ]
    out.append("")
    output_path = Path(output_path)
    output_path.write_text("\n".join(out))
    return output_path


def csv_to_dss(
    csv_dir: Union[str, Path],
    output_path: Union[str, Path],
    t: Optional[int] = None,
    opf_result: Any = None,
) -> Path:
    """Load a DistOPF CSV case directory and write it as an OpenDSS file."""
    return case_to_dss(create_case(Path(csv_dir)), output_path, t, opf_result)


def run_opendss(case: Case, opf_result: Any = None) -> PowerFlowResult:
    """Solve each time step of ``case`` (or each period of ``opf_result``) in OpenDSS.

    Returns per-unit voltage magnitudes in the FBS ``voltages`` layout.
    """
    from opendssdirect import dss

    frames = _opf_frames(opf_result)
    times = _replay_times(case, [f for n, f in frames.items() if n != "voltages"])
    bus = case.bus_data
    secondary = bus["phases"].astype(str).str.contains("s").to_numpy()
    rows, converged = [], True
    with tempfile.TemporaryDirectory() as tmp:
        for t in times:
            path = case_to_dss(case, Path(tmp) / f"case_{t}.dss", t, opf_result)
            dss.Text.Command("Clear")
            dss.Text.Command(f"Redirect {path}")
            converged = converged and bool(dss.Solution.Converged())
            for (i, name), is_sec in zip(bus[["id", "name"]].values, secondary):
                row = {"id": i, "name": name, "t": t}
                row.update({p: np.nan for p in PHASE_NODE})
                dss.Circuit.SetActiveBus(str(name))
                nodes = dss.Bus.Nodes()
                vmag = dss.Bus.puVmagAngle()[0::2]
                phases = ("s1", "s2") if is_sec else ("a", "b", "c")
                for node, value in zip(nodes, vmag):
                    if 1 <= node <= len(phases):
                        row[phases[node - 1]] = value
                rows.append(row)
    return PowerFlowResult(
        voltages=pd.DataFrame(rows),
        solver="opendss",
        backend="opendss",
        result_type="opendss",
        converged=converged,
        case=case,
    )
