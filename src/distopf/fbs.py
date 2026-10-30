"""Multi-period forward/backward sweep (FBS) power flow.

Static network data is prepared once, sweeps run level by level on arrays, and each
time step only refreshes the schedule-dependent injections. Each step starts from the
previous step's converged voltages. Schedules follow the Pyomo OPF conventions.

Saved OPF decisions (generator, battery and ``OUT`` boundary-load setpoints) can be
replayed through the exact power flow with :func:`run_fbs_with_opf_setpoints`,
:func:`replay_exact_power_flow_from_opf_result`, :func:`run_fbs_from_saved_results`
and :func:`replay_exact_power_flow`.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, NamedTuple, Optional, Sequence

import numpy as np
import pandas as pd

from distopf.results import PowerFlowResult

if TYPE_CHECKING:
    from distopf.api import Case

_PHASES = ("a", "b", "c", "s1", "s2")
_PHASE_IDX = {phase: i for i, phase in enumerate(_PHASES)}
_LOAD_SLOTS = (*_PHASES, "s1s2")
_SWING_TYPES = ("SWING", "SWING_FREE", "IN")
_PHASE_RE = re.compile(r"[A-Za-z]\d|.")
_ANGLES = np.exp(1j * np.array([0.0, -2 * np.pi / 3, 2 * np.pi / 3]))
_EPS = 1e-10

# Backwards-compatible names from the original fbs.py.
PHASE_IDX_MAP = _PHASE_IDX


def _parse_phases(phases_str: str) -> list[str]:
    """Split a phase string such as ``"abc"`` or ``"s1s2"`` into phase names."""
    return _PHASE_RE.findall(phases_str)


def _column(df: pd.DataFrame, name: str) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy(float)
    return np.zeros(len(df))


def _complex(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    out = np.empty(len(real), dtype=complex)
    out.real = real
    out.imag = imag
    return out


def _table(lead: dict, values: np.ndarray) -> pd.DataFrame:
    """Build a result frame from lead columns and a (steps, rows, 5) phase array."""
    data = dict(lead)
    for k, phase in enumerate(_PHASES):
        data[phase] = values[:, :, k].reshape(-1)
    return pd.DataFrame(data)


def _at_period(frame: Optional[pd.DataFrame], t: int) -> Optional[pd.DataFrame]:
    """Rows of ``frame`` for period ``t``; untimed frames apply to every period."""
    if frame is None:
        return None
    if "t" not in frame.columns:
        return frame
    return frame[frame["t"] == t]


class _Level(NamedTuple):
    rows: slice
    parent: np.ndarray
    starts: np.ndarray
    group_parent: np.ndarray
    tap: np.ndarray
    has_tap: bool
    tri: np.ndarray
    tri_phase: np.ndarray
    z: np.ndarray


class _Step(NamedTuple):
    pl: np.ndarray
    ql: np.ndarray
    sec_load: np.ndarray
    gen: np.ndarray
    bat: np.ndarray
    v_swing: np.ndarray


class FBS:
    """Forward/backward sweep over ``times``.

    ``times`` defaults to ``range(case.start_step, case.start_step + case.n_steps)``.
    """

    def __init__(self, case: Case, times: Optional[Sequence[int]] = None):
        self.case = case
        if times is None:
            start = int(case.start_step)
            self.times = list(range(start, start + int(case.n_steps)))
        else:
            self.times = [int(t) for t in times]
        if not self.times:
            raise ValueError("At least one time step is required")
        self.result: Optional[PowerFlowResult] = None
        self._volts: Optional[np.ndarray] = None
        self._currents: Optional[np.ndarray] = None
        self._build_topology()
        self._build_network()
        self._build_devices()
        self._build_schedules()

    # ------------------------------------------------------------------ setup

    def _build_topology(self) -> None:
        bus = self.case.bus_data
        branch = self.case.branch_data
        swing = bus.index[bus["bus_type"].isin(_SWING_TYPES)]
        if len(swing) != 1:
            raise ValueError("Exactly one swing bus must be defined")
        root = int(bus.at[swing[0], "id"])

        children: dict[int, list[int]] = {}
        for fb, tb in zip(
            branch["fb"].astype(int).tolist(), branch["tb"].astype(int).tolist()
        ):
            children.setdefault(fb, []).append(tb)

        ids = [root]
        parents = [-1]
        bounds: list[tuple[int, int]] = []
        seen = {root}
        frontier = [0]
        while frontier:
            nxt = []
            for pos in frontier:
                for child in children.get(ids[pos], ()):
                    if child not in seen:
                        seen.add(child)
                        ids.append(child)
                        parents.append(pos)
                        nxt.append(len(ids) - 1)
            if nxt:
                bounds.append((nxt[0], nxt[-1] + 1))
            frontier = nxt
        self.node_ids = np.array(ids, dtype=np.int64)
        self.parent = np.array(parents, dtype=np.int64)
        self._bounds = bounds

    def _build_network(self) -> None:
        case = self.case
        bus = case.bus_data
        branch = case.branch_data
        n = len(self.node_ids)

        bus_pos = pd.Index(bus["id"].to_numpy(np.int64)).get_indexer(self.node_ids)
        if (bus_pos < 0).any():
            raise ValueError("Branch endpoints are missing from bus_data")
        self._bus_pos = bus_pos
        self._names = bus["name"].to_numpy(dtype=object)[bus_pos]

        phases = bus["phases"].astype(str).str.lower().to_numpy()
        uniq, inverse = np.unique(phases, return_inverse=True)
        table = np.zeros((len(uniq), 5), dtype=bool)
        for k, text in enumerate(uniq):
            for phase in _PHASE_RE.findall(text):
                table[k, _PHASE_IDX[phase]] = True
        conn = table[inverse][bus_pos]
        sec = conn[:, 3] | conn[:, 4]
        self.connected = conn
        self._is_sec = sec
        self._cn = conn & ~sec[:, None]
        self._sec_idx = np.flatnonzero(sec)

        tri = np.zeros(n, dtype=bool)
        tri[1:] = conn[self.parent[1:], :3].any(axis=1) & sec[1:]
        primary = np.zeros(n, dtype=np.int64)
        if tri.any():
            labels = bus["primary_phase"].to_numpy()[bus_pos]
            primary[tri] = [_PHASE_IDX[label] for label in labels[tri]]
        self._tri = tri
        self._primary = primary

        self._cvr_p = _column(bus, "cvr_p")[bus_pos]
        self._cvr_q = _column(bus, "cvr_q")[bus_pos]

        v_nom = np.column_stack([_column(bus, f"v_{p}") for p in "abc"])[bus_pos]
        v_init = np.zeros((n, 5), dtype=complex)
        v_init[:, :3] = v_nom * _ANGLES
        self._v_init = v_init

        tap = np.ones((n, 5), dtype=complex)
        reg = case.reg_data
        if reg is not None and len(reg):
            node_pos = {int(i): k for k, i in enumerate(self.node_ids)}
            for rec in reg.drop_duplicates(subset="tb", keep="first").to_dict(
                "records"
            ):
                k = node_pos.get(int(rec["tb"]))
                if k is None:
                    continue
                for phase in str(rec["phases"]).lower():
                    tap[k, "abc".index(phase)] = rec[f"ratio_{phase}"]
        self._tap = tap

        z = np.zeros((n, 5, 5), dtype=complex)
        tb_pos = pd.Index(self.node_ids).get_indexer(branch["tb"].to_numpy(np.int64))
        sel = np.flatnonzero(tb_pos > 0)
        rows = tb_pos[sel]

        def zc(r: str, x: str) -> np.ndarray:
            return _complex(_column(branch, r)[sel], _column(branch, x)[sel])

        z[rows, 0, 0] = zc("r_aa", "x_aa")
        z[rows, 1, 1] = zc("r_bb", "x_bb")
        z[rows, 2, 2] = zc("r_cc", "x_cc")
        z[rows, 3, 3] = zc("r_s1s1", "x_s1s1")
        z[rows, 4, 4] = -zc("r_s2s2", "x_s2s2")
        z[rows, 0, 1] = z[rows, 1, 0] = zc("r_ab", "x_ab")
        z[rows, 0, 2] = z[rows, 2, 0] = zc("r_ac", "x_ac")
        z[rows, 1, 2] = z[rows, 2, 1] = zc("r_bc", "x_bc")
        # The s2 voltage and current are sign-flipped, so the mutual term is antisymmetric.
        mutual = zc("r_s1s2", "x_s1s2")
        z[rows, 3, 4] = mutual
        z[rows, 4, 3] = -mutual
        self._z = z

        self._levels = []
        for start, end in self._bounds:
            par = self.parent[start:end]
            starts = np.flatnonzero(np.r_[True, par[1:] != par[:-1]])
            tri_loc = np.flatnonzero(tri[start:end])
            self._levels.append(
                _Level(
                    rows=slice(start, end),
                    parent=par,
                    starts=starts,
                    group_parent=par[starts],
                    tap=tap[start:end],
                    has_tap=bool(np.any(tap[start:end] != 1)),
                    tri=tri_loc,
                    tri_phase=primary[start:end][tri_loc],
                    z=z[start:end],
                )
            )

    def _build_devices(self) -> None:
        case = self.case
        node_pos = pd.Index(self.node_ids)
        n_phase = len(_PHASES)

        gen = case.gen_data
        self._gen_rows = np.zeros(0, dtype=np.int64)
        self._gen_nodes = np.zeros(0, dtype=np.int64)
        self._gen_p = np.zeros((0, n_phase))
        self._gen_q = np.zeros((0, n_phase))
        if gen is not None and len(gen):
            self._gen_p = np.column_stack([_column(gen, f"p_{p}") for p in _PHASES])
            self._gen_q = np.column_stack([_column(gen, f"q_{p}") for p in _PHASES])
            gen_pos = node_pos.get_indexer(gen["id"].to_numpy(np.int64))
            last = {int(n): g for g, n in enumerate(gen_pos) if n >= 0}
            self._gen_nodes = np.array(list(last), dtype=np.int64)
            self._gen_rows = np.array(list(last.values()), dtype=np.int64)
        self._gen_sec = np.flatnonzero(self._is_sec[self._gen_nodes])

        bat = case.bat_data
        bat_nodes: dict[int, np.ndarray] = {}
        bat_legs: dict[int, np.ndarray] = {}
        if bat is not None and len(bat):
            for rec in bat.to_dict("records"):
                legs = [
                    p
                    for p in _PHASE_RE.findall(str(rec.get("phases", "abc")).lower())
                    if p in _PHASE_IDX
                ]
                values = np.zeros(n_phase, dtype=complex)
                if any(f"p_{p}" in rec or f"q_{p}" in rec for p in _PHASE_IDX):
                    for p in legs:
                        values[_PHASE_IDX[p]] = complex(
                            rec.get(f"p_{p}", 0), rec.get(f"q_{p}", 0)
                        )
                elif legs:
                    share = complex(rec.get("p", 0), rec.get("q", 0)) / len(legs)
                    for p in legs:
                        values[_PHASE_IDX[p]] = share
                mask = np.zeros(n_phase, dtype=bool)
                mask[np.array([_PHASE_IDX[p] for p in legs], dtype=np.int64)] = True
                k = node_pos.get_indexer([int(rec["id"])])[0]
                if k >= 0:
                    bat_nodes[int(k)] = values
                    bat_legs[int(k)] = mask
        self._bat_nodes = np.array(list(bat_nodes), dtype=np.int64)
        self._bat = np.array(list(bat_nodes.values()), dtype=complex).reshape(
            -1, n_phase
        )
        self._bat_legs = np.array(list(bat_legs.values()), dtype=bool).reshape(
            -1, n_phase
        )

        cap = case.cap_data
        cap_nodes: dict[int, np.ndarray] = {}
        if cap is not None and len(cap):
            for rec in cap.to_dict("records"):
                q = np.zeros(n_phase)
                q[:3] = [rec.get("q_a", 0), rec.get("q_b", 0), rec.get("q_c", 0)]
                k = node_pos.get_indexer([int(rec["id"])])[0]
                if k >= 0:
                    cap_nodes[int(k)] = q
        self._cap_nodes = np.array(list(cap_nodes), dtype=np.int64)
        self._cap = np.array(list(cap_nodes.values()), dtype=float).reshape(-1, n_phase)

    def _schedule_column(self, column: str) -> np.ndarray:
        sched = self._sched
        missing = [t for t in self.times if t not in sched.index]
        if missing:
            raise ValueError(
                f"schedule {column!r} has no row for time steps {missing[:5]}"
            )
        values = pd.to_numeric(sched.loc[self.times, column], errors="raise")
        return values.to_numpy(float)

    def _build_schedules(self) -> None:
        case = self.case
        bus = case.bus_data
        gen = case.gen_data
        sched = getattr(case, "schedules", None)
        if sched is None or getattr(case, "ignore_schedule", False):
            sched = pd.DataFrame()
        self._sched = sched
        columns = set(sched.columns)
        n_steps = len(self.times)

        self._pl_bus = np.column_stack([_column(bus, f"pl_{s}") for s in _LOAD_SLOTS])
        self._ql_bus = np.column_stack([_column(bus, f"ql_{s}") for s in _LOAD_SLOTS])
        if "load_shape" in bus.columns:
            shapes = bus["load_shape"].fillna("").astype(str).str.strip()
        else:
            shapes = pd.Series("", index=bus.index)
        shapes = shapes.replace("", "default").to_numpy()
        uniq, self._shape_idx = np.unique(shapes, return_inverse=True)
        self._node_shape = self._shape_idx[self._bus_pos]
        self._pl_node = self._pl_bus[self._bus_pos]
        self._ql_node = self._ql_bus[self._bus_pos]

        self._mp = np.ones((n_steps, len(uniq), 6))
        self._mq = np.ones((n_steps, len(uniq), 6))
        if len(sched):
            for si, shape in enumerate(uniq):
                for slot, phase in enumerate(_PHASES):
                    for kind, out in (("p", self._mp), ("q", self._mq)):
                        column = (
                            shape if shape in columns else f"{shape}.{phase}.{kind}"
                        )
                        if column not in columns:
                            continue
                        values = self._schedule_column(column)
                        if np.isinf(values).any():
                            raise ValueError(f"schedule {column!r} must be finite")
                        out[:, si, slot] = np.where(np.isnan(values), 1.0, values)
        # A line-to-line load is scaled by the mean of the leg multipliers.
        self._mp[:, :, 5] = (self._mp[:, :, 3] + self._mp[:, :, 4]) / 2
        self._mq[:, :, 5] = (self._mq[:, :, 3] + self._mq[:, :, 4]) / 2

        self._gen_mult = np.ones((n_steps, len(self._gen_p)))
        has_schedules = not sched.empty and any(c != "time" for c in sched.columns)
        if has_schedules and gen is not None and len(gen):
            if "gen_shape" in gen.columns:
                shape_col = gen["gen_shape"].fillna("").astype(str).str.strip()
            else:
                shape_col = pd.Series("", index=gen.index)
            for g, shape in enumerate(shape_col.to_numpy()):
                if shape:
                    if shape not in columns:
                        raise ValueError(
                            f"gen_shape {shape!r} is not a column of case.schedules"
                        )
                elif "PV" in columns:
                    shape = "PV"
                else:
                    continue
                values = self._schedule_column(shape)
                if not np.isfinite(values).all():
                    raise ValueError(
                        f"schedule {shape!r} has blank or non-numeric values"
                    )
                if (values < 0).any():
                    raise ValueError(f"schedule {shape!r} has negative values")
                self._gen_mult[:, g] = values

        # Per-step injections. These start from the case + schedules and may be
        # overwritten by :meth:`apply_setpoints`.
        self._gen_p_t = self._gen_p[None] * self._gen_mult[:, :, None]
        self._gen_q_t = np.repeat(self._gen_q[None], n_steps, axis=0)
        self._bat_t = np.repeat(self._bat[None], n_steps, axis=0)
        # Bus-ordered (n_steps, n_bus, 6) fixed loads; NaN means "use the schedule".
        self._pl_fixed: Optional[np.ndarray] = None
        self._ql_fixed: Optional[np.ndarray] = None

        base = np.array([float(bus[f"v_{p}"].iloc[self._swing_row()]) for p in "abc"])
        swing = np.tile(base, (n_steps, 1))
        for k, phase in enumerate("abc"):
            column = f"v_{phase}"
            if column in columns:
                present = [t for t in self.times if t in sched.index]
                values = pd.to_numeric(sched.loc[present, column], errors="coerce")
                for t, value in zip(present, values):
                    if pd.notna(value):
                        swing[self.times.index(t), k] = float(value)
        self._v_swing = np.zeros((n_steps, 5), dtype=complex)
        self._v_swing[:, :3] = swing * _ANGLES

    def _swing_row(self) -> int:
        bus = self.case.bus_data
        return int(np.flatnonzero(bus["bus_type"].isin(_SWING_TYPES).to_numpy())[0])

    # ------------------------------------------------------------ setpoints

    def apply_setpoints(
        self,
        *,
        p_gens: Optional[pd.DataFrame] = None,
        q_gens: Optional[pd.DataFrame] = None,
        p_bats: Optional[pd.DataFrame] = None,
        q_bats: Optional[pd.DataFrame] = None,
        p_loads: Optional[pd.DataFrame] = None,
        q_loads: Optional[pd.DataFrame] = None,
    ) -> None:
        """Overwrite per-step injections with saved OPF decisions.

        Frames use the result layout: an ``id`` column, an optional ``t`` column and
        phase columns ``a, b, c, s1, s2``. Frames without ``t`` apply to every step.
        NaN entries keep the case/schedule value. Generator setpoints replace the
        scheduled values, so the generator schedule is not applied on top of them.
        Loads are applied only at ``OUT`` buses, which are free variables in a
        decomposed OPF. CVR is disabled at those buses, matching the OPF.
        Call this before :meth:`solve`.
        """
        loads_applied = False
        for ti, t in enumerate(self.times):
            self._override_gens(ti, _at_period(p_gens, t), self._gen_p_t, "active")
            self._override_gens(ti, _at_period(q_gens, t), self._gen_q_t, "reactive")
            self._override_bats(ti, _at_period(p_bats, t), real=True)
            self._override_bats(ti, _at_period(q_bats, t), real=False)
            for frame, prefix in ((p_loads, "pl"), (q_loads, "ql")):
                applied = self._override_out_loads(ti, _at_period(frame, t), prefix)
                loads_applied = applied or loads_applied
        if loads_applied:
            out = self._out_nodes()
            self._cvr_p[out] = 0.0
            self._cvr_q[out] = 0.0

    def _out_nodes(self) -> np.ndarray:
        bus = self.case.bus_data
        if "bus_type" not in bus.columns:
            return np.zeros(len(self.node_ids), dtype=bool)
        return bus["bus_type"].to_numpy()[self._bus_pos] == "OUT"

    def _override_gens(
        self, ti: int, frame: Optional[pd.DataFrame], target: np.ndarray, label: str
    ) -> None:
        gen = self.case.gen_data
        if frame is None or gen is None or not len(gen):
            return
        if "id" not in frame.columns:
            raise ValueError(
                f"Saved {label} power generation setpoints must contain an 'id' column"
            )
        saved = frame.drop_duplicates("id", keep="first").set_index("id")
        gen_ids = gen["id"].to_numpy()
        missing = sorted(set(gen_ids.tolist()) - set(saved.index.tolist()))
        if missing:
            raise ValueError(
                f"Saved {label} power generation setpoints for t={self.times[ti]} "
                f"are missing generator id(s): {missing}"
            )
        phases = [p for p in _PHASES if p in saved.columns]
        if not phases:
            return
        cols = [_PHASE_IDX[p] for p in phases]
        values = saved.loc[gen_ids, phases].apply(pd.to_numeric, errors="coerce")
        values = values.to_numpy(float)
        step = target[ti]  # view into the per-step array
        step[:, cols] = np.where(np.isnan(values), step[:, cols], values)

    def _override_bats(
        self, ti: int, frame: Optional[pd.DataFrame], real: bool
    ) -> None:
        if frame is None or not len(frame) or not self._bat_nodes.size:
            return
        if "id" not in frame.columns:
            raise ValueError("Saved battery setpoints must contain an 'id' column")
        prefix = "p" if real else "q"
        saved = frame.drop_duplicates("id", keep="first").set_index("id")
        part = self._bat_t.real if real else self._bat_t.imag  # writable views
        for j, k in enumerate(self._bat_nodes):
            bat_id = int(self.node_ids[k])
            if bat_id not in saved.index:
                continue
            row = saved.loc[bat_id]
            values = np.full(len(_PHASES), np.nan)
            for phase in _PHASES:
                for column in (phase, f"{prefix}_{phase}"):
                    if column in row.index and pd.notna(row[column]):
                        values[_PHASE_IDX[phase]] = float(row[column])
                        break
            legs = self._bat_legs[j]
            if np.isnan(values[legs]).all():
                continue
            part[ti, j, legs] = np.nan_to_num(values[legs])

    def _override_out_loads(
        self, ti: int, frame: Optional[pd.DataFrame], prefix: str
    ) -> bool:
        bus = self.case.bus_data
        if frame is None or not len(frame) or "bus_type" not in bus.columns:
            return False
        is_out = bus["bus_type"] == "OUT"
        if not is_out.any():
            return False
        if "id" not in frame.columns:
            raise ValueError("Saved load setpoints must contain an 'id' column")
        selected = frame[frame["id"].isin(bus.loc[is_out, "id"])]
        selected = selected.drop_duplicates("id", keep="first")
        if selected.empty:
            return True
        name = f"_{prefix}_fixed"
        fixed = getattr(self, name)
        if fixed is None:
            fixed = np.full((len(self.times), len(bus), len(_LOAD_SLOTS)), np.nan)
            setattr(self, name, fixed)
        rows = pd.Index(bus["id"]).get_indexer(selected["id"])
        for phase in _PHASES:
            if phase not in selected.columns:
                continue
            values = pd.to_numeric(selected[phase], errors="coerce").to_numpy(float)
            given = ~np.isnan(values)
            fixed[ti, rows[given], _PHASE_IDX[phase]] = values[given]
            if phase in ("s1", "s2"):
                # Result s1/s2 loads already include half of the s1s2 load.
                fixed[ti, rows[given], 5] = 0.0
        return True

    # ------------------------------------------------------------------ solve

    def _step(self, ti: int) -> _Step:
        shape = self._node_shape
        pl = self._pl_node * self._mp[ti][shape]
        ql = self._ql_node * self._mq[ti][shape]
        if self._pl_fixed is not None:
            fixed = self._pl_fixed[ti][self._bus_pos]
            pl = np.where(np.isnan(fixed), pl, fixed)
        if self._ql_fixed is not None:
            fixed = self._ql_fixed[ti][self._bus_pos]
            ql = np.where(np.isnan(fixed), ql, fixed)
        has_load = np.abs(pl.sum(axis=1) + ql.sum(axis=1)) > _EPS
        pl[~has_load] = 0.0
        ql[~has_load] = 0.0
        sec = self._sec_idx
        rows = self._gen_rows
        gen = _complex(
            self._gen_p_t[ti][rows].ravel(),
            self._gen_q_t[ti][rows].ravel(),
        ).reshape(-1, 5)
        return _Step(
            pl=pl[:, :5],
            ql=ql[:, :5],
            sec_load=pl[sec, 3:6] + 1j * ql[sec, 3:6],
            gen=gen,
            bat=self._bat_t[ti],
            v_swing=self._v_swing[ti],
        )

    def _injection(self, v: np.ndarray, st: _Step) -> np.ndarray:
        av = np.abs(v)
        av2 = av * av
        nz = av > _EPS
        cn = self._cn
        with np.errstate(divide="ignore", invalid="ignore"):
            dv2 = av2 - 1.0
            p = st.pl + self._cvr_p[:, None] * st.pl / 2 * dv2
            q = st.ql + self._cvr_q[:, None] * st.ql / 2 * dv2
            s = p + 1j * q
            inj = -np.where(cn & nz & (np.abs(s) > _EPS), np.conj(s / v), 0)

            gn = self._gen_nodes
            if gn.size:
                ok = cn[gn] & nz[gn] & (np.abs(st.gen) > _EPS)
                inj[gn] += np.where(ok, np.conj(st.gen / v[gn]), 0)
            bn = self._bat_nodes
            if bn.size:
                ok = cn[bn] & nz[bn] & (np.abs(st.bat) > _EPS)
                inj[bn] += np.where(ok, np.conj(st.bat / v[bn]), 0)
            cp = self._cap_nodes
            if cp.size:
                ok = cn[cp] & nz[cp] & (np.abs(self._cap) > _EPS)
                s_cap = -1j * av2[cp] * self._cap
                inj[cp] -= np.where(ok, np.conj(s_cap / v[cp]), 0)

            si = self._sec_idx
            if si.size:
                v3, v4 = v[si, 3], v[si, 4]
                v_load = np.stack([v3, v4, v3 + v4], axis=1)
                a3, a4 = av2[si, 3], av2[si, 4]
                v2 = np.stack([a3, a4, (a3 + a4) / 2], axis=1)
                pn, qn = st.sec_load.real, st.sec_load.imag
                p = pn + self._cvr_p[si, None] * pn / 2 * (v2 - 1)
                q = qn + self._cvr_q[si, None] * qn / 2 * (v2 - 1)
                ok = (np.abs(v_load) > _EPS).all(axis=1)
                i_load = np.where(ok[:, None], np.conj((p + 1j * q) / v_load), 0)
                inj[si, 3] += -(i_load[:, 0] + i_load[:, 2])
                inj[si, 4] += i_load[:, 1] + i_load[:, 2]

                pos = self._gen_sec
                if pos.size:
                    nodes = gn[pos]
                    vg = v[nodes, 3:5]
                    ok = (np.abs(vg) > _EPS).all(axis=1)
                    i_gen = np.where(ok[:, None], np.conj(st.gen[pos, 3:5] / vg), 0)
                    inj[nodes, 3] += i_gen[:, 0]
                    inj[nodes, 4] += -i_gen[:, 1]
        return inj

    def _backward(self, v: np.ndarray, st: _Step) -> np.ndarray:
        inj = self._injection(v, st)
        branch = np.zeros_like(v)
        child_sum = np.zeros_like(v)
        for lv in reversed(self._levels):
            cur = child_sum[lv.rows] - inj[lv.rows]
            branch[lv.rows] = cur
            send = cur * lv.tap if lv.has_tap else cur
            if lv.tri.size:
                diff = cur[lv.tri, 3] - cur[lv.tri, 4]
                send[lv.tri] = 0
                send[lv.tri, lv.tri_phase] = diff
            child_sum[lv.group_parent] += np.add.reduceat(send, lv.starts, axis=0)
        return branch

    def _forward(self, branch: np.ndarray, v_swing: np.ndarray) -> np.ndarray:
        v = np.empty_like(branch)
        v[0] = v_swing
        for lv in self._levels:
            vp = v[lv.parent]
            base = vp * lv.tap if lv.has_tap else vp
            if lv.tri.size:
                phase_v = vp[lv.tri, lv.tri_phase]
                base[lv.tri] = 0
                base[lv.tri, 3] = phase_v
                base[lv.tri, 4] = phase_v
            cur = base - (lv.z @ branch[lv.rows][:, :, None])[:, :, 0]
            if lv.tri.size:
                cur[lv.tri, :3] = 0
            v[lv.rows] = cur
        return v

    def solve(
        self,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = False,
        initial_voltages: Optional[dict[int, np.ndarray]] = None,
        initial_currents: Optional[dict[int, np.ndarray]] = None,
    ) -> PowerFlowResult:
        """Solve every step in ``self.times``.

        ``initial_voltages`` (``{bus_id: 5-element complex array}``) warm-starts the
        first step; later steps start from the previous step's solution.
        ``initial_currents`` is accepted for compatibility with the original
        ``FBS.solve`` but is not needed, because each sweep starts from voltages.
        """
        del initial_currents
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        n_steps, n = len(self.times), len(self.node_ids)
        volts = np.empty((n_steps, n, 5), dtype=complex)
        currents = np.empty_like(volts)
        steps: list[dict] = []

        v_prev = None
        if initial_voltages is not None:
            v_prev = self._v_init.copy()
            pos = {int(i): k for k, i in enumerate(self.node_ids)}
            for bus_id, value in initial_voltages.items():
                k = pos.get(int(bus_id))
                if k is not None:
                    v_prev[k] = np.asarray(value, dtype=complex)[:5]

        for ti, t in enumerate(self.times):
            st = self._step(ti)
            start = time.perf_counter()
            v_old = (self._v_init if v_prev is None else v_prev).copy()
            v_old[0] = st.v_swing
            converged = False
            for iteration in range(max_iterations):
                branch = self._backward(v_old, st)
                v_new = self._forward(branch, st.v_swing)
                if np.max(np.abs(v_new - v_old)) < tolerance:
                    converged = True
                    break
                v_old = v_new
            elapsed = time.perf_counter() - start
            volts[ti], currents[ti], v_prev = v_new, branch, v_new
            steps.append(
                {
                    "t": t,
                    "iterations": iteration + 1 if converged else max_iterations,
                    "solve_time": elapsed,
                    "converged": converged,
                }
            )
            if verbose:
                status = "converged" if converged else "NOT converged"
                print(
                    f"t={t}: {status}, iterations={steps[-1]['iterations']} "
                    f"({elapsed:.4f}s)"
                )

        self._volts, self._currents = volts, currents
        self.result = self._results(volts, currents, steps)
        return self.result

    # ---------------------------------------------------------------- results

    def _device_frames(self) -> dict[str, Optional[pd.DataFrame]]:
        """Per-step generator and load frames in the layout used by fbs.py."""
        case = self.case
        times = self.times
        out: dict[str, Optional[pd.DataFrame]] = {}

        gen = case.gen_data
        empty_cols = ["id", "name", "t", "a", "b", "c", "s1", "s2"]
        for prefix, label, per_step in (
            ("p", "active_power_generation", self._gen_p_t),
            ("q", "reactive_power_generation", self._gen_q_t),
        ):
            names = [f"{prefix}_{p}" for p in _PHASES]
            present = [
                c for c in names if c in (gen.columns if gen is not None else [])
            ]
            if gen is None or not len(gen) or not present:
                out[label] = pd.DataFrame(columns=empty_cols)
                continue
            lead = ["id"] + (["name"] if "name" in gen.columns else [])
            template = gen[lead + present].rename(
                columns={f"{prefix}_{p}": p for p in _PHASES}
            )
            template.insert(2, "t", 0)
            frames = []
            for ti, t in enumerate(times):
                frame = template.copy()
                frame["t"] = t
                values = per_step[ti]
                for name in present:
                    frame[name[2:]] = values[:, _PHASE_IDX[name[2:]]]
                frames.append(frame)
            out[label] = pd.concat(frames, ignore_index=True)

        bus = case.bus_data
        for prefix, label, array, mult, fixed in (
            ("pl", "active_power_loads", self._pl_bus, self._mp, self._pl_fixed),
            ("ql", "reactive_power_loads", self._ql_bus, self._mq, self._ql_fixed),
        ):
            present = [p for p in _PHASES if f"{prefix}_{p}" in bus.columns]
            if not len(bus) or not present:
                out[label] = None
                continue
            split = f"{prefix}_s1s2" in bus.columns
            lead = ["id"] + (["name"] if "name" in bus.columns else [])
            template = bus[lead + [f"{prefix}_{p}" for p in present]].rename(
                columns={f"{prefix}_{p}": p for p in present}
            )
            template.insert(2, "t", 0)
            frames = []
            for ti, t in enumerate(times):
                scaled = array * mult[ti][self._shape_idx]
                if fixed is not None:
                    scaled = np.where(np.isnan(fixed[ti]), scaled, fixed[ti])
                frame = template.copy()
                frame["t"] = t
                for phase in present:
                    k = _PHASE_IDX[phase]
                    values = scaled[:, k]
                    if phase in ("s1", "s2") and split:
                        values = (
                            values
                            + np.where(np.isnan(scaled[:, 5]), 0.0, scaled[:, 5]) / 2
                        )
                    frame[phase] = values
                frames.append(frame)
            out[label] = pd.concat(frames, ignore_index=True)
        return out

    def _branch_nodes(self) -> tuple[np.ndarray, np.ndarray]:
        nodes = np.arange(1, len(self.node_ids))
        nodes = nodes[np.argsort(self.node_ids[nodes], kind="stable")]
        return nodes, self.parent[nodes]

    def _branch_lead(self, nodes: np.ndarray, parents: np.ndarray) -> dict:
        ids, names, n_steps = self.node_ids, self._names, len(self.times)
        return {
            "fb": np.tile(ids[parents], n_steps),
            "tb": np.tile(ids[nodes], n_steps),
            "from_name": np.tile(names[parents], n_steps),
            "to_name": np.tile(names[nodes], n_steps),
            "t": np.repeat(np.asarray(self.times, dtype=np.int64), len(nodes)),
        }

    def _flow_tables(
        self, volts: np.ndarray, currents: np.ndarray, from_side: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        nodes, parents = self._branch_nodes()
        conn = self.connected
        phases = conn[parents] & conn[nodes]
        tri = np.flatnonzero(self._tri[nodes])
        phases[tri] = conn[nodes][tri]
        if from_side:
            v = volts[:, parents, :].copy()
            if tri.size:
                phase_v = v[:, tri, self._primary[nodes][tri]]
                v[:, tri, :] = 0
                v[:, tri, 3] = phase_v
                v[:, tri, 4] = phase_v
            ratio = self._tap[nodes].copy()
        else:
            v = volts[:, nodes, :]
            ratio = np.ones((len(nodes), 5), dtype=complex)
        # s2 current points downstream but sits on the minus voltage reference.
        ratio[:, 4] = -ratio[:, 4]
        power = ratio[None] * v * np.conj(currents[:, nodes, :])
        lead = self._branch_lead(nodes, parents)
        mask = phases[None]
        return (
            _table(lead, np.where(mask, power.real, np.nan)),
            _table(lead, np.where(mask, power.imag, np.nan)),
        )

    def _results(
        self, volts: np.ndarray, currents: np.ndarray, steps: list[dict]
    ) -> PowerFlowResult:
        n_steps, n = volts.shape[:2]
        ids = self.node_ids
        order = np.argsort(ids, kind="stable")
        conn = self.connected

        v_sorted = volts[:, order, :]
        mask = conn[order][None]
        lead = {
            "id": np.tile(ids[order], n_steps),
            "name": np.tile(self._names[order], n_steps),
            "t": np.repeat(np.asarray(self.times, dtype=np.int64), n),
        }
        voltages = _table(lead, np.where(mask, np.abs(v_sorted), np.nan))
        angles = _table(lead, np.where(mask, np.angle(v_sorted, deg=True), np.nan))

        active, reactive = self._flow_tables(volts, currents, from_side=True)
        nodes, parents = self._branch_nodes()
        lead = self._branch_lead(nodes, parents)
        i_branch = currents[:, nodes, :]
        mask = conn[nodes][None]
        mags = _table(lead, np.where(mask, np.abs(i_branch), np.nan))
        i_ang = np.mod(np.angle(i_branch, deg=True), 360)
        cur_angles = _table(lead, np.where(mask, i_ang, np.nan))

        frames = self._device_frames()
        return PowerFlowResult(
            voltages=voltages,
            voltage_angles=angles,
            active_power_flows=active,
            reactive_power_flows=reactive,
            currents=mags,
            current_angles=cur_angles,
            active_power_generation=frames["active_power_generation"],
            reactive_power_generation=frames["reactive_power_generation"],
            active_power_loads=frames["active_power_loads"],
            reactive_power_loads=frames["reactive_power_loads"],
            converged=all(s["converged"] for s in steps),
            iterations=sum(s["iterations"] for s in steps),
            solve_time=sum(s["solve_time"] for s in steps),
            solver="fbs",
            result_type="fbs",
            metadata={"steps": steps},
            model=self,  # plotting helpers expect result.model
            case=self.case,
        )

    # ------------------------------------------------- FBS-compatible getters

    def _solved(self) -> PowerFlowResult:
        if self.result is None:
            raise ValueError("No results available. Run solve() first.")
        return self.result

    def results(self) -> PowerFlowResult:
        return self._solved()

    def get_voltages(self) -> pd.DataFrame:
        return self._solved().voltages

    def get_voltage_angles(self) -> pd.DataFrame:
        return self._solved().voltage_angles

    def get_currents(self) -> pd.DataFrame:
        return self._solved().currents

    def get_current_angles(self) -> pd.DataFrame:
        return self._solved().current_angles

    def get_p_flows(self, from_side: bool = True) -> pd.DataFrame:
        if from_side:
            return self._solved().active_power_flows
        self._solved()
        return self._flow_tables(self._volts, self._currents, from_side=False)[0]

    def get_q_flows(self, from_side: bool = True) -> pd.DataFrame:
        if from_side:
            return self._solved().reactive_power_flows
        self._solved()
        return self._flow_tables(self._volts, self._currents, from_side=False)[1]

    # ---------------------------------- attributes of the original FBS class
    # These expose the last solved step in the original dict-based layout, for
    # plotting helpers and other code that reads ``result.model``.

    @property
    def bus_data(self) -> pd.DataFrame:
        return self.case.bus_data

    @property
    def branch_data(self) -> pd.DataFrame:
        return self.case.branch_data

    @property
    def gen_data(self):
        return self.case.gen_data

    @property
    def cap_data(self):
        return self.case.cap_data

    @property
    def reg_data(self):
        return self.case.reg_data

    @property
    def bat_data(self):
        return self.case.bat_data

    @property
    def swing_bus(self) -> int:
        return int(self.node_ids[0])

    @property
    def topology(self) -> dict:
        ids = self.node_ids
        children: dict[int, list[int]] = {}
        parent: dict[int, int] = {}
        for k in range(1, len(ids)):
            fb, tb = int(ids[self.parent[k]]), int(ids[k])
            children.setdefault(fb, []).append(tb)
            parent[tb] = fb
        return {
            "nodes": sorted(int(i) for i in ids),
            "children": children,
            "parent": parent,
        }

    @property
    def phase_connections(self) -> dict[int, list[int]]:
        return {
            int(i): np.flatnonzero(self.connected[k]).tolist()
            for k, i in enumerate(self.node_ids)
        }

    @property
    def voltages(self) -> dict[int, np.ndarray]:
        if self._volts is None:
            return {}
        return {int(i): self._volts[-1, k].copy() for k, i in enumerate(self.node_ids)}

    @property
    def currents(self) -> dict[int, np.ndarray]:
        if self._currents is None:
            return {}
        return {
            int(self.node_ids[k]): self._currents[-1, k].copy()
            for k in range(1, len(self.node_ids))
        }

    @property
    def converged(self) -> bool:
        return bool(self.result.converged) if self.result is not None else False

    @property
    def iterations(self) -> int:
        return int(self.result.iterations or 0) if self.result is not None else 0

    @property
    def solve_time(self) -> Optional[float]:
        return self.result.solve_time if self.result is not None else None


def fbs_solve(
    case: Case,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> PowerFlowResult:
    """Solve every step of ``case`` and return one result with ``t``-labelled frames."""
    return FBS(case).solve(
        max_iterations=max_iterations, tolerance=tolerance, verbose=verbose
    )


# -------------------------------------------------------------------------
# Exact power-flow replay of OPF setpoints
# -------------------------------------------------------------------------

_SETPOINT_FIELDS = (
    "active_power_generation",
    "reactive_power_generation",
    "battery_active_power",
    "battery_reactive_power",
    "active_power_loads",
    "reactive_power_loads",
)


def _replay_times(case: Case, frames: Iterable[Optional[pd.DataFrame]]) -> list[int]:
    """Periods labelled in the setpoint frames, else the case's configured horizon."""
    labelled = {
        int(value)
        for frame in frames
        if frame is not None and len(frame) and "t" in frame.columns
        for value in frame["t"].dropna()
    }
    if labelled:
        return sorted(labelled)
    start = int(getattr(case, "start_step", 0) or 0)
    n_steps = int(getattr(case, "n_steps", 1) or 1)
    return list(range(start, start + max(n_steps, 1)))


def _label_periods(frame: pd.DataFrame, times: list[int]) -> pd.DataFrame:
    if "t" in frame.columns:
        return frame[frame["t"].isin(times)].reset_index(drop=True)
    return pd.concat([frame.assign(t=t) for t in times], ignore_index=True)


def _solve_with_setpoints(
    case: Case,
    frames: dict[str, Optional[pd.DataFrame]],
    *,
    max_iterations: int,
    tolerance: float,
    verbose: bool,
) -> PowerFlowResult:
    times = _replay_times(case, frames.values())
    solver = FBS(case, times=times)
    solver.apply_setpoints(
        p_gens=frames.get("active_power_generation"),
        q_gens=frames.get("reactive_power_generation"),
        p_bats=frames.get("battery_active_power"),
        q_bats=frames.get("battery_reactive_power"),
        p_loads=frames.get("active_power_loads"),
        q_loads=frames.get("reactive_power_loads"),
    )
    result = solver.solve(
        max_iterations=max_iterations, tolerance=tolerance, verbose=verbose
    )
    for name in ("battery_active_power", "battery_reactive_power"):
        frame = frames.get(name)
        if frame is not None and len(frame):
            setattr(result, name, _label_periods(frame.copy(), times))
    return result


def _tag_exact(result: PowerFlowResult, **extra) -> PowerFlowResult:
    result.metadata = {
        **(result.metadata or {}),
        "analysis": "exact_power_flow_replay",
        **extra,
        "reference_solver": "fbs",
    }
    return result


def run_fbs_with_opf_setpoints(
    case: Case,
    opf_result: Optional[PowerFlowResult] = None,
    *,
    p_gens: Optional[pd.DataFrame] = None,
    q_gens: Optional[pd.DataFrame] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> PowerFlowResult:
    """Replay OPF generator, battery and boundary-load setpoints through FBS.

    Ordinary loads, generator ``p`` and the swing voltage come from ``case`` and its
    schedules. Saved generator and battery values then replace the case values.
    Saved loads are applied only at ``OUT`` boundary buses, where they are free
    variables in a decomposed OPF. A supplied generator frame must cover every
    generator. Explicit ``p_gens`` / ``q_gens`` take precedence over ``opf_result``.
    """
    frames = {name: getattr(opf_result, name, None) for name in _SETPOINT_FIELDS}
    if p_gens is not None:
        frames["active_power_generation"] = p_gens
    if q_gens is not None:
        frames["reactive_power_generation"] = q_gens
    return _solve_with_setpoints(
        case,
        frames,
        max_iterations=max_iterations,
        tolerance=tolerance,
        verbose=verbose,
    )


def replay_exact_power_flow_from_opf_result(
    case: Case,
    opf_result: PowerFlowResult,
    *,
    output_dir: Optional[Path | str] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> PowerFlowResult:
    """Run exact FBS using an in-memory OPF result's setpoints.

    The optimization is not run again. Every period labelled in the OPF result is
    solved in one :class:`FBS` instance, and each period is warm-started from
    the previous one. When ``output_dir`` is given, the result is saved there.
    """
    result = run_fbs_with_opf_setpoints(
        case,
        opf_result,
        max_iterations=max_iterations,
        tolerance=tolerance,
        verbose=verbose,
    )
    _tag_exact(result)
    if output_dir is not None:
        result.save(output_dir)
    return result


# ------------------------------------------------------- saved-result replay

_GEN_FILES = ("active_power_generation", "reactive_power_generation")
_LOAD_FILES = ("active_power_loads", "reactive_power_loads")
_BAT_FILES = ("battery_active_power", "battery_reactive_power")
_CASE_KWARGS = {
    "start_step",
    "n_steps",
    "delta_t",
    "ignore_schedule",
    "ignore_gen",
    "ignore_bat",
    "ignore_cap",
    "ignore_reg",
}


def _read_frame(results_path: Path, name: str) -> Optional[pd.DataFrame]:
    frame = pd.read_csv(results_path / f"{name}.csv")
    if frame.empty:
        return None
    if "id" not in frame.columns:
        raise ValueError(f"Saved setpoint file {name}.csv must contain an 'id' column")
    return frame


def _validate_ids(name, frame, target, target_label) -> None:
    if frame is None or frame.empty:
        return
    if target is None or len(target) == 0:
        raise ValueError(
            f"Saved {name} setpoints contain IDs, but the selected case has no "
            f"{target_label}"
        )
    unknown = set(frame["id"].dropna()) - set(target["id"].dropna())
    if unknown:
        raise ValueError(
            f"Saved {name} IDs are incompatible with the selected case; "
            f"unknown IDs: {sorted(unknown)}"
        )


def _case_from_snapshot(results_path: Path) -> Case:
    input_path = results_path / "input"
    if not input_path.is_dir():
        raise FileNotFoundError(
            f"No case supplied and saved input snapshot is missing: {input_path}"
        )
    run_config = {}
    config_path = results_path / "run_config.json"
    if config_path.is_file():
        try:
            run_config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read {config_path}: {exc}") from exc
    from distopf.api import create_case

    case_info = run_config.get("case", {})
    kwargs = case_info.get("kwargs", {}) if isinstance(case_info, dict) else {}
    kwargs = {key: value for key, value in kwargs.items() if key in _CASE_KWARGS}
    return create_case(input_path, model_type="csv", **kwargs)


def _read_saved_setpoints(
    results_path: Path, case: Optional[Case]
) -> tuple[Case, dict[str, Optional[pd.DataFrame]]]:
    if not results_path.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_path}")

    # Generator files are always written by OPF results; load files are optional
    # but must come as a pair; battery files are optional.
    missing = [
        f"{n}.csv" for n in _GEN_FILES if not (results_path / f"{n}.csv").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Saved results are missing required setpoint file(s): " + ", ".join(missing)
        )
    present_loads = [n for n in _LOAD_FILES if (results_path / f"{n}.csv").is_file()]
    if present_loads and len(present_loads) != len(_LOAD_FILES):
        missing = [f"{n}.csv" for n in _LOAD_FILES if n not in present_loads]
        raise FileNotFoundError(
            "Saved load setpoints are incomplete; missing: " + ", ".join(missing)
        )

    if case is None:
        case = _case_from_snapshot(results_path)

    frames: dict[str, Optional[pd.DataFrame]] = {
        n: _read_frame(results_path, n) for n in _GEN_FILES
    }
    for n in _LOAD_FILES:
        frames[n] = _read_frame(results_path, n) if present_loads else None

    if case.gen_data is not None and len(case.gen_data):
        for n in _GEN_FILES:
            if frames[n] is None:
                raise ValueError(
                    f"Saved {n}.csv is empty but the selected case has generators"
                )
    for n in _LOAD_FILES:
        _validate_ids(n, frames[n], case.bus_data, "buses")
    for n in _GEN_FILES:
        _validate_ids(n, frames[n], case.gen_data, "generators")

    # Battery setpoints are only consumed when the case has batteries.
    has_bats = case.bat_data is not None and len(case.bat_data) > 0
    for n in _BAT_FILES:
        frames[n] = None
        if has_bats and (results_path / f"{n}.csv").is_file():
            frames[n] = _read_frame(results_path, n)
            _validate_ids(n, frames[n], case.bat_data, "batteries")
    return case, frames


def run_fbs_from_saved_results(
    results_dir: Path | str,
    case: Optional[Case] = None,
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> PowerFlowResult:
    """Run FBS using setpoints saved by :meth:`PowerFlowResult.save`.

    The recorded OPF is never re-executed. When ``case`` is omitted, the saved
    ``input`` snapshot is rebuilt with :func:`distopf.api.create_case`.

    Raises
    ------
    FileNotFoundError
        If the result directory, input snapshot or a required setpoint CSV is
        missing.
    ValueError
        If the saved setpoints cannot be applied to the selected case.
    """
    case, frames = _read_saved_setpoints(Path(results_dir), case)
    return _solve_with_setpoints(
        case,
        frames,
        max_iterations=max_iterations,
        tolerance=tolerance,
        verbose=verbose,
    )


def replay_exact_power_flow(
    results_dir: Path | str,
    *,
    output_dir: Optional[Path | str] = None,
    case: Optional[Case] = None,
    save: bool = True,
    overwrite: bool = False,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> PowerFlowResult:
    """Replay a saved OPF operating point through the exact FBS power flow.

    The saved OPF result is treated as a set of operating-point decisions; the
    recorded optimization is not run again. By default, the result is saved in an
    ``exact`` subdirectory of ``results_dir``.
    """
    results_path = Path(results_dir)
    exact_path = Path(output_dir) if output_dir is not None else results_path / "exact"
    if save and exact_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Exact power-flow result directory already exists: {exact_path}. "
                "Pass overwrite=True to replace it."
            )
        import shutil

        shutil.rmtree(exact_path)

    result = run_fbs_from_saved_results(
        results_path,
        case=case,
        max_iterations=max_iterations,
        tolerance=tolerance,
        verbose=verbose,
    )
    _tag_exact(result, source_results_dir=str(results_path))
    if save:
        result.save(exact_path)
    return result


# -------------------------------------------------------------------------
# Comparison with a reference solver and/or OpenDSS
# -------------------------------------------------------------------------


def _comparison_stats(comparison_df: pd.DataFrame) -> dict:
    if len(comparison_df) == 0:
        return {
            "max_abs_diff": 0,
            "mean_abs_diff": 0,
            "max_rel_diff": 0,
            "mean_rel_diff": 0,
        }
    return {
        "max_abs_diff": comparison_df["abs_diff"].max(),
        "mean_abs_diff": comparison_df["abs_diff"].mean(),
        "max_rel_diff": comparison_df["rel_diff_pct"].max(),
        "mean_rel_diff": comparison_df["rel_diff_pct"].mean(),
    }


def _compare_with_ref_frame(
    fbs_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    keys: list[str],
    labels: dict[str, str],
    rel_floor: float,
) -> pd.DataFrame:
    """Phase-by-phase comparison of an FBS frame with a reference frame.

    Rows are matched on ``keys``; the first matching reference row is used.
    ``labels`` maps output column names to FBS column names.
    """
    rows = []
    if ref_df is None or len(ref_df) == 0:
        return pd.DataFrame(rows)
    ref = ref_df.drop_duplicates(keys, keep="first").set_index(keys)
    for _, row in fbs_df.iterrows():
        key = tuple(row[k] for k in keys) if len(keys) > 1 else row[keys[0]]
        if key not in ref.index:
            continue
        ref_row = ref.loc[key]
        for phase in "abc":
            fbs_val = row.get(phase, np.nan)
            ref_val = ref_row.get(phase, np.nan)
            if pd.isna(fbs_val) and pd.isna(ref_val):
                continue
            fbs_val = 0.0 if pd.isna(fbs_val) else float(fbs_val)
            ref_val = 0.0 if pd.isna(ref_val) else float(ref_val)
            if abs(ref_val) <= 1e-6:  # only compare significant values
                continue
            diff = abs(fbs_val - ref_val)
            rows.append(
                {
                    **{out: row[src] for out, src in labels.items()},
                    "phase": phase,
                    "fbs_value": fbs_val,
                    "ref_value": ref_val,
                    "abs_diff": diff,
                    "rel_diff_pct": diff / max(abs(ref_val), rel_floor) * 100,
                }
            )
    return pd.DataFrame(rows)


def compare_with_reference(
    fbs: FBS,
    model=None,
    dss_parser=None,
    verbose: bool = True,
    t: Optional[int] = None,
) -> dict:
    """Compare one period of an FBS solution with a reference solver and/or OpenDSS.

    Parameters
    ----------
    fbs : FBS
        A solved FBS instance.
    model : object, optional
        Reference solver with get_voltages, get_p_flows, get_q_flows methods.
    dss_parser : DSSToCSVConverter, optional
        OpenDSS parser with get_v_solved, get_apparent_power_flows methods.
    verbose : bool
        Print a comparison summary.
    t : int, optional
        Period of the FBS solution to compare. Defaults to the first solved step.
    """
    fbs.results()  # raises if solve() has not been run
    t = fbs.times[0] if t is None else t
    fbs_voltages = _at_period(fbs.get_voltages(), t).reset_index(drop=True)
    fbs_p_flows = _at_period(fbs.get_p_flows(), t).reset_index(drop=True)
    fbs_q_flows = _at_period(fbs.get_q_flows(), t).reset_index(drop=True)
    comparisons: dict = {}

    if model is not None:
        from distopf.matrix_models.matrix_bess.objectives import cp_obj_none
        from distopf.matrix_models.matrix_bess.solvers import cvxpy_solve

        res = cvxpy_solve(model, cp_obj_none)
        bus_labels = {"bus_id": "id", "bus_name": "name"}
        flow_labels = {"from_bus": "fb", "to_bus": "tb"}
        comparisons["reference"] = {
            "voltage_comparison": _compare_with_ref_frame(
                fbs_voltages, model.get_voltages(res.x), ["id"], bus_labels, 1e-6
            ),
            "p_flow_comparison": _compare_with_ref_frame(
                fbs_p_flows, model.get_p_flows(res.x), ["fb", "tb"], flow_labels, 1e-3
            ),
            "q_flow_comparison": _compare_with_ref_frame(
                fbs_q_flows, model.get_q_flows(res.x), ["fb", "tb"], flow_labels, 1e-3
            ),
        }

    if dss_parser is not None:
        dss_voltages = dss_parser.get_v_solved().reset_index(drop=True)
        dss_power_flows = dss_parser.get_apparent_power_flows()

        voltage_comparison_dss = []
        for _, fbs_row in fbs_voltages.iterrows():
            # Match by name first, then fall back to positional ID.
            dss_row = None
            matches = dss_voltages[dss_voltages["name"] == fbs_row["name"]]
            if len(matches) > 0:
                dss_row = matches.iloc[0]
            elif int(fbs_row["id"]) < len(dss_voltages):
                dss_row = dss_voltages.iloc[int(fbs_row["id"])]
            if dss_row is None:
                continue
            for phase in "abc":
                fbs_val = fbs_row[phase]
                dss_val = dss_row[phase] if phase in dss_row.index else np.nan
                if pd.isna(fbs_val) or pd.isna(dss_val) or abs(dss_val) <= 1e-6:
                    continue
                diff = abs(fbs_val - dss_val)
                voltage_comparison_dss.append(
                    {
                        "bus_id": fbs_row["id"],
                        "bus_name": fbs_row["name"],
                        "phase": phase,
                        "fbs_value": fbs_val,
                        "dss_value": dss_val,
                        "abs_diff": diff,
                        "rel_diff_pct": diff / max(abs(dss_val), 1e-6) * 100,
                    }
                )

        p_index = fbs_p_flows.drop_duplicates(["fb", "tb"]).set_index(["fb", "tb"])
        q_index = fbs_q_flows.drop_duplicates(["fb", "tb"]).set_index(["fb", "tb"])
        power_flow_comparison_dss = []
        for _, dss_flow in dss_power_flows.iterrows():
            key = (dss_flow["fb"], dss_flow["tb"])
            if key not in p_index.index or key not in q_index.index:
                continue
            fbs_p_row, fbs_q_row = p_index.loc[key], q_index.loc[key]
            for phase in "abc":
                dss_complex = dss_flow[phase]
                if not (pd.notna(dss_complex) and isinstance(dss_complex, complex)):
                    continue
                for kind, dss_val, fbs_val in (
                    ("P", dss_complex.real, fbs_p_row[phase]),
                    ("Q", dss_complex.imag, fbs_q_row[phase]),
                ):
                    if abs(dss_val) <= 1e-6:
                        continue
                    fbs_val = 0.0 if pd.isna(fbs_val) else fbs_val
                    diff = abs(fbs_val - dss_val)
                    power_flow_comparison_dss.append(
                        {
                            "from_bus": key[0],
                            "to_bus": key[1],
                            "phase": phase,
                            "type": kind,
                            "fbs_value": fbs_val,
                            "dss_value": dss_val,
                            "abs_diff": diff,
                            "rel_diff_pct": diff / max(abs(dss_val), 1e-3) * 100,
                        }
                    )

        comparisons["opendss"] = {
            "voltage_comparison": pd.DataFrame(voltage_comparison_dss),
            "power_flow_comparison": pd.DataFrame(
                power_flow_comparison_dss,
                columns=[
                    "from_bus",
                    "to_bus",
                    "phase",
                    "type",
                    "fbs_value",
                    "dss_value",
                    "abs_diff",
                    "rel_diff_pct",
                ],
            ),
        }

    if verbose:
        print("\n" + "=" * 80)
        print(f"COMPREHENSIVE COMPARISON RESULTS (t={t})")
        print("=" * 80)

        if "reference" in comparisons:
            ref = comparisons["reference"]
            v_stats = _comparison_stats(ref["voltage_comparison"])
            p_stats = _comparison_stats(ref["p_flow_comparison"])
            q_stats = _comparison_stats(ref["q_flow_comparison"])
            print("\nCOMPARISON WITH REFERENCE SOLVER:")
            print(f"  Voltage - Max Abs Diff: {v_stats['max_abs_diff']:.6f} pu")
            print(f"  Voltage - Mean Abs Diff: {v_stats['mean_abs_diff']:.6f} pu")
            print(f"  Voltage - Max Rel Diff: {v_stats['max_rel_diff']:.3f}%")
            print(f"  P Flow - Max Abs Diff: {p_stats['max_abs_diff']:.6f} MW")
            print(f"  P Flow - Mean Abs Diff: {p_stats['mean_abs_diff']:.6f} MW")
            print(f"  Q Flow - Max Abs Diff: {q_stats['max_abs_diff']:.6f} MVar")
            print(f"  Q Flow - Mean Abs Diff: {q_stats['mean_abs_diff']:.6f} MVar")

        if "opendss" in comparisons:
            dss = comparisons["opendss"]
            flows = dss["power_flow_comparison"]
            v_stats = _comparison_stats(dss["voltage_comparison"])
            p_stats = _comparison_stats(flows[flows["type"] == "P"])
            q_stats = _comparison_stats(flows[flows["type"] == "Q"])
            print("\nCOMPARISON WITH OPENDSS:")
            print(f"  Voltage - Max Abs Diff: {v_stats['max_abs_diff']:.6f} pu")
            print(f"  Voltage - Mean Abs Diff: {v_stats['mean_abs_diff']:.6f} pu")
            print(f"  Voltage - Max Rel Diff: {v_stats['max_rel_diff']:.3f}%")
            print(f"  P Flow - Max Abs Diff: {p_stats['max_abs_diff']:.6f} MW")
            print(f"  P Flow - Mean Abs Diff: {p_stats['mean_abs_diff']:.6f} MW")
            print(f"  Q Flow - Max Abs Diff: {q_stats['max_abs_diff']:.6f} MVAr")
            print(f"  Q Flow - Mean Abs Diff: {q_stats['mean_abs_diff']:.6f} MVAr")

            if len(dss["voltage_comparison"]) > 0:
                print("\n  Largest voltage differences vs OpenDSS:")
                worst = dss["voltage_comparison"].nlargest(3, "abs_diff")
                for _, row in worst.iterrows():
                    print(
                        f"    Bus {row['bus_id']} ({row['bus_name']}) "
                        f"Phase {row['phase'].upper()}: "
                        f"FBS={row['fbs_value']:.6f}, DSS={row['dss_value']:.6f}, "
                        f"Diff={row['abs_diff']:.6f} ({row['rel_diff_pct']:.2f}%)"
                    )
            if len(flows) > 0:
                print("\n  Largest P flow differences vs OpenDSS:")
                worst = flows[flows["type"] == "P"].nlargest(3, "abs_diff")
                for _, row in worst.iterrows():
                    print(
                        f"    Branch {row['from_bus']}->{row['to_bus']} "
                        f"Phase {row['phase'].upper()}: "
                        f"FBS={row['fbs_value']:.6f}, DSS={row['dss_value']:.6f}, "
                        f"Diff={row['abs_diff']:.6f} ({row['rel_diff_pct']:.2f}%)"
                    )

    return comparisons
