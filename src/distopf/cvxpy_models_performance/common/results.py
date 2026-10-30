"""Result container and extraction for vectorized CVXPY models.

DataFrames are built directly from the variable arrays. Layouts are identical to
``distopf.cvxpy_models.common.results`` (wide tables with one column per phase).

Dual convention: for a constraint stored as one inequality or equality the value
is ``constraint.dual_value``. For ranged constraints the value is
``dual(lower) - dual(upper)``. Cone and PSD constraints have no tabulated dual.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from distopf.cvxpy_models_performance.common.model import (
    ConstraintRecord,
    CvxModel,
    VarSpec,
)

_DEVICE_BUS_MAP = {
    "p_gen": "gen_bus_by_device",
    "q_gen": "gen_bus_by_device",
    "q_cap": "cap_bus_by_device",
    "u_cap": "cap_bus_by_device",
    "z_cap": "cap_bus_by_device",
    "p_bat": "bat_bus_by_device",
    "q_bat": "bat_bus_by_device",
    "p_charge": "bat_bus_by_device",
    "p_discharge": "bat_bus_by_device",
    "soc": "bat_bus_by_device",
}
_ZONE_VARS = {"p_max": ["zone", "resource"], "p_zr": ["zone", "resource", "t"]}


def _values_2d(spec: VarSpec) -> np.ndarray:
    value = spec.variable.value
    if value is None:
        return np.full(spec.variable.shape, np.nan)
    return np.asarray(value, dtype=float)


def _long_table(
    model: CvxModel,
    keys: list,
    values: np.ndarray,
    device_buses: dict | None = None,
    value_name: str = "value",
    device_columns: bool = False,
) -> pd.DataFrame:
    """Tidy table with one row per (key, time), key-major, from a ``(n_key, T)`` array.

    Key columns depend on the key width: ``(id,)`` and ``(id, phase)`` give
    ``id, name, t, phase``; ``(fb, tb, phase)`` gives ``fb, tb, from_name,
    to_name, t, phase``. Device-owned quantities (``device_buses`` given) use
    ``(device, phase)`` keys and give ``device_name, id, name, t, phase``.
    ``device_columns`` gives ``device_name, t, phase`` without bus information.
    """
    times = list(model.time_set)
    name_map = model.name_map
    keys = [k if isinstance(k, tuple) else (k,) for k in keys]
    n_t = len(times)
    width = len(keys[0]) if keys else 0
    key_cols: list[Any] = [
        list(np.repeat(np.array([k[i] for k in keys], dtype=object), n_t))
        for i in range(width)
    ]
    frame = {"t": times * len(keys)}
    flat = values.reshape(-1)

    out: dict[str, Any]
    if device_buses is not None:
        devices = list(key_cols[0]) if key_cols else []
        buses = [device_buses[d] for d in devices]
        out = {
            "device_name": devices,
            "id": buses,
            "name": [name_map[b] for b in buses],
            **frame,
            "phase": list(key_cols[1]) if width > 1 else ["value"] * len(flat),
        }
    elif device_columns:
        out = {
            "device_name": list(key_cols[0]) if key_cols else [],
            **frame,
            "phase": list(key_cols[1]) if key_cols else [],
        }
    elif width == 3:
        out = {
            "fb": list(key_cols[0]),
            "tb": list(key_cols[1]),
            "from_name": [name_map.get(i, i) for i in key_cols[0]],
            "to_name": [name_map.get(i, i) for i in key_cols[1]],
            **frame,
            "phase": list(key_cols[2]),
        }
    elif width in (1, 2):
        out = {
            "id": list(key_cols[0]),
            "name": [name_map.get(i, i) for i in key_cols[0]],
            **frame,
            "phase": list(key_cols[1]) if width == 2 else ["value"] * len(flat),
        }
    else:
        return pd.DataFrame(columns=["id", "name", "t", "phase", value_name])
    out[value_name] = flat
    return pd.DataFrame(out)


def _tidy(model: CvxModel, spec: VarSpec) -> pd.DataFrame:
    """Tidy frame for a variable, mirroring ``pyomo_models`` ``get_values_tidy``."""
    name = spec.name
    values = _values_2d(spec)
    if name == "u_reg":
        times = list(model.time_set)
        name_map = model.name_map
        taps = list(spec.axes[1])
        rows = [
            (fb, tb, name_map[fb], name_map[tb], t, ph, tap, values[i, k, j])
            for i, (fb, tb, ph) in enumerate(spec.axes[0])
            for k, tap in enumerate(taps)
            for j, t in enumerate(times)
        ]
        columns = ["fb", "tb", "from_name", "to_name", "t", "phase", "tap", "value"]
        return pd.DataFrame(rows, columns=columns)

    keys = list(spec.axes[0])
    bus_map_name = _DEVICE_BUS_MAP.get(name)
    return _long_table(
        model,
        keys,
        values,
        device_buses=getattr(model, bus_map_name, None) if bus_map_name else None,
        device_columns=name in ("p_mpssd", "q_mpssd"),
    )


def get_values(model: CvxModel, name: str) -> pd.DataFrame:
    """Variable values in wide format (one column per phase)."""
    spec = model.var_specs[name]
    df = _tidy(model, spec)
    if name == "u_reg" or df.empty:
        return df
    if {"fb", "tb", "from_name", "to_name"}.issubset(df.columns):
        index_cols = ["fb", "tb", "from_name", "to_name", "t"]
    elif "device_name" in df.columns:
        index_cols = ["device_name"]
        index_cols.extend(c for c in ("id", "name") if c in df.columns)
        index_cols.append("t")
    else:
        index_cols = ["id", "name", "t"]
    df = df.pivot(index=index_cols, columns="phase", values="value").reset_index()
    df.columns.name = None
    return df


def get_voltages(model: CvxModel) -> pd.DataFrame:
    """Voltage magnitudes (square root of ``v2``) with columns id, name, t, a, b, c."""
    df = _tidy(model, model.var_specs["v2"])
    df["value"] = np.sqrt(df["value"].to_numpy(dtype=float))
    df = df.pivot(
        index=["id", "name", "t"], columns="phase", values="value"
    ).reset_index()
    df.columns.name = None
    return df


def _zone_values(model: CvxModel, spec: VarSpec) -> pd.DataFrame:
    columns = _ZONE_VARS[spec.name]
    values = _values_2d(spec)
    times = list(model.time_set)
    rows = []
    zones, resources = spec.axes[0], spec.axes[1]
    for i, zone in enumerate(zones):
        for k, resource in enumerate(resources):
            if spec.name == "p_max":
                rows.append([zone, resource, values[i, k]])
            else:
                for j, t in enumerate(times):
                    rows.append([zone, resource, t, values[i, k, j]])
    return pd.DataFrame(rows, columns=[*columns, "value"])


def _piece_dual(pieces: list, shape: tuple) -> np.ndarray | None:
    values = []
    for piece in pieces:
        dual = piece.dual_value
        if dual is None or isinstance(dual, (list, tuple)):
            return None
        dual = np.asarray(dual, dtype=float)
        if dual.shape != shape:
            return None
        values.append(dual)
    if len(values) == 2:
        return values[0] - values[1]
    return values[0]


class CvxResult:
    """Result container for vectorized CVXPY OPF solutions."""

    def __init__(
        self,
        model: CvxModel,
        problem: Any,
        solve_time: float | None = None,
        log: str = "",
        objective_value: float | None = None,
        extract_duals: bool = False,
    ):
        self.model = model
        self.results = problem
        self.problem = problem
        self.log = log
        self.voltages = get_voltages(model)
        self.objective_value = objective_value
        self.solve_time = solve_time
        for name in model.var_names:
            if name in _ZONE_VARS:
                setattr(self, name, _zone_values(model, model.var_specs[name]))
            else:
                setattr(self, name, get_values(model, name))

        self.validate_u_reg()
        u_reg = getattr(self, "u_reg", None)
        if u_reg is not None:
            self.reg_taps = u_reg.loc[u_reg.value.round() == 1].copy()
            self.reg_taps["tap"] -= 16
            self.reg_taps = self.reg_taps.drop(columns=["value"]).reset_index(drop=True)
            self.reg_taps = self.reg_taps.pivot(
                index=["fb", "tb", "from_name", "to_name", "t"],
                columns="phase",
                values="tap",
            ).reset_index()

        self._duals_available = extract_duals
        if extract_duals:
            self._populate_common_duals()

    def validate_u_reg(self):
        """Binary variables come back as floats near 0/1; snap and verify."""
        u_reg_df = getattr(self, "u_reg", None)
        if u_reg_df is None:
            return
        snapped = u_reg_df["value"].round()
        if (
            not ((u_reg_df["value"] - snapped).abs() < 1e-6).all()
            or not snapped.isin([0, 1]).all()
        ):
            raise ValueError("u_reg DataFrame contains non-binary values.")
        u_reg_df["value"] = snapped

    def _populate_common_duals(self) -> None:
        for name in (
            "power_balance_p",
            "power_balance_q",
            "voltage_drop",
            "voltage_limits",
        ):
            if name in self.model.constraint_names:
                duals_df = self.get_dual(name)
                if not duals_df.empty:
                    setattr(self, f"dual_{name}", duals_df)

    def get_dual(self, constraint_name: str) -> pd.DataFrame:
        """Dual values of a named constraint as a tidy DataFrame."""
        model = self.model
        if constraint_name not in model.constraint_names:
            return pd.DataFrame()
        record: ConstraintRecord = getattr(model, constraint_name)
        if not record.rows or not record.pieces:
            return pd.DataFrame()
        n_t = len(model.time_set)
        dual = _piece_dual(record.pieces, (len(record.rows), n_t))
        if dual is None:
            return pd.DataFrame()
        return _dual_frame(model, constraint_name, record.rows, dual)

    def get_all_duals(self) -> dict[str, pd.DataFrame]:
        duals = {}
        for name in self.model.constraint_names:
            df = self.get_dual(name)
            if not df.empty:
                duals[name] = df
        return duals


def _dual_device_map(model: CvxModel, name: str) -> dict | None:
    if name.startswith("gen_"):
        return getattr(model, "gen_bus_by_device", None)
    if name.startswith("mpssd_"):
        return getattr(model, "mpssd_bus_by_device", None)
    if name.startswith(("cap_", "capacitor_")) or name == "z_cap_bounds":
        return getattr(model, "cap_bus_by_device", None)
    if name.startswith(("bat_", "battery_")) or name in ("storage", "net_discharge"):
        return getattr(model, "bat_bus_by_device", None)
    return None


def _dual_frame(
    model: CvxModel, name: str, rows: list, dual: np.ndarray
) -> pd.DataFrame:
    """Tidy dual table. Device constraints also report device, bus id and bus name."""
    keys = [r if isinstance(r, tuple) else (r,) for r in rows]
    width = len(keys[0])
    bus_by_device = _dual_device_map(model, name)
    if bus_by_device is not None and width in (1, 2):
        return _long_table(
            model, keys, dual, device_buses=bus_by_device, value_name="dual"
        ).drop(columns=[] if width == 2 else ["phase"])
    if width > 3:
        return pd.DataFrame()
    return _long_table(model, keys, dual, value_name="dual")


__all__ = ["CvxResult", "get_values", "get_voltages"]
