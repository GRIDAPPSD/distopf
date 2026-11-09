import copy
import re
import warnings

import pandas as pd
from typing import Optional


GEN_COLUMN_RENAMES = {
    "pa": "p_a",
    "pb": "p_b",
    "pc": "p_c",
    "qa": "q_a",
    "qb": "q_b",
    "qc": "q_c",
    "sa_max": "s_a_max",
    "sb_max": "s_b_max",
    "sc_max": "s_c_max",
    "qa_max": "q_a_max",
    "qb_max": "q_b_max",
    "qc_max": "q_c_max",
    "qa_min": "q_a_min",
    "qb_min": "q_b_min",
    "qc_min": "q_c_min",
    "ps1": "p_s1",
    "ps2": "p_s2",
    "ps1s2": "p_s1s2",
    "qs1": "q_s1",
    "qs2": "q_s2",
    "qs1s2": "q_s1s2",
    "ss1_max": "s_s1_max",
    "ss2_max": "s_s2_max",
    "ss1s2_max": "s_s1s2_max",
    "qs1_max": "q_s1_max",
    "qs2_max": "q_s2_max",
    "qs1s2_max": "q_s1s2_max",
    "qs1_min": "q_s1_min",
    "qs2_min": "q_s2_min",
    "qs1s2_min": "q_s1s2_min",
}

CAP_COLUMN_RENAMES = {
    "qa": "q_a",
    "qb": "q_b",
    "qc": "q_c",
}

BRANCH_COLUMN_RENAMES = {
    "raa": "r_aa",
    "rab": "r_ab",
    "rac": "r_ac",
    "rbb": "r_bb",
    "rbc": "r_bc",
    "rcc": "r_cc",
    "xaa": "x_aa",
    "xab": "x_ab",
    "xac": "x_ac",
    "xbb": "x_bb",
    "xbc": "x_bc",
    "xcc": "x_cc",
    "sa_max": "s_a_max",
    "sb_max": "s_b_max",
    "sc_max": "s_c_max",
}


REG_RANGE_DEFAULTS = {"ratio_min": 0.9, "ratio_max": 1.1, "num_taps": 32}


def _rename_known_columns(df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    available = {
        k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns
    }
    if available:
        return df.rename(columns=available)
    return df


def get(s: pd.Series, i, default=None):
    """
    Get value at index i from a Series. Return default if it does not exist.
    Parameters
    ----------
    s : pd.Series
    i : index or key for eries
    default : value to return if it fails

    Returns
    -------
    value: value at index i or default if it doesn't exist.
    """
    try:
        return s.loc[i]
    except (KeyError, ValueError, IndexError):
        return default


def require_legacy_device_schema(case, solver_name: str) -> None:
    """Raise if any device table uses the multi-device ``device_name`` schema."""
    for table in ("gen_data", "cap_data", "bat_data"):
        data = getattr(case, table, None)
        if data is not None and "device_name" in data.columns:
            raise ValueError(
                f"{table} uses the 'device_name'/'bus_name' schema, which {solver_name} "
                "does not support. Use the 'pyomo' or 'cvxpy' wrapper, or the legacy "
                "'id'/'name' schema."
            )


_DEVICE_TABLES = ("gen_data", "cap_data", "bat_data")
_POWER_COLUMN = re.compile(r"[pq](_(a|b|c|s1|s2))?")
_PHASE_TOKEN = re.compile(r"s1|s2|[abc]")


def legacy_device_case(case, include_gen_shape: bool = True):
    """Return ``case`` with ``device_name`` tables merged to one legacy row per bus.

    FBS and ``plot_network`` read one generator, capacitor and battery row per bus,
    keyed by bus ``id``. The merged tables keep only what they read: summed
    ``p``/``q`` columns, the union of phases and, when ``include_gen_shape`` is
    True, the generator shape (FBS needs one shape per bus and raises otherwise).
    Returns ``case`` itself when there is nothing to merge.
    """
    to_merge = [
        table
        for table in _DEVICE_TABLES
        if getattr(case, table, None) is not None
        and "device_name" in getattr(case, table).columns
    ]
    if not to_merge:
        return case
    bus_ids = dict(zip(case.bus_data["name"].astype(str), case.bus_data["id"]))
    legacy = copy.copy(case)
    for table in to_merge:
        merged = _merge_devices_by_bus(
            table, getattr(case, table), bus_ids, include_gen_shape
        )
        setattr(legacy, table, merged)
    return legacy


def _merge_devices_by_bus(
    table: str, data: pd.DataFrame, bus_ids: dict, include_gen_shape: bool
) -> pd.DataFrame:
    bus_names = data["bus_name"].astype(str)
    ids = bus_names.map(bus_ids)
    unknown = sorted(bus_names[ids.isna()].unique())
    if unknown:
        raise ValueError(f"{table} references unknown bus names: {unknown}")
    by_bus = data.groupby(ids.astype(int).rename("id"))
    power = [column for column in data.columns if _POWER_COLUMN.fullmatch(column)]
    merged = by_bus[power].sum() if power else pd.DataFrame(index=by_bus.size().index)
    merged["name"] = by_bus["bus_name"].first().astype(str)
    merged["phases"] = by_bus["phases"].agg(_union_phases)
    if include_gen_shape and "gen_shape" in data.columns:
        shapes = data["gen_shape"].fillna("").astype(str).str.strip()
        shapes_by_bus = shapes.groupby(ids.astype(int)).unique()
        mixed = shapes_by_bus[shapes_by_bus.map(len) > 1]
        if len(mixed):
            raise ValueError(
                "The FBS warm start needs one gen_shape per bus, but buses "
                f"{list(mixed.index)} have generators with different shapes. "
                "Pass initialize=None to skip the warm start."
            )
        merged["gen_shape"] = shapes_by_bus.str[0]
    return merged.reset_index()


def _union_phases(phases: pd.Series) -> str:
    seen: list[str] = []
    for value in phases.astype(str).str.lower():
        for token in _PHASE_TOKEN.findall(value):
            if token not in seen:
                seen.append(token)
    return "".join(seen)


def handle_gen_input(gen_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    if gen_data is None:
        return pd.DataFrame(
            columns=[
                "id",
                "name",
                "p_a",
                "p_b",
                "p_c",
                "q_a",
                "q_b",
                "q_c",
                "s_a_max",
                "s_b_max",
                "s_c_max",
                "phases",
                "q_a_max",
                "q_b_max",
                "q_c_max",
                "q_a_min",
                "q_b_min",
                "q_c_min",
                "control_variable",
                "gen_shape",
            ]
        )
    gen_data = _rename_known_columns(gen_data, GEN_COLUMN_RENAMES)
    if "device_name" in gen_data.columns:
        return gen_data.reset_index(drop=True)
    if "control_variable" not in gen_data.columns:
        gen_data["control_variable"] = ""
    else:
        gen_data["control_variable"] = gen_data["control_variable"].fillna("")
    if "gen_shape" not in gen_data.columns:
        gen_data["gen_shape"] = "PV"
        warnings.warn(
            "gen_data has no 'gen_shape' column. Defaulting to 'PV'. "
            "If a 'PV' column exists in schedules, it will be applied to all generators. "
            "Add a 'gen_shape' column to gen_data to control per-generator schedule shapes.",
            UserWarning,
            stacklevel=2,
        )
    gen = gen_data.sort_values(by="id", ignore_index=True)
    gen.index = gen.id.to_numpy() - 1
    return gen


def handle_cap_input(cap_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    if cap_data is None:
        return pd.DataFrame(
            columns=[
                "id",
                "name",
                "q_a",
                "q_b",
                "q_c",
                "phases",
            ]
        )
    cap_data = _rename_known_columns(cap_data, CAP_COLUMN_RENAMES)
    if "device_name" in cap_data.columns:
        return cap_data.reset_index(drop=True)
    cap = cap_data.sort_values(by="id", ignore_index=True)
    cap.index = cap.id.to_numpy() - 1
    return cap


def handle_reg_input(reg_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Normalize regulator data, treating tap positions as authoritative."""
    if reg_data is None:
        return pd.DataFrame(
            columns=[
                "fb",
                "tb",
                "phases",
                "tap_a",
                "tap_b",
                "tap_c",
                "ratio_a",
                "ratio_b",
                "ratio_c",
                *REG_RANGE_DEFAULTS,
            ]
        )
    reg = reg_data.sort_values(by="tb", ignore_index=True)
    reg.index = reg.tb.to_numpy() - 1
    for key, default in REG_RANGE_DEFAULTS.items():
        if key not in reg.columns:
            reg[key] = default
        reg[key] = pd.to_numeric(reg[key], errors="coerce").fillna(default)
    step = (reg["ratio_max"] - reg["ratio_min"]) / reg["num_taps"]
    for ph in "abc":
        tap_key = f"tap_{ph}"
        ratio_key = f"ratio_{ph}"
        if tap_key not in reg.columns:
            if ratio_key not in reg.columns:
                continue
            ratio_values = reg[ratio_key]
            ratio_provided = ratio_values.notna() & ratio_values.astype(
                str
            ).str.strip().ne("")
            if ratio_provided.any():
                raise ValueError(
                    f"{tap_key} is required; {ratio_key} is derived from {tap_key}"
                )
            reg[tap_key] = 0.0

        tap_values = reg[tap_key]
        tap_missing = tap_values.isna() | tap_values.astype(str).str.strip().eq("")
        if ratio_key in reg.columns:
            ratio_values = reg[ratio_key]
            ratio_provided = ratio_values.notna() & ratio_values.astype(
                str
            ).str.strip().ne("")
            if (ratio_provided & tap_missing).any():
                raise ValueError(
                    f"{tap_key} is required; {ratio_key} is derived from {tap_key}"
                )
        reg.loc[tap_missing, tap_key] = 0.0
        reg[ratio_key] = 1 + step * reg[tap_key]
    return reg


def handle_branch_input(branch_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    if branch_data is None:
        raise ValueError("Branch data must be provided.")
    branch_data = _rename_known_columns(branch_data, BRANCH_COLUMN_RENAMES)
    branch = branch_data.sort_values(by="tb", ignore_index=True)
    branch = pd.DataFrame(branch.loc[branch.status != "OPEN", :])
    return branch


def handle_bus_input(bus_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    if bus_data is None:
        raise ValueError("Bus data must be provided.")
    type_dict = {
        "id": int,
        "name": str,
        "pl_a": float,
        "ql_a": float,
        "pl_b": float,
        "ql_b": float,
        "pl_c": float,
        "ql_c": float,
        "bus_type": str,
        "v_a": float,
        "v_b": float,
        "v_c": float,
        "v_ln_base": float,
        "s_base": float,
        "v_min": float,
        "v_max": float,
        "cvr_p": float,
        "cvr_q": float,
        "phases": str,
        "has_gen": bool,
        "has_load": bool,
        "has_cap": bool,
        "latitude": float,
        "longitude": float,
        "load_shape": str,
    }
    if "load_shape" not in bus_data.columns:
        bus_data["load_shape"] = "default"
    for c, t in type_dict.items():
        if c not in bus_data.columns:
            bus_data[c] = t()
    bus = bus_data.astype(type_dict)
    bus = bus.sort_values(by="id", ignore_index=True)
    bus.index = bus.id.to_numpy() - 1
    return bus


def handle_schedules_input(loadshape_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    if loadshape_data is None:
        return pd.DataFrame(
            columns=[
                "time",
                "M",
            ]
        )
    loadshape = loadshape_data.sort_values(by="time", ignore_index=True)
    loadshape.index = loadshape.time.to_numpy()
    return loadshape


def handle_pv_loadshape_input(
    pv_loadshape_data: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if pv_loadshape_data is None:
        return pd.DataFrame(
            columns=[
                "time",
                "PV",
            ]
        )
    pv_loadshape = pv_loadshape_data.sort_values(by="time", ignore_index=True)
    pv_loadshape.index = pv_loadshape.time.to_numpy()
    return pv_loadshape


def handle_bat_input_depricated(bat_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    if bat_data is None:
        return pd.DataFrame(
            columns=[
                "id",
                "name",
                "nc_a",
                "nc_b",
                "nc_c",
                "nd_a",
                "nd_b",
                "nd_c",
                "hmax_a",
                "hmax_b",
                "hmax_c",
                "Pb_max_a",
                "Pb_max_b",
                "Pb_max_c",
                "bmin_a",
                "bmin_b",
                "bmin_c",
                "bmax_a",
                "bmax_b",
                "bmax_c",
                "b0_a",
                "b0_b",
                "b0_c",
                "phases",
            ]
        )
    if "b0_a" not in bat_data.columns:
        bat_data["b0_a"] = bat_data.bmin_a
    if "b0_b" not in bat_data.columns:
        bat_data["b0_b"] = bat_data.bmin_a
    if "b0_c" not in bat_data.columns:
        bat_data["b0_c"] = bat_data.bmin_a
    bat = bat_data.sort_values(by="id", ignore_index=True)
    bat.index = bat.id.to_numpy() - 1
    return bat


def handle_bat_input(bat_data: Optional[pd.DataFrame]) -> pd.DataFrame:
    if bat_data is None:
        return pd.DataFrame(
            columns=[
                "id",
                "name",
                "s_max",
                "phases",
                "energy_capacity",
                "min_soc",
                "max_soc",
                "start_soc",
                "charge_efficiency",
                "discharge_efficiency",
                "control_variable",
            ]
        )
    if "device_name" in bat_data.columns:
        return bat_data.reset_index(drop=True)
    bat = bat_data.sort_values(by="id", ignore_index=True)
    bat.index = bat.id.to_numpy() - 1
    return bat
