import logging
import multiprocessing as mp
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

import distopf as opf
from distopf.api import Case
from distopf.distributed.spatial.decompose import decompose
from distopf.results import PowerFlowResult


logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s | %(message)s"))
    logger.addHandler(_handler)


BOUNDARY_KEYS = ["name", "t"]
PHASES = ["a", "b", "c"]


# =============================================================================
# ADMM boundary data structures
# =============================================================================


@dataclass
class BoundaryVars:
    """Legacy-compatible representation of one area's ENApp boundary values."""

    s_up: pd.DataFrame
    v_down: pd.DataFrame


@dataclass
class CutLocalVars:
    """The four local variable copies associated with one inter-area cut.

    For a cut from ``up_area`` to ``down_area``:

    - ``v_up`` is the upstream area's copy of the boundary voltage.
    - ``v_down`` is the downstream area's swing-voltage copy.
    - ``s_up`` is the upstream area's boundary-load/flow copy.
    - ``s_down`` is the downstream area's swing-power copy.

    Power is represented as complex values P + jQ. Voltage is represented
    using the same real-valued voltage quantity returned by the OPF result.
    """

    v_up: pd.DataFrame
    v_down: pd.DataFrame
    s_up: pd.DataFrame
    s_down: pd.DataFrame


@dataclass
class CutDualVars:
    """Scaled ADMM dual variables for one inter-area cut."""

    u_v_up: pd.DataFrame
    u_v_down: pd.DataFrame
    u_s_up: pd.DataFrame
    u_s_down: pd.DataFrame


@dataclass
class CutConsensusVars:
    """ADMM consensus variables for one inter-area cut."""

    z_v: pd.DataFrame
    z_s: pd.DataFrame


CutKey = tuple[str, str]


# =============================================================================
# Boundary DataFrame utilities
# =============================================================================


def _empty_boundary_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=BOUNDARY_KEYS + PHASES)


def _validate_boundary_frame(
    frame: pd.DataFrame,
    field_name: str,
    *,
    allow_empty: bool = False,
) -> None:
    required = set(BOUNDARY_KEYS + PHASES)
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"{field_name} is missing required columns: {sorted(missing)}")

    if frame.empty:
        if allow_empty:
            return
        raise ValueError(f"{field_name} contains no boundary rows")

    if frame.duplicated(BOUNDARY_KEYS).any():
        duplicate_rows = frame.loc[
            frame.duplicated(BOUNDARY_KEYS, keep=False),
            BOUNDARY_KEYS,
        ]
        raise ValueError(
            f"{field_name} contains duplicate boundary rows by "
            f"{BOUNDARY_KEYS}: {duplicate_rows.to_dict('records')}"
        )


def _validate_matching_boundary_frames(
    first: pd.DataFrame,
    second: pd.DataFrame,
    first_name: str,
    second_name: str,
) -> None:
    _validate_boundary_frame(first, first_name)
    _validate_boundary_frame(second, second_name)

    first_keys = pd.MultiIndex.from_frame(first[BOUNDARY_KEYS])
    second_keys = pd.MultiIndex.from_frame(second[BOUNDARY_KEYS])

    missing_from_first = second_keys.difference(first_keys)
    missing_from_second = first_keys.difference(second_keys)

    if len(missing_from_first) or len(missing_from_second):
        raise ValueError(
            f"Boundary keys do not match between {first_name} and {second_name}. "
            f"Missing from {first_name}: {missing_from_first.tolist()}; "
            f"missing from {second_name}: {missing_from_second.tolist()}"
        )


def _sort_boundary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.loc[:, BOUNDARY_KEYS + PHASES]
        .sort_values(BOUNDARY_KEYS)
        .reset_index(drop=True)
    )


def _rename_boundary(frame: pd.DataFrame, boundary_name: str) -> pd.DataFrame:
    result = frame.loc[:, BOUNDARY_KEYS + PHASES].copy()
    result["name"] = str(boundary_name)
    return _sort_boundary_frame(result)


def _zero_like(frame: pd.DataFrame, *, complex_values: bool) -> pd.DataFrame:
    result = frame.loc[:, BOUNDARY_KEYS + PHASES].copy()
    value = 0.0 + 0.0j if complex_values else 0.0
    result.loc[:, PHASES] = value
    return _sort_boundary_frame(result)


def _binary_boundary_operation(
    first: pd.DataFrame,
    second: pd.DataFrame,
    operation: Callable,
    first_name: str,
    second_name: str,
) -> pd.DataFrame:
    _validate_matching_boundary_frames(
        first,
        second,
        first_name,
        second_name,
    )

    merged = first.merge(
        second,
        on=BOUNDARY_KEYS,
        how="inner",
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )

    result = merged.loc[:, BOUNDARY_KEYS].copy()

    for phase in PHASES:
        result[phase] = operation(
            merged[f"{phase}_first"],
            merged[f"{phase}_second"],
        )

    return _sort_boundary_frame(result)


def _boundary_add(
    first: pd.DataFrame,
    second: pd.DataFrame,
    first_name: str = "first",
    second_name: str = "second",
) -> pd.DataFrame:
    return _binary_boundary_operation(
        first,
        second,
        lambda a, b: a + b,
        first_name,
        second_name,
    )


def _boundary_subtract(
    first: pd.DataFrame,
    second: pd.DataFrame,
    first_name: str = "first",
    second_name: str = "second",
) -> pd.DataFrame:
    return _binary_boundary_operation(
        first,
        second,
        lambda a, b: a - b,
        first_name,
        second_name,
    )


def _boundary_average(
    first: pd.DataFrame,
    second: pd.DataFrame,
    first_name: str = "first",
    second_name: str = "second",
) -> pd.DataFrame:
    return _binary_boundary_operation(
        first,
        second,
        lambda a, b: 0.5 * (a + b),
        first_name,
        second_name,
    )


def _boundary_scale(frame: pd.DataFrame, scale: float) -> pd.DataFrame:
    result = frame.loc[:, BOUNDARY_KEYS + PHASES].copy()
    result.loc[:, PHASES] = result.loc[:, PHASES] * scale
    return _sort_boundary_frame(result)


def _boundary_max_abs(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0

    values = frame.loc[:, PHASES].to_numpy()

    if values.size == 0:
        return 0.0

    magnitudes = np.abs(values.astype(complex))
    finite = magnitudes[np.isfinite(magnitudes)]

    if finite.size == 0:
        return 0.0

    return float(np.max(finite))


def _force_real_voltage(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, BOUNDARY_KEYS + PHASES].copy()

    for phase in PHASES:
        values = np.asarray(result[phase])
        imaginary_error = np.max(np.abs(np.imag(values))) if values.size else 0.0

        if imaginary_error > 1e-10:
            raise ValueError(
                "A voltage boundary frame acquired a non-negligible imaginary "
                f"component for phase {phase}: {imaginary_error}"
            )

        result[phase] = np.real(values).astype(float)

    return _sort_boundary_frame(result)


# =============================================================================
# Boundary-result parsing
# =============================================================================


def _get_swing_name(case: Case) -> str:
    if case.bus_data is None:
        raise ValueError("Case has no bus_data")

    swing_names = (
        case.bus_data.loc[
            case.bus_data.bus_type.isin([opf.SWING_BUS, opf.SWING_FREE, "IN"]),
            "name",
        ]
        .astype(str)
        .tolist()
    )

    if len(swing_names) != 1:
        raise ValueError(f"Expected exactly one swing bus, found {len(swing_names)}")

    return swing_names[0]


def parse_v_up(case: Case, result: PowerFlowResult) -> pd.DataFrame:
    """Parse the solved voltage at an area's swing bus."""

    swing = _get_swing_name(case)
    voltages = result.voltages

    if voltages is None:
        return _empty_boundary_frame()

    selected = voltages.loc[
        voltages["name"].astype(str) == swing,
        ["name", "t"] + PHASES,
    ].copy()

    return _force_real_voltage(selected)


def parse_v_swing_local(
    case: Case,
    result: PowerFlowResult,
) -> pd.DataFrame:
    return parse_v_up(case, result)


def parse_v_dn(
    case: Case,
    result: PowerFlowResult,
    down_buses: list[str],
) -> pd.DataFrame:
    """Parse the upstream area's voltage copies at downstream dummy buses."""

    if case.bus_data is None:
        raise ValueError("Case has no bus_data")

    voltages = result.voltages

    if voltages is None:
        return _empty_boundary_frame()

    down_bus_names = {str(name) for name in down_buses}

    selected = voltages.loc[
        voltages["name"].astype(str).isin(down_bus_names),
        ["name", "t"] + PHASES,
    ].copy()

    return _force_real_voltage(selected)


def _combine_p_q_frames(
    p: pd.DataFrame,
    q: pd.DataFrame,
    endpoint_column: str,
    output_name_column: str,
) -> pd.DataFrame:
    columns = [endpoint_column, "t"] + PHASES

    p_selected = p.loc[:, columns].copy()
    q_selected = q.loc[:, columns].copy()

    # Sum multiple branches if more than one branch contributes to the same
    # boundary endpoint and time.
    p_grouped = p_selected.groupby([endpoint_column, "t"], as_index=False)[PHASES].sum()
    q_grouped = q_selected.groupby([endpoint_column, "t"], as_index=False)[PHASES].sum()

    merged = p_grouped.merge(
        q_grouped,
        on=[endpoint_column, "t"],
        suffixes=("_p", "_q"),
        how="inner",
        validate="one_to_one",
    )

    result = merged.loc[:, [endpoint_column, "t"]].copy()
    result["name"] = result[endpoint_column].astype(str)

    for phase in PHASES:
        result[phase] = merged[f"{phase}_p"].astype(float) + 1j * merged[
            f"{phase}_q"
        ].astype(float)

    result = result.rename(columns={"name": output_name_column})

    if output_name_column != "name":
        result["name"] = result[output_name_column].astype(str)

    return _sort_boundary_frame(result)


def parse_s_dn(
    case: Case,
    result: PowerFlowResult,
    down_buses: list[str],
) -> pd.DataFrame:
    """Parse power flowing into downstream dummy-load boundary buses."""

    del case  # The result contains all data needed here.

    p = result.active_power_flows
    q = result.reactive_power_flows

    if p is None or q is None:
        return _empty_boundary_frame()

    down_bus_names = {str(name) for name in down_buses}

    p_selected = p.loc[p["to_name"].astype(str).isin(down_bus_names)].copy()
    q_selected = q.loc[q["to_name"].astype(str).isin(down_bus_names)].copy()

    if p_selected.empty or q_selected.empty:
        return _empty_boundary_frame()

    return _combine_p_q_frames(
        p_selected,
        q_selected,
        endpoint_column="to_name",
        output_name_column="name",
    )


def parse_s_load_local(
    case: Case,
    result: PowerFlowResult,
    down_buses: list[str],
) -> pd.DataFrame:
    return parse_s_dn(case, result, down_buses)


def parse_s_up(
    case: Case,
    result: PowerFlowResult,
) -> pd.DataFrame:
    """Parse total power leaving an area's swing bus."""

    swing = _get_swing_name(case)
    p = result.active_power_flows
    q = result.reactive_power_flows

    if p is None or q is None:
        return _empty_boundary_frame()

    p_selected = p.loc[p["from_name"].astype(str) == swing].copy()
    q_selected = q.loc[q["from_name"].astype(str) == swing].copy()

    if p_selected.empty or q_selected.empty:
        return _empty_boundary_frame()

    result = _combine_p_q_frames(
        p_selected,
        q_selected,
        endpoint_column="from_name",
        output_name_column="name",
    )

    result["name"] = swing
    return _sort_boundary_frame(result)


# =============================================================================
# ADMM cut extraction, consensus, and dual updates
# =============================================================================


def extract_cut_local_vars(
    cases: dict[str, Case],
    all_results: dict[str, PowerFlowResult],
    area_info: dict[str, dict[str, list]],
) -> dict[CutKey, CutLocalVars]:
    """Extract both areas' local copies for every inter-area cut."""

    cut_vars: dict[CutKey, CutLocalVars] = {}

    for down_area, info in area_info.items():
        for up_area in info.get("up_areas", []):
            if up_area not in cases:
                raise KeyError(
                    f"Upstream area {up_area!r} for area {down_area!r} "
                    "is not present in the decomposed cases"
                )

            if down_area not in all_results or up_area not in all_results:
                raise KeyError(
                    f"Missing local OPF result for cut {up_area!r} -> {down_area!r}"
                )

            # Downstream area's local copies.
            v_down = parse_v_swing_local(
                cases[down_area],
                all_results[down_area],
            )
            s_down = parse_s_up(
                cases[down_area],
                all_results[down_area],
            )

            # Upstream area's local copies.
            v_up = parse_v_dn(
                cases[up_area],
                all_results[up_area],
                [down_area],
            )
            s_up = parse_s_load_local(
                cases[up_area],
                all_results[up_area],
                [down_area],
            )

            # Use the downstream area name as the canonical cut identifier.
            v_down = _rename_boundary(v_down, down_area)
            s_down = _rename_boundary(s_down, down_area)
            v_up = _rename_boundary(v_up, down_area)
            s_up = _rename_boundary(s_up, down_area)

            cut_name = f"{up_area}->{down_area}"

            _validate_matching_boundary_frames(
                v_up,
                v_down,
                f"{cut_name}.v_up",
                f"{cut_name}.v_down",
            )
            _validate_matching_boundary_frames(
                s_up,
                s_down,
                f"{cut_name}.s_up",
                f"{cut_name}.s_down",
            )

            cut_vars[(up_area, down_area)] = CutLocalVars(
                v_up=_force_real_voltage(v_up),
                v_down=_force_real_voltage(v_down),
                s_up=s_up,
                s_down=s_down,
            )

    return cut_vars


def initialize_cut_duals(
    cut_vars: dict[CutKey, CutLocalVars],
) -> dict[CutKey, CutDualVars]:
    """Initialize all scaled ADMM dual variables to zero."""

    return {
        cut: CutDualVars(
            u_v_up=_zero_like(local.v_up, complex_values=False),
            u_v_down=_zero_like(local.v_down, complex_values=False),
            u_s_up=_zero_like(local.s_up, complex_values=True),
            u_s_down=_zero_like(local.s_down, complex_values=True),
        )
        for cut, local in cut_vars.items()
    }


def calculate_consensus(
    cut_vars: dict[CutKey, CutLocalVars],
    duals: dict[CutKey, CutDualVars],
) -> dict[CutKey, CutConsensusVars]:
    """Perform the ADMM consensus update.

    For each cut and each boundary quantity:

        z = 0.5 * ((x_up + u_up) + (x_down + u_down))
    """

    consensus: dict[CutKey, CutConsensusVars] = {}

    for cut, local in cut_vars.items():
        if cut not in duals:
            raise KeyError(f"Missing ADMM dual variables for cut {cut}")

        dual = duals[cut]

        v_up_shifted = _boundary_add(
            local.v_up,
            dual.u_v_up,
            "v_up",
            "u_v_up",
        )
        v_down_shifted = _boundary_add(
            local.v_down,
            dual.u_v_down,
            "v_down",
            "u_v_down",
        )

        s_up_shifted = _boundary_add(
            local.s_up,
            dual.u_s_up,
            "s_up",
            "u_s_up",
        )
        s_down_shifted = _boundary_add(
            local.s_down,
            dual.u_s_down,
            "s_down",
            "u_s_down",
        )

        z_v = _boundary_average(
            v_up_shifted,
            v_down_shifted,
            "v_up + u_v_up",
            "v_down + u_v_down",
        )
        z_s = _boundary_average(
            s_up_shifted,
            s_down_shifted,
            "s_up + u_s_up",
            "s_down + u_s_down",
        )

        consensus[cut] = CutConsensusVars(
            z_v=_force_real_voltage(z_v),
            z_s=z_s,
        )

    return consensus


def update_cut_duals(
    duals: dict[CutKey, CutDualVars],
    cut_vars: dict[CutKey, CutLocalVars],
    consensus: dict[CutKey, CutConsensusVars],
) -> dict[CutKey, CutDualVars]:
    """Apply the scaled dual updates u <- u + x - z."""

    updated: dict[CutKey, CutDualVars] = {}

    for cut, local in cut_vars.items():
        dual = duals[cut]
        z = consensus[cut]

        updated[cut] = CutDualVars(
            u_v_up=_force_real_voltage(
                _boundary_add(
                    dual.u_v_up,
                    _boundary_subtract(
                        local.v_up,
                        z.z_v,
                        "v_up",
                        "z_v",
                    ),
                    "u_v_up",
                    "v_up - z_v",
                )
            ),
            u_v_down=_force_real_voltage(
                _boundary_add(
                    dual.u_v_down,
                    _boundary_subtract(
                        local.v_down,
                        z.z_v,
                        "v_down",
                        "z_v",
                    ),
                    "u_v_down",
                    "v_down - z_v",
                )
            ),
            u_s_up=_boundary_add(
                dual.u_s_up,
                _boundary_subtract(
                    local.s_up,
                    z.z_s,
                    "s_up",
                    "z_s",
                ),
                "u_s_up",
                "s_up - z_s",
            ),
            u_s_down=_boundary_add(
                dual.u_s_down,
                _boundary_subtract(
                    local.s_down,
                    z.z_s,
                    "s_down",
                    "z_s",
                ),
                "u_s_down",
                "s_down - z_s",
            ),
        )

    return updated


def calculate_primal_residual(
    cut_vars: dict[CutKey, CutLocalVars],
) -> float:
    """Calculate max boundary-copy mismatch across all cuts."""

    residual = 0.0

    for cut, local in cut_vars.items():
        v_mismatch = _boundary_subtract(
            local.v_up,
            local.v_down,
            f"{cut}.v_up",
            f"{cut}.v_down",
        )
        s_mismatch = _boundary_subtract(
            local.s_up,
            local.s_down,
            f"{cut}.s_up",
            f"{cut}.s_down",
        )

        residual = max(
            residual,
            _boundary_max_abs(v_mismatch),
            _boundary_max_abs(s_mismatch),
        )

    return residual


def calculate_dual_residual(
    consensus: dict[CutKey, CutConsensusVars],
    previous_consensus: Optional[dict[CutKey, CutConsensusVars]],
    rho_v: float,
    rho_s: float,
) -> float:
    """Calculate max scaled consensus change.

    This uses the infinity-norm form:

        max(rho_v * ||z_v^k - z_v^(k-1)||_inf,
            rho_s * ||z_s^k - z_s^(k-1)||_inf)
    """

    if previous_consensus is None:
        return float("inf") if consensus else 0.0

    if set(consensus) != set(previous_consensus):
        raise ValueError("The set of ADMM cuts changed between iterations")

    residual = 0.0

    for cut, z in consensus.items():
        previous = previous_consensus[cut]

        delta_v = _boundary_subtract(
            z.z_v,
            previous.z_v,
            f"{cut}.z_v",
            f"{cut}.previous_z_v",
        )
        delta_s = _boundary_subtract(
            z.z_s,
            previous.z_s,
            f"{cut}.z_s",
            f"{cut}.previous_z_s",
        )

        residual = max(
            residual,
            rho_v * _boundary_max_abs(delta_v),
            rho_s * _boundary_max_abs(delta_s),
        )

    return residual


# =============================================================================
# Schedule/target updates
# =============================================================================


def _ensure_schedule_columns(
    schedules: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    schedules = schedules.copy()

    for column in columns:
        if column not in schedules.columns:
            schedules[column] = np.nan

    return schedules


def _validate_schedule_times(
    schedules: pd.DataFrame,
    values: pd.DataFrame,
    context: str,
) -> None:
    if "time" not in schedules.columns:
        raise ValueError(f"Schedules have no 'time' column while writing {context}")

    if schedules["time"].duplicated().any():
        duplicates = schedules.loc[
            schedules["time"].duplicated(keep=False),
            "time",
        ].tolist()
        raise ValueError(
            f"Schedules contain duplicate times while writing {context}: {duplicates}"
        )

    missing_times = pd.Index(values.index).difference(pd.Index(schedules["time"]))

    if len(missing_times):
        raise ValueError(
            f"Schedule times are missing while writing {context}: "
            f"{missing_times.tolist()}"
        )


def add_v_swing_to_schedules(
    schedules: pd.DataFrame,
    voltage: pd.DataFrame,
    receiving_area: str,
) -> pd.DataFrame:
    """Write the downstream swing-voltage ADMM target."""

    voltage = _force_real_voltage(voltage)

    rows = voltage.loc[
        voltage["name"].astype(str) == str(receiving_area),
        ["t"] + PHASES,
    ].copy()

    if rows.empty:
        raise ValueError(
            f"No swing-voltage target rows found for area {receiving_area}"
        )

    if rows.duplicated(["t"]).any():
        raise ValueError(f"Duplicate swing-voltage targets for area {receiving_area}")

    prepared = rows.rename(
        columns={
            "t": "time",
            "a": "v_a",
            "b": "v_b",
            "c": "v_c",
        }
    ).set_index("time")

    target_columns = ["v_a", "v_b", "v_c"]
    schedules = _ensure_schedule_columns(schedules, target_columns)
    _validate_schedule_times(
        schedules,
        prepared,
        f"swing-voltage target for {receiving_area}",
    )

    schedules = schedules.set_index("time")
    schedules.loc[prepared.index, target_columns] = prepared[target_columns]
    return schedules.reset_index()


def add_v_down_to_schedules(
    schedules: pd.DataFrame,
    voltage: pd.DataFrame,
    downstream_area: str,
) -> pd.DataFrame:
    """Write the upstream area's dummy-bus voltage ADMM target."""

    voltage = _force_real_voltage(voltage)

    rows = voltage.loc[
        voltage["name"].astype(str) == str(downstream_area),
        ["t"] + PHASES,
    ].copy()

    if rows.empty:
        raise ValueError(
            f"No downstream-voltage target rows found for {downstream_area}"
        )

    if rows.duplicated(["t"]).any():
        raise ValueError(f"Duplicate downstream-voltage targets for {downstream_area}")

    target_columns = [f"{downstream_area}.{phase}.v" for phase in PHASES]

    prepared = rows.rename(columns={"t": "time"}).set_index("time")
    prepared.columns = target_columns

    schedules = _ensure_schedule_columns(schedules, target_columns)
    _validate_schedule_times(
        schedules,
        prepared,
        f"boundary-voltage target for {downstream_area}",
    )

    schedules = schedules.set_index("time")
    schedules.loc[prepared.index, target_columns] = prepared[target_columns]
    return schedules.reset_index()


def add_s_to_schedules(
    schedules: pd.DataFrame,
    power: pd.DataFrame,
    downstream_area: str,
) -> pd.DataFrame:
    """Write the upstream dummy-load power ADMM target."""

    rows = power.loc[
        power["name"].astype(str) == str(downstream_area),
        ["t"] + PHASES,
    ].copy()

    if rows.empty:
        raise ValueError(
            f"No boundary-load power target rows found for {downstream_area}"
        )

    if rows.duplicated(["t"]).any():
        raise ValueError(f"Duplicate boundary-load power targets for {downstream_area}")

    p_columns = [f"{downstream_area}.{phase}.p" for phase in PHASES]
    q_columns = [f"{downstream_area}.{phase}.q" for phase in PHASES]

    prepared = rows.rename(columns={"t": "time"}).set_index("time")

    p_data = prepared[PHASES].apply(np.real)
    p_data.columns = p_columns

    q_data = prepared[PHASES].apply(np.imag)
    q_data.columns = q_columns

    target_data = pd.concat([p_data, q_data], axis=1)
    target_columns = p_columns + q_columns

    schedules = _ensure_schedule_columns(schedules, target_columns)
    _validate_schedule_times(
        schedules,
        target_data,
        f"boundary-load target for {downstream_area}",
    )

    schedules = schedules.set_index("time")
    schedules.loc[target_data.index, target_columns] = target_data[target_columns]
    return schedules.reset_index()


def add_s_swing_to_schedules(
    schedules: pd.DataFrame,
    power: pd.DataFrame,
    receiving_area: str,
) -> pd.DataFrame:
    """Write the downstream area's swing-power ADMM target.

    The OPF API is expected to read these columns when applying its
    swing-power boundary penalty:

        s_a_p, s_b_p, s_c_p
        s_a_q, s_b_q, s_c_q
    """

    rows = power.loc[
        power["name"].astype(str) == str(receiving_area),
        ["t"] + PHASES,
    ].copy()

    if rows.empty:
        raise ValueError(f"No swing-power target rows found for area {receiving_area}")

    if rows.duplicated(["t"]).any():
        raise ValueError(f"Duplicate swing-power targets for area {receiving_area}")

    p_columns = ["s_a_p", "s_b_p", "s_c_p"]
    q_columns = ["s_a_q", "s_b_q", "s_c_q"]

    prepared = rows.rename(columns={"t": "time"}).set_index("time")

    p_data = prepared[PHASES].apply(np.real)
    p_data.columns = p_columns

    q_data = prepared[PHASES].apply(np.imag)
    q_data.columns = q_columns

    target_data = pd.concat([p_data, q_data], axis=1)
    target_columns = p_columns + q_columns

    schedules = _ensure_schedule_columns(schedules, target_columns)
    _validate_schedule_times(
        schedules,
        target_data,
        f"swing-power target for {receiving_area}",
    )

    schedules = schedules.set_index("time")
    schedules.loc[target_data.index, target_columns] = target_data[target_columns]
    return schedules.reset_index()


def send_admm_targets(
    cases: dict[str, Case],
    consensus: dict[CutKey, CutConsensusVars],
    duals: dict[CutKey, CutDualVars],
) -> dict[str, Case]:
    """Write each area's next proximal target, z - u, into its schedules."""

    for (up_area, down_area), z in consensus.items():
        dual = duals[(up_area, down_area)]

        # Upstream area's voltage target at its downstream dummy bus.
        v_target_up = _force_real_voltage(
            _boundary_subtract(
                z.z_v,
                dual.u_v_up,
                "z_v",
                "u_v_up",
            )
        )

        # Downstream area's swing-voltage target.
        v_target_down = _force_real_voltage(
            _boundary_subtract(
                z.z_v,
                dual.u_v_down,
                "z_v",
                "u_v_down",
            )
        )

        # Upstream area's dummy-load power target.
        s_target_up = _boundary_subtract(
            z.z_s,
            dual.u_s_up,
            "z_s",
            "u_s_up",
        )

        # Downstream area's swing-power target.
        s_target_down = _boundary_subtract(
            z.z_s,
            dual.u_s_down,
            "z_s",
            "u_s_down",
        )

        cases[up_area].schedules = add_v_down_to_schedules(
            cases[up_area].schedules,
            v_target_up,
            down_area,
        )
        cases[up_area].schedules = add_s_to_schedules(
            cases[up_area].schedules,
            s_target_up,
            down_area,
        )

        cases[down_area].schedules = add_v_swing_to_schedules(
            cases[down_area].schedules,
            v_target_down,
            down_area,
        )
        cases[down_area].schedules = add_s_swing_to_schedules(
            cases[down_area].schedules,
            s_target_down,
            down_area,
        )

    return cases


# =============================================================================
# Area solve utilities
# =============================================================================


def _solve_pool(
    area_name: str,
    area_case: Case,
    objective: Callable,
    kwargs: dict,
):
    try:
        result = area_case.run_opf(objective=objective, **kwargs)

        # Multiprocessing workers must return pickle-safe objects.
        if hasattr(result, "raw_result"):
            result.raw_result = None
        if hasattr(result, "model"):
            result.model = None

        return result

    except Exception:
        logger.exception("ADMM solve failed for area %s", area_name)
        return None


def _solve_all_parallel(
    cases: dict[str, Case],
    objective: Callable,
    **kwargs,
) -> dict[str, Optional[PowerFlowResult]]:
    area_names = list(cases)

    args = [
        (area_name, cases[area_name], objective, kwargs) for area_name in area_names
    ]

    with mp.Pool() as pool:
        results = pool.starmap(_solve_pool, args)

    return dict(zip(area_names, results))


def _solve_all_loop(
    cases: dict[str, Case],
    objective: Callable,
    **kwargs,
) -> dict[str, Optional[PowerFlowResult]]:
    all_results: dict[str, Optional[PowerFlowResult]] = {}

    for area_name, area_case in cases.items():
        try:
            all_results[area_name] = area_case.run_opf(
                objective=objective,
                **kwargs,
            )
        except Exception:
            logger.exception("ADMM solve failed for area %s", area_name)
            all_results[area_name] = None

    return all_results


# =============================================================================
# Result aggregation
# =============================================================================


def _concat_field(
    res_list: list,
    field: str,
    valid_names: Optional[set],
    name_cols: tuple = ("name",),
    bus_id_col: Optional[str] = None,
    branch_cols: Optional[tuple] = None,
) -> Optional[pd.DataFrame]:
    frames = [
        getattr(result, field)
        for result in res_list
        if getattr(result, field, None) is not None
    ]

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    is_branch_data = branch_cols is not None

    from_col = None
    to_col = None

    if is_branch_data:
        from_col = next(
            (column for column in ("from_name", "fb") if column in df.columns),
            None,
        )
        to_col = next(
            (column for column in ("to_name", "tb", "name") if column in df.columns),
            None,
        )

    if valid_names is not None:
        if is_branch_data and from_col and to_col:
            df = df.loc[
                df[from_col].astype(str).isin(valid_names)
                & df[to_col].astype(str).isin(valid_names)
            ]
        elif not is_branch_data:
            name_col = next(
                (column for column in name_cols if column in df.columns),
                None,
            )
            if name_col is not None:
                df = df.loc[df[name_col].astype(str).isin(valid_names)]

    if df.empty:
        return None

    if is_branch_data:
        key = [column for column in (from_col, to_col) if column is not None]
    elif bus_id_col and bus_id_col in df.columns:
        key = [bus_id_col]
    else:
        key = [column for column in ("name", "id") if column in df.columns]

    if key:
        if "t" in df.columns:
            key.append("t")
        df = df.drop_duplicates(subset=key)

    if is_branch_data and "tb" in df.columns:
        sort_columns = ["tb"]
        if "fb" in df.columns:
            sort_columns.append("fb")
        df = df.sort_values(sort_columns, na_position="last")
    elif bus_id_col and bus_id_col in df.columns:
        df = df.sort_values([bus_id_col], na_position="last")

    return df.reset_index(drop=True)


def combine_powerflow_results(
    results: Iterable[PowerFlowResult],
    case_ref: Optional[Case] = None,
    objective_value: Optional[float] = None,
) -> Optional[PowerFlowResult]:
    res_list = list(results)

    if not res_list:
        return None

    valid_names: Optional[set] = (
        set(case_ref.bus_data["name"].astype(str).tolist())
        if case_ref is not None and case_ref.bus_data is not None
        else None
    )

    def bus(
        field: str,
        *,
        id_col: str = "id",
    ) -> Optional[pd.DataFrame]:
        return _concat_field(
            res_list,
            field,
            valid_names,
            name_cols=("name",),
            bus_id_col=id_col,
        )

    def branch(field: str) -> Optional[pd.DataFrame]:
        return _concat_field(
            res_list,
            field,
            valid_names,
            name_cols=(),
            branch_cols=("from_name", "to_name", "fb", "tb"),
        )

    def scalar_concat(field: str) -> Optional[pd.DataFrame]:
        frames = [
            getattr(result, field, None)
            for result in res_list
            if getattr(result, field, None) is not None
        ]

        if not frames:
            return None

        frame = pd.concat(frames, ignore_index=True)

        if frame.empty:
            return None

        return frame.drop_duplicates().reset_index(drop=True)

    voltages = bus("voltages", id_col="id")
    voltage_angles = bus("voltage_angles", id_col="id")
    active_loads = bus("active_power_loads", id_col="id")
    reactive_loads = bus("reactive_power_loads", id_col="id")
    p_gens = bus("active_power_generation", id_col="id")
    q_gens = bus("reactive_power_generation", id_col="id")
    p_bats = bus("battery_active_power", id_col="id")
    q_bats = bus("battery_reactive_power", id_col="id")
    p_discharge = bus("p_discharge", id_col="id")
    p_charge = bus("p_charge", id_col="id")
    soc = bus("soc", id_col="id")
    q_caps = bus("capacitor_reactive_power", id_col="id")
    currents_df = bus("currents", id_col="id")
    current_angles_df = bus("current_angles", id_col="id")

    p_flows = branch("active_power_flows")
    q_flows = branch("reactive_power_flows")
    tap_ratios = branch("tap_ratios")
    reg_taps = branch("reg_taps")

    z_caps = scalar_concat("z_caps")
    u_caps = scalar_concat("u_caps")

    dual_p = branch("dual_power_balance_p")
    dual_q = branch("dual_power_balance_q")
    dual_vd = branch("dual_voltage_drop")
    dual_vlo = branch("dual_voltage_limits_lower")
    dual_vhi = branch("dual_voltage_limits_upper")

    if objective_value is None:
        objective_values = [
            float(result.objective_value)
            for result in res_list
            if getattr(result, "objective_value", None) is not None
        ]
        objective_value = sum(objective_values) if objective_values else None

    converged_all = all(getattr(result, "converged", True) for result in res_list)

    return PowerFlowResult(
        voltages=voltages,
        voltage_angles=voltage_angles,
        active_power_flows=p_flows,
        reactive_power_flows=q_flows,
        active_power_generation=p_gens,
        reactive_power_generation=q_gens,
        active_power_loads=active_loads,
        reactive_power_loads=reactive_loads,
        battery_active_power=p_bats,
        battery_reactive_power=q_bats,
        p_discharge=p_discharge,
        p_charge=p_charge,
        soc=soc,
        capacitor_reactive_power=q_caps,
        tap_ratios=tap_ratios,
        reg_taps=reg_taps,
        z_caps=z_caps,
        u_caps=u_caps,
        currents=currents_df,
        current_angles=current_angles_df,
        dual_power_balance_p=dual_p,
        dual_power_balance_q=dual_q,
        dual_voltage_drop=dual_vd,
        dual_voltage_limits_lower=dual_vlo,
        dual_voltage_limits_upper=dual_vhi,
        objective_value=objective_value,
        converged=converged_all,
        solver="admm",
        result_type="opf",
        case=case_ref,
    )


def _get_root_areas(
    area_info: dict[str, dict[str, list]],
) -> set[str]:
    return {
        area_name for area_name, info in area_info.items() if not info.get("up_areas")
    }


def _aggregate_root_objective(
    results: dict[str, PowerFlowResult],
    root_areas: set[str],
) -> Optional[float]:
    """Preserve the objective aggregation behavior of the original code.

    If each area's OPF objective contains a distinct portion of the original
    global objective, this should instead sum all area objectives after
    removing augmented-Lagrangian penalty terms.
    """

    objective_values = [
        result.objective_value
        for area_name, result in results.items()
        if area_name in root_areas
        and result is not None
        and getattr(result, "objective_value", None) is not None
    ]

    return sum(objective_values) if objective_values else None


def _remap_area_results_to_global_ids(
    results: dict[str, PowerFlowResult],
    case: Case,
    area_info: dict[str, dict[str, list]],
) -> None:
    if case.bus_data is None:
        return

    name_to_global_id = {
        str(row["name"]): int(row["id"])
        for _, row in case.bus_data.loc[:, ["id", "name"]].iterrows()
    }

    for result in results.values():
        if result is None:
            continue

        voltages = getattr(result, "voltages", None)

        if voltages is not None and "name" in voltages.columns:
            voltages = voltages.copy()
            voltages["id"] = voltages["name"].astype(str).map(name_to_global_id)
            result.voltages = voltages

        for flow_attribute in (
            "active_power_flows",
            "reactive_power_flows",
        ):
            flows = getattr(result, flow_attribute, None)

            if flows is None:
                continue

            if not {"from_name", "to_name"}.issubset(flows.columns):
                continue

            flows = flows.copy()
            flows["fb"] = flows["from_name"].astype(str).map(name_to_global_id)
            flows["tb"] = flows["to_name"].astype(str).map(name_to_global_id)
            setattr(result, flow_attribute, flows)

    _reconstruct_boundary_flows(
        results,
        area_info,
        case,
        name_to_global_id,
    )


def _reconstruct_boundary_flows(
    all_results: dict[str, PowerFlowResult],
    area_info: dict[str, dict[str, list]],
    case: Case,
    name_to_global_id: dict[str, int],
) -> None:
    if case.branch_data is None:
        return

    branch_data = case.branch_data

    if not {"from_name", "to_name"}.issubset(branch_data.columns):
        return

    for area_name, info in area_info.items():
        up_buses = info.get("up_buses", [])

        if not up_buses:
            continue

        for up_area in info.get("up_areas", []):
            source_bus = str(up_buses[0])

            real_branch = branch_data.loc[
                branch_data["to_name"].astype(str) == source_bus
            ]

            if real_branch.empty:
                continue

            real_from_name = str(real_branch.iloc[0]["from_name"])
            real_fb = name_to_global_id.get(real_from_name)
            real_tb = name_to_global_id.get(source_bus)

            if real_fb is None or real_tb is None:
                continue

            area_result = all_results.get(area_name)

            if area_result is None:
                continue

            for flow_attribute in (
                "active_power_flows",
                "reactive_power_flows",
            ):
                frame = getattr(area_result, flow_attribute, None)

                if frame is None or "from_name" not in frame.columns:
                    continue

                mask = frame["from_name"].astype(str) == str(up_area)

                if not mask.any():
                    continue

                frame = frame.copy()
                frame.loc[mask, "from_name"] = real_from_name
                frame.loc[mask, "to_name"] = source_bus
                frame.loc[mask, "fb"] = real_fb
                frame.loc[mask, "tb"] = real_tb

                setattr(area_result, flow_attribute, frame)


def _finalize_admm_result(
    result: Optional[PowerFlowResult],
    *,
    case: Case,
    objective_value: Optional[float],
    converged: bool,
    iterations: int,
    runtime: float,
    solver_status: str,
    termination_condition: str,
    area_results: dict[str, PowerFlowResult],
    primal_residuals: list[float],
    dual_residuals: list[float],
    parallel_used: bool,
    area_solve_failed: bool,
    failed_areas: set[str],
    rho_v: float,
    rho_s: float,
    duals: Optional[dict[CutKey, CutDualVars]] = None,
    consensus: Optional[dict[CutKey, CutConsensusVars]] = None,
) -> PowerFlowResult:
    if result is None:
        result = PowerFlowResult(
            objective_value=objective_value,
            converged=converged,
            solver="admm",
            result_type="opf",
            case=case,
        )

    result.case = case
    result.objective_value = objective_value
    result.solve_time = runtime
    result.iterations = iterations
    result.converged = converged
    result.solver = "admm"
    result.result_type = "opf"
    result.solver_status = solver_status
    result.termination_condition = termination_condition
    result.backend = "admm"

    result.raw_result = {
        "area_results": area_results,
        "primal_residual_per_iter": primal_residuals,
        "dual_residual_per_iter": dual_residuals,
        # Compatibility alias for existing ENApp consumers.
        "boundary_error_per_iter": primal_residuals,
        "admm_iterations": iterations,
        "admm_runtime": runtime,
        "admm_parallel_used": parallel_used,
        "area_solve_failed": area_solve_failed,
        "failed_areas": sorted(failed_areas),
        "rho_v": rho_v,
        "rho_s": rho_s,
        "cut_duals": duals,
        "cut_consensus": consensus,
    }

    return result


def _build_partial_result(
    *,
    all_results: dict[str, PowerFlowResult],
    case: Case,
    area_info: dict[str, dict[str, list]],
    root_areas: set[str],
    tic: float,
    iterations: int,
    primal_residuals: list[float],
    dual_residuals: list[float],
    parallel_used: bool,
    failed_areas: set[str],
    rho_v: float,
    rho_s: float,
    duals: Optional[dict[CutKey, CutDualVars]],
    consensus: Optional[dict[CutKey, CutConsensusVars]],
) -> PowerFlowResult:
    _remap_area_results_to_global_ids(
        all_results,
        case,
        area_info,
    )

    objective_value = _aggregate_root_objective(
        all_results,
        root_areas,
    )

    partial_result = combine_powerflow_results(
        all_results.values(),
        case_ref=case,
        objective_value=objective_value,
    )

    runtime = perf_counter() - tic

    return _finalize_admm_result(
        partial_result,
        case=case,
        objective_value=objective_value,
        converged=False,
        iterations=iterations,
        runtime=runtime,
        solver_status="failure",
        termination_condition="incomplete_solve",
        area_results=all_results,
        primal_residuals=primal_residuals,
        dual_residuals=dual_residuals,
        parallel_used=parallel_used,
        area_solve_failed=True,
        failed_areas=failed_areas,
        rho_v=rho_v,
        rho_s=rho_s,
        duals=duals,
        consensus=consensus,
    )


# =============================================================================
# Main ADMM solver
# =============================================================================


def solve_admm(
    case: Case,
    area_info: dict[str, dict[str, list]],
    objective: Callable,
    rho: float = 1e3,
    rho_v: Optional[float] = None,
    rho_s: Optional[float] = None,
    tol: float = 1e-6,
    primal_tol: Optional[float] = None,
    dual_tol: Optional[float] = None,
    max_iterations: int = 100,
    parallel: bool = True,
    solve_callback: Optional[Callable] = None,
    iteration_callback: Optional[Callable] = None,
    verbose: bool = False,
    **kwargs,
) -> PowerFlowResult:
    """Solve a decomposed OPF using consensus ADMM.

    Each inter-area cut has two local copies of voltage and power:

    - Upstream boundary voltage and downstream swing voltage.
    - Upstream boundary-load power and downstream swing power.

    The OPF API is assumed to support these keyword arguments:

    ``free_swing_voltage``
        Allows the downstream area's swing voltage to vary.

    ``swing_voltage_slack_penalty``
        Penalty coefficient for the downstream swing-voltage target.

    ``boundary_voltage_slack_penalty``
        Penalty coefficient for the upstream dummy-bus voltage target.

    ``free_boundary_load``
        Allows upstream dummy boundary loads to vary.

    ``boundary_load_slack_penalty``
        Penalty coefficient for upstream boundary-load power targets.

    ``swing_power_slack_penalty``
        Penalty coefficient for downstream swing-power targets.

    The corresponding target values are written into the area's schedules.

    Parameters
    ----------
    rho
        Default penalty coefficient used for both voltage and power.
    rho_v
        Voltage penalty coefficient. Defaults to ``rho``.
    rho_s
        Power penalty coefficient. Defaults to ``rho``.
    tol
        Default stopping tolerance for both residuals.
    primal_tol
        Optional primal-residual tolerance. Defaults to ``tol``.
    dual_tol
        Optional dual-residual tolerance. Defaults to ``tol``.
    """

    logger.setLevel(logging.INFO if verbose else logging.WARNING)

    if max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")

    rho_v = float(rho if rho_v is None else rho_v)
    rho_s = float(rho if rho_s is None else rho_s)

    if rho_v <= 0:
        raise ValueError("rho_v must be positive")
    if rho_s <= 0:
        raise ValueError("rho_s must be positive")

    primal_tol = float(tol if primal_tol is None else primal_tol)
    dual_tol = float(tol if dual_tol is None else dual_tol)

    if primal_tol < 0:
        raise ValueError("primal_tol must be nonnegative")
    if dual_tol < 0:
        raise ValueError("dual_tol must be nonnegative")

    controlled_arguments = {
        "free_swing_voltage",
        "swing_voltage_slack_penalty",
        "boundary_voltage_slack_penalty",
        "free_boundary_load",
        "boundary_load_slack_penalty",
        "swing_power_slack_penalty",
    }

    supplied_controlled_arguments = controlled_arguments.intersection(kwargs)

    if supplied_controlled_arguments:
        raise TypeError(
            "The following arguments are controlled by solve_admm: "
            f"{sorted(supplied_controlled_arguments)}"
        )

    tic = perf_counter()

    sources = {area_name: info["up_buses"][0] for area_name, info in area_info.items()}
    cases = decompose(case, sources)

    missing_case_info = set(cases) - set(area_info)
    if missing_case_info:
        raise ValueError(
            f"area_info is missing decomposed areas: {sorted(missing_case_info)}"
        )

    solve_kwargs = {
        **kwargs,
        # Downstream-side swing voltage penalty.
        "free_swing_voltage": True,
        "swing_voltage_slack_penalty": rho_v,
        # Upstream-side boundary voltage penalty.
        "boundary_voltage_slack_penalty": rho_v,
        # Upstream-side dummy-load power penalty.
        "free_boundary_load": True,
        "boundary_load_slack_penalty": rho_s,
        # Downstream-side swing-power penalty.
        "swing_power_slack_penalty": rho_s,
    }

    root_areas = _get_root_areas(area_info)

    all_results: dict[str, PowerFlowResult] = {}
    duals: Optional[dict[CutKey, CutDualVars]] = None
    consensus: Optional[dict[CutKey, CutConsensusVars]] = None
    previous_consensus: Optional[dict[CutKey, CutConsensusVars]] = None

    primal_residuals: list[float] = []
    dual_residuals: list[float] = []

    failed_areas: set[str] = set()
    any_area_solve_failed = False
    converged = False

    parallel_used = parallel and solve_callback is None
    iterations = 0

    for iterations in range(1, max_iterations + 1):
        if solve_callback is not None:
            iteration_results = solve_callback(
                cases,
                objective,
                **solve_kwargs,
            )
        elif parallel:
            iteration_results = _solve_all_parallel(
                cases,
                objective,
                **solve_kwargs,
            )
        else:
            iteration_results = _solve_all_loop(
                cases,
                objective,
                **solve_kwargs,
            )

        if set(iteration_results) != set(cases):
            missing = set(cases) - set(iteration_results)
            extra = set(iteration_results) - set(cases)
            raise ValueError(
                "The area solve callback returned the wrong set of areas. "
                f"Missing: {sorted(missing)}; extra: {sorted(extra)}"
            )

        iteration_failed_areas = {
            area_name
            for area_name, result in iteration_results.items()
            if result is None
        }

        if iteration_failed_areas:
            any_area_solve_failed = True
            failed_areas.update(iteration_failed_areas)

            for area_name in sorted(iteration_failed_areas):
                logger.warning(
                    "ADMM area %s failed on iteration %d",
                    area_name,
                    iterations,
                )

        for area_name, result in iteration_results.items():
            if result is not None:
                all_results[area_name] = result

        # If an area has never solved, there is no local copy from which to
        # perform a valid consensus or dual update.
        if len(all_results) < len(cases):
            logger.warning(
                "Only %d/%d ADMM areas have produced a successful result; "
                "returning a partial result.",
                len(all_results),
                len(cases),
            )

            return _build_partial_result(
                all_results=all_results,
                case=case,
                area_info=area_info,
                root_areas=root_areas,
                tic=tic,
                iterations=iterations,
                primal_residuals=primal_residuals,
                dual_residuals=dual_residuals,
                parallel_used=parallel_used,
                failed_areas=failed_areas,
                rho_v=rho_v,
                rho_s=rho_s,
                duals=duals,
                consensus=consensus,
            )

        # Do not update ADMM state from a mixture of current and stale local
        # results. Retry the local solves on the next iteration instead.
        if iteration_failed_areas:
            logger.warning(
                "Skipping ADMM consensus and dual updates on iteration %d "
                "because one or more area solves failed.",
                iterations,
            )
            continue

        # 1. Extract the latest local boundary copies.
        cut_vars = extract_cut_local_vars(
            cases,
            all_results,
            area_info,
        )

        # A single-area/no-cut problem has no consensus constraints.
        if not cut_vars:
            primal_residuals.append(0.0)
            dual_residuals.append(0.0)

            if iteration_callback is not None:
                iteration_callback(
                    iterations,
                    cases,
                    all_results,
                    cut_vars,
                    consensus,
                    duals,
                    0.0,
                    0.0,
                )

            converged = True
            break

        # Initialize the scaled dual variables once, using exact cut shapes.
        if duals is None:
            duals = initialize_cut_duals(cut_vars)

        # 2. Consensus update:
        #       z = average(x_i + u_i)
        consensus = calculate_consensus(
            cut_vars,
            duals,
        )

        # 3. Compute residuals before mutating the dual state.
        primal_residual = calculate_primal_residual(cut_vars)
        dual_residual = calculate_dual_residual(
            consensus,
            previous_consensus,
            rho_v,
            rho_s,
        )

        primal_residuals.append(primal_residual)
        dual_residuals.append(dual_residual)

        logger.info(
            "ADMM iteration %d: primal residual=%.6e, dual residual=%.6e",
            iterations,
            primal_residual,
            dual_residual,
        )

        # 4. Scaled dual update:
        #       u_i <- u_i + x_i - z
        duals = update_cut_duals(
            duals,
            cut_vars,
            consensus,
        )

        # 5. Write z - u targets for the next set of local OPF solves.
        cases = send_admm_targets(
            cases,
            consensus,
            duals,
        )

        if iteration_callback is not None:
            iteration_callback(
                iterations,
                cases,
                all_results,
                cut_vars,
                consensus,
                duals,
                primal_residual,
                dual_residual,
            )

        if primal_residual <= primal_tol and dual_residual <= dual_tol:
            logger.info(
                "ADMM converged after %d iterations",
                iterations,
            )
            converged = True
            break

        previous_consensus = deepcopy(consensus)

    _remap_area_results_to_global_ids(
        all_results,
        case,
        area_info,
    )

    objective_value = _aggregate_root_objective(
        all_results,
        root_areas,
    )

    aggregated_result = combine_powerflow_results(
        all_results.values(),
        case_ref=case,
        objective_value=objective_value,
    )

    runtime = perf_counter() - tic

    if converged:
        solver_status = "optimal"
        termination_condition = "converged"
    elif any_area_solve_failed:
        solver_status = "failure"
        termination_condition = "area_solve_failure"
    else:
        solver_status = "max_iterations"
        termination_condition = "max_iterations"

    return _finalize_admm_result(
        aggregated_result,
        case=case,
        objective_value=objective_value,
        converged=converged,
        iterations=iterations,
        runtime=runtime,
        solver_status=solver_status,
        termination_condition=termination_condition,
        area_results=all_results,
        primal_residuals=primal_residuals,
        dual_residuals=dual_residuals,
        parallel_used=parallel_used,
        area_solve_failed=any_area_solve_failed,
        failed_areas=failed_areas,
        rho_v=rho_v,
        rho_s=rho_s,
        duals=duals,
        consensus=consensus,
    )
