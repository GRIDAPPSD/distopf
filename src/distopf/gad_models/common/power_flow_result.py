"""Convert a solved model result into the package-wide ``PowerFlowResult``.

Works on any result shaped like ``PyoResult`` (one DataFrame attribute per
model variable), so the CVXPY ``CvxResult`` uses it too.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from distopf.results import PowerFlowResult

# Device states that are averaged (not summed) when aggregated to a bus.
_STATE_VARIABLES = ("soc", "u_cap", "z_cap")
_VALUE_COLUMNS = ("a", "b", "c", "s1", "s2", "value")
_DIAGONAL_PAIRS = {"aa": "a", "bb": "b", "cc": "c", "s1s1": "s1", "s2s2": "s2"}


def to_power_flow_result(
    raw: Any,
    case: Any,
    *,
    solver: str = "",
    current_angles: Optional[pd.DataFrame] = None,
) -> PowerFlowResult:
    """Build a ``PowerFlowResult`` from a solved ``PyoResult`` (or ``CvxResult``).

    Device-level variables are aggregated to buses, regulator taps are
    converted to ratios and squared currents to magnitudes.

    Parameters
    ----------
    raw : PyoResult or CvxResult
        Result returned by ``solve``.
    case : Case
        Case the model was built from (regulator steps and device weights).
    solver : str
        Solver name stored on the result.
    current_angles : DataFrame, optional
        Branch current angles (degrees), e.g. from an FBS warm start.

    Examples
    --------
    >>> raw = solve(model, "ipopt")
    >>> result = to_power_flow_result(raw, case, solver="ipopt")
    >>> result.plot_network()
    """
    reg_taps = getattr(raw, "reg_taps", None)

    def bus_variable(variable: str) -> Optional[pd.DataFrame]:
        frame = getattr(raw, variable, None)
        return None if frame is None else bus_values(frame, case, variable)

    return PowerFlowResult(
        voltages=raw.voltages,
        active_power_flows=raw.p_flow,
        reactive_power_flows=raw.q_flow,
        currents=currents_from_l_flow(getattr(raw, "l_flow", None)),
        current_angles=current_angles,
        active_power_generation=bus_variable("p_gen"),
        reactive_power_generation=bus_variable("q_gen"),
        active_power_loads=getattr(raw, "p_load", None),
        reactive_power_loads=getattr(raw, "q_load", None),
        capacitor_reactive_power=bus_variable("q_cap"),
        tap_ratios=tap_ratios(reg_taps, case.reg_data),
        reg_taps=reg_taps,
        u_caps=bus_variable("u_cap"),
        z_caps=bus_variable("z_cap"),
        battery_active_power=bus_variable("p_bat"),
        battery_reactive_power=bus_variable("q_bat"),
        p_charge=bus_variable("p_charge"),
        p_discharge=bus_variable("p_discharge"),
        soc=bus_variable("soc"),
        dual_power_balance_p=getattr(raw, "dual_power_balance_p", None),
        dual_power_balance_q=getattr(raw, "dual_power_balance_q", None),
        dual_voltage_drop=getattr(raw, "dual_voltage_drop", None),
        dual_voltage_limits_lower=getattr(raw, "dual_voltage_limits_lower", None),
        dual_voltage_limits_upper=getattr(raw, "dual_voltage_limits_upper", None),
        objective_value=getattr(raw, "objective_value", None),
        converged=True,
        solver=solver,
        solve_time=getattr(raw, "solve_time", None),
        result_type="opf",
        raw_result=raw,
        model=raw.model,
        case=case,
        log=getattr(raw, "log", ""),
    )


def bus_values(frame: pd.DataFrame, case: Any, variable: str) -> pd.DataFrame:
    """Aggregate a device-level variable table to the bus-level result contract.

    Powers are summed. States (``soc``, ``u_cap``, ``z_cap``) are averaged,
    weighted by battery energy capacity or capacitor rating, so that a bus
    state times the bus total matches the sum over its devices. Tables
    without ``device_name`` are returned unchanged.

    Example: ``bus_values(raw.p_gen, case, "p_gen")``.
    """
    if "device_name" not in frame.columns:
        return frame
    keys = ["id", "name", "t"]
    columns = [column for column in _VALUE_COLUMNS if column in frame]
    if variable not in _STATE_VARIABLES:
        return frame.groupby(keys, as_index=False)[columns].sum(min_count=1)
    weights = _device_state_weights(case, variable, frame, columns)
    weights = weights.where(frame[columns].notna())
    groups = [frame[key] for key in keys]
    numerator = (frame[columns] * weights).groupby(groups).sum()
    denominator = weights.groupby(groups).sum()
    return (numerator / denominator).reset_index()


def _device_state_weights(
    case: Any, variable: str, frame: pd.DataFrame, columns: list[str]
) -> pd.DataFrame:
    """Per-row weights for averaging device states over a bus.

    Uses ``energy_capacity`` for batteries and ``q_{phase}`` for capacitors.
    Legacy tables (no ``device_name``) have one device per bus, so equal
    weights are used.
    """
    table = case.bat_data if variable == "soc" else case.cap_data
    weights = pd.DataFrame(1.0, index=frame.index, columns=columns)
    if table is None or "device_name" not in table.columns:
        return weights
    devices = table.set_index("device_name")
    for column in columns:
        source = "energy_capacity" if variable == "soc" else f"q_{column}"
        if source in devices.columns:
            weights[column] = frame["device_name"].map(devices[source]).abs()
    return weights


def currents_from_l_flow(l_flow: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Branch current magnitudes ``sqrt(L)`` from the squared-current variable.

    Returns None when there is no ``l_flow`` (e.g. LinDistFlow).
    """
    if l_flow is None:
        return None
    id_cols = [
        col for col in ("fb", "tb", "from_name", "to_name", "t") if col in l_flow
    ]
    currents = l_flow[id_cols].copy()
    for pair, phase in _DIAGONAL_PAIRS.items():
        if pair in l_flow:
            currents[phase] = np.sqrt(l_flow[pair].clip(lower=0.0))
    return currents


def tap_ratios(
    reg_taps: Optional[pd.DataFrame], reg_data: pd.DataFrame
) -> Optional[pd.DataFrame]:
    """Regulator voltage ratios ``1 + step * tap`` from integer tap positions."""
    if reg_taps is None:
        return None
    ratios = reg_taps.copy()
    reg = reg_data.set_index(["fb", "tb"])
    step = (reg["ratio_max"] - reg["ratio_min"]) / reg["num_taps"]
    step = step.reindex(pd.MultiIndex.from_frame(reg_taps[["fb", "tb"]]))
    for phase in "abc":
        ratios[phase] = 1 + step.to_numpy() * reg_taps[phase]
    return ratios
