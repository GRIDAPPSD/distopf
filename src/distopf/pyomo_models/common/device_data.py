"""Shared parsing and validation helpers for device-provider CSV tables."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import pandas as pd

from distopf.pyomo_models.common.data import parse_phases
from distopf.pyomo_models.common.model_types import (
    ControlVariable,
    CONTROL_VARIABLE_MAP,
)

TOL = 1e-9


class InfeasibleCaseError(ValueError):
    """The input data is well-formed but provably infeasible."""


class DeviceDataWarning(UserWarning):
    """Suspicious but usable device data."""


# --------------------------------------------------------------------------- #
# Cell parsing
# --------------------------------------------------------------------------- #


def cell_blank(row: pd.Series, key: str) -> bool:
    """True if the column is missing or the cell is blank."""
    value = row.get(key)
    return (
        value is None
        or pd.isna(value)
        or (isinstance(value, str) and not value.strip())
    )


def cell_text(value: Any) -> str:
    """Return a stripped string; missing or blank cells give ''."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def cell_num(row: pd.Series, key: str, default: Any) -> Any:
    """Return row[key] as float, or default for missing columns and blank cells."""
    return default if cell_blank(row, key) else float(row[key])


def cell_opt_float(
    row: pd.Series, key: str, label: str, errors: list[str]
) -> float | None:
    """Parse an optional numeric cell. Blank gives None; junk records an error."""
    if cell_blank(row, key):
        return None
    value = row[key]
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label}: {key}={value!r} is not a number")
        return None
    if not math.isfinite(number):
        errors.append(f"{label}: {key} must be finite")
        return None
    return number


def cell_required_float(
    row: pd.Series,
    key: str,
    label: str,
    errors: list[str],
    *,
    gt: float | None = None,
    ge: float | None = None,
) -> float | None:
    """Parse a required numeric cell with optional lower bounds."""
    if cell_blank(row, key):
        errors.append(f"{label}: {key} is required")
        return None
    value = cell_opt_float(row, key, label, errors)
    if value is None:
        return None
    if gt is not None and value <= gt:
        errors.append(f"{label}: {key}={value} must be > {gt}")
        return None
    if ge is not None and value < ge:
        errors.append(f"{label}: {key}={value} must be >= {ge}")
        return None
    return value


def cell_flag(row: pd.Series, key: str, default: bool = False) -> bool:
    """Parse a True/False cell. Missing columns and blank cells give default."""
    if cell_blank(row, key):
        return default
    value = row[key]
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1"):
            return True
        if text in ("false", "0"):
            return False
        raise ValueError(f"Cannot interpret {key}={value!r} as True/False")
    return bool(value)


def cell_control(row: pd.Series, key: str = "control_variable") -> ControlVariable:
    """Parse a control mode. Missing or blank means ControlVariable.NONE."""
    raw = row.get(key)
    text = cell_text(raw).upper()
    if text == "":
        return ControlVariable.NONE
    try:
        return ControlVariable(CONTROL_VARIABLE_MAP[text])
    except KeyError:
        raise ValueError(
            f"Unknown {key} {raw!r}; expected one of "
            f"{sorted(k for k in CONTROL_VARIABLE_MAP if k)} or blank"
        ) from None


def preview(items: Iterable[Any], n: int = 5) -> str:
    """Short printable list for error messages."""
    items = list(items)
    head = ", ".join(map(str, items[:n]))
    return f"[{head}{', ...' if len(items) > n else ''}] ({len(items)} total)"


# --------------------------------------------------------------------------- #
# Network context
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NetworkContext:
    """Network facts that device data is validated against."""

    bus_name_to_id: Mapping[str, int]
    injectable: frozenset[tuple[int, str]]  # (bus, phase) pairs with an incoming branch
    swing_buses: frozenset[int]
    times: tuple[Any, ...]

    @classmethod
    def from_model(cls, model: Any, provider: str) -> NetworkContext:
        required = (
            "bus_name_to_id_map",
            "branch_phase_set",
            "swing_bus_set",
            "time_set",
        )
        missing = [attr for attr in required if not hasattr(model, attr)]
        if missing:
            raise RuntimeError(
                f"{provider} needs model attributes {missing}; the network "
                "provider must create components first"
            )
        return cls(
            bus_name_to_id=dict(model.bus_name_to_id_map),
            injectable=frozenset((tb, ph) for _, tb, ph in model.branch_phase_set),
            swing_buses=frozenset(model.swing_bus_set),
            times=tuple(model.time_set),
        )


# --------------------------------------------------------------------------- #
# Validation building blocks
# --------------------------------------------------------------------------- #


def check_columns(
    data: pd.DataFrame,
    table: str,
    required: set[str],
    known: set[str],
    legacy: set[str],
    legacy_hint: str,
) -> None:
    """Structural checks. Raise immediately because row checks depend on them."""
    cols = set(data.columns)
    if missing := required - cols:
        raise ValueError(f"{table}: missing columns {sorted(missing)}")
    if old := legacy & cols:
        raise ValueError(f"{table}: removed columns {sorted(old)}; {legacy_hint}")
    if unknown := cols - known:
        raise ValueError(f"{table}: unrecognized columns (typo?): {sorted(unknown)}")


def check_device_names(data: pd.DataFrame, errors: list[str]) -> None:
    names = data["device_name"].map(cell_text)
    if (names == "").any():
        errors.append("blank device_name values")
    dupes = sorted(set(names[names.duplicated() & (names != "")]))
    if dupes:
        errors.append(f"duplicate device_name values: {dupes}")


def row_label(row: pd.Series, idx: Any) -> str:
    name = cell_text(row.get("device_name"))
    return f"device {name!r}" if name else f"row {idx}"


def check_phases(
    label: str, raw: Any, allowed: Iterable[str], errors: list[str]
) -> list[str] | None:
    """Parse and validate a phases cell. Returns None (with an error) if invalid."""
    allowed = list(allowed)
    text = cell_text(raw)
    try:
        phases = list(parse_phases(text)) if text else []
    except Exception as exc:
        errors.append(f"{label}: cannot parse phases {raw!r} ({exc})")
        return None
    if not phases or len(set(phases)) != len(phases) or not set(phases) <= set(allowed):
        errors.append(f"{label}: phases {raw!r} must be distinct phases from {allowed}")
        return None
    return phases


def check_connection(
    label: str, bus_name: str, phases: list[str], ctx: NetworkContext, errors: list[str]
) -> None:
    """Every device port must enter an AC power-balance constraint."""
    if bus_name not in ctx.bus_name_to_id:
        errors.append(f"{label}: bus_name {bus_name!r} not found in network")
        return
    bus = ctx.bus_name_to_id[bus_name]
    if bus in ctx.swing_buses:
        errors.append(
            f"{label}: bus {bus_name!r} is a swing/boundary bus; "
            "device injections there are not modeled"
        )
        return
    absent = [ph for ph in phases if (bus, ph) not in ctx.injectable]
    if absent:
        errors.append(
            f"{label}: phases {absent} have no incoming branch at bus {bus_name!r}; "
            "power on those ports would not enter any power balance"
        )


def finish_validation(
    table: str, errors: list[str], infeasible: list[str], warns: list[str]
) -> None:
    """Emit warnings, then raise data errors first and infeasibility second."""
    for message in warns:
        warnings.warn(f"{table}: {message}", DeviceDataWarning, stacklevel=3)
    if errors:
        raise ValueError(f"Invalid {table}:\n" + "\n".join(f"  - {e}" for e in errors))
    if infeasible:
        raise InfeasibleCaseError(
            f"{table} is infeasible:\n" + "\n".join(f"  - {m}" for m in infeasible)
        )
