"""Fixed and switched shunt capacitors (CVXPY).

Mirror of ``pyomo_models.devices.capacitor``. Tables require device_name,
bus_name, phases, and nonnegative q_{phase} for each connected phase.
Switched capacitors use binary ``u_cap`` with a McCormick envelope for
``z_cap = u_cap * v2``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from distopf.cvxpy_models.common.model import (
    CvxModel,
    IndexSet,
    ParamDict,
    VarDict,
    add_constraint,
    index_product,
)
from distopf.pyomo_models.common.data import parse_phases
from distopf.pyomo_models.common.device_data import (
    NetworkContext,
    cell_required_float,
    cell_text,
    check_columns,
    check_connection,
    check_device_names,
    check_phases,
    finish_validation,
    row_label,
)

_PHASES = ("a", "b", "c", "s1", "s2")
_REQUIRED = {"device_name", "bus_name", "phases"}
_KNOWN = _REQUIRED | {"s_base"} | {f"q_{phase}" for phase in _PHASES}


def _capacitor_data(model: Any, case: Any) -> pd.DataFrame:
    data = getattr(case, "cap_data", None)
    if data is None:
        return pd.DataFrame()
    if not data.empty and "device_name" not in data and "id" in data:
        names_by_bus = {bus: name for name, bus in model.bus_name_to_id_map.items()}
        bus_names = data["id"].map(names_by_bus)
        if "bus_name" in data and not data["bus_name"].map(cell_text).equals(
            bus_names.map(cell_text)
        ):
            raise ValueError("cap_data has conflicting legacy 'id' and 'bus_name'")
        data = data.copy()
        data["bus_name"] = bus_names
        data["device_name"] = [f"cap_{index}" for index in range(len(data))]
        data = data.drop(
            columns=[column for column in ("id", "name") if column in data]
        )
    return data


def validate_cap_data(data: pd.DataFrame, ctx: NetworkContext) -> None:
    """Validate capacitor names, connections, and nominal reactive power."""
    if data.empty:
        return
    check_columns(
        data,
        "cap_data",
        _REQUIRED,
        _KNOWN,
        {"id", "name", "bus_id"},
        "capacitors use 'device_name' and 'bus_name'",
    )
    errors: list[str] = []
    check_device_names(data, errors)
    for index, row in data.iterrows():
        label = row_label(row, index)
        phases = check_phases(label, row["phases"], _PHASES, errors)
        if phases is None:
            continue
        check_connection(label, cell_text(row["bus_name"]), phases, ctx, errors)
        for phase in phases:
            cell_required_float(row, f"q_{phase}", label, errors, ge=0)
    finish_validation("cap_data", errors, [], [])


def create_capacitor_parameters(model: Any, case: Any) -> None:
    """Create capacitor parameters from case data."""
    q_data = {
        (cell_text(row["device_name"]), phase): float(row[f"q_{phase}"])
        for _, row in _capacitor_data(model, case).iterrows()
        for phase in parse_phases(cell_text(row["phases"]))
    }
    model.cap_q_nom = ParamDict(q_data)


class CapacitorProvider:
    """Own capacitor components, voltage-dependent constraints, and Q injection."""

    name = "capacitors"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        ctx = NetworkContext.from_model(model, "CapacitorProvider")
        data = _capacitor_data(model, case)
        validate_cap_data(data, ctx)
        bus_by_device: dict[str, int] = {}
        ports: list[tuple[str, str]] = []
        devices_by_bus_phase: dict[tuple[int, str], list[str]] = {}
        for _, row in data.iterrows():
            device = cell_text(row["device_name"])
            bus = ctx.bus_name_to_id[cell_text(row["bus_name"])]
            bus_by_device[device] = bus
            for phase in parse_phases(cell_text(row["phases"])):
                ports.append((device, phase))
                devices_by_bus_phase.setdefault((bus, phase), []).append(device)
        model.cap_device_set = IndexSet(list(bus_by_device), dimen=1)
        model.cap_device_phase_set = IndexSet(ports, dimen=2)
        model.cap_bus_by_device = bus_by_device
        model.cap_devices_by_bus_phase = devices_by_bus_phase
        create_capacitor_parameters(model, case)
        keys = index_product(model.cap_device_phase_set, model.time_set)
        model.q_cap = VarDict(keys, initialize=0.0)
        if getattr(model, "cap_mi_enabled", False):
            model.u_cap = VarDict(keys, boolean=True, initialize=1.0)
            model.z_cap = VarDict(keys, nonneg=True, initialize=1.0)

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return 0

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return sum(
            model.q_cap[device, phase, time]
            for device in model.cap_devices_by_bus_phase.get((bus, phase), [])
        )

    def add_constraints(self, model: Any, config: Any) -> None:
        if len(model.cap_device_phase_set) == 0:
            return
        if getattr(model, "cap_mi_enabled", False):
            add_capacitor_mi_constraints(model)
            add_capacitor_mccormick_constraints(model)
            add_capacitor_z_bounds(model)
        else:
            add_capacitor_constraints(model)


def add_capacitor_constraints(m: CvxModel) -> None:
    """Scale each device's nominal Q by its bus's squared voltage."""

    def capacitor_rule(m, device, ph, t):
        bus = m.cap_bus_by_device[device]
        return m.q_cap[device, ph, t] == m.cap_q_nom[device, ph] * m.v2[bus, ph, t]

    add_constraint(
        m, "capacitor_injection", (m.cap_device_phase_set, m.time_set), capacitor_rule
    )


def add_capacitor_mi_constraints(m: CvxModel) -> None:
    """Set Q = nominal Q * z, where z represents switch state * squared voltage."""

    def capacitor_q_rule(m, device, ph, t):
        return (
            m.q_cap[device, ph, t] == m.cap_q_nom[device, ph] * m.z_cap[device, ph, t]
        )

    add_constraint(
        m,
        "capacitor_mi_injection",
        (m.cap_device_phase_set, m.time_set),
        capacitor_q_rule,
    )


def add_capacitor_mccormick_constraints(m: CvxModel) -> None:
    """Enforce z = u * v2 for binary u using the connected bus's voltage limits."""

    def mccormick_upper_1(m, device, ph, t):
        v2_max = m.v_max[m.cap_bus_by_device[device], ph] ** 2
        return m.z_cap[device, ph, t] <= v2_max * m.u_cap[device, ph, t]

    def mccormick_upper_2(m, device, ph, t):
        return m.z_cap[device, ph, t] <= m.v2[m.cap_bus_by_device[device], ph, t]

    def mccormick_lower_1(m, device, ph, t):
        v2_max = m.v_max[m.cap_bus_by_device[device], ph] ** 2
        return m.z_cap[device, ph, t] >= m.v2[
            m.cap_bus_by_device[device], ph, t
        ] - v2_max * (1 - m.u_cap[device, ph, t])

    def mccormick_lower_2(m, device, ph, t):
        v2_min = m.v_min[m.cap_bus_by_device[device], ph] ** 2
        return m.z_cap[device, ph, t] >= v2_min * m.u_cap[device, ph, t]

    index = (m.cap_device_phase_set, m.time_set)
    add_constraint(m, "cap_mccormick_u1", index, mccormick_upper_1)
    add_constraint(m, "cap_mccormick_u2", index, mccormick_upper_2)
    add_constraint(m, "cap_mccormick_l1", index, mccormick_lower_1)
    add_constraint(m, "cap_mccormick_l2", index, mccormick_lower_2)


def add_capacitor_z_bounds(m: CvxModel) -> None:
    """Bound each device's auxiliary variable between zero and its bus's v_max^2."""

    def z_cap_bounds(m, device, ph, t):
        v2_max = m.v_max[m.cap_bus_by_device[device], ph] ** 2
        return (0, m.z_cap[device, ph, t], v2_max)

    add_constraint(
        m, "z_cap_bounds", (m.cap_device_phase_set, m.time_set), z_cap_bounds
    )


__all__ = ["CapacitorProvider", "create_capacitor_parameters", "validate_cap_data"]
