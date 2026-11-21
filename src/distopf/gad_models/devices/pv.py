"""PV device provider (GAD schema v1): one total P/Q decision per PV system.

CSV schema (case.pv_data, one row per OpenDSS PVSystem)
-------------------------------------------------------
Required columns:
    device_name     Unique name, e.g. "pvsystem.pv1".
    bus_name        AC bus; several PV systems may share a bus.
    phases          e.g. "abc", "a", "s1s2". The totals are split equally over them.
    pmpp            Panel rating at 1 pu irradiance, total (>= 0).
    s_max           Inverter rating, total (> 0).

Optional columns (blank = default):
    irradiance      Irradiance in pu (default 1).
    pmpp_limit      Cap on P as a fraction of pmpp (OpenDSS %Pmpp / 100, default 1).
    q               Reactive-power setpoint, total (default 0).
    q_min, q_max    Inverter reactive limits, total (default -s_max, s_max).
    control_variable  "PQ", "P", "Q", or blank (= NONE: P and Q fixed, no rating).
    pv_shape        Schedule column that scales P (and Q, see below). Blank is constant.
                    The name must match the column exactly.
    cost            Cost per unit of active power (default 0).
    type            Constant "pv".

Available P is min(pmpp * irradiance * pv_shape(t), pmpp_limit * pmpp); the cap
applies to P only. Fixed Q (modes NONE and P) is q * qmult(t), where qmult is the
schedule column "<pv_shape>.q" if it exists, otherwise the P multiplier, so Q is
proportional to P at a fixed power factor. Temperature and efficiency curves are
ignored.

Variables are p_pv[device, t] and q_pv[device, t] (totals); each phase of a device
injects total / number of phases. Limits, rating and control modes are shared with
generators (``devices/injector.py``).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from distopf.gad_models.common.device_data import (
    NetworkContext,
    cell_num,
    cell_opt_float,
    cell_required_float,
)
from distopf.gad_models.devices import injector
from distopf.gad_models.devices.injector import PowerTerms

_TABLE = "pv_data"
_PREFIX = "pv"


def _power_terms(row: pd.Series, label: str, errors: list[str]) -> PowerTerms | None:
    pmpp = cell_required_float(row, "pmpp", label, errors, ge=0.0)
    cell_opt_float(row, "irradiance", label, errors)  # records a non-numeric cell
    cell_opt_float(row, "pmpp_limit", label, errors)
    if pmpp is None:
        return None
    irradiance = cell_num(row, "irradiance", 1.0)
    pmpp_limit = cell_num(row, "pmpp_limit", 1.0)
    if irradiance < 0 or pmpp_limit < 0:
        errors.append(f"{label}: irradiance and pmpp_limit must be >= 0")
        return None
    return PowerTerms(scale=pmpp * irradiance, cap=pmpp_limit * pmpp, floor=0.0)


class PVProvider:
    """PV systems: available power from pmpp, irradiance and shape."""

    name = "pv"

    def create_components(self, model: Any, case: Any, config: Any) -> None:
        ctx = NetworkContext.from_model(model, "PVProvider")
        devices = injector.read_injectors(
            injector.device_table(case, _TABLE),
            _TABLE,
            "pv_shape",
            _power_terms,
            ctx,
            injector.case_schedules(case),
            limits_enforced=not injector.cfg(config, "equality_only", False),
        )
        injector.add_injector_components(model, _PREFIX, devices, ctx.times)

    def active_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return injector.injection(model, _PREFIX, "p", bus, phase, time)

    def reactive_power_injection(
        self, model: Any, bus: int, phase: str, time: Any
    ) -> Any:
        return injector.injection(model, _PREFIX, "q", bus, phase, time)

    def add_constraints(self, model: Any, config: Any) -> None:
        injector.add_injector_constraints(model, _PREFIX, config)


__all__ = ["PVProvider"]
