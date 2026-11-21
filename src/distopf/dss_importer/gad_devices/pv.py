"""Extract OpenDSS PVSystem elements into the GAD `pv_data` table (schema v1)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pandas as pd

from distopf.dss_importer.gad_devices.injector import (
    bus_and_phases,
    injector_table,
    read_property,
    register_shape,
    to_per_unit,
)
from distopf.gad_models.common.schema import PV_COLUMNS, empty_device_table

if TYPE_CHECKING:
    from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter

# Contract columns of `pv_data` (defined once in gad_models.common.schema).
COLUMNS = PV_COLUMNS


def _pv_reactive_power_kvar(conv: GadDSSToCSVConverter, available_kw: float) -> float:
    """Reactive power (kvar, generation positive) of the active PVSystem.

    With a power factor set (|pf| < 1) OpenDSS makes q = P * tan(acos(|pf|)),
    absorbing vars when pf < 0, at the present irradiance. The ``kvar`` property
    cannot be read directly because OpenDSS recomputes it from the irradiance at
    the current time step (0 at night), so it is derived here from the nominal
    available power (Pmpp * irradiance; %Pmpp caps P only). Otherwise ``kvar`` is
    the user's value. The model scales q with the PV shape, as OpenDSS does.
    """
    pf = conv.dss.PVsystems.pf()
    if abs(pf) >= 1:
        return conv.dss.PVsystems.kvar()
    return math.copysign(available_kw * math.tan(math.acos(abs(pf))), pf)


def extract_pv(conv: GadDSSToCSVConverter) -> pd.DataFrame:
    """One row per OpenDSS PVSystem, with exactly the `pv_data` contract columns.

    ``pmpp``, ``s_max``, ``q`` and the q limits are totals (per unit). Available
    power in the model is ``pmpp * irradiance * pv_shape(t)``, capped at
    ``pmpp_limit * pmpp`` (temperature and efficiency curves are ignored).
    """
    rows = []
    dss = conv.dss
    flag = dss.PVsystems.First()
    while flag:
        element = f"PVSystem.{dss.PVsystems.Name()}"
        device_name = element.lower()
        bus_name, phases = bus_and_phases(conv, device_name)
        pmpp_kw = dss.PVsystems.Pmpp()
        irradiance = dss.PVsystems.Irradiance()
        pmpp_limit = float(read_property(conv, element, "%Pmpp")) / 100
        rows.append(
            dict(
                device_name=device_name,
                type="pv",
                bus_name=bus_name,
                phases=phases,
                pmpp=to_per_unit(conv, pmpp_kw),
                s_max=to_per_unit(conv, dss.PVsystems.kVARated()),
                irradiance=irradiance,
                pmpp_limit=pmpp_limit,
                q=to_per_unit(
                    conv,
                    _pv_reactive_power_kvar(conv, pmpp_kw * irradiance),
                ),
                q_min=-to_per_unit(conv, float(read_property(conv, element, "kvarMaxAbs"))),
                q_max=to_per_unit(conv, float(read_property(conv, element, "kvarMax"))),
                control_variable="",
                pv_shape=register_shape(conv, element),
                cost=0.0,
            )
        )
        flag = dss.PVsystems.Next()
    return injector_table(rows, COLUMNS, empty_device_table("pv_data"))
