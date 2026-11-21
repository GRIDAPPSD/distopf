"""Extract OpenDSS Generator elements into the GAD `gen_data` table (schema v1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from distopf.dss_importer.gad_devices.injector import (
    bus_and_phases,
    injector_table,
    read_property,
    register_shape,
    to_per_unit,
)
from distopf.gad_models.common.schema import GEN_COLUMNS, empty_device_table

if TYPE_CHECKING:
    from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter

# Contract columns of `gen_data` (defined once in gad_models.common.schema).
COLUMNS = GEN_COLUMNS

_VOLTAGE_REGULATING_MODEL = 3


def extract_generator(conv: GadDSSToCSVConverter) -> pd.DataFrame:
    """One row per OpenDSS Generator, with exactly the `gen_data` contract columns.

    ``p``, ``q`` and ``s_max`` are totals (per unit). ``q_min``/``q_max`` are
    OpenDSS's Minkvar/Maxkvar (default: -/+ 2 * kvar). Model=3 raises.
    """
    rows = []
    dss = conv.dss
    flag = dss.Generators.First()
    while flag:
        element = f"Generator.{dss.Generators.Name()}"
        device_name = element.lower()
        if dss.Generators.Model() == _VOLTAGE_REGULATING_MODEL:
            raise ValueError(
                f"{device_name}: Model=3 (voltage-regulating generator) is not supported"
            )
        bus_name, phases = bus_and_phases(conv, device_name)
        p = to_per_unit(conv, dss.Generators.kW())
        rows.append(
            dict(
                device_name=device_name,
                type="generator",
                bus_name=bus_name,
                phases=phases,
                p=p,
                s_max=to_per_unit(conv, dss.Generators.kVARated()),
                p_min=0.0,
                p_max=p,
                q=to_per_unit(conv, dss.Generators.kvar()),
                q_min=to_per_unit(conv, float(read_property(conv, element, "Minkvar"))),
                q_max=to_per_unit(conv, float(read_property(conv, element, "Maxkvar"))),
                control_variable="",
                gen_shape=register_shape(conv, element),
                cost=0.0,
            )
        )
        flag = dss.Generators.Next()
    return injector_table(rows, COLUMNS, empty_device_table("gen_data"))
