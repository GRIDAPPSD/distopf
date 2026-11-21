"""Extract OpenDSS PVSystem elements into the GAD `pv_data` table (schema v1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from distopf.gad_models.common.schema import PV_COLUMNS, empty_device_table

if TYPE_CHECKING:
    from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter

# Contract columns of `pv_data` (defined once in gad_models.common.schema).
COLUMNS = PV_COLUMNS


def extract_pv(conv: GadDSSToCSVConverter) -> pd.DataFrame:
    """One row per OpenDSS PVSystem, with exactly the `pv_data` contract columns.

    Not implemented yet: returns an empty table.
    """
    return empty_device_table("pv_data")
