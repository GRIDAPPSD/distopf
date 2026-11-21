"""Extract OpenDSS Capacitor elements into the GAD `cap_data` table (schema v1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from distopf.gad_models.common.schema import CAP_COLUMNS, empty_device_table

if TYPE_CHECKING:
    from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter

# Contract columns of `cap_data` (defined once in gad_models.common.schema).
COLUMNS = CAP_COLUMNS


def extract_capacitor(conv: GadDSSToCSVConverter) -> pd.DataFrame:
    """One row per OpenDSS Capacitor, with exactly the `cap_data` contract columns.

    Not implemented yet: returns an empty table.
    """
    return empty_device_table("cap_data")
