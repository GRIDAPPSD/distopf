"""Extract OpenDSS Storage elements into the GAD `bat_data` table (schema v1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from distopf.gad_models.common.schema import BAT_COLUMNS, empty_device_table

if TYPE_CHECKING:
    from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter

# Contract columns of `bat_data` (defined once in gad_models.common.schema).
COLUMNS = BAT_COLUMNS


def extract_storage(conv: GadDSSToCSVConverter) -> pd.DataFrame:
    """One row per OpenDSS Storage, with exactly the `bat_data` contract columns.

    Not implemented yet: returns an empty table.
    """
    return empty_device_table("bat_data")
