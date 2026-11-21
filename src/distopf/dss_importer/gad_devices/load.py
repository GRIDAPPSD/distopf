"""Extract OpenDSS Load elements into the GAD `load_data` table (schema v1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from distopf.gad_models.common.schema import LOAD_COLUMNS, empty_device_table

if TYPE_CHECKING:
    from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter

# Contract columns of `load_data` (defined once in gad_models.common.schema).
COLUMNS = LOAD_COLUMNS


def extract_load(conv: GadDSSToCSVConverter) -> pd.DataFrame:
    """One row per OpenDSS Load, with exactly the `load_data` contract columns.

    Not implemented yet: returns an empty table.
    """
    return empty_device_table("load_data")
