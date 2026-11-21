"""Extract OpenDSS regulator transformers into the GAD `reg_data` table (schema v1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from distopf.gad_models.common.schema import REG_COLUMNS, empty_device_table

if TYPE_CHECKING:
    from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter

# Contract columns of `reg_data` (defined once in gad_models.common.schema).
COLUMNS = REG_COLUMNS


def extract_regulator(conv: GadDSSToCSVConverter) -> pd.DataFrame:
    """One row per regulator Transformer, with exactly the `reg_data` contract columns.

    Not implemented yet: returns an empty table.
    """
    return empty_device_table("reg_data")
