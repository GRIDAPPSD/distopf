"""Extract OpenDSS Generator elements into the GAD `gen_data` table (schema v1)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from distopf.gad_models.common.schema import GEN_COLUMNS, empty_device_table

if TYPE_CHECKING:
    from distopf.dss_importer.gad_dss_to_csv_converter import GadDSSToCSVConverter

# Contract columns of `gen_data` (defined once in gad_models.common.schema).
COLUMNS = GEN_COLUMNS


def extract_generator(conv: GadDSSToCSVConverter) -> pd.DataFrame:
    """One row per OpenDSS Generator, with exactly the `gen_data` contract columns.

    Not implemented yet: returns an empty table.
    """
    return empty_device_table("gen_data")
