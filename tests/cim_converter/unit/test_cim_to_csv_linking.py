# tests/unit/test_cim_to_csv_linking.py
import pandas as pd
# import pytest
from distopf.cim_importer.cim_to_csv_converter import CIMToCSVConverter


def make_minimal_bus_df():
    return pd.DataFrame(
        [
            {"name": "sourcebus", "bus_type": "SWING", "mrid": "m1", "phases": "abc"},
            {"name": "650", "bus_type": "PQ", "mrid": "m2", "phases": "abc"},
            {"name": "rg60", "bus_type": "PQ", "mrid": "m3", "phases": "abc"},
        ]
    )


def make_minimal_branch_df():
    return pd.DataFrame(
        [
            {"from_name": "sourcebus", "to_name": "650", "name": "e1", "phases": "abc"},
            {"from_name": "650", "to_name": "rg60", "name": "e2", "phases": "abc"},
        ]
    )


def test_link_dataframes_assigns_ids_and_maps_fb_tb():
    conv = CIMToCSVConverter(cim_file="does_not_matter_for_test")
    bus_df = make_minimal_bus_df()
    branch_df = make_minimal_branch_df()
    data = dict(
        bus_data=bus_df,
        branch_data=branch_df,
        gen_data=pd.DataFrame(),
        cap_data=pd.DataFrame(),
        reg_data=pd.DataFrame(),
    )
    data_out = conv._link_dataframes(data)
    bus_df_out = data_out["bus_data"]
    branch_df_out = data_out["branch_data"]
    # ids should be ints and start at 1
    assert set(bus_df_out["id"]) == set([1, 2, 3])
    # fb and tb should exist and be ints on branch_df_out
    assert branch_df_out["fb"].dtype.kind in "iu"
    assert branch_df_out["tb"].dtype.kind in "iu"
    # The first node should be the swing bus id==1
    assert int(bus_df_out.loc[bus_df_out["name"] == "sourcebus", "id"].iloc[0]) == 1
